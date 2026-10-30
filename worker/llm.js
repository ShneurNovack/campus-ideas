// The language-model side of the edge. Every route here follows the same rules:
//   - the model never sees a sign-up count. The engine turns numbers into bands ("top fifth of this center's
//     events") before anything reaches a prompt, and `guard` strips any count-like number from what comes back.
//   - the model decides narrow things (is this the same event, which title is clearer, write one sentence); every
//     verdict that is a number is computed in code from the matched past events, never by the model.
//   - JSON is validated against a flat schema and the call is retried once with the error, then falls to a
//     second model. JSON mode on Workers AI is a request, not a guarantee.
const MODELS = { main: "@cf/google/gemma-4-26b-a4b-it", fallback: "@cf/qwen/qwen3-30b-a3b-fp8", reranker: "@cf/baai/bge-reranker-base" };

export const json = (data, status = 200, headers = {}) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json; charset=utf-8", "x-robots-tag": "noindex, nofollow", "cache-control": "no-store", ...headers } });

// ---- the privacy guard. Numbers next to sign-up words are the one thing that must never leave the model.
const COUNT_WORDS = /\b(rsvp|rsvps|sign[- ]?ups?|signed[- ]up|registr|attend|attendance|people came|showed up|turnout|guests|students came|headcount)\b/i;
export function guard(text) {
  if (typeof text !== "string") return text;
  let out = text;
  // a number within 40 characters of a sign-up word, in either direction
  out = out.replace(/(\d[\d,]{1,})(?=[^.]{0,40}\b(rsvp|rsvps|sign[- ]?ups?|signed[- ]up|registr|attend|attendance|turnout|headcount|showed up|people came|guests)\b)/gi, "[number removed]");
  out = out.replace(/\b(rsvp|rsvps|sign[- ]?ups?|signed[- ]up|registr\w*|attend\w*|turnout|headcount|showed up|people came|guests)\b([^.]{0,40}?)(\d[\d,]{1,})/gi, (m, a, b, c) => `${a}${b}[number removed]`);
  return out;
}
export const leaks = (text) => typeof text === "string" && /\d[\d,]{1,}/.test(text) && COUNT_WORDS.test(text) && /(\d[\d,]{1,})[^.]{0,40}(rsvp|sign|regist|attend|turnout|headcount|guests)|(rsvp|sign|regist|attend|turnout|headcount|guests)[^.]{0,40}\d[\d,]{1,}/i.test(text);

function tidy(t, n = 600) {
  return String(t || "").slice(0, n).replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, "").replace(/[\u0000-\u001F\u007F]/g, " ").trim();
}

// ---- one JSON call with validation, retry, and fallback model
async function callJson(env, { system, user, schema, check, maxTokens = 900, temperature = 0.2 }) {
  const attempts = [[MODELS.main, ""], [MODELS.main, "retry"], [MODELS.fallback, ""]];
  let lastErr = "";
  for (const [model, mode] of attempts) {
    const messages = [{ role: "system", content: system }, { role: "user", content: mode === "retry" ? `${user}\n\nYour previous answer was not valid: ${lastErr}. Answer again with only the JSON object.` : user }];
    // Gemma 4 thinks before it answers and spends the whole budget on the thought unless thinking is turned off
    // at the template; Qwen3 on Workers AI answers in JSON mode as it is (and goes blank with the same flag).
    const extra = model === MODELS.main ? { chat_template_kwargs: { enable_thinking: false } } : {};
    let res;
    try {
      res = await env.AI.run(model, { messages, max_tokens: maxTokens, temperature, response_format: { type: "json_schema", json_schema: schema }, ...extra });
    } catch (e) {
      try { res = await env.AI.run(model, { messages, max_tokens: maxTokens, temperature, ...extra }); } catch (e2) { lastErr = String(e2?.message || e2); continue; }
    }
    const raw = textOf(res);
    const parsed = parseJson(raw);
    if (!parsed) { lastErr = "not JSON"; continue; }
    const problem = check ? check(parsed) : null;
    if (problem) { lastErr = problem; continue; }
    return { data: parsed, model };
  }
  throw new Error(`the model did not return usable JSON (${lastErr})`);
}

// Workers AI answers in two shapes: {response} for the older models, OpenAI-style {choices[0].message.content}
// for the chat models. The content may also be a parsed object already.
function textOf(res) {
  if (!res) return "";
  if (typeof res === "string") return res;
  const c = res.choices?.[0]?.message?.content;
  if (typeof c === "string") return c;
  if (c && typeof c === "object") return JSON.stringify(c);
  if (typeof res.response === "string") return res.response;
  if (res.response && typeof res.response === "object") return JSON.stringify(res.response);
  return "";
}

function parseJson(raw) {
  if (!raw) return null;
  const s = String(raw).replace(/^\s*```(?:json)?\s*|\s*```\s*$/g, "").trim();
  try { return JSON.parse(s); } catch (_) { /* fall through */ }
  const m = s.match(/\{[\s\S]*\}/);
  if (m) { try { return JSON.parse(m[0]); } catch (_) { /* fall through */ } }
  return null;
}

// ---- POST /api/rerank { query, texts[] }  -> { scores[] }
// bge-reranker-base scores a query against each text; used on the top 50 of the browser's hybrid search.
export async function rerank(request, env) {
  const body = await request.json().catch(() => null);
  const query = tidy(body?.query, 300);
  const texts = Array.isArray(body?.texts) ? body.texts.slice(0, 60).map((t) => tidy(t, 400)) : null;
  if (!query || !texts?.length) return json({ error: "query and texts[] required" }, 400);
  const res = await env.AI.run(MODELS.reranker, { query, contexts: texts.map((text) => ({ text })) });
  const scores = new Array(texts.length).fill(0);
  for (const r of res?.response || []) if (typeof r.id === "number") scores[r.id] = r.score;
  return json({ scores });
}

// ---- POST /api/expand { text } -> { titles[] }
// Five to ten other ways centers might have titled the same event. Used to widen retrieval without pretending
// the model knows the corpus.
export async function expand(request, env) {
  const body = await request.json().catch(() => null);
  const text = tidy(body?.text, 200);
  if (!text) return json({ error: "text required" }, 400);
  const { data } = await callJson(env, {
    system: "You help search a list of Jewish campus event titles. Given an event idea, write other short titles a Chabad on Campus center might have used for the same kind of event: synonyms, Hebrew or Yiddish terms (Shabbat/Shabbos, Sukkot/Sukkos, BBQ/cookout/grill night), common branded names. Output JSON only.",
    user: `Idea: ${text}\n\nReturn {"titles": [8 short alternative titles]}`,
    schema: { type: "object", properties: { titles: { type: "array", items: { type: "string" } } }, required: ["titles"] },
    check: (d) => (Array.isArray(d.titles) && d.titles.length >= 3 ? null : "titles must be a list of at least 3"),
    maxTokens: 300, temperature: 0.4,
  });
  return json({ titles: data.titles.map((t) => tidy(t, 80)).filter(Boolean).slice(0, 10) });
}

// ---- POST /api/judge { idea, candidates: [{id, name, blurb}] } -> { labels: [{id, label, why}] }
// The one judgement the model makes: is each candidate the same program, a variant, or something different.
// Asked twice in opposite orders; a candidate keeps a label only when both orders agree, otherwise "unsure".
export async function judge(request, env) {
  const body = await request.json().catch(() => null);
  const idea = tidy(body?.idea, 300);
  const cands = Array.isArray(body?.candidates) ? body.candidates.slice(0, 12).map((c, i) => ({ id: String(c.id ?? i), name: tidy(c.name, 120), blurb: tidy(c.blurb, 220) })) : null;
  if (!idea || !cands?.length) return json({ error: "idea and candidates[] required" }, 400);
  const schema = { type: "object", properties: { labels: { type: "array", items: { type: "object", properties: { id: { type: "string" }, label: { type: "string", enum: ["same", "variant", "different"] }, why: { type: "string" } }, required: ["id", "label", "why"] } } }, required: ["labels"] };
  const system = "You compare a proposed campus event with past events from Jewish student centers. For each past event say whether it is the SAME program (same activity and format, wording may differ: a cookout is a BBQ), a VARIANT (same core activity with a different twist, audience or holiday), or DIFFERENT. Judge the activity, not the words. Give a reason of at most 12 words. Output JSON only.";
  const ask = (list) => callJson(env, {
    system, user: `Proposed event: ${idea}\n\nPast events:\n${list.map((c) => `[${c.id}] ${c.name}${c.blurb ? ` :: ${c.blurb}` : ""}`).join("\n")}\n\nReturn {"labels":[{"id":"...","label":"same|variant|different","why":"..."}]} with one entry per past event.`,
    schema, check: (d) => (Array.isArray(d.labels) && d.labels.length >= Math.min(list.length, 3) ? null : "labels must cover the past events"), maxTokens: 700, temperature: 0,
  });
  const [a, b] = await Promise.all([ask(cands), ask([...cands].reverse())]);
  const byId = (r) => Object.fromEntries((r.data.labels || []).map((x) => [String(x.id), x]));
  const A = byId(a), B = byId(b);
  const labels = cands.map((c) => {
    const x = A[c.id], y = B[c.id];
    const label = x && y && x.label === y.label ? x.label : x && !y ? x.label : y && !x ? y.label : "unsure";
    return { id: c.id, label, why: guard(tidy((x || y)?.why || "", 90)) };
  });
  return json({ labels, model: a.model });
}

// ---- POST /api/generate { brief, exemplars: [{name, hook, band}], avoid: [names], n } -> { ideas: [...] }
// Idea generation grounded in diverse past winners. Exemplars arrive with center names removed and outcomes
// as bands. The model writes titles and one-line pitches in a batch: list, make distinct, then expand.
export async function generate(request, env) {
  const body = await request.json().catch(() => null);
  const brief = body?.brief && typeof body.brief === "object" ? body.brief : null;
  const ex = Array.isArray(body?.exemplars) ? body.exemplars.slice(0, 14).map((e) => ({ name: tidy(e.name, 90), hook: tidy(e.hook, 160), band: tidy(e.band, 40) })) : [];
  const avoid = Array.isArray(body?.avoid) ? body.avoid.slice(0, 40).map((s) => tidy(s, 80)) : [];
  const n = Math.min(20, Math.max(6, Number(body?.n) || 12));
  if (!brief) return json({ error: "brief required" }, 400);
  const lines = [];
  if (brief.audience) lines.push(`Audience: ${tidy(brief.audience, 60)}`);
  if (brief.when) lines.push(`When: ${tidy(brief.when, 80)}`);
  if (brief.season) lines.push(`Time of year: ${tidy(brief.season, 80)}`);
  if (brief.kind) lines.push(`Kind of event: ${tidy(brief.kind, 60)}`);
  if (brief.vibe) lines.push(`Direction from the planner: ${tidy(brief.vibe, 200)}`);
  if (brief.center) lines.push(`About this center (bands only): ${tidy(brief.center, 300)}`);
  const persona = ["a sophomore who comes for the food and stays for the people", "a grad student with little free time", "a freshman who has never been to Chabad", "a student leader who wants to bring friends", "an Israeli student far from home", "a senior who has been to everything"][Math.floor(Math.random() * 6)];
  const schema = { type: "object", properties: { ideas: { type: "array", items: { type: "object", properties: { title: { type: "string" }, pitch: { type: "string" }, format: { type: "string" }, hook: { type: "string" }, builds_on: { type: "string" } }, required: ["title", "pitch", "format", "hook"] } } }, required: ["ideas"] };
  const { data, model } = await callJson(env, {
    system: `You write event ideas for a Chabad on Campus center. You think like ${persona}. Work in three steps inside your head: list ${n + 6} candidate titles, cross out any two that are too alike or that copy an example's title, then expand the best ${n}. Each idea: a title of at most 7 words, a one-sentence pitch a student would read on a flyer, the format (dinner, class, social, trip, service, cooking, speaker, holiday program), the hook (the one thing that makes it different), and which example it builds on, if any. No prices, no dates, no numbers. Write Shabbos, not Shabbat. Output JSON only.`,
    user: `${lines.join("\n")}\n\nPast events that did well at centers like this one (names only, outcome as a band):\n${ex.map((e) => `- ${e.name}${e.hook ? ` (${e.hook})` : ""}${e.band ? ` [${e.band}]` : ""}`).join("\n") || "- none given"}\n${avoid.length ? `\nAlready run here, do not repeat: ${avoid.join("; ")}\n` : ""}\nReturn {"ideas":[{"title","pitch","format","hook","builds_on"}]} with ${n} ideas.`,
    schema, check: (d) => (Array.isArray(d.ideas) && d.ideas.length >= Math.min(n, 5) && d.ideas.every((i) => i.title && i.pitch) ? null : `ideas must be a list of ${n} with title and pitch`), maxTokens: 1800, temperature: 0.8,
  });
  const ideas = data.ideas.slice(0, n).map((i) => ({ title: guard(tidy(i.title, 80)), pitch: guard(tidy(i.pitch, 220)), format: tidy(i.format, 30), hook: guard(tidy(i.hook, 120)), builds_on: tidy(i.builds_on, 90) }));
  return json({ ideas, model, persona });
}

// ---- POST /api/explain { idea, verdict, comparables: [{name, label}] } -> { text }
// One paragraph in plain words, from a verdict that code already computed. The model gets bands, never counts.
export async function explain(request, env) {
  const body = await request.json().catch(() => null);
  const idea = tidy(body?.idea, 300), verdict = tidy(body?.verdict, 400);
  const comps = Array.isArray(body?.comparables) ? body.comparables.slice(0, 8).map((c) => `${tidy(c.name, 80)} (${tidy(c.label, 20)})`) : [];
  if (!idea || !verdict) return json({ error: "idea and verdict required" }, 400);
  const { data } = await callJson(env, {
    system: "You write two plain sentences for a campus rabbi about an event idea, from facts you are given. Do not invent facts. Never write any number. Write Shabbos, not Shabbat. Output JSON only.",
    user: `Idea: ${idea}\nWhat the data says: ${verdict}\nClosest past events: ${comps.join("; ") || "none"}\n\nReturn {"text": "two sentences"}`,
    schema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    check: (d) => (typeof d.text === "string" && d.text.length > 20 ? null : "text missing"), maxTokens: 220, temperature: 0.3,
  });
  const text = guard(tidy(data.text, 500));
  return json({ text: leaks(text) ? "" : text });
}

export const LLM_MODELS = MODELS;
