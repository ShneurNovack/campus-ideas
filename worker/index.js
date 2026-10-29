// Campus Ideas edge.
//   GET  /data/<key>            public blobs the engine wrote (page payloads, the event catalog, the vector file)
//   POST /api/embed {text}      one query -> multilingual embedding (Workers AI, bge-m3) projected into the search space
//   POST /analyst/embed         batch embeddings for the offline engine, raw float32
//   PUT  /analyst/put?key=      the engine stores a blob        GET /analyst/get?key=  reads one back
//   GET  /health
// Everything else is the app (static assets). The Worker does no analysis. The engine (engine/, Python, run by
// GitHub Actions) computes everything and writes blobs; the browser does the vector search itself.
// /analyst/* needs the ANALYST_KEY secret in x-analyst-key, or a GitHub OIDC token from this repo's workflow.

import { verifyGithubToken } from "./oidc.js";
import { rerank, expand, judge, generate, explain, LLM_MODELS } from "./llm.js";

// Embedding models the engine may ask for. The engine tests them against each other on its own data and names
// the winner in public/embed.json, which the public search route then follows.
const MODELS = { "bge-m3": "@cf/baai/bge-m3", "qwen3": "@cf/qwen/qwen3-embedding-0.6b", "gemma": "@cf/google/embeddinggemma-300m", "bge-large": "@cf/baai/bge-large-en-v1.5" };
const MODEL = MODELS["bge-m3"];
const json = (data, status = 200, headers = {}) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json; charset=utf-8", "x-robots-tag": "noindex, nofollow", ...headers } });

// Slicing can cut an emoji in half, and the model rejects malformed text. Drop lone surrogates and control characters.
function tidy(t, n = 900) {
  const s = String(t || "").slice(0, n)
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, "")
    .replace(/[\u0000-\u001F\u007F]/g, " ").trim();
  return s || "untitled";
}

function sameKey(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

async function authed(request, env) {
  const key = request.headers.get("x-analyst-key");
  if (key && env.ANALYST_KEY && sameKey(key, env.ANALYST_KEY)) return true;
  const auth = request.headers.get("authorization") || "";
  if (auth.startsWith("Bearer ")) return !!(await verifyGithubToken(auth.slice(7), env));
  return false;
}

async function embedMany(env, texts, MODEL = MODELS["bge-m3"]) {
  let out = null, dims = 0;
  let n = 0;
  for (let i = 0; i < texts.length; i += 32) {
    const batch = texts.slice(i, i + 32);
    let rows;
    try { rows = (await env.AI.run(MODEL, { text: batch })).data; }
    catch (_) {
      // One bad input should not sink the batch: embed one by one and fall back to a plain version of the text.
      rows = [];
      for (const t of batch) {
        let res;
        try { res = await env.AI.run(MODEL, { text: [t] }); } catch (_e) { res = await env.AI.run(MODEL, { text: [t.replace(/[^\p{L}\p{N}\s.,'-]/gu, " ").trim() || "untitled"] }); }
        rows.push(res.data[0]);
      }
    }
    if (!out) { dims = rows[0].length; out = new Float32Array(texts.length * dims); }
    for (const v of rows) { out.set(v, n * dims); n++; }
  }
  return { out, dims };
}

// The projection the engine fitted: mean (1024) and components (k x 1024), float32, written to public/proj.bin.
// Layout: [d_in, k, mean..., components row-major...]. Loaded once per isolate and refreshed when the file changes.
// A second copy can sit under public/next/ so a new engine output can be tried (?space=next) before it replaces the live one.
const projs = {};
async function projection(env, dir = "public/") {
  const now = Date.now();
  let proj = projs[dir] || { at: 0, etag: "", mean: null, comp: null, k: 0 };
  if (proj.comp && now - proj.at < 300e3) return proj;
  const head = await env.BUCKET.head(dir + "proj.bin");
  if (!head) return null;
  if (proj.comp && head.etag === proj.etag) { proj.at = now; return proj; }
  const obj = await env.BUCKET.get(dir + "proj.bin");
  const f = new Float32Array(await obj.arrayBuffer());
  const d = f[0] | 0, k = f[1] | 0;
  let model = MODEL;
  try { const cfg = await env.BUCKET.get(dir + "embed.json"); if (cfg) { const c = await cfg.json(); if (MODELS[c.model]) model = MODELS[c.model]; } } catch (_) { /* keep the default */ }
  proj = { at: now, etag: head.etag, mean: f.subarray(2, 2 + d), comp: f.subarray(2 + d, 2 + d + k * d), k, d, model };
  projs[dir] = proj;
  return proj;
}

const hits = new Map(); // best-effort per-isolate rate limit for the public embed route
function limited(ip) {
  const now = Date.now(), list = (hits.get(ip) || []).filter((t) => now - t < 60e3);
  list.push(now); hits.set(ip, list);
  if (hits.size > 5000) hits.clear();
  return list.length > 60;
}

async function queryEmbed(request, env) {
  const body = await request.json().catch(() => null);
  const text = tidy(body?.text, 600);
  if (!body?.text || text === "untitled") return json({ error: "text required" }, 400);
  const p = await projection(env, new URL(request.url).searchParams.get("space") === "next" ? "public/next/" : "public/");
  if (!p) return json({ error: "engine has not published a projection yet" }, 503);
  const v = (await env.AI.run(p.model || MODEL, { text: [text] })).data[0];
  if (v.length !== p.d) return json({ error: "the published projection does not match the model" }, 503);
  // the engine works with unit vectors, so bring the query to unit length before the engine's mean is taken off
  { let n0 = 0; for (let i = 0; i < v.length; i++) n0 += v[i] * v[i]; n0 = Math.sqrt(n0) || 1; for (let i = 0; i < v.length; i++) v[i] /= n0; }
  const out = new Float32Array(p.k);
  let norm = 0;
  for (let j = 0; j < p.k; j++) {
    let s = 0; const row = j * p.d;
    for (let i = 0; i < p.d; i++) s += (v[i] - p.mean[i]) * p.comp[row + i];
    out[j] = s; norm += s * s;
  }
  norm = Math.sqrt(norm) || 1;
  for (let j = 0; j < p.k; j++) out[j] /= norm;
  const bytes = new Uint8Array(out.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return json({ model: p.model || MODEL, dims: p.k, v: btoa(bin) }, 200, { "cache-control": "private, max-age=3600" });
}

async function data(request, env, url) {
  const key = decodeURIComponent(url.pathname.slice("/data/".length));
  if (!key || key.includes("..")) return json({ error: "bad key" }, 400);
  const obj = await env.BUCKET.get("public/" + key, { onlyIf: request.headers });
  if (!obj) return json({ error: "not found" }, 404);
  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set("etag", obj.httpEtag);
  headers.set("cache-control", url.searchParams.has("v") ? "public, max-age=31536000, immutable" : "public, max-age=60, stale-while-revalidate=600");
  headers.set("x-robots-tag", "noindex, nofollow");
  if (!obj.body) return new Response(null, { status: 304, headers });
  return new Response(obj.body, { headers });
}

async function analyst(request, env, url) {
  if (!(await authed(request, env))) return json({ error: "unauthorized" }, 401);
  const key = url.searchParams.get("key") || "";
  const okKey = /^(public|private)\/[\w.\-\/]+$/.test(key) && !key.includes("..");
  if (url.pathname === "/analyst/embed" && request.method === "POST") {
    const body = await request.json().catch(() => null);
    const texts = Array.isArray(body?.texts) ? body.texts.slice(0, 96).map((t) => tidy(t)) : null;
    if (!texts || !texts.length) return json({ error: "texts[] required" }, 400);
    const model = MODELS[body.model || "bge-m3"];
    if (!model) return json({ error: "unknown model", models: Object.keys(MODELS) }, 400);
    const { out, dims } = await embedMany(env, texts, model);
    return new Response(out.buffer, { headers: { "content-type": "application/octet-stream", "x-count": String(texts.length), "x-dims": String(dims) } });
  }
  if (url.pathname === "/analyst/put" && request.method === "PUT") {
    if (!okKey) return json({ error: "bad key" }, 400);
    const put = await env.BUCKET.put(key, request.body, { httpMetadata: { contentType: request.headers.get("content-type") || "application/octet-stream" } });
    return json({ ok: true, key, size: put.size, etag: put.etag });
  }
  if (url.pathname === "/analyst/get" && request.method === "GET") {
    if (!okKey) return json({ error: "bad key" }, 400);
    const obj = await env.BUCKET.get(key);
    if (!obj) return json({ error: "not found" }, 404);
    const headers = new Headers(); obj.writeHttpMetadata(headers);
    return new Response(obj.body, { headers });
  }
  if (url.pathname === "/analyst/list" && request.method === "GET") {
    const list = await env.BUCKET.list({ prefix: url.searchParams.get("prefix") || "", limit: 1000 });
    return json({ objects: list.objects.map((o) => ({ key: o.key, size: o.size, uploaded: o.uploaded })) });
  }
  return json({ error: "not found" }, 404);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      if (url.pathname === "/health") return json({ ok: true, service: "campus-ideas", embed: MODEL, llm: LLM_MODELS });
      if (url.pathname.startsWith("/data/") && request.method === "GET") return await data(request, env, url);
      if (url.pathname === "/api/embed" && request.method === "POST") return await queryEmbed(request, env);
      if (request.method === "POST" && url.pathname.startsWith("/api/")) {
        if (limited(request.headers.get("cf-connecting-ip") || "x")) return json({ error: "slow down" }, 429);
        if (url.pathname === "/api/rerank") return await rerank(request, env);
        if (url.pathname === "/api/expand") return await expand(request, env);
        if (url.pathname === "/api/judge") return await judge(request, env);
        if (url.pathname === "/api/generate") return await generate(request, env);
        if (url.pathname === "/api/explain") return await explain(request, env);
      }
      if (url.pathname.startsWith("/analyst/")) return await analyst(request, env, url);
      if (url.pathname.startsWith("/api/")) return json({ error: "not found" }, 404);
      return env.ASSETS.fetch(request);
    } catch (e) {
      return json({ error: String(e?.message || e) }, 500);
    }
  },
};
