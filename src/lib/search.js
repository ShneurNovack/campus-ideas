// Retrieval, in a cascade. Each step is cheap and narrow; nothing here decides a verdict.
//   1. meaning: one 256-dim int8 vector per program (vectors.bin); the typed idea is embedded by the Worker (bge-m3)
//      and compared with every program here, a few milliseconds.
//   2. wording: character n-grams of program names, built here, so exact phrases carry over the line.
//   3. rerank: the top 50 go to a cross-encoder (bge-reranker-base) on the Worker, which reads idea and program
//      together and reorders them.
//   4. judge: the top 10 are labeled same / variant / different by the language model, asked in both orders;
//      only labels that agree are kept.
// The verdict ("12 centers ran this; it tends to run above their usual") is then counted in code from the matches.
import { load, world } from './data';

export const LAM = 0.4;
export const CLOSE = 0.60, RELATED = 0.45, MATCH = 0.36;
export const blend = (sim, wording) => (sim + LAM * wording) / (1 + LAM);
export const strength = (sim, wording) => Math.min(1, sim + (wording >= 0.5 ? LAM * wording : 0));
export const isMatch = (sim, wording) => sim >= RELATED || blend(sim, wording) >= MATCH;

let ready = null;
export function searchIndex() {
  if (!ready) {
    ready = Promise.all([load('vectors.bin'), load('proj.bin'), world()]).then(([buf, projBuf, W]) => {
      const head = new Float32Array(projBuf, 0, 2), dims = head[1];
      const M = new Int8Array(buf), n = M.length / dims;
      // vectors.bin rows follow P (every program, including the few the app does not list); map pid -> row
      const norm = new Float32Array(n);
      for (let i = 0; i < n; i++) { let s = 0; const o = i * dims; for (let j = 0; j < dims; j++) s += M[o + j] * M[o + j]; norm[i] = Math.sqrt(s) || 1; }
      const rows = W.programs.map((p) => p.pid);        // pid is the row index in P
      const names = W.programs.map((p) => p.name);
      return { M, n, dims, norm, W, rows, lex: wordingIndex(names) };
    }).catch((e) => { ready = null; throw e; });
  }
  return ready;
}

// ---------------------------------------------------------------- wording
const NGRAM = [3, 4, 5], MAX_DF = 0.3, MIN_DF = 2, WHOLE = 2;
function grams(text, add) {
  const words = (text || '').toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}]+/gu, ' ').trim().split(' ');
  for (const w of words) {
    if (!w) continue;
    const p = ` ${w} `;
    for (const k of NGRAM) for (let a = 0; a + k <= p.length; a++) add(p.slice(a, a + k));
    if (WHOLE && p.length > 5) add(p);
  }
}
const count = (text) => { const m = new Map(); grams(text, (g) => m.set(g, (m.get(g) || 0) + 1)); return m; };
export function wordingIndex(names) {
  const n = names.length, df = new Map();
  const per = names.map((nm) => { const m = count(nm); for (const g of m.keys()) df.set(g, (df.get(g) || 0) + 1); return m; });
  const Ids = n < 65536 ? Uint16Array : Uint32Array, post = new Map();
  for (const [g, d] of df) if (d >= MIN_DF && d <= MAX_DF * n) post.set(g, { ids: new Ids(d), tf: null, idf: (Math.log((1 + n) / (1 + d)) + 1) * (g.length > 5 ? WHOLE : 1), k: 0 });
  const len = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const [g, tf] of per[i]) {
      const e = post.get(g);
      if (!e) continue;
      if (tf > 1) { if (!e.tf) e.tf = new Uint8Array(e.ids.length).fill(1); e.tf[e.k] = Math.min(255, tf); }
      e.ids[e.k++] = i;
      const w = (1 + Math.log(tf)) * e.idf; s += w * w;
    }
    len[i] = Math.sqrt(s) || 1;
  }
  return { n, post, len };
}
export function wordingMatch(ix, text) {
  const { n, post, len } = ix.lex, out = new Float32Array(n);
  let qs = 0;
  for (const [g, tf] of count(text)) {
    const e = post.get(g);
    if (!e) continue;
    const w = (1 + Math.log(tf)) * e.idf, { ids, tf: tfs } = e;
    qs += w * w;
    if (tfs) for (let k = 0; k < ids.length; k++) out[ids[k]] += w * (1 + Math.log(tfs[k])) * e.idf;
    else { const wd = w * e.idf; for (let k = 0; k < ids.length; k++) out[ids[k]] += wd; }
  }
  if (!qs) return out;
  qs = Math.sqrt(qs);
  for (let i = 0; i < n; i++) if (out[i]) out[i] = Math.min(1, out[i] / (qs * len[i]));
  return out;
}

// ---------------------------------------------------------------- meaning
const memo = new Map();
export async function embed(text) {
  const key = text.trim().toLowerCase();
  if (memo.has(key)) return memo.get(key);
  const res = await fetch('/api/embed', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: text.trim() }) });
  if (!res.ok) throw new Error(res.status === 429 ? 'Too many searches at once. Wait a moment and try again.' : `The search service did not answer (${res.status})`);
  const { v } = await res.json();
  const bin = atob(v), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const vec = new Float32Array(bytes.buffer);
  memo.set(key, vec);
  return vec;
}
// cosine of q against every listed program (index = position in W.programs)
export function scoreAll(ix, q) {
  const { M, dims, norm, rows } = ix, out = new Float32Array(rows.length);
  let ql = 0; for (let j = 0; j < dims; j++) ql += q[j] * q[j];
  ql = Math.sqrt(ql) || 1;
  for (let a = 0; a < rows.length; a++) { const i = rows[a]; let s = 0; const o = i * dims; for (let j = 0; j < dims; j++) s += M[o + j] * q[j]; out[a] = s / (norm[i] * ql); }
  return out;
}
export function vectorOfRow(ix, i) {
  const { M, dims, norm } = ix, v = new Float32Array(dims), o = i * dims;
  for (let j = 0; j < dims; j++) v[j] = M[o + j] / norm[i];
  return v;
}
export function meanVector(ix, pids) {
  const { dims } = ix, v = new Float32Array(dims);
  let k = 0;
  for (const pid of pids) { if (pid == null || pid >= ix.n) continue; const u = vectorOfRow(ix, pid); for (let j = 0; j < dims; j++) v[j] += u[j]; k++; }
  if (!k) return null;
  let s = 0; for (let j = 0; j < dims; j++) s += v[j] * v[j];
  s = Math.sqrt(s) || 1; for (let j = 0; j < dims; j++) v[j] /= s;
  return v;
}

export async function searchText(text, { limit = 60, filter = null, lexical = LAM } = {}) {
  const [ix, q] = await Promise.all([searchIndex(), embed(text)]);
  const sims = scoreAll(ix, q), words = lexical ? wordingMatch(ix, text) : null;
  const idx = [];
  for (let a = 0; a < sims.length; a++) if (!filter || filter(ix.W.programs[a])) idx.push(a);
  const score = (a) => (words ? (sims[a] + lexical * words[a]) / (1 + lexical) : sims[a]);
  idx.sort((a, b) => score(b) - score(a));
  return { hits: idx.slice(0, limit).map((a) => ({ a, p: ix.W.programs[a], sim: sims[a], wording: words ? words[a] : 0, score: score(a) })), sims, q, ix };
}

// ---------------------------------------------------------------- the Worker's narrow model calls
async function post(path, body) {
  const res = await fetch(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || (res.status === 429 ? 'Too many requests at once. Wait a moment.' : `The model did not answer (${res.status})`)); }
  return res.json();
}
export const rerank = (query, texts) => post('/api/rerank', { query, texts });
export const judge = (idea, candidates) => post('/api/judge', { idea, candidates });
export const expand = (text) => post('/api/expand', { text });
export const generate = (brief, exemplars, avoid, n) => post('/api/generate', { brief, exemplars, avoid, n });
export const explain = (idea, verdict, comparables) => post('/api/explain', { idea, verdict, comparables });

// The full cascade for one typed idea. Returns matches grouped by the judge's label, plus the raw hits.
// Steps 3 and 4 are skipped (with a note) when the Worker is busy, so the page still answers from steps 1 and 2.
export async function cascade(text, { onStep } = {}) {
  onStep?.('Reading past events');
  const r = await searchText(text, { limit: 60 });
  const top = r.hits.filter((h) => isMatch(h.sim, h.wording)).slice(0, 50);
  let ordered = top, notes = [];
  if (top.length) {
    onStep?.('Reading the closest ones more carefully');
    try {
      const { scores } = await rerank(text, top.map((h) => `${h.p.name}${h.p.short ? `. ${h.p.short}` : ''}`));
      ordered = top.map((h, k) => ({ ...h, rr: scores[k] })).sort((a, b) => (b.rr - a.rr) || (b.score - a.score));
    } catch (e) { notes.push('ranked by meaning only'); }
  }
  const ten = ordered.slice(0, 10);
  let labels = new Map();
  if (ten.length) {
    onStep?.('Deciding which are the same event');
    try {
      const { labels: L } = await judge(text, ten.map((h) => ({ id: String(h.p.pid), name: h.p.name, blurb: h.p.short })));
      for (const l of L) labels.set(Number(l.id), l);
    } catch (e) { notes.push('matches judged by distance only'); for (const h of ten) labels.set(h.p.pid, { label: strength(h.sim, h.wording) >= CLOSE ? 'same' : 'variant', why: '' }); }
  }
  const same = [], variant = [], different = [];
  for (const h of ten) {
    const l = labels.get(h.p.pid)?.label || 'unsure';
    (l === 'same' ? same : l === 'variant' ? variant : different).push({ ...h, label: l, why: labels.get(h.p.pid)?.why || '' });
  }
  // the rest of the related hits beyond the judged ten, by distance alone
  const more = ordered.slice(10).filter((h) => strength(h.sim, h.wording) >= RELATED);
  return { same, variant, different, more, hits: r.hits, q: r.q, sims: r.sims, ix: r.ix, notes };
}
