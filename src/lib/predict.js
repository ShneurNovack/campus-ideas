// What a typed idea would do at the chosen center, from the engine's published expectations.
// The engine ran the neighbor system (engine/neighbors.py) for every center against every idea with four or more
// centers behind it and stored where each would land among that center's own events, as a percentile. For a typed
// idea the browser finds the ideas it is closest to (cosine on the search vectors, to the 4th power) and takes the
// weighted percentile. The answer is a band in words, never a number.
import { meanVector, scoreAll } from './search';
import { bandOf } from './data';

const POWER = 4, K = 12;
let ideaVecs = null;
function ideaVectors(ix) {
  if (ideaVecs && ideaVecs.ix === ix) return ideaVecs;
  const { W } = ix, list = [];
  for (const id of W.predIdeas) {
    const i = W.ideaById.get(id);
    if (!i) continue;
    const v = meanVector(ix, i.rep);
    if (v) list.push({ id, v, i });
  }
  ideaVecs = { ix, list };
  return ideaVecs;
}

// cosine of q against every predictable idea
export function ideaSims(ix, q) {
  const { list } = ideaVectors(ix), out = new Float32Array(list.length);
  let ql = 0; for (let j = 0; j < q.length; j++) ql += q[j] * q[j];
  ql = Math.sqrt(ql) || 1;
  for (let a = 0; a < list.length; a++) { const v = list[a].v; let s = 0; for (let j = 0; j < v.length; j++) s += v[j] * q[j]; out[a] = s / ql; }
  return out;
}

// { pct, band, lo, hi, near: [{idea, sim, w}], weight } or null when nothing is close enough / no center
export function expectAt(ix, centerId, q) {
  if (!centerId || !q) return null;
  const { list } = ideaVectors(ix), sims = ideaSims(ix, q);
  const order = Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, K);
  let sw = 0, sp = 0, slo = 0, shi = 0; const near = [];
  for (const a of order) {
    const s = sims[a];
    if (s < 0.3) continue;
    const p = ix.W.pred(centerId, list[a].id);
    if (!p) continue;
    const w = s ** POWER;
    sw += w; sp += w * p.pct; slo += w * p.lo; shi += w * p.hi;
    near.push({ idea: list[a].i, sim: s, w, pct: p.pct });
  }
  if (!sw || near.length === 0) return null;
  const pct = sp / sw;
  return { pct, band: bandOf(pct), lo: slo / sw, hi: shi / sw, near, weight: sw, sure: near[0].sim >= 0.6 ? 'close' : near[0].sim >= 0.45 ? 'related' : 'loose' };
}

// The kind of event a text is most like: majority among the 20 closest programs, weighted by cosine.
export function kindOf(ix, sims) {
  const idx = Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, 20);
  const tally = new Map();
  for (const a of idx) { const t = ix.W.programs[a].type; tally.set(t, (tally.get(t) || 0) + Math.max(0, sims[a])); }
  let best = null, bv = -1;
  for (const [t, v] of tally) if (v > bv) { best = t; bv = v; }
  return best;
}

// How the matched past events landed at their own centers: counted, never averaged from sign-ups.
export function tally(matches) {
  const bands = { top: 0, upper: 0, lower: 0, bottom: 0 }, centers = new Set(), years = new Set();
  let measured = 0;
  for (const h of matches) {
    centers.add(h.p.center);
    for (const y of h.p.years || []) years.add(y);
    if (h.p.band) { bands[h.p.band]++; measured++; }
  }
  const high = bands.top + bands.upper;
  return { centers: centers.size, measured, high, bands, years: [...years].sort(), shareHigh: measured ? high / measured : null,
    read: measured < 3 ? null : high / measured >= 0.7 ? 'Usually one of a center\'s bigger events' : high / measured >= 0.5 ? 'Often above a center\'s average' : high / measured >= 0.3 ? 'Mixed results' : 'Usually one of a center\'s smaller events' };
}
