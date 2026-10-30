// Data layer. The engine writes a handful of files; the app reads them once. Nothing in these files is a sign-up
// count: centers come as bands and neighbor lists, outcomes as evidence tiers and effect bands, expectations as a
// percentile band per center and idea (pred.bin). The app shows none of a center's bands to anyone.
import { useEffect, useState } from 'react';

const cache = new Map();
export function load(name) {
  if (!cache.has(name)) {
    cache.set(name, fetch(`/data/${name}`).then((r) => {
      if (!r.ok) throw new Error(`${name} did not load (${r.status})`);
      return name.endsWith('.bin') ? r.arrayBuffer() : r.json();
    }).catch((e) => { cache.delete(name); throw e; }));
  }
  return cache.get(name);
}

let worldP = null;
// Everything the pages share: ideas, programs, centers, predictions, with lookups.
export function world() {
  if (!worldP) {
    worldP = Promise.all([load('ideas.json'), load('programs.json'), load('centers.json'), load('pred.bin'), load('engine.json')]).then(([I, Pr, C, predBuf, engine]) => {
      const ideas = I.ideas, programs = Pr.programs, centers = C.centers;
      const ideaById = new Map(ideas.map((i) => [i.id, i]));
      const programById = new Map(programs.map((p) => [p.pid, p]));
      const centerById = new Map(centers.map((c) => [c.id, c]));
      const byCenter = new Map();
      for (const p of programs) { if (!byCenter.has(p.center)) byCenter.set(p.center, []); byCenter.get(p.center).push(p); }
      // predictions: three uint8 planes (center x candidate idea): the percentile, and the 25th and 75th of the voters
      const n = C.pred.ids.length, m = C.pred.ideas.length, U = new Uint8Array(predBuf);
      const cpos = new Map(C.pred.ids.map((id, k) => [id, k]));
      const ipos = new Map(C.pred.ideas.map((id, k) => [id, k]));
      const pred = (centerId, ideaId) => {
        const a = cpos.get(centerId), b = ipos.get(ideaId);
        if (a == null || b == null) return null;
        const v = U[a * m + b];
        if (v === C.pred.none) return null;
        return { pct: v / C.pred.scale, lo: U[n * m + a * m + b] / C.pred.scale, hi: U[2 * n * m + a * m + b] / C.pred.scale };
      };
      const types = I.types;
      return { ideas, ideaById, programs, programById, centers, centerById, byCenter, pred, predIdeas: C.pred.ideas, bands: C.pred.bands, types, tiers: I.tiers, effectBands: I.bands, facets: C.facets, engine };
    }).catch((e) => { worldP = null; throw e; });
  }
  return worldP;
}

export function useWorld() {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  useEffect(() => { let live = true; world().then((data) => live && setState({ data, error: null, loading: false })).catch((error) => live && setState({ data: null, error, loading: false })); return () => { live = false; }; }, []);
  return state;
}
export function useData(...names) {
  const key = names.join('|');
  const [state, setState] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    Promise.all(names.map((n) => load(n))).then((data) => live && setState({ data, error: null, loading: false })).catch((error) => live && setState({ data: null, error, loading: false }));
    return () => { live = false; };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  return state;
}

// "My center" lives in this browser only. The app uses it and never describes it.
const KEY = 'ci.center';
const listeners = new Set();
export function getMyCenter() { try { const v = localStorage.getItem(KEY); return v ? Number(v) : null; } catch (_) { return null; } }
export function setMyCenter(id) { try { if (id) localStorage.setItem(KEY, String(id)); else localStorage.removeItem(KEY); } catch (_) { /* private mode */ } listeners.forEach((f) => f(id || null)); }
export function useMyCenter() {
  const [id, setId] = useState(getMyCenter());
  useEffect(() => { listeners.add(setId); return () => listeners.delete(setId); }, []);
  return [id, setMyCenter];
}

// ---- words
export const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
export const DAY3 = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
export const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
export const hourLabel = (h) => (h === 0 ? '12am' : h < 12 ? `${h}am` : h === 12 ? '12pm' : `${h - 12}pm`);
export const plural = (n, one, many) => `${Number(n).toLocaleString()} ${n === 1 ? one : many || one + 's'}`;
export const centerLabel = (c) => (c ? c.nick || c.campus || c.name : '');
// Jewish terms the Chabad way
export const shabbos = (s) => (s || '').replace(/Shabbat/g, 'Shabbos').replace(/shabbat/g, 'shabbos').replace(/Sukkot/g, 'Sukkos').replace(/Shavuot/g, 'Shavuos').replace(/Passover/g, 'Pesach');

// Where an idea would land among the asking center's own events, in words. Never a number.
export const BAND_WORDS = { top: 'Likely one of your biggest events', upper: 'Likely above your average event', lower: 'Likely a little under your average', bottom: 'Likely one of your smaller events' };
export const BAND_SHORT = { top: 'Top quarter', upper: 'Upper half', lower: 'Lower half', bottom: 'Bottom quarter' };
export function bandOf(pct) {
  if (pct == null) return null;
  return pct >= 0.75 ? 'top' : pct >= 0.5 ? 'upper' : pct >= 0.25 ? 'lower' : 'bottom';
}
// network-level outcome, from lift.py bands
export const EFFECT_WORDS = { 'well above usual': 'Runs well above a center\'s usual', 'above usual': 'Runs above a center\'s usual', 'about usual': 'Runs about like a usual event', 'below usual': 'Runs below a center\'s usual' };
export const TIER_WORDS = { strong: 'Strong evidence', good: 'Good evidence', thin: 'Thin evidence', none: 'Too little to tell' };
export const KIND_LABEL = { shabbat_dinner: 'Shabbos dinner', shabbat_lunch: 'Shabbos lunch', shabbat_services: 'Shabbos services', shabbat_other: 'Shabbos program', holiday_meal: 'Holiday meal', holiday_event: 'Holiday event', holiday_service: 'Holiday service',
  class: 'Class', meal: 'Meal', social: 'Social', cooking: 'Cooking or baking', ritual: 'Mitzvah or prayer', speaker: 'Speaker', israel: 'Israel', chesed: 'Chesed', trip: 'Trip', outreach: 'Tabling', other: 'Other', resource: 'Resource', togo: 'To-go' };
export const AUD_LABEL = { women: 'Women', men: 'Men', grads: 'Grad students', freshmen: 'Freshmen', seniors: 'Seniors', parents: 'Parents', community: 'Community', greek: 'Greek life', faculty: 'Faculty', israelis: 'Israelis', athletes: 'Athletes' };

// Current season, for defaults: the Jewish calendar matters more than the month here, so the engine's holiday
// windows are approximated with civil months and the user can override.
export function seasonNow(d = new Date()) {
  const m = d.getMonth() + 1;
  if (m === 8 || m === 9) return 'Start of fall semester, High Holidays and Sukkos approaching';
  if (m === 10) return 'Mid fall, after Sukkos';
  if (m === 11) return 'Late fall, Thanksgiving, Chanukah approaching';
  if (m === 12) return 'Chanukah and finals';
  if (m === 1) return 'Start of spring semester';
  if (m === 2) return 'Mid spring semester, Purim approaching';
  if (m === 3) return 'Purim, Pesach approaching';
  if (m === 4) return 'Pesach, Yom HaShoah, Yom HaAtzmaut';
  if (m === 5) return 'End of year, finals, Lag BaOmer, Shavuos';
  return 'Summer';
}
