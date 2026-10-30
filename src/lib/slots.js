// Reading the slot model (engine/slots.py) for one kind of event, blended with the chosen center's own history.
// Effects are in percentile points among a center's own events: +0.05 means events in that slot land five points
// higher than that center's usual for the kind. Small numbers are the honest size of the effect.
import { DAY3, DAYS } from './data';

export const BLOCK_WORDS = { morning: 'morning', lunch: 'lunchtime', afternoon: 'afternoon', early: 'early evening', evening: 'evening', late: 'late night' };
export const BLOCK_HOURS = { morning: '7 to 11am', lunch: '12 to 1pm', afternoon: '2 to 4pm', early: '5 to 6pm', evening: '7 to 8pm', late: '9pm on' };

// the kind of the slot tables that fits an event type
export function slotKind(S, type) {
  if (!type) return null;
  if (S.kinds[type]) return type;
  const map = { shabbat_services: 'shabbat_dinner', shabbat_other: 'shabbat_dinner', other: 'social', outreach: 'social', ritual: 'class', resource: null, togo: null };
  return map[type] && S.kinds[map[type]] ? map[type] : null;
}

// cells for a kind: network effect, frequency share, reliability; blended with the center's own cells when given
export function cells(S, kind, center) {
  const K = S.kinds[kind];
  if (!K) return [];
  const own = new Map((center?.slots?.[kind]?.cells || []).map((c) => [`${c.wd}:${c.block}`, c]));
  return K.grid.map((c) => {
    const o = own.get(`${c.wd}:${c.block}`);
    const effect = o ? o.effect : c.effect;
    return { ...c, net: c.effect, effect, ownN: o?.n || 0, ownScored: o?.scored || 0, usedHere: !!o, day: DAYS[c.wd], day3: DAY3[c.wd], blockWord: BLOCK_WORDS[c.block], hours: BLOCK_HOURS[c.block] };
  });
}

// the best slots for a kind at a center: effect first, among slots the network uses enough to trust
export function best(S, kind, center, { limit = 3, days = null } = {}) {
  let all = cells(S, kind, center).filter((c) => c.ok || c.ownScored >= 4);
  if (days) all = all.filter((c) => days.includes(c.wd));
  const score = (c) => (c.effect == null ? -1 : c.effect) + 0.02 * Math.min(1, c.share * 10) + (c.usedHere ? 0.01 : 0);
  return all.sort((a, b) => score(b) - score(a)).slice(0, limit);
}

export function words(c, kind, K) {
  const e = c.effect;
  const size = e == null ? 'no readable effect' : Math.abs(e) < 0.015 ? 'about the same as other slots' : `${Math.abs(Math.round(e * 100))} points ${e > 0 ? 'above' : 'below'} the usual for this kind`;
  const common = c.share >= 0.12 ? 'a common slot' : c.share >= 0.05 ? 'a fairly common slot' : 'an unusual slot';
  return `${c.day} ${c.blockWord}: ${size}, ${common} (${Math.round(c.share * 100)}% of ${K?.label?.toLowerCase() || 'these events'} run here)`;
}
