// Plan my week. Choose what kinds of events you want to hold and which days are open; the planner places each one
// in the slot where that kind lands highest for a center like yours, keeps events off each other's days when it
// can, and shows the trade it made. Coordinate ascent over the slot tables: tiny problem, exact enough.
import { useMemo, useState } from 'react';
import { CalendarDays } from 'lucide-react';
import { useWorld, useMyCenter, useData, DAYS, DAY3, centerLabel } from '../lib/data';
import { cells, BLOCK_WORDS, BLOCK_HOURS } from '../lib/slots';
import { PageHead, Section, Chips, Icon, Loading, ErrorNote } from '../components/bits';

const SAME_DAY = 0.04;        // the cost of a second event on a day, in percentile points (about one slot's worth of advantage)
const BACK_TO_BACK = 0.015;   // the cost of events on consecutive days

export default function Week() {
  const { data: W } = useWorld();
  const { data, error } = useData('slots.json');
  const S = data?.[0];
  const [my] = useMyCenter();
  const center = W && my ? W.centerById.get(my) : null;
  const allKinds = S ? Object.keys(S.kinds) : [];
  const defaults = useMemo(() => {
    if (!S) return [];
    const strong = center ? center.kinds.filter((k) => k.read === 'strength').map((k) => k.key).filter((k) => S.kinds[k]) : [];
    const base = ['shabbat_dinner', 'class', 'meal'];
    return [...new Set([...base, ...strong])].slice(0, 4);
  }, [S, center]);
  const [kinds, setKinds] = useState(null);
  const [days, setDays] = useState([0, 1, 2, 3, 4, 6]);
  const chosen = kinds ?? defaults;

  const plan = useMemo(() => (S ? solve(S, center, chosen, days) : null), [S, center, chosen, days]);

  return (
    <>
      <PageHead title="Plan my week" lede="Pick the kinds of events you want to hold this week and the days you have. Each one goes where that kind lands highest for a center like yours, spread across the week so they do not eat each other." />
      <ErrorNote error={error} what="The time tables" />
      {!S && <Loading />}
      {S && (
        <>
          <div className="mt-12 grid gap-10 lg:grid-cols-2">
            <div><span className="c-label">This week I want to run</span><Chips multi value={chosen} onChange={(v) => setKinds(v.slice(0, 6))} options={allKinds.map((k) => [k, S.kinds[k].label])} label="Kinds of events" /><p className="mt-3 c-small c-muted">Up to six. {center && defaults.length > 3 ? 'Your strengths are in by default.' : ''}</p></div>
            <div><span className="c-label">Days I have</span><Chips multi value={days} onChange={setDays} options={DAYS.map((d, i) => [i, d])} label="Open days" /><p className="mt-3 c-small c-muted">Shabbos dinners and lunches stay on Friday and Shabbos whatever you pick.</p></div>
          </div>
          {plan && (
            <Section title="The week" sub={center ? `Tuned for ${centerLabel(center)}, with the network filling in where its own history is thin.` : 'Network view. Choose your center to tune it.'} tight>
              <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(7, minmax(0, 1fr))' }}>
                {DAYS.map((d, wd) => (
                  <div key={d} className="min-h-[9rem] p-3" style={{ background: days.includes(wd) || plan.some((p) => p.wd === wd) ? 'var(--fog)' : 'transparent' }}>
                    <p className="c-small c-muted">{DAY3[wd]}</p>
                    {plan.filter((p) => p.wd === wd).sort((a, b) => a.order - b.order).map((p) => (
                      <div key={p.kind} className="mt-3 bg-white p-2.5">
                        <div className="flex items-center gap-2"><Icon text="" type={p.kind} size={16} /><p className="font-semibold c-small" style={{ lineHeight: 1.25 }}>{S.kinds[p.kind].label}</p></div>
                        <p className="c-small c-muted mt-1">{BLOCK_WORDS[p.block]}, {BLOCK_HOURS[p.block]}</p>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              <div className="mt-10 grid gap-4 md:grid-cols-2">
                {plan.map((p) => (
                  <div key={p.kind} className="flex gap-4">
                    <Icon text="" type={p.kind} size={22} />
                    <div>
                      <p className="font-semibold">{S.kinds[p.kind].label}: {DAYS[p.wd]} {BLOCK_WORDS[p.block]}</p>
                      <p className="c-small c-muted mt-1" style={{ lineHeight: 1.5 }}>{p.why}</p>
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-8 c-small c-muted" style={{ lineHeight: 1.55 }}>The planner adds up where each kind lands in each slot and takes away a little for a second event on the same day and a little less for back-to-back days. It does not know your room, your staff or your budget, and the slot is the smaller part of what decides turnout.</p>
            </Section>
          )}
        </>
      )}
    </>
  );
}

function solve(S, center, kinds, days) {
  const FIXED = { shabbat_dinner: [4], shabbat_lunch: [5], shabbat_services: [4, 5] };
  const options = {};
  for (const k of kinds) {
    if (!S.kinds[k]) continue;
    const allow = FIXED[k] || (days.length ? days : [0, 1, 2, 3, 4, 5, 6]);
    let cs = cells(S, k, center).filter((c) => allow.includes(c.wd) && (c.ok || c.ownScored >= 4 || c.n >= 10));
    if (!cs.length) cs = cells(S, k, center).filter((c) => allow.includes(c.wd));
    options[k] = cs.map((c) => ({ ...c, base: (c.effect ?? 0) + 0.01 * Math.min(1, c.share * 8) }));
  }
  const ks = Object.keys(options);
  const pick = Object.fromEntries(ks.map((k) => [k, options[k].reduce((b, c) => (c.base > (b?.base ?? -9) ? c : b), null)]));
  const total = (assign) => {
    let s = 0; const perDay = {};
    for (const k of ks) { const c = assign[k]; if (!c) continue; s += c.base; perDay[c.wd] = (perDay[c.wd] || 0) + 1; }
    for (const d in perDay) s -= SAME_DAY * Math.max(0, perDay[d] - 1) * (perDay[d] - 1);
    const used = Object.keys(perDay).map(Number).sort();
    for (let i = 1; i < used.length; i++) if (used[i] - used[i - 1] === 1) s -= BACK_TO_BACK;
    return s;
  };
  for (let pass = 0; pass < 4; pass++) {
    for (const k of ks) {
      let bestC = pick[k], bestS = total(pick);
      for (const c of options[k]) { const t = { ...pick, [k]: c }; const v = total(t); if (v > bestS + 1e-9) { bestS = v; bestC = c; } }
      pick[k] = bestC;
    }
  }
  const order = { morning: 0, lunch: 1, afternoon: 2, early: 3, evening: 4, late: 5 };
  return ks.filter((k) => pick[k]).map((k) => {
    const c = pick[k], top = options[k].slice().sort((a, b) => b.base - a.base)[0];
    const e = c.effect;
    const why = `${e == null ? 'No readable effect for this slot' : Math.abs(e) < 0.015 ? 'Lands about like any other slot for this kind' : `Lands ${Math.abs(Math.round(e * 100))} points ${e > 0 ? 'above' : 'below'} the usual for this kind`}; ${Math.round(c.share * 100)}% of the network's ${S.kinds[k].label.toLowerCase()} run here.${top && top !== c && top.base - c.base > 0.005 ? ` Moved off ${DAYS[top.wd]} ${BLOCK_WORDS[top.block]} to keep the week spread out.` : ''}${c.usedHere ? ' You have used this slot before.' : ''}`;
    return { kind: k, wd: c.wd, block: c.block, order: order[c.block], why };
  });
}
