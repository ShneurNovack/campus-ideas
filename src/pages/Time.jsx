// Best time for this event. Type it or pick its kind; get the three slots where events like it land highest among
// their centers' own events, with how common each slot is, and the full week as two maps.
import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Clock } from 'lucide-react';
import { useWorld, useMyCenter, useData, DAY3, DAYS, shabbos, KIND_LABEL, centerLabel } from '../lib/data';
import { searchText } from '../lib/search';
import { kindOf } from '../lib/predict';
import { slotKind, cells, best, words, BLOCK_WORDS, BLOCK_HOURS } from '../lib/slots';
import { PageHead, Section, Chips, Icon, Kind, Loading, ErrorNote, Empty } from '../components/bits';

export default function Time() {
  const { data: W } = useWorld();
  const { data, error } = useData('slots.json');
  const S = data?.[0];
  const [my] = useMyCenter();
  const [params, setParams] = useSearchParams();
  const [text, setText] = useState(params.get('q') || '');
  const [kind, setKind] = useState(params.get('kind') || null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const center = W && my ? W.centerById.get(my) : null;
  const kinds = S ? Object.keys(S.kinds) : [];

  async function go() {
    if (!text.trim()) return;
    setBusy(true); setErr(null);
    try {
      const r = await searchText(text, { limit: 20 });
      const k = slotKind(S, kindOf(r.ix, r.sims));
      setKind(k); setParams({ q: text, kind: k || '' }, { replace: true });
    } catch (e) { setErr(e); }
    setBusy(false);
  }
  useEffect(() => { if (S && W && params.get('q') && !kind) go(); }, [S, W]); // eslint-disable-line react-hooks/exhaustive-deps

  const K = S && kind ? S.kinds[kind] : null;
  const top = useMemo(() => (S && kind ? best(S, kind, center, { limit: 3 }) : []), [S, kind, center]);
  const grid = useMemo(() => (S && kind ? cells(S, kind, center) : []), [S, kind, center]);
  const ownN = center?.slots?.[kind]?.n || 0;

  return (
    <>
      <PageHead title="When should it be?" lede="Every event is placed among its own center's events, so a slot only looks good when centers did better there than in their other slots. Pick the event; the week lights up." />
      <ErrorNote error={error} what="The time tables" />
      <form className="mt-12 flex flex-col sm:flex-row gap-3" onSubmit={(e) => { e.preventDefault(); go(); }}>
        <input className="c-input text-lg" placeholder="Describe the event, or pick a kind below" value={text} onChange={(e) => setText(e.target.value)} aria-label="Event" />
        <button type="submit" className="c-btn c-btn-go" disabled={!S || busy || !text.trim()}><Clock size={18} aria-hidden="true" />{busy ? 'Reading' : 'Find the time'}</button>
      </form>
      <ErrorNote error={err} what="The lookup" />
      {S && <div className="mt-6"><Chips value={kind} onChange={(k) => { setKind(k); setParams({ q: text, kind: k }, { replace: true }); }} options={kinds.map((k) => [k, S.kinds[k].label])} label="Kind of event" /></div>}
      {!S && <Loading />}
      {S && !kind && <p className="mt-10 c-muted">Pick a kind of event or describe one.</p>}
      {K && (
        <>
          <Section title={`${K.label}: the three best slots`} sub={center ? `Blended for ${centerLabel(center)}: ${ownN ? `its own ${K.label.toLowerCase()} count more the more of them it has held` : 'it has not held enough of these for its own history to count, so this is the network'}.` : 'Across the network. Choose your center to blend in its own history.'} tight>
            {top.length ? (
              <div className="grid gap-5 md:grid-cols-3">
                {top.map((c, k) => (
                  <div key={`${c.wd}${c.block}`} className="c-card">
                    <p className="c-small c-muted">{k === 0 ? 'Best' : k === 1 ? 'Second' : 'Third'}</p>
                    <p className="c-h2 mt-2">{c.day}</p>
                    <p className="font-semibold mt-1">{BLOCK_WORDS[c.block][0].toUpperCase() + BLOCK_WORDS[c.block].slice(1)}, {c.hours}</p>
                    <p className="mt-4 c-small" style={{ lineHeight: 1.5 }}>{words(c, kind, K).split(': ')[1]}</p>
                    {c.usedHere && <p className="mt-2 c-small c-muted">You have used this slot for {K.label.toLowerCase()} before.</p>}
                  </div>
                ))}
              </div>
            ) : <Empty>Too few events of this kind have readable results to call a slot.</Empty>}
          </Section>
          <Section title="The whole week" sub="Left: how events of this kind land in each slot, against their own center's usual. Right: how often the network uses each slot. Dark means more. Slots with too few events to trust are shown lighter." tight>
            <div className="grid gap-8 lg:grid-cols-2">
              <Heat title="Where they land" grid={grid} value={(c) => c.effect} fmt={(v) => (v == null ? '' : `${v > 0 ? '+' : ''}${Math.round(v * 100)}`)} diverge />
              <Heat title="How often they run" grid={grid} value={(c) => c.share} fmt={(v) => (v == null || v < 0.005 ? '' : `${Math.round(v * 100)}%`)} />
            </div>
            <p className="mt-6 c-small c-muted" style={{ lineHeight: 1.55 }}>Numbers on the left are percentile points among a center's own events (a 5 means five points higher than that center's usual for this kind). The effects are small on purpose: most of what decides turnout is the event, not the slot. {S.method.min_n} scored events from {S.method.min_centers} centers are needed before a slot is trusted.</p>
          </Section>
        </>
      )}
    </>
  );
}

export function Heat({ title, grid, value, fmt, diverge }) {
  const blocks = ['morning', 'lunch', 'afternoon', 'early', 'evening', 'late'];
  const by = new Map(grid.map((c) => [`${c.wd}:${c.block}`, c]));
  const vals = grid.map(value).filter((v) => v != null);
  const max = Math.max(0.0001, ...vals.map(Math.abs));
  return (
    <div>
      <p className="font-semibold mb-4">{title}</p>
      <div className="grid gap-1" style={{ gridTemplateColumns: '5.4rem repeat(7, minmax(0, 1fr))' }}>
        <div />
        {DAY3.map((d) => <div key={d} className="c-small c-muted text-center pb-1">{d}</div>)}
        {blocks.map((b) => (
          <Row key={b} b={b} by={by} value={value} fmt={fmt} max={max} diverge={diverge} />
        ))}
      </div>
    </div>
  );
}
function Row({ b, by, value, fmt, max, diverge }) {
  return (
    <>
      <div className="c-small c-muted self-center pr-2" style={{ lineHeight: 1.2 }}>{BLOCK_WORDS[b][0].toUpperCase() + BLOCK_WORDS[b].slice(1)}<br /><span style={{ fontSize: '0.72rem' }}>{BLOCK_HOURS[b]}</span></div>
      {[0, 1, 2, 3, 4, 5, 6].map((wd) => {
        const c = by.get(`${wd}:${b}`), v = c ? value(c) : null;
        const t = v == null ? 0 : Math.min(1, Math.abs(v) / max);
        const bg = v == null ? 'var(--fog)' : diverge && v < 0 ? `rgba(180,83,10,${0.12 + 0.75 * t})` : `rgba(15,110,86,${0.1 + 0.85 * t})`;
        const faint = c && !c.ok && !c.usedHere;
        return <div key={wd} className="c-cell" style={{ background: bg, color: t > 0.55 ? '#fff' : 'var(--ink)', opacity: faint ? 0.45 : 1 }} title={c ? `${c.day} ${c.blockWord}: ${c.n} events from ${c.centers} centers` : ''}>{c && c.n >= 5 ? fmt(v) : ''}</div>;
      })}
    </>
  );
}
