// When things run. Up-to-date time-of-week and time-of-day tables for every kind of event: how often the network
// uses each slot, and where events in that slot land against their own center's usual.
import { useMemo, useState } from 'react';
import { useData, DAYS, DAY3 } from '../lib/data';
import { BLOCK_WORDS, BLOCK_HOURS } from '../lib/slots';
import { Heat } from './Time';
import { PageHead, Section, Chips, Loading, ErrorNote } from '../components/bits';

export default function Metrics() {
  const { data, error } = useData('slots.json');
  const S = data?.[0];
  const [kind, setKind] = useState('all');
  const kinds = S ? Object.keys(S.kinds) : [];
  const K = S ? (kind === 'all' ? { label: 'Everything a center picks the day for', grid: S.overall.grid, days: S.overall.days, blocks: S.overall.blocks, n: S.overall.n } : S.kinds[kind]) : null;
  const grid = useMemo(() => (K ? K.grid.map((c) => ({ ...c, day: DAYS[c.wd], day3: DAY3[c.wd], blockWord: BLOCK_WORDS[c.block], hours: BLOCK_HOURS[c.block] })) : []), [K]);
  return (
    <>
      <PageHead title="When things run" lede="Two different questions, kept apart: how often the network holds each kind of event in each slot, and how events in that slot land against their own center's usual. The last three years, refreshed daily." />
      <ErrorNote error={error} what="The time tables" />
      {!S && <Loading />}
      {S && K && (
        <>
          <div className="mt-12"><Chips value={kind} onChange={setKind} options={[['all', 'Everything free to schedule'], ...kinds.map((k) => [k, S.kinds[k].label])]} label="Kind" /></div>
          <Section title={K.label} sub={`${K.n.toLocaleString()} events${K.centers ? ` from ${K.centers} centers` : ''}.`} tight>
            <div className="grid gap-8 lg:grid-cols-2">
              <Heat title="How often they run" grid={grid} value={(c) => c.share} fmt={(v) => (v == null || v < 0.005 ? '' : `${Math.round(v * 100)}%`)} />
              <Heat title="Where they land" grid={grid} value={(c) => c.effect} fmt={(v) => (v == null ? '' : `${v > 0 ? '+' : ''}${Math.round(v * 100)}`)} diverge />
            </div>
          </Section>
          <Section title="By day and by time" tight>
            <div className="grid gap-10 md:grid-cols-2">
              <Bars title="Day of the week" rows={K.days.map((d) => ({ label: DAYS[d.wd], ...d }))} />
              <Bars title="Time of day" rows={['morning', 'lunch', 'afternoon', 'early', 'evening', 'late'].map((b) => K.blocks.find((x) => x.block === b)).filter(Boolean).map((d) => ({ label: `${BLOCK_WORDS[d.block][0].toUpperCase()}${BLOCK_WORDS[d.block].slice(1)} (${BLOCK_HOURS[d.block]})`, ...d }))} />
            </div>
            <p className="mt-8 c-small c-muted" style={{ lineHeight: 1.55 }}>The grey bar is the share of events held then. The number beside it is where those events land, in percentile points against their own center's usual for the kind, shrunk toward zero when there is little evidence and tested by shuffling sign-ups within each center. A slot is trusted at {S.method.min_n} scored events from {S.method.min_centers} centers.</p>
          </Section>
        </>
      )}
    </>
  );
}

function Bars({ title, rows }) {
  const max = Math.max(0.0001, ...rows.map((r) => r.share));
  return (
    <div>
      <p className="font-semibold mb-4">{title}</p>
      <div className="space-y-3">
        {rows.map((r) => (
          <div key={r.label} className="grid items-center gap-3" style={{ gridTemplateColumns: '9.5rem 1fr 4.5rem' }}>
            <span className="c-small">{r.label}</span>
            <div className="c-bar-fog"><div className="c-bar" style={{ width: `${(100 * r.share) / max}%`, background: 'var(--ink)' }} /></div>
            <span className="c-small text-right" style={{ color: r.effect == null ? 'var(--grey)' : r.effect > 0.01 ? 'var(--go)' : r.effect < -0.01 ? 'var(--warm)' : 'var(--grey)', opacity: r.ok ? 1 : 0.5 }}>{r.effect == null ? '' : `${r.effect > 0 ? '+' : ''}${Math.round(r.effect * 100)}`}{r.ok ? '' : r.effect == null ? '' : ' ?'}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
