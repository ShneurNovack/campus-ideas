// The idea generator. The model writes; the data chooses what the model sees and grades what it writes.
//   1. exemplars: past ideas that fit the brief (kind, season, audience) with real evidence behind them (lift.py
//      tiers), preferring ones the chosen center has not run and that the neighbor system expects to land high
//      there. Picked with diminishing returns on similarity (MMR) so the model sees breadth, not twelve dinners.
//   2. the model gets names and effect bands, never a count and never a center's name.
//   3. every idea that comes back is placed by the same machinery as the tester: nearest past ideas, expected band
//      at the chosen center, and the closest real events, so a made-up idea arrives with its track record.
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, RefreshCw, ArrowRight } from 'lucide-react';
import { useWorld, useMyCenter, KIND_LABEL, AUD_LABEL, seasonNow, shabbos, centerLabel } from '../lib/data';
import { searchIndex, embed, scoreAll, generate, meanVector } from '../lib/search';
import { expectAt } from '../lib/predict';
import { PageHead, Section, Chips, Expect, Icon, Loading, ErrorNote } from '../components/bits';

const KINDS = [['any', 'Anything'], ['shabbat_dinner', 'Shabbos dinner'], ['holiday', 'Holiday program'], ['class', 'Class or learning'], ['meal', 'Food event'], ['social', 'Social'], ['cooking', 'Cooking or baking'], ['speaker', 'Speaker'], ['israel', 'Israel'], ['chesed', 'Chesed'], ['trip', 'Trip']];
const KIND_TYPES = { holiday: ['holiday_meal', 'holiday_event', 'holiday_service'], shabbat_dinner: ['shabbat_dinner', 'shabbat_lunch'] };
const AUDS = [['any', 'Everyone'], ['freshmen', 'Freshmen'], ['grads', 'Grad students'], ['women', 'Women'], ['men', 'Men'], ['greek', 'Greek life'], ['israelis', 'Israelis'], ['parents', 'Parents'], ['seniors', 'Seniors'], ['community', 'Community']];
const WHENS = [['any', 'Any time'], ['weeknight', 'A weeknight'], ['lunch', 'Lunchtime'], ['friday', 'Friday night'], ['shabbos', 'Shabbos day'], ['sunday', 'Sunday'], ['weekend', 'Saturday night']];

export default function Home() {
  const { data: W, error } = useWorld();
  const [my] = useMyCenter();
  const [kind, setKind] = useState('any');
  const [aud, setAud] = useState('any');
  const [when, setWhen] = useState('any');
  const [season, setSeason] = useState(seasonNow());
  const [vibe, setVibe] = useState('');
  const [state, setState] = useState({ busy: false, step: '', ideas: null, error: null, persona: null });
  const center = W && my ? W.centerById.get(my) : null;

  async function go() {
    if (!W) return;
    setState({ busy: true, step: 'Choosing what to show the model', ideas: null, error: null });
    try {
      const ix = await searchIndex();
      const ex = exemplars(W, ix, { kind, aud, season, center });
      const avoid = center ? (W.byCenter.get(center.id) || []).slice().sort((a, b) => b.n - a.n).slice(0, 30).map((p) => p.name) : [];
      const brief = { audience: aud === 'any' ? 'Jewish students, everyone welcome' : AUD_LABEL[aud], when: when === 'any' ? null : WHENS.find((w) => w[0] === when)[1], season, kind: kind === 'any' ? null : KINDS.find((k) => k[0] === kind)[1], vibe: vibe.trim() || null,
        center: center ? centerBrief(center, W) : null };
      setState((s) => ({ ...s, step: 'Writing ideas' }));
      const res = await generate(brief, ex.map((e) => ({ name: e.label, hook: e.hook, band: e.band })), avoid, 12);
      setState((s) => ({ ...s, step: 'Checking each one against the record' }));
      const placed = await Promise.all(res.ideas.map(async (i) => {
        const q = await embed(`${i.title}. ${i.pitch}`);
        const sims = scoreAll(ix, q);
        const order = Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, 5);
        const closest = order.filter((a) => sims[a] >= 0.45).map((a) => ({ p: ix.W.programs[a], sim: sims[a] }));
        const ex2 = expectAt(ix, center?.id, q);
        const kindGuess = closest[0]?.p.type || null;
        return { ...i, closest, expect: ex2, kind: kindGuess, novelty: closest.length ? (closest[0].sim >= 0.7 ? 'Has been run before' : closest[0].sim >= 0.55 ? 'Close to things that ran' : 'New ground') : 'New ground' };
      }));
      setState({ busy: false, step: '', ideas: placed, error: null, persona: res.persona, exemplars: ex });
    } catch (e) { setState({ busy: false, step: '', ideas: null, error: e }); }
  }

  return (
    <>
      <PageHead title={<>What should we run next?</>} lede="Tell it who the event is for and roughly when. It reads what has worked across the Chabad on Campus network, writes new ideas in that spirit, and grades each one against the record before you see it." />
      <ErrorNote error={error} />
      <div className="mt-12 grid gap-10 lg:grid-cols-[1fr_1fr]">
        <div className="space-y-8">
          <div><span className="c-label">Kind of event</span><Chips value={kind} onChange={setKind} options={KINDS} label="Kind of event" /></div>
          <div><span className="c-label">Who it is for</span><Chips value={aud} onChange={setAud} options={AUDS} label="Audience" /></div>
          <div><span className="c-label">When</span><Chips value={when} onChange={setWhen} options={WHENS} label="When" /></div>
        </div>
        <div className="space-y-8">
          <label className="block"><span className="c-label">Time of year</span><input className="c-input" value={season} onChange={(e) => setSeason(e.target.value)} /></label>
          <label className="block"><span className="c-label">Anything else</span><textarea className="c-input" placeholder="Low budget. Something outdoors. We want to reach students who have never come." value={vibe} onChange={(e) => setVibe(e.target.value)} /></label>
          <div className="flex flex-wrap items-center gap-4">
            <button type="button" className="c-btn c-btn-go" onClick={go} disabled={!W || state.busy}><Sparkles size={18} aria-hidden="true" />{state.busy ? state.step : 'Give me ideas'}</button>
            {!center && W && <span className="c-small c-muted">Choose your center (top right) to have ideas tuned and graded for it.</span>}
          </div>
        </div>
      </div>
      <ErrorNote error={state.error} what="The idea run" />
      {state.busy && <Loading rows={4} h={140} />}
      {state.ideas && (
        <Section title="Twelve ideas" sub={`Each one is placed against ${W.engine.counts.ran.toLocaleString()} past events. "Expected" is where it would likely land among your own events, from centers like yours that ran things like it.`}
          action={<button type="button" className="c-btn c-btn-soft" onClick={go}><RefreshCw size={16} aria-hidden="true" />Another twelve</button>} tight>
          <div className="grid gap-5 md:grid-cols-2">
            {state.ideas.map((i, k) => <IdeaCard key={k} idea={i} center={center} />)}
          </div>
          {state.exemplars && <p className="mt-10 c-small c-muted" style={{ lineHeight: 1.55 }}>Written in the spirit of {state.exemplars.length} past ideas with real evidence behind them: {state.exemplars.map((e) => shabbos(e.label)).join(', ')}.</p>}
        </Section>
      )}
    </>
  );
}

function IdeaCard({ idea, center }) {
  return (
    <article className="c-card flex gap-5">
      <Icon text={`${idea.title} ${idea.pitch} ${idea.format}`} type={idea.kind} />
      <div className="min-w-0 flex-1">
        <h3 className="c-h3">{shabbos(idea.title)}</h3>
        <p className="mt-2" style={{ lineHeight: 1.5 }}>{shabbos(idea.pitch)}</p>
        {idea.hook && <p className="mt-2 c-small c-muted" style={{ lineHeight: 1.5 }}>The hook: {shabbos(idea.hook)}</p>}
        <div className="mt-4 flex flex-wrap gap-2">
          {idea.format && <span className="c-tag">{idea.format}</span>}
          <span className="c-tag">{idea.novelty}</span>
          {center && idea.expect && <Expect band={idea.expect.band} />}
        </div>
        {idea.closest.length > 0 && <p className="mt-4 c-small c-muted" style={{ lineHeight: 1.5 }}>Closest real events: {idea.closest.slice(0, 3).map((c) => shabbos(c.p.name)).join(' · ')}</p>}
        <Link to={`/test?q=${encodeURIComponent(idea.title)}`} className="c-link c-small mt-3 inline-flex items-center gap-1">Test this idea <ArrowRight size={13} aria-hidden="true" /></Link>
      </div>
    </article>
  );
}

// ---- exemplar choice
function centerBrief(c, W) {
  const f = Object.fromEntries(W.facets.map((x) => [x.key, Object.fromEntries(x.values.map((v) => [v.key, v.label]))]));
  const parts = [];
  if (c.cat.size && c.cat.size !== 'unknown') parts.push(`turnout band: ${f.size[c.cat.size]}`);
  if (c.cat.pace && c.cat.pace !== 'unknown') parts.push(`pace: ${f.pace[c.cat.pace]}`);
  if (c.cat.strength && !['unknown', 'even'].includes(c.cat.strength)) parts.push(`what lands high here: ${f.strength[c.cat.strength]}`);
  const strong = c.kinds.filter((k) => k.read === 'strength').map((k) => k.label);
  if (strong.length) parts.push(`strong kinds: ${strong.join(', ')}`);
  return parts.join('; ') || null;
}
const MONTH_OF = { january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9, november: 10, december: 11 };
function exemplars(W, ix, { kind, aud, season, center }) {
  const types = kind === 'any' ? null : KIND_TYPES[kind] || [kind];
  const m = new Date().getMonth();
  const months = [m, (m + 1) % 12, (m + 2) % 12];
  const ran = center ? new Set((W.byCenter.get(center.id) || []).map((p) => p.idea).filter((x) => x != null)) : new Set();
  const pool = W.ideas.filter((i) => (!types || types.includes(i.type)) && i.tier !== 'none' && i.band && i.band !== 'below usual' && i.centers >= 3)
    .map((i) => {
      let s = (i.tier === 'strong' ? 1 : i.tier === 'good' ? 0.7 : 0.4) + (i.band.startsWith('well') ? 0.6 : i.band === 'above usual' ? 0.3 : 0);
      const tot = (i.months || []).reduce((a, b) => a + b, 0) || 1;
      const inSeason = months.reduce((a, mm) => a + (i.months?.[mm] || 0), 0) / tot;
      s += 0.8 * inSeason;
      if (aud !== 'any' && (i.aud || []).includes(aud)) s += 0.8;
      if (center) {
        const p = W.pred(center.id, i.id);
        if (p) s += p.pct >= 0.75 ? 0.6 : p.pct >= 0.5 ? 0.3 : -0.3;
        if (ran.has(i.id)) s -= 0.5;
      }
      if (i.status === 'rising') s += 0.2;
      return { i, s };
    }).sort((a, b) => b.s - a.s).slice(0, 60);
  // MMR on the search vectors: each pick must be at most 0.55 like any earlier pick
  const picked = [], vecs = [];
  for (const { i } of pool) {
    const v = meanVector(ix, i.rep);
    if (!v) continue;
    let maxSim = 0;
    for (const u of vecs) { let s = 0; for (let j = 0; j < v.length; j++) s += v[j] * u[j]; maxSim = Math.max(maxSim, s); }
    if (maxSim > 0.55) continue;
    picked.push(i); vecs.push(v);
    if (picked.length >= 12) break;
  }
  return picked.map((i) => ({ label: i.label, hook: i.names?.length > 1 ? `also called ${i.names.slice(1, 3).join(' / ')}` : i.when && i.when !== 'Any time of year' ? i.when : '', band: i.band }));
}
