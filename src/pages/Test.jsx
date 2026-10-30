// The idea tester. Type an event; find out whether it has been done, how it went where it was done, and where it
// would likely land for you. Retrieval is a cascade (search.js); every number in the verdict is counted in code
// from the matched events; the model only decides which past events are the same thing and writes two sentences.
import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { FlaskConical } from 'lucide-react';
import { useWorld, useMyCenter, KIND_LABEL, shabbos, BAND_WORDS, centerLabel } from '../lib/data';
import { cascade, explain } from '../lib/search';
import { expectAt, kindOf, tally } from '../lib/predict';
import { PageHead, Section, Icon, Expect, Kind, Loading, ErrorNote, Empty } from '../components/bits';

const EXAMPLES = ['Sushi making night', 'Mental health Shabbos', 'Pickleball tournament', 'Challah bake for finals', 'Israeli cooking class', 'Trivia night with pizza'];

export default function Test() {
  const { data: W } = useWorld();
  const [my] = useMyCenter();
  const [params, setParams] = useSearchParams();
  const [text, setText] = useState(params.get('q') || '');
  const [run, setRun] = useState({ busy: false, step: '', result: null, error: null });
  const center = W && my ? W.centerById.get(my) : null;

  async function go(q = text) {
    if (!q.trim() || !W) return;
    setParams({ q }, { replace: true });
    setRun({ busy: true, step: 'Reading past events', result: null, error: null });
    try {
      const r = await cascade(q, { onStep: (s) => setRun((x) => ({ ...x, step: s })) });
      const same = [...r.same], variants = [...r.variant];
      const t = tally(same), tv = tally([...same, ...variants]);
      const expect = expectAt(r.ix, center?.id, r.q);
      const kind = kindOf(r.ix, r.sims);
      const years = t.years;
      const verdict = same.length === 0 ? (variants.length ? 'Nothing quite like this has run, though there are relatives.' : 'Nothing like this is in the record.')
        : `${same.length === 1 ? 'One center has' : `${t.centers} centers have`} run this${years.length ? ` (${years[0]}${years.length > 1 ? ` to ${years[years.length - 1]}` : ''})` : ''}.${t.read ? ` ${t.read}.` : ''}`;
      let words = '';
      try {
        const facts = [verdict, variants.length ? `${variants.length} related variants exist.` : '', expect ? `For the asking center it would likely land: ${BAND_WORDS[expect.band].toLowerCase()}.` : ''].filter(Boolean).join(' ');
        words = (await explain(q, facts, [...same, ...variants].slice(0, 6).map((h) => ({ name: h.p.name, label: h.label })))).text;
      } catch (_) { /* two sentences are a nicety */ }
      setRun({ busy: false, step: '', error: null, result: { q, same, variants, different: r.different, more: r.more, t, tv, expect, kind, verdict, words, notes: r.notes } });
    } catch (e) { setRun({ busy: false, step: '', result: null, error: e }); }
  }
  useEffect(() => { if (W && params.get('q') && !run.result && !run.busy) go(params.get('q')); }, [W]); // eslint-disable-line react-hooks/exhaustive-deps

  const R = run.result;
  return (
    <>
      <PageHead title="Test an idea" lede="Type the event the way you would put it on a flyer. The tester finds every center that has run something like it, says how those went, and where it would likely land among your own events." />
      <form className="mt-12 flex flex-col sm:flex-row gap-3" onSubmit={(e) => { e.preventDefault(); go(); }}>
        <input className="c-input text-lg" placeholder="Sushi in the Sukkah" value={text} onChange={(e) => setText(e.target.value)} aria-label="Event idea" />
        <button type="submit" className="c-btn c-btn-go sm:w-auto" disabled={!W || run.busy || !text.trim()}><FlaskConical size={18} aria-hidden="true" />{run.busy ? run.step : 'Test it'}</button>
      </form>
      {!R && !run.busy && <p className="mt-5 c-small c-muted">Try: {EXAMPLES.map((e, k) => <span key={e}>{k ? ' · ' : ''}<button type="button" className="c-link" onClick={() => { setText(e); go(e); }}>{e}</button></span>)}</p>}
      <ErrorNote error={run.error} what="The test" />
      {run.busy && <Loading rows={3} />}
      {R && (
        <>
          <Section title={shabbos(R.verdict)} tight>
            <div className="grid gap-5 md:grid-cols-[1.4fr_1fr]">
              <div className="c-card">
                <div className="flex gap-5">
                  <Icon text={R.q} type={R.kind} />
                  <div>
                    <div className="flex flex-wrap gap-2"><Kind type={R.kind} />{center && R.expect && <Expect band={R.expect.band} />}{R.tv.measured >= 3 && R.tv.read && <span className="c-tag">{R.tv.read}</span>}</div>
                    {R.words && <p className="mt-4" style={{ lineHeight: 1.55 }}>{shabbos(R.words)}</p>}
                    {!R.words && <p className="mt-4" style={{ lineHeight: 1.55 }}>{shabbos(R.verdict)}</p>}
                    {center && R.expect && <p className="mt-3 c-small c-muted" style={{ lineHeight: 1.5 }}>The expectation for {centerLabel(center)} comes from {R.expect.near.length} past ideas like this one, weighted by how alike their centers are to yours. {R.expect.sure === 'loose' ? 'The closest of them is a loose match, so read it lightly.' : ''}</p>}
                    {center && !R.expect && <p className="mt-3 c-small c-muted">Nothing close enough has run at centers like yours to say where this would land.</p>}
                    {R.notes.length > 0 && <p className="mt-3 c-small c-muted">Note: {R.notes.join('; ')}.</p>}
                  </div>
                </div>
              </div>
              <div className="c-card">
                <p className="font-semibold">How the matches went</p>
                {R.tv.measured ? <Bands bands={R.tv.bands} /> : <p className="c-muted mt-2 c-small">No matched event had a readable result.</p>}
                <p className="mt-3 c-small c-muted" style={{ lineHeight: 1.5 }}>Each matched event is placed among its own center's events, from smallest to biggest, and the placements are counted here without saying which center is which.</p>
              </div>
            </div>
          </Section>
          <Matches title="The same event" items={R.same} note="Same activity and format, under whatever name." />
          <Matches title="Relatives" items={[...R.variants, ...R.more.slice(0, 8)]} note="Same core idea with a different twist, audience or holiday." />
          {R.same.length + R.variants.length === 0 && <Empty>Nothing in {W.engine.counts.ran.toLocaleString()} past events is the same program. That is either a gap or a reason, and the data cannot tell which. The ideas page can suggest the closest things that did work.</Empty>}
        </>
      )}
    </>
  );
}

function Bands({ bands }) {
  const total = Object.values(bands).reduce((a, b) => a + b, 0) || 1;
  const rows = [['top', 'Top quarter of its center\'s events'], ['upper', 'Upper half'], ['lower', 'Lower half'], ['bottom', 'Bottom quarter']];
  return (
    <div className="mt-4 space-y-3">
      {rows.map(([k, l]) => <div key={k}><div className="flex justify-between c-small"><span>{l}</span><span className="c-muted">{bands[k]}</span></div><div className="mt-1 c-bar-fog"><div className="c-bar" style={{ width: `${(100 * bands[k]) / total}%` }} /></div></div>)}
    </div>
  );
}

function Matches({ title, items, note }) {
  const { data: W } = useWorld();
  if (!items.length || !W) return null;
  return (
    <Section title={title} sub={note} tight>
      <div className="grid gap-4 md:grid-cols-2">
        {items.map((h) => {
          const c = W.centerById.get(h.p.center);
          return (
            <article key={h.p.pid} className="flex gap-4 py-2">
              <Icon text={h.p.name} type={h.p.type} size={22} />
              <div className="min-w-0">
                <p className="font-semibold">{shabbos(h.p.name)}</p>
                <p className="c-small c-muted mt-0.5">{c ? c.name : 'A center'}{h.p.years?.length ? ` · ${h.p.years[0]}${h.p.years.length > 1 ? ` to ${h.p.years[h.p.years.length - 1]}` : ''}` : ''}{h.p.n > 1 ? ` · ran ${h.p.n} times` : ''}</p>
                {h.p.short && <p className="c-small mt-1.5" style={{ lineHeight: 1.45 }}>{shabbos(h.p.short)}</p>}
                {h.why && <p className="c-small c-muted mt-1.5">{h.why}</p>}
              </div>
            </article>
          );
        })}
      </div>
    </Section>
  );
}
