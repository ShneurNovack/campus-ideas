// Shabbos dinner themes: what is spreading, what belongs to this time of year, and how each theme tends to land
// against a center's plain dinners. Trend is counted in centers (engine/trends.py), never in events.
import { useMemo, useState } from 'react';
import { Wine, TrendingUp, Snowflake, Sparkles } from 'lucide-react';
import { useWorld, useMyCenter, useData, shabbos, MONTHS } from '../lib/data';
import { PageHead, Section, Chips, Icon, Tier, Effect, Expect, Loading, ErrorNote, Empty } from '../components/bits';

const VIEWS = [['season', 'Right for this time of year'], ['rising', 'Spreading'], ['proven', 'Proven'], ['all', 'Every theme']];

export default function Shabbos() {
  const { data: W } = useWorld();
  const { data, error } = useData('themes.json');
  const T = data?.[0];
  const [my] = useMyCenter();
  const center = W && my ? W.centerById.get(my) : null;
  const [view, setView] = useState('season');
  const month = new Date().getMonth();
  const mine = useMemo(() => {
    if (!center || !W) return new Set();
    // themes this center has run: through its programs' idea labels is indirect; names are the honest signal here
    const names = (W.byCenter.get(center.id) || []).filter((p) => p.type === 'shabbat_dinner').map((p) => p.name.toLowerCase());
    return new Set((T?.themes || []).filter((t) => t.names.some((n) => names.some((m) => m.includes(n.toLowerCase().replace(/shabbat|shabbos|dinner/g, '').trim()) && n.toLowerCase().replace(/shabbat|shabbos|dinner/g, '').trim().length > 3))).map((t) => t.key));
  }, [center, W, T]);

  const list = useMemo(() => {
    if (!T) return [];
    let L = T.themes.filter((t) => t.centers_ever >= 3);
    if (view === 'season') L = L.filter((t) => t.seasonal && (t.months_ahead <= 2 || t.peak.includes(MONTHS[month]))).sort((a, b) => a.months_ahead - b.months_ahead || b.centers_now - a.centers_now);
    if (view === 'rising') L = L.filter((t) => t.status === 'rising' || t.adopters >= 4).sort((a, b) => (b.centers_now - b.centers_before) - (a.centers_now - a.centers_before));
    if (view === 'proven') L = L.filter((t) => ['strong', 'good'].includes(t.tier) && t.band && t.band !== 'below usual').sort((a, b) => (a.tier === 'strong' ? 0 : 1) - (b.tier === 'strong' ? 0 : 1) || b.centers_ever - a.centers_ever);
    if (view === 'all') L = L.sort((a, b) => b.centers_ever - a.centers_ever);
    return L;
  }, [T, view, month]);

  return (
    <>
      <PageHead title="Shabbos dinner themes" lede="Every themed dinner in the network, read against the same center's plain dinners. Which themes are spreading from campus to campus, which belong to this month, and which reliably bring more people than an ordinary Friday night." />
      <ErrorNote error={error} what="Themes" />
      {!T && <Loading />}
      {T && (
        <>
          <div className="mt-12"><Chips value={view} onChange={setView} options={VIEWS} label="View" /></div>
          <Section title={VIEWS.find((v) => v[0] === view)[1]} sub={view === 'season' ? 'Themes whose peak month is now or within the next two months, soonest first.' : view === 'rising' ? 'More distinct centers ran these in the last twelve months than in the twelve before, or at least four centers tried them for the first time.' : view === 'proven' ? 'Good or strong evidence, from many centers, that the theme lands above a center\'s plain dinners.' : `${list.length} themes used by three or more centers.`} tight>
            {list.length ? <div className="grid gap-5 md:grid-cols-2">{list.map((t) => <ThemeCard key={t.key} t={t} ran={mine.has(t.key)} center={center} W={W} />)}</div> : <Empty>Nothing fits this view right now.</Empty>}
          </Section>
          <p className="mt-12 c-small c-muted" style={{ lineHeight: 1.55 }}>Trend counts centers, not dinners, so one center that runs a theme weekly does not make a movement. A theme tied to a Jewish holiday can look like it appeared from nowhere when the holiday happens to fall on Shabbos; those are marked.</p>
        </>
      )}
    </>
  );
}

function ThemeCard({ t, ran, center, W }) {
  const calendarArtifact = t.status === 'calendar';
  const status = calendarArtifact ? 'On the calendar this year' : t.status === 'rising' ? 'Spreading' : t.status === 'fading' ? 'Fading' : t.status === 'steady' ? 'Steady' : 'Rare';
  const idea = W ? W.ideas.find((i) => i.type === 'shabbat_dinner' && i.label.toLowerCase() === t.label.toLowerCase()) : null;
  const p = center && idea ? W.pred(center.id, idea.id) : null;
  return (
    <article className="c-card flex gap-5">
      <Icon text={t.label} type="shabbat_dinner" />
      <div className="min-w-0 flex-1">
        <h3 className="c-h3">{shabbos(t.label)}</h3>
        <p className="mt-1.5 c-small c-muted">{t.centers_ever} {t.centers_ever === 1 ? 'center has' : 'centers have'} run it{t.adopters ? `, ${t.adopters} for the first time this year` : ''}.{t.seasonal && t.peak.length ? ` Peaks in ${t.peak.join(' and ')}.` : ''}{t.when && !t.seasonal ? ` ${t.when}.` : ''}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <span className={`c-tag ${t.status === 'rising' && !calendarArtifact ? 'c-tag-go' : ''}`}>{t.status === 'rising' && !calendarArtifact && <TrendingUp size={13} aria-hidden="true" />}{status}</span>
          <Effect band={t.band} /><Tier tier={t.tier} />
          {p && <Expect band={p.pct >= 0.75 ? 'top' : p.pct >= 0.5 ? 'upper' : p.pct >= 0.25 ? 'lower' : 'bottom'} />}
          {ran && <span className="c-tag">You have run this</span>}
        </div>
        {t.names.length > 1 && <p className="mt-3 c-small c-muted">Also called {t.names.slice(1, 4).map(shabbos).join(', ')}.</p>}
        {t.descs?.[0] && <p className="mt-3 c-small" style={{ lineHeight: 1.5 }}>How one center put it: “{shabbos(t.descs[0]).slice(0, 180)}{t.descs[0].length > 180 ? '…' : ''}”</p>}
        {t.paired?.length > 0 && <p className="mt-2 c-small c-muted">Often paired with {t.paired.slice(0, 3).map((x) => x.id).join(', ')}.</p>}
      </div>
    </article>
  );
}
