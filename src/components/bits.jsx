import { useState } from 'react';
import { Utensils, BookOpen, Users, ChefHat, Flame, Mic2, Bus, Star, HandHeart, Flag, Wine, Sun, Sparkles, Candy, Tent, Music, Trophy, Palette, Heart, Coffee, Gamepad2, Film, Leaf, Shirt, Gift, ScrollText } from 'lucide-react';
import { TIER_WORDS, EFFECT_WORDS, BAND_WORDS, KIND_LABEL } from '../lib/data';

export function PageHead({ title, lede, action, children }) {
  return (
    <div className="pt-12 sm:pt-20 flex flex-wrap items-end justify-between gap-8">
      <div className="min-w-0 max-w-[760px]"><h1 className="c-h1">{title}</h1>{lede && <p className="c-lede mt-6">{lede}</p>}{children}</div>
      {action}
    </div>
  );
}
export function Section({ title, sub, action, children, id, tight }) {
  return (
    <section className={tight ? 'mt-14 sm:mt-16' : 'mt-20 sm:mt-28'} aria-labelledby={id}>
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3 mb-7">
        <div><h2 id={id} className="c-h2">{title}</h2>{sub && <p className="mt-2 c-muted max-w-[62ch]" style={{ lineHeight: 1.5 }}>{sub}</p>}</div>
        {action}
      </div>
      {children}
    </section>
  );
}
export const Empty = ({ children }) => <p className="c-card c-muted" style={{ lineHeight: 1.55 }}>{children}</p>;
export const ErrorNote = ({ error, what = 'This page' }) => (error ? <p className="mt-8 c-note" style={{ background: '#FBE9E7', color: 'var(--bad)' }} role="alert">{what} did not load: {error.message}. Reload to try again.</p> : null);
export const Loading = ({ rows = 3, h = 110 }) => <div className="space-y-5 mt-12" aria-busy="true" aria-label="Loading">{Array.from({ length: rows }).map((_, i) => <div key={i} className="c-skel" style={{ height: h }} />)}</div>;

export function Chips({ value, onChange, options, label, multi }) {
  const on = (k) => (multi ? value.includes(k) : value === k);
  const flip = (k) => (multi ? onChange(on(k) ? value.filter((x) => x !== k) : [...value, k]) : onChange(k));
  return <div className="flex flex-wrap gap-2" role="group" aria-label={label}>{options.map(([k, l]) => <button key={String(k)} type="button" className="c-chip" aria-pressed={on(k)} onClick={() => flip(k)}>{l}</button>)}</div>;
}

export const Tier = ({ tier }) => (tier && tier !== 'none' ? <span className={`c-tag ${tier === 'strong' ? 'c-tag-ink' : ''}`}>{TIER_WORDS[tier]}</span> : null);
export const Effect = ({ band }) => (band ? <span className={`c-tag ${band.startsWith('well') ? 'c-tag-go' : band === 'above usual' ? 'c-tag-go' : band === 'below usual' ? 'c-tag-warm' : ''}`}>{EFFECT_WORDS[band]}</span> : null);
export const Expect = ({ band }) => (band ? <span className={`c-tag ${band === 'top' ? 'c-tag-ink' : band === 'upper' ? 'c-tag-go' : ''}`}>{BAND_WORDS[band]}</span> : null);
export const Kind = ({ type }) => (type ? <span className="c-tag">{KIND_LABEL[type] || type}</span> : null);

// One icon per kind of event, and a second pass on words for the generator's formats. Never a flyer.
const WORDS = [
  [/sushi|food|dinner|meal|lunch|brunch|breakfast|pizza|bbq|cookout|grill|falafel|shawarma|taco|wings|bagel/i, Utensils],
  [/bake|challah|cook|kitchen|chef|hamantash|latke|matzah|donut|sufganiyot|cupcake|cookie/i, ChefHat],
  [/class|torah|learn|parsha|talmud|kabbalah|mysticism|lecture|study|sinai|jlearn/i, BookOpen],
  [/speaker|talk|conversation|panel|author|survivor|story/i, Mic2],
  [/trip|hike|retreat|shabbaton|travel|israel trip|ski|beach|kayak|camping/i, Bus],
  [/israel|idf|iron|unity/i, Flag],
  [/chesed|volunteer|mitzvah|pack|drive|donate|soup kitchen/i, HandHeart],
  [/chanukah|hanukkah|menorah|candle|havdalah|lag/i, Flame],
  [/sukk|tent/i, Tent],
  [/purim|costume|carnival|party|megillah/i, Candy],
  [/music|concert|karaoke|kumzitz|jam|sing|dj/i, Music],
  [/game|trivia|bingo|poker|escape|casino|quiz/i, Gamepad2],
  [/movie|film|screening/i, Film],
  [/paint|art|craft|canvas|pottery|design|candle making/i, Palette],
  [/sport|football|basketball|soccer|super bowl|tailgate|march madness|golf|pickleball|bowling|dodgeball/i, Trophy],
  [/coffee|cafe|tea|boba|smoothie/i, Coffee],
  [/women|men|girls|guys|mom|dad|family|parents|date/i, Users],
  [/shabbat|shabbos|friday|kabbalat/i, Wine],
  [/rosh|yom kippur|shofar|tashlich|high holiday|pesach|passover|seder|shavuos|shavuot|tu b|holiday/i, Star],
  [/garden|plant|outdoor|farm|nature|stars/i, Leaf],
  [/shirt|merch|swag|tie dye/i, Shirt],
  [/gift|care package|bag/i, Gift],
  [/mezuzah|tefillin|torah scroll|siyum/i, ScrollText],
  [/wellness|yoga|mental|health|meditation|self/i, Heart],
  [/welcome|kickoff|orientation|move in|open house|fair/i, Sun],
];
const BY_KIND = { shabbat_dinner: Wine, shabbat_lunch: Wine, shabbat_services: Wine, shabbat_other: Wine, holiday_meal: Star, holiday_event: Star, holiday_service: Star, class: BookOpen, meal: Utensils, social: Users, cooking: ChefHat, ritual: ScrollText, speaker: Mic2, israel: Flag, chesed: HandHeart, trip: Bus, outreach: Sun, other: Sparkles };
export function iconFor(text, type) {
  for (const [rx, I] of WORDS) if (rx.test(text || '')) return I;
  return BY_KIND[type] || Sparkles;
}
export function Icon({ text, type, size = 28, className = '' }) {
  const I = iconFor(text, type);
  return <span className={`inline-flex items-center justify-center shrink-0 ${className}`} style={{ width: size * 1.7, height: size * 1.7, background: 'var(--paper)' }} aria-hidden="true"><I size={size} strokeWidth={1.7} /></span>;
}

export function Details({ summary, children }) {
  const [open, setOpen] = useState(false);
  return <div><button type="button" className="c-link c-small" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'Hide' : summary}</button>{open && <div className="mt-3">{children}</div>}</div>;
}
