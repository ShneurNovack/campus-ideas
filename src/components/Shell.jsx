import { useEffect, useMemo, useRef, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { Sparkles, FlaskConical, Clock, CalendarDays, Wine, BarChart3, Mic2, Info, MapPin, X, Search } from 'lucide-react';
import { useWorld, useMyCenter, centerLabel } from '../lib/data';

const NAV = [
  ['/', 'Ideas', Sparkles], ['/test', 'Test an idea', FlaskConical], ['/time', 'Best time', Clock], ['/week', 'Plan my week', CalendarDays],
  ['/shabbos', 'Shabbos themes', Wine], ['/metrics', 'When things run', BarChart3], ['/guests', 'Guests', Mic2], ['/about', 'How it works', Info],
];

export default function Shell() {
  const { pathname } = useLocation();
  const [pick, setPick] = useState(false);
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return (
    <div className="min-h-screen flex flex-col">
      <header className="max-w-[1180px] w-full mx-auto px-5 sm:px-8 pt-7 flex items-center justify-between gap-6">
        <NavLink to="/" className="c-display text-[1.35rem] font-bold">Campus Ideas</NavLink>
        <CenterButton onOpen={() => setPick(true)} />
      </header>
      <nav className="max-w-[1180px] w-full mx-auto px-5 sm:px-8 mt-6 overflow-x-auto c-scroll" aria-label="Pages">
        <div className="flex gap-x-7 gap-y-2 whitespace-nowrap text-[0.95rem]">
          {NAV.map(([to, label, Icon]) => <NavLink key={to} to={to} end={to === '/'} className="inline-flex items-center gap-2 py-2"><Icon size={16} strokeWidth={2.2} aria-hidden="true" />{label}</NavLink>)}
        </div>
      </nav>
      <main className="max-w-[1180px] w-full mx-auto px-5 sm:px-8 flex-1 pb-28"><Outlet /></main>
      <footer className="max-w-[1180px] w-full mx-auto px-5 sm:px-8 pb-12 c-small c-muted">
        Built on the public event listings of Chabad on Campus centers. The app never shows what it knows about any one center, including yours.
      </footer>
      {pick && <CenterPicker onClose={() => setPick(false)} />}
    </div>
  );
}

function CenterButton({ onOpen }) {
  const { data: W } = useWorld();
  const [my] = useMyCenter();
  const c = W && my ? W.centerById.get(my) : null;
  return (
    <button type="button" className="c-btn c-btn-soft c-btn-sm" onClick={onOpen} aria-haspopup="dialog">
      <MapPin size={15} aria-hidden="true" />{c ? centerLabel(c) : 'Choose your center'}
    </button>
  );
}

export function CenterPicker({ onClose }) {
  const { data: W } = useWorld();
  const [my, setMy] = useMyCenter();
  const [q, setQ] = useState('');
  const ref = useRef(null);
  useEffect(() => { ref.current?.focus(); const k = (e) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k); }, [onClose]);
  const list = useMemo(() => {
    if (!W) return [];
    const s = q.trim().toLowerCase();
    const all = [...W.centers].sort((a, b) => centerLabel(a).localeCompare(centerLabel(b)));
    return s ? all.filter((c) => `${c.name} ${c.nick} ${c.campus} ${c.state || ''}`.toLowerCase().includes(s)) : all;
  }, [W, q]);
  return (
    <div role="dialog" aria-modal="true" aria-label="Choose your center" className="fixed inset-0 z-50 flex items-start justify-center p-4 sm:p-10" style={{ background: 'rgba(11,11,12,0.55)' }} onClick={onClose}>
      <div className="w-full max-w-[640px] bg-white p-6 sm:p-8 max-h-[86vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-6">
          <div><h2 className="c-h2">Your center</h2><p className="c-muted mt-2" style={{ lineHeight: 1.5 }}>Ideas, timing and the week plan are tuned to the center you choose. It stays in this browser. Nothing about it is shown on screen.</p></div>
          <button type="button" className="c-btn c-btn-soft c-btn-sm" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        <div className="mt-6 relative"><Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 c-muted" aria-hidden="true" /><input ref={ref} className="c-input pl-10" placeholder="Search by school or city" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search centers" /></div>
        <div className="mt-4 overflow-y-auto flex-1 c-scroll">
          {my && <button type="button" className="w-full text-left py-2.5 px-3 font-semibold c-muted hover:bg-[var(--fog)]" onClick={() => { setMy(null); onClose(); }}>No center (network view)</button>}
          {list.map((c) => (
            <button key={c.id} type="button" className="w-full text-left py-2.5 px-3 hover:bg-[var(--fog)] flex items-baseline justify-between gap-4" aria-pressed={c.id === my} style={c.id === my ? { background: 'var(--fog)' } : undefined} onClick={() => { setMy(c.id); onClose(); }}>
              <span className="font-semibold">{c.name}</span><span className="c-small c-muted shrink-0">{c.state || ''}</span>
            </button>
          ))}
          {!list.length && <p className="c-muted p-3">No center matches.</p>}
        </div>
      </div>
    </div>
  );
}
