// Guests and speakers: who gets brought to campus, how widely, and how those events land.
import { useState } from 'react';
import { Mic2 } from 'lucide-react';
import { useData, shabbos } from '../lib/data';
import { PageHead, Section, Chips, Tier, Effect, Loading, ErrorNote, Empty } from '../components/bits';

const ROLES = { speaker: 'Speaker', survivor: 'Survivor', performer: 'Performer', chef: 'Chef', author: 'Author', athlete: 'Athlete', politician: 'Public figure', soldier: 'Soldier', scholar: 'Scholar', other: 'Guest' };

export default function Guests() {
  const { data, error } = useData('guests.json');
  const G = data?.[0];
  const [role, setRole] = useState('all');
  const list = G ? G.guests.filter((g) => role === 'all' || g.role === role) : [];
  const roles = G ? [...new Set(G.guests.map((g) => g.role))] : [];
  return (
    <>
      <PageHead title="Guests and speakers" lede="People centers have brought to campus, read out of the event listings themselves. Who has been to more than one center, who gets brought back, and how those evenings land." />
      <ErrorNote error={error} what="Guests" />
      {!G && <Loading />}
      {G && (
        <>
          <div className="mt-12"><Chips value={role} onChange={setRole} options={[['all', 'Everyone'], ...roles.map((r) => [r, ROLES[r] || r])]} label="Role" /></div>
          <Section title={`${list.length} ${list.length === 1 ? 'guest' : 'guests'}`} sub="Listed when two centers hosted them, or one center brought them back. Events that name nobody, and a center's own staff, are left out." tight>
            {list.length ? (
              <div className="grid gap-5 md:grid-cols-2">
                {list.map((g) => (
                  <article key={g.name} className="c-card flex gap-5">
                    <span className="inline-flex items-center justify-center shrink-0 bg-white" style={{ width: 48, height: 48 }} aria-hidden="true"><Mic2 size={24} strokeWidth={1.7} /></span>
                    <div className="min-w-0">
                      <h3 className="c-h3">{g.name}</h3>
                      <p className="mt-1.5 c-small c-muted">{ROLES[g.role] || g.role} · {g.centers} {g.centers === 1 ? 'center' : 'centers'}, {g.events} {g.events === 1 ? 'event' : 'events'} · {g.years.join(', ')}</p>
                      <div className="mt-3 flex flex-wrap gap-2"><Effect band={g.band} /><Tier tier={g.tier} />{g.recent ? <span className="c-tag">Active this year</span> : null}</div>
                      {g.titles?.length > 0 && <p className="mt-3 c-small" style={{ lineHeight: 1.5 }}>{g.titles.slice(0, 3).map(shabbos).join(' · ')}</p>}
                    </div>
                  </article>
                ))}
              </div>
            ) : <Empty>No guests in this group.</Empty>}
          </Section>
          <p className="mt-12 c-small c-muted" style={{ lineHeight: 1.55 }}>Speakers are a small part of the sign-up record ({G.read.toLocaleString()} listings named a person). A guest's effect band is read the same way as an idea's: against each hosting center's usual for that kind of event, with thin evidence marked as thin.</p>
        </>
      )}
    </>
  );
}
