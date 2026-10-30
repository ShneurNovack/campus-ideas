import { useWorld } from '../lib/data';
import { PageHead, Section, Loading } from '../components/bits';

export default function About() {
  const { data: W } = useWorld();
  const e = W?.engine;
  return (
    <>
      <PageHead title="How it works" lede="Public event listings in, ideas and timing out. No center's numbers are shown to anyone, including the center itself." />
      {!e && <Loading />}
      {e && (
        <>
          <Section title="The data" tight>
            <div className="max-w-[68ch] space-y-5" style={{ lineHeight: 1.6 }}>
              <p>The engine reads the public, unauthenticated event listings of every Chabad on Campus center that uses the shared sign-up system: {e.counts.events.toLocaleString()} events at {e.counts.centers} centers since 2020, of which {e.counts.ran.toLocaleString()} ran. It is refreshed once a day. Sign-ups are not attendance, and many centers take sign-ups only for some events; those gaps are treated as unknown, never as empty rooms.</p>
              <p>Nothing here needs a login and nothing is scraped from social media. The app shows event names and blurbs the way centers posted them, and never a flyer.</p>
            </div>
          </Section>
          <Section title="How an outcome is read" tight>
            <div className="max-w-[68ch] space-y-5" style={{ lineHeight: 1.6 }}>
              <p>No event is compared with another center's. Each event is read against the same center's own median for the same kind of event within a year either side, and also placed among all of that center's events from smallest to biggest. Everything on these pages is built from those two within-center readings.</p>
              <p>An idea's effect across the network is the average of its events' lifts, pulled toward the average for its kind when there is little evidence, and tested by shuffling each center's sign-ups among its own events of that kind {e.lift.draws.toLocaleString()} times. The p-values are corrected for the number of ideas tested in each kind. That gives the tiers: strong, good, thin, or too little to tell. Of {e.lift.summary.ideas} ideas, {e.lift.summary.strong} have strong evidence and {e.lift.summary.good} good.</p>
              <p>Where an idea would land for a given center comes from the neighbor system: the past programs most like the idea, weighted by how alike their centers are to the asking center on turnout, pace, Jewish population and the profile of what lands high there, averaged on the percentile scale. Replayed on the last year, its top fifth of predictions landed in the top third of their center's events {Math.round((e.predictor?.test?.full?.top_fifth_in_top_third ?? 0.71) * 100)}% of the time and its bottom fifth {Math.round((e.predictor?.test?.full?.bottom_fifth_in_top_third ?? 0.06) * 100)}% of the time. The app shows the answer as a band, never a number.</p>
            </div>
          </Section>
          <Section title="Where the language model comes in" tight>
            <div className="max-w-[68ch] space-y-5" style={{ lineHeight: 1.6 }}>
              <p>A model (Gemma 4, with Qwen3 as a fallback, both on Cloudflare) does four narrow jobs: decides whether a past event is the same program as a typed idea, writes alternative titles to widen a search, writes new ideas in the spirit of past winners it is shown, and turns a verdict the code already computed into two plain sentences. It is never given a sign-up count or a center's name, and anything it writes is checked for numbers before it is shown.</p>
              <p>Search itself is not the model. A typed idea is embedded (bge-m3), compared with every past program in the browser, matched on wording with character n-grams, and the fifty closest are reordered by a cross-encoder before the model sees ten of them.</p>
            </div>
          </Section>
          <Section title="Timing" tight>
            <div className="max-w-[68ch] space-y-5" style={{ lineHeight: 1.6 }}>
              <p>Time tables keep two questions apart: how often the network uses a slot, and where events in that slot land against their own center's usual for the kind. Effects are shrunk toward zero with little evidence and tested by shuffling within each center. They are small, and shown small, because the event matters more than the hour.</p>
              <p>Shabbos themes are counted in centers, never in dinners. A theme is spreading when more distinct centers ran it in the last twelve months than the twelve before.</p>
            </div>
          </Section>
          <p className="mt-16 c-small c-muted">Engine run {new Date(e.generated).toLocaleString()}. {e.privacy}.</p>
        </>
      )}
    </>
  );
}
