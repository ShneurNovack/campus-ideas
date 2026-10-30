# How Campus Ideas works

Campus Ideas is an idea generator, idea tester and scheduling helper for Chabad on Campus centers, built on the public
event listings of every center that uses the shared sign-up system. It shares its data stages with Campus Playbook
(prepare, types, embed, score, themes, programs, ideas, centers and neighbors; see the Playbook's `docs/ENGINE.md`
for those) and adds the stages below. The app never shows a sign-up count, a center's bands, or any flyer.

## The privacy line

`engine/app.py` is the only module that writes what the browser can read. The rule it enforces: no file carries a
sign-up count, a median, a spread or a per-center total. Centers are published as five bands (turnout at big events,
pace, Jewish students, strongest family of events, stage), a tracking flag, and a list of their twelve nearest
centers. The bands feed the generator's prompt and the neighbor weighting; no page shows them. Outcomes are
published as effect bands and evidence tiers. What a center can expect from an idea is published as one byte per
center and idea (`pred.bin`): a percentile among that center's own events, shown only as a band in words.

The language model is never given a count or a center's name. `worker/llm.js` strips any number that appears near a
sign-up word from everything the model writes, and the app checks again before showing a sentence.

## New stages (`python -m engine.run`, or `python -m engine.run app` from the saved state)

**lift** (`engine/lift.py`). Every scored event already has a within-center lift (its sign-ups against the same
center's median for the same kind within a year either side). Ideas, Shabbos themes and guests are read from those
lifts with three corrections: empirical-Bayes shrinkage toward the kind's mean (between-group variance by the method
of moments per kind), a within-center permutation test (each event's lift redrawn from its own center-kind pool,
2,000 draws), and Benjamini-Hochberg correction per kind. The result is a tier (strong: q at most 0.05 from five or
more centers and eight events; good: q at most 0.15 from three centers and five events; thin; none) and an effect
band (well above usual at x1.5, above usual at x1.15, about usual, below usual at x0.87). First run: 992 ideas, 48
strong, 32 good; 73 themes, 24 strong.

**slots** (`engine/slots.py`). For each of fifteen kinds of event, two maps over weekday x six time blocks: the share
of events held there (all events, last three years), and where events in that slot land against their own center's
usual for the kind (percentile points, within-center deviations, shrunk by n/(n+30), permutation-tested within
center-kind). A slot is trusted at 25 scored events from 6 centers. Each center also gets its own cells per kind,
blended with the network by evidence (own weight k against 6 for the network). The effects are a few percentile
points; the app says so.

**trends** (`engine/trends.py`). Shabbos themes counted in centers: rising when more distinct centers ran the theme
in the last twelve months than the twelve before (minimum three), fading at 0.7x, "calendar" when a holiday theme
appears because the holiday fell on Shabbos this year. Seasonality is the concentration of the top two months and
how far ahead the peak is.

**speakers** (`engine/speakers.py`). A wide regex picks the events that look like they feature a person (6,142 of
54,607). The Worker's `/analyst/extract` route has the model read those in batches of twenty and write the real
names and a role; answers are cached by event id (`speakers_cache.json`, cached in the daily workflow), so a daily
run reads only new events. Names are folded with rapidfuzz (token sort ratio 88) and read through lift. First run:
183 people named, 31 published (two centers, or one center twice). Guests are a small part of the record.

**app** (`engine/app.py`). Payloads: `ideas.json`, `programs.json` (public text only), `centers.json` (bands,
neighbors, picks as bands), `pred.bin`, `slots.json`, `themes.json`, `guests.json`, `engine.json`, and the search
index (`vectors.bin`, `proj.bin`, `embed.json`, same as the Playbook). Pushed to the Worker's bucket under `public/`.

## In the browser

- **Search** (`src/lib/search.js`): the typed text is embedded by the Worker (bge-m3), compared with every program's
  256-dim int8 vector, and matched on character n-grams of names. The top 50 are reordered by a cross-encoder
  (`/api/rerank`, bge-reranker-base). The top 10 are labeled same / variant / different by the model (`/api/judge`),
  asked in both orders; only agreeing labels are kept. If the Worker is busy, the page answers from the first two
  steps and says so.
- **Expectation** (`src/lib/predict.js`): the twelve closest published ideas (cosine to the 4th power) vote with
  their `pred.bin` percentile for the chosen center. Shown as a band.
- **Generator** (`src/pages/Home.jsx`): exemplars are ideas with evidence that fit the kind, season and audience,
  preferring ones the chosen center has not run and that the neighbor system expects to land high there, thinned
  so no two are closer than 0.55. The model (`/api/generate`) writes twelve ideas from names and bands; each is then
  embedded and placed like a tested idea.
- **Best time / Plan my week** (`src/lib/slots.js`, `src/pages/Week.jsx`): the slot maps, blended with the
  center's own history. The planner is coordinate ascent over the kinds chosen, with a cost for a second event on a
  day and a smaller one for back-to-back days. The report proposed CP-SAT; the problem is six events over 42 slots,
  so a few passes of coordinate ascent reach the same answer in the browser.

## Models

Workers AI: `@cf/google/gemma-4-26b-a4b-it` for every JSON call (with `chat_template_kwargs.enable_thinking: false`,
without which it spends its whole budget thinking), `@cf/qwen/qwen3-30b-a3b-fp8` as the fallback, `@cf/baai/
bge-reranker-base` for reranking, `@cf/baai/bge-m3` for embeddings. JSON mode (`response_format: json_schema`) is
honored by both chat models; answers are still validated and retried once with the error.

## Running it

```
pip install -r engine/requirements.txt
python -m engine.pull             # public API -> engine/data/raw
python -m engine.run push         # everything, then publish the app payloads
python -m engine.run app push     # only the Campus Ideas stages from the saved state
```

`ANALYST_KEY` (the Worker's secret) is needed for the speaker extraction and the push; the daily workflow uses a
GitHub OIDC token instead.
