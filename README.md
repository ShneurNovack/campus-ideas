# Campus Ideas

An idea generator for Chabad on Campus centers, built on the public sign-up data of every center in the network
(about 54,000 events since 2020). Pick your center and it suggests events, tests an idea you type, finds the best
time for it, plans a week, and ranks Shabbos dinner themes. It uses a center's own history to answer, but never
shows that history to anyone: every readout is a band ("would land in your top fifth"), never a count.

- `engine/` the offline engine (Python, runs daily in GitHub Actions): pulls the public API, scores events against
  their own center, embeds them, mines themes, fits the lift and slot models, builds plans and writes JSON.
- `worker/` the Cloudflare Worker: serves the app and the engine's files, embeds and reranks typed ideas, and makes
  a few small language-model calls (comparables, synonyms, idea writing). The model never sees a number.
- `src/` the app (React).

How it works and why: [docs/ENGINE.md](docs/ENGINE.md). Deploys from `main` through Cloudflare Workers Builds.
