"""Run the whole engine:
prepare -> types -> embed -> score -> themes -> programs -> ideas -> centers -> timing -> home -> niches -> export.
Every stage is arithmetic, statistics or embeddings. There is no language-model call anywhere.

  python -m engine.pull        fetch the public API into data/raw (full backfill or refresh)
  python -m engine.run         compute everything into data/out
  python -m engine.run push    compute, then upload the app payloads to the Worker's bucket
  python -m engine.run app     only the Campus Ideas stages (lift, slots, trends, guests, payloads) from the saved state
"""
import pickle, sys, time
from .common import WORK, log
from . import prepare, types, score, embed, programs, themes, ideas, centers as centers_mod, neighbors, timing, home, niches, export, lift, slots, trends, speakers, app


def main(push=False):
    t0 = time.time()
    events, centers, pulled_at, term_info = prepare.run()
    today = pulled_at.date()
    events = types.run(events)
    cache = embed.Cache("emb_name", model=embed.MODEL)          # event names (and theme phrases)
    blurbs = embed.Cache("emb_blurb", model=embed.MODEL)
    events, T, texts, space = programs.attach_text(events, centers, cache, blurbs)
    events, type_info = programs.knn_types(events, T)
    events, centers, score_info = score.run(events, centers)
    theme_blob, sd = themes.build(events, centers, cache, today, T)
    events["tkeys"] = [[] for _ in range(len(events))]
    events.loc[sd.index, "tkeys"] = sd["tkeys"]
    events["one_off"] = False; events.loc[sd.index, "one_off"] = sd["one_off"]
    events["tlift"] = float("nan"); events.loc[sd.index, "tlift"] = sd["tlift"]
    P, V, events = programs.build(events, T)
    # themes were found before programs existed; give every event they mention its program
    ev_pid = events.set_index("id")["pid"].to_dict()
    for t in theme_blob["themes"]:
        for f in t["flyers"]:
            f["pid"] = int(ev_pid[f["id"]])
        t["occ"] = [o + [int(ev_pid[o[4]])] for o in t["occ"]]
    for f in theme_blob["fridays"]:
        for x in f["live"]:
            x["pid"] = int(ev_pid[x["id"]])
    for x in theme_blob["one_offs"]:
        x["pid"] = int(ev_pid[x["id"]])
    idea_list, P = ideas.build(P, V, events, centers, today)
    centers_blob, centers = centers_mod.build(centers, events, P, V, idea_list, today)
    state = dict(events=events, centers=centers, P=P, V=V, T=T, space=space, today=today, pulled_at=pulled_at, term_info=term_info, type_info=type_info, score_info=score_info,
                 theme_blob=theme_blob, themes=theme_blob, ideas=idea_list, centers_blob=centers_blob)
    # replay the past to see how well the neighbor system would have called it (settings were picked on the earlier year)
    state["check"] = {"test": neighbors.validate(events, centers, P, V, today, back=365), "tuning": neighbors.validate(events, centers, P, V, today, back=730)}
    log("neighbor system, last year replayed:", state["check"]["test"] and state["check"]["test"]["full"])
    state["timing"] = timing.build(events, centers, today)
    state["home"] = home.build(events, centers, P, idea_list, theme_blob, today)
    state["coming"] = home.coming_up(events, centers, P, today)
    state["niches"] = niches.build(events, P, today)
    pickle.dump(state, open(f"{WORK}/state.pkl", "wb"), protocol=4)
    export.write_all(state)                     # the Playbook payloads and the engine's self-checks (not published here)
    engine = build_app(state, push)
    log("engine finished in", round(time.time() - t0, 1), "s")
    return engine


def build_app(state, push=False):
    """The Campus Ideas stages: careful lifts, the time model, theme trends, guests, and the app payloads."""
    events, today = state["events"], state["today"]
    L = lift.run(events, state["P"], state["ideas"], state["theme_blob"], today)
    log("lift:", L["method"]["summary"])
    S = slots.build(events, today)
    T = trends.build(events, state["theme_blob"], L["themes"], today)
    G = speakers.run(events, lift.Pools(events), today)
    engine = app.write(state, L, S, T, G)
    if push:
        app.push()
    return engine


if __name__ == "__main__":
    if "app" in sys.argv[1:]:        # rebuild only the Campus Ideas stages from the saved state
        build_app(pickle.load(open(f"{WORK}/state.pkl", "rb")), push="push" in sys.argv[1:])
    else:
        main(push="push" in sys.argv[1:])
