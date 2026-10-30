"""Payloads for Campus Ideas. This is the only module that writes what the app can read, so it is also where the
privacy line is drawn:

  - no file here carries a sign-up count, a median, a spread or a per-center total. Centers are described by bands
    (categories.py) and a tracking flag; outcomes by effect bands and evidence tiers (lift.py).
  - what a center can expect from an idea is published as a percentile band among its own events ("would land in
    the top quarter of your events"), computed from the neighbor system (neighbors.py). The percentile is stored for
    every center and idea as one byte, so the browser never asks the server what it knows about a center.
  - event text (names, blurbs) is the public text the centers posted. Flyer images are not referenced anywhere.

Everything else (the search index, the model name) is the same as campus-playbook.
"""
import os, datetime as dt
import numpy as np
from collections import Counter
from . import embed, neighbors as nb, export, categories as cat
from .common import DATA, WORK, dump, r, log, clean
from .types import TYPES, EVENT_TYPES

APP = os.path.join(DATA, "app")
os.makedirs(APP, exist_ok=True)
KEEP_FACETS = ("size", "pace", "jewish", "strength", "stage")    # the bands the generator and neighbors use; the app shows none of them
PCT_BANDS = [(0.75, "top", "Top quarter of your events"), (0.5, "upper", "Upper half"), (0.25, "lower", "Lower half"), (0.0, "bottom", "Bottom quarter")]


def pct_band(p):
    if p is None or p != p:
        return None
    for cut, key, _ in PCT_BANDS:
        if p >= cut:
            return key
    return "bottom"


def predictions(state, cand):
    """centers x ideas percentile, through the neighbor system, as uint8 (0..200 = 0..1 in halves of a percent)."""
    P, V, events, centers, today = state["P"], state["V"], state["events"], state["centers"], state["today"]
    tab, _ = nb.center_table(events, centers, today)
    ids, CS, _ = nb.similarity(tab)
    pos = {c: i for i, c in enumerate(ids)}
    rows, y, n, ctr = nb.program_targets(P)
    Qi = embed.unit(np.stack([V[i["programs"]].mean(0) for i in cand]))
    Wt = np.clip(Qi @ V[rows].T, 0, None) ** nb.POWER * np.sqrt(n / (n + 1.0))[None, :]
    ci = np.array([pos[c] for c in ctr])
    placed = [c for c in ids if tab.at[c, "n3"] >= 15 or tab.at[c, "jpop"] == tab.at[c, "jpop"]]
    M = np.full((len(ids), len(cand)), 255, dtype=np.uint8)
    lo = np.full_like(M, 255); hi = np.full_like(M, 255)
    for c in placed:
        a = pos[c]
        pred, info = nb.pool(Wt * np.where(ctr == c, nb.OWN, CS[a][ci])[None, :], y, nb.K, quartiles=True)
        M[a] = np.clip(np.round(pred * 200), 0, 200).astype(np.uint8)
        lo[a] = np.clip(np.round(info["lo"] * 200), 0, 200).astype(np.uint8)
        hi[a] = np.clip(np.round(info["hi"] * 200), 0, 200).astype(np.uint8)
    return ids, M, lo, hi


def write(state, L, S, T, G):
    P, V, events, centers, ideas, today = state["P"], state["V"], state["events"], state["centers"], state["ideas"], state["today"]
    prof = state["centers_blob"]["profiles"]
    sizes = {}
    # ---- search index: the same reduced program vectors as campus-playbook
    q8, proj, Xn, search_info = export.project(V, state["space"]["mu_name"])
    q8.tofile(f"{APP}/vectors.bin"); proj.tofile(f"{APP}/proj.bin")
    dump({"model": state["space"]["model"], "names": state["space"]["names"], "blurbs": state["space"]["blurbs"]}, f"{APP}/embed.json")

    # ---- programs: public text only, with a percentile band for the program at its own center
    def band_of(pc, k):
        return pct_band(pc) if pc == pc and k and k >= 1 else None
    progs = []
    for row in P.itertuples():
        if row.type in ("resource", "togo") or (row.n + row.n_up) == 0:
            continue
        progs.append({"pid": int(row.pid), "name": clean(row.name)[:120], "short": clean(row.short or row.desc or "")[:200], "center": int(row.center), "type": row.type, "hol": row.hol if isinstance(row.hol, str) else None,
                      "n": int(row.n), "idea": int(row.idea) if row.idea == row.idea and row.idea is not None and row.idea >= 0 else None, "band": band_of(row.pc, row.n_pc),
                      "years": [int(y) for y in row.years][-4:], "wd": int(row.wd) if row.wd == row.wd else None, "hour": int(row.hour) if row.hour == row.hour else None, "aud": list(row.aud or [])})
    sizes["programs.json"] = dump({"programs": progs}, f"{APP}/programs.json")

    # ---- ideas: network facts, evidence tiers, effect bands, seasonality
    cand = [i for i in ideas if not i.get("staple") and i["type"] in EVENT_TYPES and i["centers"] >= 4]
    cand_pos = {i["id"]: j for j, i in enumerate(cand)}
    idea_rows = []
    for i in ideas:
        lr = L["ideas"].get(i["id"], {})
        idea_rows.append({"id": i["id"], "label": i["label"], "type": i["type"], "hol": i.get("hol"), "centers": i["centers"], "n": i["n"], "scored": lr.get("n", 0),
                          "tier": lr.get("tier", "none"), "band": lr.get("band"), "hit": r(lr.get("hit"), 2) if lr.get("hit") is not None else None,
                          "status": i.get("status"), "when": i.get("when"), "months": i.get("months"), "wd": i.get("wd"), "hour": i.get("hour"), "aud": i.get("aud", []),
                          "years": [{"ay": y["ay"], "centers": y["centers"]} for y in i.get("years", [])], "names": [n["name"] for n in i.get("names", [])[:5]],
                          "rep": [int(p) for p in i.get("rep", [])[:4]], "related": i.get("related", [])[:6], "xy": i.get("xy"), "pred": cand_pos.get(i["id"])})
    sizes["ideas.json"] = dump({"ideas": idea_rows, "types": TYPES, "tiers": L["method"]["tiers"], "bands": L["method"]["bands"]}, f"{APP}/ideas.json")

    # ---- predictions
    ids, M, lo, hi = predictions(state, cand)
    np.stack([M, lo, hi]).tofile(f"{APP}/pred.bin")
    sizes["pred.bin"] = M.size * 3

    # ---- centers: bands, nearest centers, tracking flag, strengths as words. No numbers.
    kind_word = lambda edge: "strength" if edge >= 0.04 else "weaker" if edge <= -0.04 else "usual"
    crow = {}
    txt = lambda v: clean(v) if isinstance(v, str) else ""
    for c, row in centers.iterrows():
        p = prof.get(int(c), {})
        kinds = [{"key": k["key"], "label": k["label"], "read": kind_word(k["edge"])} for k in p.get("kinds", []) if k["n"] >= 5]
        crow[int(c)] = {"id": int(c), "name": txt(row["name"]), "nick": txt(row.get("nick")), "campus": txt(row.get("campus")), "state": row.get("state") if isinstance(row.get("state"), str) else None, "region": row.get("region") if isinstance(row.get("region"), str) else None,
                        "cat": {k: v for k, v in p.get("cat", {}).items() if k in KEEP_FACETS}, "basis": p.get("basis", "none"), "near": [x["id"] for x in p.get("near", [])], "tracking": L["tracking"].get(int(c), "rare"), "kinds": kinds,
                        "picks": {"new": [{"idea": x["idea"], "band": pct_band(x["pct"])} for x in p.get("picks", {}).get("new", [])],
                                  "again": [{"idea": x["idea"], "band": pct_band(x["pct"])} for x in p.get("picks", {}).get("again", [])]},
                        "best": [{"pid": b["pid"], "band": pct_band(b["pct"])} for b in p.get("best", [])],
                        "slots": S["centers"].get(int(c), {})}
    facets = [{"key": f["key"], "label": f["label"], "values": [{"key": k, "label": lab} for k, lab, _ in f["values"]]} for f in cat.FACETS if f["key"] in KEEP_FACETS]
    sizes["centers.json"] = dump({"centers": list(crow.values()), "pred": {"ids": [int(c) for c in ids], "ideas": [i["id"] for i in cand], "scale": 200, "none": 255, "bands": [{"key": k, "label": lab, "from": cut} for cut, k, lab in PCT_BANDS]},
                                  "facets": facets}, f"{APP}/centers.json")

    # ---- time, themes, guests
    sizes["slots.json"] = dump({k: v for k, v in S.items() if k != "centers"}, f"{APP}/slots.json")
    sizes["themes.json"] = dump(T, f"{APP}/themes.json")
    sizes["guests.json"] = dump(G, f"{APP}/guests.json")

    ev = events
    engine = {"generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", "pulled_at": state["pulled_at"].isoformat() + "Z", "today": today.isoformat(),
              "counts": {"events": int(len(ev)), "ran": int(ev["ran"].sum()), "centers": int(len(centers)), "scored": int(ev["scored"].sum()), "programs": len(progs), "ideas": len(ideas),
                         "themes": len(T["themes"]), "guests": len(G["guests"]), "by_year": {int(y): int(n) for y, n in Counter(d.year for d in ev.loc[ev["ran"], "date"]).items()}},
              "lift": L["method"], "slots": S["method"], "trends": T["summary"], "search": search_info, "predictor": state.get("check"), "sizes": sizes, "model": state["space"]["model"],
              "privacy": "no sign-up counts leave the engine; centers are bands, outcomes are bands and tiers, expectations are percentile bands"}
    sizes["engine.json"] = dump(engine, f"{APP}/engine.json")
    log("app export:", {k: f"{v // 1024} KB" for k, v in sizes.items()})
    return engine


def push():
    from . import export as ex
    out = ex.OUT
    ex.OUT = APP
    try:
        ex.push()
    finally:
        ex.OUT = out
