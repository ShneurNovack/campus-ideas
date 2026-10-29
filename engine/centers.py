"""Centers. What each center is like, which centers are nearest to it, and what should do well there.

  measures    how many sign up (typical and biggest), events a week, Jewish students served, the share of them it
              draws, and which kinds of event run above its own usual (neighbors.center_table)
  groups      a band on each measure, with written cut points (categories.py)
  neighbors   the centers closest on draw, pace, Jewish numbers and profile of what does well (neighbors.similarity)
  picks       for every center, the ideas the neighbor system expects to beat its usual event, split into ones it
              has never run and ones it has
  structure   a test of whether centers form natural clumps on all measures together. They do not, which is why
              the groups are bands and likeness is left to nearest neighbors.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from . import embed, categories as cat, neighbors as nb
from .common import r, log
from .types import TYPES, EVENT_TYPES

GROUPS = {"shabbat": ["shabbat_dinner", "shabbat_lunch", "shabbat_services", "shabbat_other"], "holiday": ["holiday_meal", "holiday_service", "holiday_event"],
          "learning": ["class", "speaker"], "social": ["social", "meal", "cooking", "trip"], "other": ["israel", "chesed", "ritual", "outreach", "other"]}
GROUP_LABEL = {"shabbat": "Shabbat", "holiday": "Holidays", "learning": "Classes", "social": "Social and food", "other": "Everything else"}
MIN_EVENTS = 15
N_NEAR = 12
WHY = {"top5": "draw", "pace": "pace", "jpop": "jewish", "profile": "profile"}


def _ay(d):
    return d.year if d.month >= 7 else d.year - 1


def _display(centers, events, today):
    """Plain per-center facts the pages show (calendar mix, history by year, first and last event)."""
    ev = events[events["ran"]]
    rows = []
    for c in centers.index:
        g_all = ev[ev["center"] == c]
        g = g_all[g_all["type"].isin(EVENT_TYPES)]
        age = g["date"].map(lambda d: (today - d).days) if len(g) else pd.Series([], dtype=float)
        first, last = (g["date"].min(), g["date"].max()) if len(g) else (None, None)
        base = g[age <= 1095] if (age <= 1095).sum() >= 30 else g
        tm, tm_all = Counter(base["type"]), Counter(g_all["type"])
        sd = g[g["type"] == "shabbat_dinner"]; sdm = sd[sd["regs"] > 0]
        yrs = defaultdict(lambda: {"events": 0, "sd": []})
        for d, t, regs in zip(g["date"], g["type"], g["regs"]):
            y = _ay(d); yrs[y]["events"] += 1
            if t == "shabbat_dinner" and regs > 0:
                yrs[y]["sd"].append(regs)
        up = events[events["upcoming"] & (events["center"] == c)]
        rows.append({
            "id": c, "n": len(g), "n_all": len(g_all), "med_regs": float(g.loc[g["regs"] > 0, "regs"].median()) if (g["regs"] > 0).any() else np.nan,
            "sd_max": int(sdm["regs"].max()) if len(sdm) else 0,
            "mix": {k: sum(tm.get(t, 0) for t in ts) / max(1, len(base)) for k, ts in GROUPS.items()}, "types": {t: int(v) for t, v in tm_all.most_common()},
            "themed": float((sd["tkeys"].map(len) > 0).mean()) if len(sd) >= 8 else np.nan, "one_time": float((~g["recurring"]).mean()) if len(g) else np.nan,
            "wd": [int((g["wd"] == i).sum()) for i in range(7)], "first": first, "last": max([last] + list(up["date"])) if last is not None and len(up) else last,
            "n_now": int((age <= 365).sum()) if len(g) else 0, "n_before": int(((age > 365) & (age <= 730)).sum()) if len(g) else 0, "upcoming": int(len(up)),
            "traj": [{"ay": f"{y}-{str(y + 1)[2:]}", "events": yrs[y]["events"], "sd": r(float(np.median(yrs[y]["sd"])), 1) if len(yrs[y]["sd"]) >= 3 else None, "sd_n": len(yrs[y]["sd"])} for y in sorted(yrs) if y >= 2020],
        })
    return pd.DataFrame(rows).set_index("id")


def _structure(X):
    """Do the centers clump? k-means for 3 to 6 groups: how separate the groups are (silhouette) and how often a
    reshuffled sample of the same centers lands in the same groups (stability, 1 is always)."""
    out = []
    rng = np.random.default_rng(1)
    for k in range(3, 7):
        lab = KMeans(n_clusters=k, n_init=20, random_state=7).fit(X).labels_
        aris = []
        for b in range(20):
            ix = rng.choice(len(X), len(X), replace=True)
            aris.append(adjusted_rand_score(lab, KMeans(n_clusters=k, n_init=5, random_state=b).fit(X[ix]).predict(X)))
        out.append({"k": k, "silhouette": r(float(silhouette_score(X, lab)), 3), "stability": r(float(np.mean(aris)), 2)})
    return out


def _value_profiles(T, cats, ideas, P, readable):
    """For every band of every measure: who is in it, what is typical there, and which ideas set it apart."""
    prof = set(readable)
    ever = {i["id"]: set(P.loc[i["programs"], "center"]) & prof for i in ideas}
    S = P[P["pc"].notna() & (P["idea"] >= 0)]
    out = []
    for f in cat.FACETS:
        vals = []
        for key, label, definition in f["values"]:
            members = [c for c in T.index if cats[c][f["key"]] == key]
            if not members:
                continue
            mp = [c for c in members if c in prof]; ms = set(mp); g = T.loc[mp]
            med = lambda col, nd=0: r(float(g[col].median()), nd) if len(g) and g[col].notna().any() else None
            entry = {"key": key, "label": label, "def": definition, "n": len(members), "n_readable": len(mp),
                     "centers": [int(c) for c in sorted(members, key=lambda c: -(T.at[c, "top5"] if T.at[c, "top5"] == T.at[c, "top5"] else -1))],
                     "stats": {"typical": med("usual"), "biggest": med("top5"), "pace": med("pace", 1), "jewish": med("jpop"), "share": med("draw_share", 3), "themed": med("themed", 2)},
                     "ideas": [], "works": []}
            if key != "unknown" and len(mp) >= 5:
                n_in, n_out = len(mp), max(1, len(prof) - len(mp))
                dist = []
                for i in ideas:
                    if i.get("staple"):
                        continue
                    k_in = len(ever[i["id"]] & ms); k_out = len(ever[i["id"]]) - k_in
                    if k_in >= 3 and k_in / n_in >= 0.25 and k_in / n_in - k_out / n_out >= 0.15:
                        dist.append({"idea": i["id"], "in": k_in, "share": r(k_in / n_in, 2), "elsewhere": r(k_out / n_out, 2)})
                entry["ideas"] = sorted(dist, key=lambda x: -(x["share"] - x["elsewhere"]))[:8]
                if f["key"] in ("size", "typical", "pace", "jewish", "share", "strength"):
                    sub = S[S["center"].isin(ms)]
                    works = []
                    for iid, gg in sub.groupby("idea"):
                        idea = ideas[int(iid)]
                        if idea.get("staple") or gg["center"].nunique() < 4:
                            continue
                        w = gg["n_pc"].values / (gg["n_pc"].values + 1.0)
                        here = float(((gg["pc"].values * w).sum() + 2.0 * 0.5) / (w.sum() + 2.0))
                        allp = S[S["idea"] == iid]; wa = allp["n_pc"].values / (allp["n_pc"].values + 1.0)
                        works.append({"idea": int(iid), "pct": r(here, 2), "centers": int(gg["center"].nunique()), "runs": int(gg["n_pc"].sum()),
                                      "overall": r(float(((allp["pc"].values * wa).sum() + 1.0) / (wa.sum() + 2.0)), 2)})
                    entry["works"] = sorted([w_ for w_ in works if w_["pct"] >= 0.65], key=lambda x: -x["pct"])[:6]
            vals.append(entry)
        out.append({"key": f["key"], "label": f["label"], "question": f["question"], "basis": f["basis"], "values": vals})
    return out


def build(centers, events, P, V, ideas, today):
    tab, net = nb.center_table(events, centers, today)
    T = centers.drop(columns=[c_ for c_ in tab.columns if c_ in centers.columns]).join(tab).join(_display(centers, events, today))
    cats = {}
    for c, row in T.iterrows():
        cats[c] = {"size": cat.size(row), "typical": cat.typical(row), "pace": cat.pace(row), "jewish": cat.jewish(row), "share": cat.share(row),
                   "strength": cat.strength(row), "signups": cat.signups(row), "stage": cat.stage(row["first"], row["last"], row["n_now"], row["n_before"], today)}
    ids, CS, parts = nb.similarity(tab)
    pos = {c: i for i, c in enumerate(ids)}
    readable = [c for c in ids if T.at[c, "n3"] >= MIN_EVENTS]
    placed = [c for c in ids if T.at[c, "n3"] >= MIN_EVENTS or T.at[c, "jpop"] == T.at[c, "jpop"]]     # something to match on

    # ---- nearest centers
    near = {}
    for c in ids:
        if c not in placed:
            near[c] = []
            continue
        a = pos[c]
        order = [b for b in np.argsort(-CS[a]) if b != a and ids[b] in readable][:N_NEAR]
        near[c] = [{"id": int(ids[b]), "sim": r(float(CS[a, b]), 3), "why": [WHY[k] for k in parts if parts[k][a, b] <= 0.5 and not (parts[k][a, b] == nb.UNKNOWN_GAP)]} for b in order]

    # ---- what should do well at each center
    rows, y, n, ctr = nb.program_targets(P)
    cand = [i for i in ideas if not i.get("staple") and i["type"] in EVENT_TYPES and i["centers"] >= 4]
    Qi = embed.unit(np.stack([V[i["programs"]].mean(0) for i in cand]))
    Wt = np.clip(Qi @ V[rows].T, 0, None) ** nb.POWER * np.sqrt(n / (n + 1.0))[None, :]
    ci = np.array([pos[c] for c in ctr])
    # Has the center already run it? Yes if it is in the idea, if one of its own programs sits as close to the idea
    # as the idea's looser members do, or (for a holiday) if it already runs that kind of event for that holiday.
    own_sim = np.concatenate([V[i["programs"]] @ q for i, q in zip(cand, Qi)])
    close = float(np.clip(np.percentile(own_sim, 5), 0.5, 0.85))
    ran, kinds_run = {}, {}
    for c, g_ in P.groupby("center"):
        ran[c] = (V[g_["pid"].values] @ Qi.T).max(0) >= close
        kinds_run[c] = {(t, h) for t, h in zip(g_["type"], g_["hol"]) if isinstance(h, str)}
    twin = (Qi @ Qi.T) >= 0.68                 # two ideas this close are one pick, not two
    from .ideas import _canon
    icanon = [set(_canon(i["label"].lower()).split()) for i in cand]
    names_run = {c_: [set(_canon(n_).split()) for n_ in g_["nname"]] for c_, g_ in P.groupby("center")}
    idea_centers = [set(P.loc[i["programs"], "center"]) for i in cand]
    idea_pc = [{c_: (pc, k_, mr) for c_, pc, k_, mr in zip(P.loc[i["programs"], "center"], P.loc[i["programs"], "pc"], P.loc[i["programs"], "n_pc"], P.loc[i["programs"], "med_regs"])} for i in cand]

    def thin(items, limit):
        kept, idx = [], []
        for x in items:
            if not any(twin[x["_j"], j2] for j2 in idx):
                kept.append({k_: v for k_, v in x.items() if k_ != "_j"}); idx.append(x["_j"])
            if len(kept) >= limit:
                break
        return kept

    picks = {}
    for c in ids:
        if c not in placed:
            picks[c] = {"new": [], "again": []}
            continue
        a = pos[c]
        pred, info = nb.pool(Wt * np.where(ctr == c, nb.OWN, CS[a][ci])[None, :], y, nb.K, quartiles=True)
        spread = T.at[c, "spread"]
        spread = spread if isinstance(spread, list) else None
        nn = [x["id"] for x in near[c]]
        new, again = [], []
        for j, i in enumerate(cand):
            member = c in idea_centers[j]
            # close to something it already runs: by meaning, by name, or the same kind of event for the same holiday
            did = member or bool(ran[c][j] if c in ran else False) or (bool(icanon[j]) and any(icanon[j] <= w_ for w_ in names_run.get(c, ()))) \
                or (isinstance(i.get("hol"), str) and (i["type"], i["hol"]) in kinds_run.get(c, ()))
            at = [idea_pc[j][d] for d in nn if d in idea_pc[j]]
            scored = [v for v, k_, _ in at if v == v and k_]
            exp = nb.sign_ups(spread, pred[j])
            lo_, hi_ = (nb.sign_ups(spread, info["lo"][j]), nb.sign_ups(spread, info["hi"][j])) if spread else (None, None)
            item = {"_j": j, "idea": i["id"], "pct": r(float(pred[j]), 2), "expect": r(min(max(exp, lo_), hi_), 0) if exp is not None else None,
                    "lo": r(lo_, 0) if spread else None, "hi": r(hi_, 0) if spread else None,
                    "near": len(at), "near_pct": r(float(np.mean(scored)), 2) if scored else None, "eff": r(float(info["eff"][j]), 0)}
            if member:
                mine = idea_pc[j].get(c)
                if mine and mine[0] == mine[0] and mine[1]:
                    item["mine"] = {"pct": r(float(mine[0]), 2), "regs": r(float(mine[2]), 0) if mine[2] == mine[2] and mine[2] is not None else None, "runs": int(mine[1])}
                again.append(item)
            elif not did and len(at) >= 2:          # new here, and at least two of its nearest centers have run it
                new.append(item)
        picks[c] = {"new": thin(sorted([x for x in new if x["pct"] >= 0.6], key=lambda x: -x["pct"]), 20), "again": thin(sorted(again, key=lambda x: -x["pct"]), 12)}

    # ---- profiles
    out = {}
    for c, row in T.iterrows():
        fs = cat.family_strengths(row) if row["scored3"] >= cat.MIN_SCORED else None
        kinds = [{"key": t, "label": nb.FAM_LABEL[t], "n": int(row["nfam"][t]), "pct": r(row["level"][t], 2), "network": r(net[t], 2), "edge": r(row["strength"][t], 3)} for t in nb.FAM if row["nfam"][t] >= 3]
        mine = P[(P["center"] == c) & P["pc"].notna() & (P["n_pc"] >= 1)].sort_values(["pc", "med_regs"], ascending=False)
        out[int(c)] = {"cat": cats[c], "basis": "all" if c in readable else ("campus" if c in placed else "none"), "near": near[c], "picks": picks[c],
                       "kinds": sorted(kinds, key=lambda k: -k["edge"]), "families": {k: (r(v, 3) if v is not None else None) for k, v in fs.items()} if fs else None,
                       "best": [{"pid": int(p), "pct": r(float(l), 2), "regs": r(float(m_), 0) if m_ == m_ and m_ is not None else None, "runs": int(k_)}
                                for p, l, k_, m_ in zip(mine["pid"].head(8), mine["pc"].head(8), mine["n_pc"].head(8), mine["med_regs"].head(8))]}

    # ---- do centers clump? and a 2-D picture
    z = lambda x: (lambda v: (v - np.nanmean(v)) / (np.nanstd(v) or 1.0))(np.where(np.isfinite(x), x, np.nanmedian(x)))
    full = [c for c in readable if cats[c]["size"] != "unknown" and T.at[c, "pace"] == T.at[c, "pace"]]
    F = T.loc[full]
    blk = lambda m: m / np.sqrt((m ** 2).sum(1).mean())
    X = np.hstack([blk(np.column_stack([z(np.log(F["usual"].astype(float))), z(np.log(F["top5"].astype(float)))])), blk(z(np.log(F["pace"].astype(float)))[:, None]),
                   blk(z(np.log(F["jpop"].astype(float)))[:, None]), blk(np.array([[F.at[c, "strength"][t] for t in nb.FAM] for c in full]))])
    structure = _structure(X)
    try:
        import umap
        xy = umap.UMAP(n_neighbors=10, min_dist=0.4, random_state=7).fit_transform(X)
    except Exception:
        xy = PCA(2).fit_transform(X)
    xy = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
    for a, c in enumerate(full):
        out[int(c)]["xy"] = [r(xy[a, 0], 4), r(xy[a, 1], 4)]

    stats = {}
    for c, row in T.iterrows():
        nn_ = lambda v, nd=1: r(float(v), nd) if v == v and v is not None else None
        stats[int(c)] = {
            "n": int(row["n_all"]), "events": int(row["n"]), "typical": nn_(row["usual"]), "top5": nn_(row["top5"], 0), "biggest": int(row["biggest"]), "dinner": nn_(row["dinner"]),
            "pace": nn_(row["pace"]), "per_year": nn_(row["per_year"], 0), "weeks": int(row["weeks"]), "jewish": nn_(row["jpop"], 0), "jewish_share": nn_(row["jshare"], 3),
            "draw_share": nn_(row["draw_share"], 3), "campuses": [x["name"] for x in row["served"]][:12] if isinstance(row["served"], list) else [],
            "spread": [r(v, 1) for v in row["spread"]] if isinstance(row["spread"], list) else None,
            "rsvp_share": nn_(row["share3"], 2), "scored": int(row["scored3"]), "reliable": bool(row["reliable"]), "med_regs": nn_(row["med_regs"]), "sd_regs": nn_(row["dinner"]), "sd_max": int(row["sd_max"]),
            "themed": nn_(row["themed"], 2), "one_time": nn_(row["one_time"], 2), "mix": {k_: r(v, 3) for k_, v in row["mix"].items()} if row["n"] else None, "types": row["types"],
            "wd": row["wd"], "traj": row["traj"], "first": row["first"].isoformat() if row["first"] is not None else None, "last": row["last"].isoformat() if row["last"] is not None else None,
            "upcoming": int(row["upcoming"]), "n_now": int(row["n_now"]), "n_before": int(row["n_before"]),
        }
    for k in [f["key"] for f in cat.FACETS]:
        centers[k] = centers.index.map(lambda c, k=k: cats[c][k])
    centers["usual"] = T["usual"]
    blob = {
        "readable": len(readable), "total": int(len(T)), "min_events": MIN_EVENTS, "profiles": out, "stats": stats,
        "facets": _value_profiles(T, cats, ideas, P, readable), "group_labels": GROUP_LABEL, "kind_labels": nb.FAM_LABEL,
        "network_kinds": {t: r(v, 2) for t, v in net.items()},
        "sim": {"ids": [int(c) for c in ids], "rows": [[int(round(v * 100)) for v in CS[a]] for a in range(len(ids))]},
        "structure": {"centers": len(full), "tests": structure},
        "method": {"power": nb.POWER, "k": nb.K, "own": nb.OWN, "features": ["draw", "pace", "jewish", "profile"], "close": r(close, 3)},
    }
    log("centers: readable", len(readable), "| with picks", sum(1 for c in ids if picks[c]["new"]), "| clumping", [(s["k"], s["silhouette"], s["stability"]) for s in structure],
        "| sizes", dict(Counter(cats[c]["size"] for c in ids)), "| strengths", dict(Counter(cats[c]["strength"] for c in ids)))
    return blob, centers
