"""How much an idea, theme or guest lifts a center's turnout, read carefully.

Every scored event already carries `c_lift`: its sign-ups against the median of the same center's same-kind events
within a year either side (score.py). Averaging those raw lifts over an idea's events would reward ideas that
happen to have one lucky run at a small center. Three corrections are made here:

  shrinkage    each idea's mean log-lift is pulled toward its kind's mean by how little evidence it has
               (empirical Bayes, the between-idea variance estimated by the method of moments per kind)
  permutation  the mean is compared with what the same events would show if sign-ups were shuffled among each
               center's events of that kind: one p-value per idea, from 2,000 draws, so a center that is simply
               large or simply growing cannot make its idea look good
  BH           p-values are corrected for the number of ideas tested within each kind (Benjamini-Hochberg)

An idea is then placed in a tier and an effect band. Bands are what the app shows; numbers stay in the engine.
"""
import numpy as np
from collections import defaultdict
from .common import r

CLIP = 1.5                 # one event can move a log-lift by at most this (x4.5 either way)
DRAWS = 2000
RNG = np.random.default_rng(11)
TIERS = {"strong": "Strong evidence", "good": "Good evidence", "thin": "Thin evidence", "none": "Not enough to tell"}
BANDS = [(0.405, "well above usual"), (0.14, "above usual"), (-0.14, "about usual"), (-9, "below usual")]   # log cut points: x1.5, x1.15, x0.87


def band(log_lift):
    if log_lift is None or log_lift != log_lift:
        return None
    for cut, lab in BANDS:
        if log_lift >= cut:
            return lab
    return BANDS[-1][1]


def bh(p):
    """Benjamini-Hochberg q-values for one family."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * n / rank)
        q[i] = running
    return q


class Pools:
    """Each center's scored events of each kind, as the null distribution their sign-ups came from."""

    def __init__(self, events, value="c_lift"):
        e = events[events["scored"] & events[value].notna() & events["ran"]]
        self.logs = np.clip(np.log(e[value].values.astype(float)), -CLIP, CLIP)
        self.index = {}
        for key, idx in e.groupby(["center", "type"]).indices.items():
            self.index[key] = idx
        self.pos = {eid: i for i, eid in enumerate(e["id"].values)}
        self.key = dict(zip(e["id"].values, zip(e["center"].values, e["type"].values)))

    def observed(self, ids):
        rows = [self.pos[i] for i in ids if i in self.pos]
        return self.logs[rows], rows

    def null_means(self, ids, draws=DRAWS):
        """Draw each event's value from its own center-kind pool; the mean of the draws is one null sample."""
        pools = [self.index[self.key[i]] for i in ids if i in self.pos]
        if not pools:
            return None
        acc = np.zeros(draws)
        for pool in pools:
            acc += self.logs[pool[RNG.integers(0, len(pool), draws)]]
        return acc / len(pools)


def shrink(groups, kind_of):
    """groups: {gid: array of clipped log-lifts}. Returns {gid: (shrunk mean, raw mean, n)} with the between-group
    variance estimated within each kind."""
    by_kind = defaultdict(list)
    for gid, x in groups.items():
        if len(x):
            by_kind[kind_of[gid]].append((gid, x))
    out = {}
    for kind, items in by_kind.items():
        means = np.array([x.mean() for _, x in items]); ns = np.array([len(x) for _, x in items], float)
        within = np.array([x.var(ddof=1) if len(x) > 1 else np.nan for _, x in items])
        s2 = float(np.nanmean(within)) if np.isfinite(within).any() else 0.25
        s2 = max(s2, 0.05)
        grand = float(np.average(means, weights=ns))
        tau2 = max(float(np.average((means - grand) ** 2, weights=ns) - s2 * np.mean(1.0 / ns)), 0.005)
        for (gid, x), m, n in zip(items, means, ns):
            k = s2 / tau2
            out[gid] = (float((n * m + k * grand) / (n + k)), float(m), int(n))
    return out


def tier(q, centers, n, shrunk):
    if n < 3 or centers < 2:
        return "none"
    if q <= 0.05 and centers >= 5 and n >= 8:
        return "strong"
    if q <= 0.15 and centers >= 3 and n >= 5:
        return "good"
    return "thin"


def assess(groups, kind_of, centers_of, pools):
    """groups: {gid: list of event ids}. One result per group: shrunk log-lift, p, q, tier, bands."""
    obs = {}
    for gid, ids in groups.items():
        x, _ = pools.observed(ids)
        obs[gid] = x
    sh = shrink(obs, kind_of)
    res, fam = {}, defaultdict(list)
    for gid, ids in groups.items():
        x = obs[gid]
        if len(x) == 0:
            res[gid] = {"n": 0, "centers": int(centers_of.get(gid, 0)), "log": None, "raw": None, "p": None, "q": None, "tier": "none", "band": None}
            continue
        null = pools.null_means(ids)
        p = float((np.sum(null >= x.mean()) + 1) / (len(null) + 1)) if null is not None else 1.0
        s, raw, n = sh[gid]
        res[gid] = {"n": n, "centers": int(centers_of.get(gid, 0)), "log": s, "raw": raw, "p": p, "hit": float(np.mean(x > 0))}
        fam[kind_of[gid]].append(gid)
    for kind, gids in fam.items():
        qs = bh([res[g]["p"] for g in gids])
        for g, q in zip(gids, qs):
            res[g]["q"] = float(q)
            res[g]["tier"] = tier(q, res[g]["centers"], res[g]["n"], res[g]["log"])
            res[g]["band"] = band(res[g]["log"])
            res[g]["lift"] = r(float(np.exp(res[g]["log"])), 2)
    return res


def tracking(events, today):
    """Which centers take sign-ups steadily enough for their numbers to mean anything (a flag, never a count)."""
    e = events[events["ran"]]
    e = e[(today - e["date"]).map(lambda d: d.days) <= 1095]
    out = {}
    for c, g in e.groupby("center"):
        share = float((g["regs"] > 0).mean()) if len(g) else 0.0
        out[int(c)] = "steady" if share >= 0.5 and len(g) >= 15 else "partial" if share >= 0.15 and len(g) >= 15 else "rare"
    return out


def run(events, P, ideas, theme_blob, today):
    pools = Pools(events)
    pid_events = defaultdict(list)
    for eid, pid, scored in zip(events["id"].values, events["pid"].values, events["scored"].values):
        if pid == pid and scored:
            pid_events[int(pid)].append(int(eid))
    # ---- ideas
    groups, kind_of, centers_of = {}, {}, {}
    for i in ideas:
        ids = [e for p in i["programs"] for e in pid_events.get(int(p), [])]
        groups[i["id"]] = ids; kind_of[i["id"]] = i["type"]; centers_of[i["id"]] = i.get("centers_scored", i["centers"])
    idea_res = assess(groups, kind_of, centers_of, pools)
    # ---- Shabbos dinner themes: the themed dinners, against each center's own dinners
    tgroups, tkind, tcenters = {}, {}, {}
    tk = events[events["scored"] & (events["type"] == "shabbat_dinner")]
    by_theme = defaultdict(list); cen = defaultdict(set)
    for eid, keys, c in zip(tk["id"].values, tk["tkeys"].values, tk["center"].values):
        for k in (keys or []):
            by_theme[k].append(int(eid)); cen[k].add(c)
    for t in theme_blob["themes"]:
        k = t["key"]
        tgroups[k] = by_theme.get(k, []); tkind[k] = "theme"; tcenters[k] = len(cen.get(k, ()))
    theme_res = assess(tgroups, tkind, tcenters, pools)
    n_strong = sum(1 for v in idea_res.values() if v["tier"] == "strong")
    return {"ideas": idea_res, "themes": theme_res, "tracking": tracking(events, today),
            "method": {"clip": CLIP, "draws": DRAWS, "tiers": TIERS, "bands": [lab for _, lab in BANDS],
                       "summary": {"ideas": len(idea_res), "strong": n_strong, "good": sum(1 for v in idea_res.values() if v["tier"] == "good"),
                                   "themes_strong": sum(1 for v in theme_res.values() if v["tier"] == "strong")}}}
