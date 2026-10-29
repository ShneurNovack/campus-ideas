"""The nearest-neighbor system: how will this event do at this center?

The question is always about one center and one event. The answer comes from the past events most like it, held
at the centers most like that center, and from the center's own history:

    weight(past program j) = text likeness(event, j) ^ 4
                             x how alike the two centers are        (x 8 when j is the center's own)
                             x how much evidence j has
    prediction             = weighted average of where the 100 heaviest programs landed

"Where it landed" is a percentile: the event's place among all of its own center's events (0 is the smallest,
1 the biggest). That scale lets a center that draws 20 and a center that draws 400 vote on each other's events,
and it does not care whether a center takes sign-ups at every dinner or only at big events. Expected sign-ups
are read back through the asking center's own spread of sign-ups: "the 85th percentile of your events is 46".
Tested against predicting a ratio to the center's usual event, this scale was more accurate on sign-ups,
most of all for big events.

Two centers are alike when they are close on four things, weighted equally:
    draw      the average sign-ups of the five biggest events in three years
    pace      events a week while the center is in session
    Jewish    Jewish students on the campuses it serves
    profile   which kinds of event land high or low among the center's own events (nine kinds)
Campus type is not used.

`validate` replays the past: it stands one year back, hides everything after that day, predicts every program
that was new to its center in the following year, and compares with what happened. The settings above (power 4,
100 neighbors, own history x 8, equal feature weights) were chosen on the year before that one.
"""
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import r, log
from .types import EVENT_TYPES, TYPES

FAM = ["shabbat_dinner", "shabbat_lunch", "holiday_meal", "holiday_event", "holiday_service", "class", "meal", "social", "cooking"]
FAM_LABEL = {"shabbat_dinner": "Shabbat dinners", "shabbat_lunch": "Shabbat lunches", "holiday_meal": "Holiday meals", "holiday_event": "Holiday events",
             "holiday_service": "Holiday services", "class": "Classes", "meal": "Food and hangouts", "social": "Socials", "cooking": "Baking and cooking"}
POWER, K, OWN, TAU = 4, 100, 8.0, 1.0
FEATURES = ["top5", "pace", "jpop"]
UNKNOWN_GAP = 1.5          # standard deviations assumed between two centers when one of them lacks a measurement
PROFILE_PRIOR = 6.0        # events' worth of pull toward the network's level for a kind of event
GRID = np.linspace(0, 1, 21)   # a center's spread of sign-ups is stored at every 5th percentile
MIN_SPREAD = 15            # measured events needed before a center's spread is used


def _events(events):
    return events[events["ran"] & events["type"].isin(EVENT_TYPES)]


def center_table(events, centers, cut, col="cpct"):
    """What is known about each center on the day `cut`, from its events before that day."""
    e = _events(events)
    h = e[e["date"] < cut]
    net = h[h[col].notna()].groupby("type")[col].mean()
    rows = []
    for c, row in centers.iterrows():
        g = h[h["center"] == c]
        age = g["date"].map(lambda d: (cut - d).days) if len(g) else pd.Series([], dtype=float)
        g1, g2, g3 = g[age <= 365], g[age <= 730], g[age <= 1095]
        m3 = g3[g3["regs"] > 0]; m2 = g2[g2["regs"] > 0]
        top = np.sort(m3["regs"].values)[::-1]
        weeks = g1["date"].map(lambda d: d.isocalendar()[:2]).nunique() if len(g1) else 0
        sd3 = g3[g3["type"] == "shabbat_dinner"]; sdm = sd3[sd3["regs"] > 0]
        sc = g3[g3[col].notna()]
        level, strength, nfam = {}, {}, {}
        for t in FAM:
            x = sc.loc[sc["type"] == t, col].values
            nt = float(net.get(t, 0.5))
            level[t] = float((x.sum() + PROFILE_PRIOR * nt) / (len(x) + PROFILE_PRIOR))
            strength[t] = level[t] - nt
            nfam[t] = int(len(x))
        jp = row.get("jpop_served")
        jp = float(jp) if jp == jp and jp else np.nan
        tot = np.nansum([row.get("pop_ug"), row.get("pop_grad")])
        t5 = float(top[:5].mean()) if len(top) >= 5 else np.nan
        src = m2 if len(m2) >= MIN_SPREAD else m3
        rows.append({
            "id": c, "n3": len(g3), "n12": len(g1), "measured3": len(m3), "share3": len(m3) / max(1, len(g3)), "scored3": len(sc),
            "spread": [float(v) for v in np.quantile(src["regs"].values, GRID)] if len(src) >= MIN_SPREAD else None,
            "usual": float(np.median(src["regs"])) if len(src) >= 5 else np.nan,
            "top5": t5, "biggest": int(top[0]) if len(top) else 0,
            "dinner": float(sdm["regs"].median()) if len(sdm) >= 8 and len(sdm) / max(1, len(sd3)) >= 0.6 else np.nan,
            "pace": len(g1) / weeks if len(g1) >= 8 and weeks else np.nan, "weeks": int(weeks), "per_year": float(len(g1)),
            "jpop": jp, "jshare": float(row["jpop"] / tot) if row.get("jpop") == row.get("jpop") and tot > 0 and jp == jp and row["jpop"] >= 0.8 * jp else np.nan,
            "draw_share": t5 / jp if t5 == t5 and jp == jp and jp > 0 else np.nan,
            "level": level, "strength": strength, "nfam": nfam,
        })
    return pd.DataFrame(rows).set_index("id"), {t: float(net.get(t, 0.5)) for t in FAM}


def similarity(tab):
    """centers x centers likeness in (0, 1], and the standardized distances behind it, feature by feature."""
    ids = list(tab.index)
    X = np.array([[np.log(tab.at[c, k]) if tab.at[c, k] == tab.at[c, k] and tab.at[c, k] > 0 else np.nan for k in FEATURES] for c in ids])
    Z = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
    parts = {}
    for j, k in enumerate(FEATURES):
        d = np.abs(Z[:, j][:, None] - Z[:, j][None, :])
        parts[k] = np.where(np.isfinite(d), d, UNKNOWN_GAP)
    Pf = np.array([[tab.at[c, "strength"][t] for t in FAM] for c in ids])
    known = np.array([tab.at[c, "scored3"] >= 20 for c in ids])
    Pf = Pf / (Pf[known].std() or 1.0)
    dp = np.sqrt(((Pf[:, None, :] - Pf[None, :, :]) ** 2).mean(-1))
    parts["profile"] = np.where(known[:, None] & known[None, :], dp, UNKNOWN_GAP)
    D2 = sum(parts[k] ** 2 for k in parts)
    CS = np.exp(-D2 / (2 * TAU ** 2 * len(parts)))
    return ids, CS, parts


def sign_ups(spread, p):
    """A percentile turned back into sign-ups through a center's own spread."""
    if spread is None:
        return None
    return float(np.interp(np.clip(p, 0, 1), GRID, spread))


def program_targets(P):
    """The programs that can vote: where each landed among its center's events, and the runs behind that."""
    S = P[P["pc"].notna() & (P["n_pc"] >= 1) & P["type"].isin(EVENT_TYPES)]
    return S["pid"].values, S["pc"].values.astype(float), S["n_pc"].values.astype(float), S["center"].values


def predict(Q, q_center, D, rows, y, n, centers_of, ids, CS, k=K, power=POWER, own=OWN, quartiles=False):
    """Q: unit vectors of the events to predict. q_center: the center asking, one per row (or None for no center).
    rows: indices into D of the programs that can vote, with their results y, evidence n and centers."""
    pos = {c: i for i, c in enumerate(ids)}
    S = Q @ D[rows].T
    W = np.clip(S, 0, None) ** power * np.sqrt(n / (n + 1.0))[None, :]
    if q_center is not None:
        qi = np.array([pos.get(c, -1) for c in q_center]); ci = np.array([pos.get(c, -1) for c in centers_of])
        cw = CS[np.maximum(qi, 0)][:, np.maximum(ci, 0)]
        cw = np.where((qi[:, None] < 0) | (ci[None, :] < 0), np.exp(-UNKNOWN_GAP ** 2 / 2), cw)
        same = np.asarray(q_center)[:, None] == np.asarray(centers_of)[None, :]
        W = W * np.where(same, own, cw)
    return pool(W, y, k, quartiles)


def pool(W, y, k=K, quartiles=False):
    kk = min(k, W.shape[1] - 1)
    part = np.argpartition(-W, kk, axis=1)[:, :kk]
    w = np.take_along_axis(W, part, 1); yy = y[part]
    den = w.sum(1)
    pred = np.where(den > 0, (w * yy).sum(1) / np.maximum(den, 1e-12), 0.5)
    info = {"eff": den ** 2 / np.maximum((w ** 2).sum(1), 1e-12)}
    if quartiles:
        o = np.argsort(yy, axis=1); ys = np.take_along_axis(yy, o, 1); ws = np.take_along_axis(w, o, 1)
        cw = np.cumsum(ws, 1) / np.maximum(den, 1e-12)[:, None]
        for name, q in (("lo", 0.25), ("hi", 0.75)):
            j = np.minimum((cw < q).sum(1), kk - 1)
            info[name] = ys[np.arange(len(ys)), j]
    return pred, info


def _asof(events, cut):
    """Percentiles and spreads as they stood on `cut`, with nothing from later days."""
    e = _events(events)
    e = e[e["pid"].notna() & (e["regs"] > 0)].copy(); e["pid"] = e["pid"].astype(int)
    h = e[(e["date"] < cut) & (e["date"] >= cut - dt.timedelta(days=1095))]
    spread = {c: np.sort(g["regs"].values) for c, g in h.groupby("center") if len(g) >= MIN_SPREAD}
    def pct(c, regs):
        s = spread[c]
        return (np.searchsorted(s, regs, "left") + np.searchsorted(s, regs, "right")) / (2.0 * len(s))
    return e, spread, pct


def validate(events, centers, P, V, today, back=365):
    """Stand `back` days in the past, hide what came after, and predict every program that was new to its center
    in the hidden year. Returns the full system and the same test with pieces of it switched off."""
    cut = today - dt.timedelta(days=back)
    e, spread, pct = _asof(events, cut)
    hh = e[(e["date"] < cut) & e["center"].isin(spread)]
    hh = hh.assign(pc=[pct(c, x) for c, x in zip(hh["center"], hh["regs"])])
    tr = hh.groupby("pid")["pc"].agg(["mean", "size"])
    first = e.groupby("pid")["date"].min()
    f = e[(e["date"] >= cut) & (e["date"] < cut + dt.timedelta(days=365)) & e["center"].isin(spread)]
    f = f[f["pid"].map(first) >= cut]
    te = f.groupby("pid").agg(regs=("regs", "median"), center=("center", "first"))
    if len(te) < 200 or len(tr) < 1000:
        return None
    cen = P.set_index("pid")["center"]; typ = P.set_index("pid")["type"]
    rows = tr.index.values; y = tr["mean"].values; n = tr["size"].values.astype(float); ctr = cen.reindex(rows).values
    cte = te["center"].values; act = te["regs"].values
    yt = np.array([pct(c, x) for c, x in zip(cte, act)])
    ev2 = events.assign(cpct_asof=np.nan)
    ev2.loc[hh.index, "cpct_asof"] = hh["pc"]
    tab, _ = center_table(ev2, centers, cut, col="cpct_asof")
    ids, CS, _ = similarity(tab)
    Q = V[te.index.values]
    q = lambda c, p: float(np.quantile(spread[c], np.clip(p, 0, 1)))

    def report(pred, info=None):
        o = np.argsort(pred); f5 = max(1, len(o) // 5)
        exp = np.array([q(c, p) for c, p in zip(cte, pred)])
        err = np.abs(np.log((act + 1) / (exp + 1)))
        out = {"rho": r(spearmanr(pred, yt)[0], 3), "within_center": _within(pred, yt, cte), "rho_signups": r(spearmanr(exp, act)[0], 3),
               "top_fifth_in_top_third": r(float((yt[o[-f5:]] >= 2 / 3).mean()), 3), "bottom_fifth_in_top_third": r(float((yt[o[:f5]] >= 2 / 3).mean()), 3),
               "top_fifth_median_pct": r(float(np.median(yt[o[-f5:]])), 2), "bottom_fifth_median_pct": r(float(np.median(yt[o[:f5]])), 2),
               "typical_error": r(float(np.exp(np.median(err))), 2), "within_1_5x": r(float((err <= np.log(1.5)).mean()), 2), "n": int(len(yt))}
        if info is not None and "lo" in info:
            lo = np.array([q(c, p) for c, p in zip(cte, info["lo"])]); hi = np.array([q(c, p) for c, p in zip(cte, info["hi"])])
            out["range_covers"] = r(float(((act >= np.floor(lo)) & (act <= np.ceil(hi))).mean()), 2)
        return out

    out = {"cut": cut.isoformat(), "trained_on": int(len(tr)), "predicted": int(len(te)), "centers": int(len(set(cte)))}
    p, info = predict(Q, cte, V, rows, y, n, ctr, ids, CS, quartiles=True)
    out["full"] = report(p, info)
    flat = np.ones_like(CS)
    out["no_center_likeness"] = report(predict(Q, cte, V, rows, y, n, ctr, ids, flat)[0])
    out["no_own_history"] = report(predict(Q, cte, V, rows, y, n, ctr, ids, CS, own=0.0)[0])
    out["text_only_other_centers"] = report(predict(Q, cte, V, rows, y, n, ctr, ids, flat, own=0.0)[0])
    tmean = pd.Series(y, index=rows).groupby(typ.reindex(rows).values).mean()
    out["kind_of_event_only"] = report(np.array([tmean.get(t, 0.5) for t in typ.reindex(te.index)]))
    out["center_usual_only"] = report(np.full(len(yt), 0.5))
    return out


def _within(pred, y, cen):
    rs, ws = [], []
    for c in set(cen):
        m = cen == c
        if m.sum() >= 8 and pred[m].std() > 0 and y[m].std() > 0:
            rs.append(spearmanr(pred[m], y[m])[0]); ws.append(int(m.sum()))
    return r(float(np.average(rs, weights=ws)), 3) if rs else None
