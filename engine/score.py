"""The fair score. Raw sign-ups cannot be compared across centers, so every event is read against its own
center's normal.

  base      the center's median sign-ups for the same type of event in the year around it
  lift      (sign-ups + 2) / (base + 2)                    how far above or below the center's normal
  pct       rank of the event among that center's events of the same type, 0 to 1
  c_lift    the same against all of the center's events    lets different types be compared on one scale
  adj       lift with the usual shape of a semester taken out (week 1 is always big, December is always thin)

Only events with at least one sign-up are scored. A zero usually means the center did not use sign-ups for
that event, not that nobody came, so zeros are kept as "it ran" and left out of every performance figure.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from .types import EVENT_TYPES
from .common import log

A = 2.0          # additive smoothing on counts
WINDOW = 365     # days either side
MIN_SAME = 4     # same-type events needed for a type baseline
MIN_ANY = 6


def run(events, centers):
    n = len(events)
    ev = events
    ev["measured"] = ev["ran"] & (ev["regs"] >= 1) & ev["type"].isin(EVENT_TYPES + ["other"])
    base = np.full(n, np.nan); pct = np.full(n, np.nan); cbase = np.full(n, np.nan); nsame = np.zeros(n, int)
    cpct = np.full(n, np.nan)      # where the event lands among ALL of its center's events around that time, 0 to 1
    ord_ = (ev["dt"].values.astype("datetime64[D]").astype(int))
    for c, idx in ev.groupby("center").indices.items():
        idx = np.asarray(idx)
        m = ev["measured"].values[idx]
        mi = idx[m]
        if len(mi) < 3:
            continue
        days = ord_[mi]; regs = ev["regs"].values[mi].astype(float); types = ev["type"].values[mi]
        order = np.argsort(days, kind="stable"); mi, days, regs, types = mi[order], days[order], regs[order], types[order]
        for t in set(types):
            sel = types == t
            d_t, r_t, i_t = days[sel], regs[sel], mi[sel]
            lo = np.searchsorted(d_t, d_t - WINDOW, "left"); hi = np.searchsorted(d_t, d_t + WINDOW, "right")
            for k in range(len(i_t)):
                w = np.concatenate([r_t[lo[k]:k], r_t[k + 1:hi[k]]])
                nsame[i_t[k]] = len(w)
                if len(w) >= MIN_SAME:
                    base[i_t[k]] = np.median(w)
                    if len(w) >= 5:
                        pct[i_t[k]] = ((w < r_t[k]).sum() + 0.5 * (w == r_t[k]).sum()) / len(w)
        lo = np.searchsorted(days, days - WINDOW, "left"); hi = np.searchsorted(days, days + WINDOW, "right")
        for k in range(len(mi)):
            w = np.concatenate([regs[lo[k]:k], regs[k + 1:hi[k]]])
            if len(w) >= MIN_ANY:
                cbase[mi[k]] = np.median(w)
                cpct[mi[k]] = ((w < regs[k]).sum() + 0.5 * (w == regs[k]).sum()) / len(w)
                if np.isnan(pct[mi[k]]):
                    pct[mi[k]] = ((w < regs[k]).sum() + 0.5 * (w == regs[k]).sum()) / len(w)
    ev["base"], ev["cbase"], ev["pct"], ev["nsame"] = base, cbase, pct, nsame
    ev["cpct"] = cpct
    b = np.where(np.isnan(base), cbase, base)
    ev["lift"] = np.where(ev["measured"] & ~np.isnan(b), (ev["regs"] + A) / (b + A), np.nan)
    ev["c_lift"] = np.where(ev["measured"] & ~np.isnan(cbase), (ev["regs"] + A) / (cbase + A), np.nan)
    ev["scored"] = ev["measured"] & ~np.isnan(b) & (b >= 3)

    # the usual shape of a semester, per type: median lift by week of term, smoothed
    wf = {}
    s = ev[ev["scored"] & ev["season"].isin(["fall", "spring"]) & (ev["tweek"] <= 18)]
    for (t, sea), g in s.groupby(["type", "season"]):
        if len(g) < 300:
            continue
        med = g.groupby("tweek")["lift"].agg(lambda x: np.exp(np.mean(np.clip(np.log(x), -1.5, 1.5)))).reindex(range(1, 19))
        cnt = g.groupby("tweek")["lift"].size().reindex(range(1, 19)).fillna(0)
        med = med.where(cnt >= 15).interpolate(limit_direction="both")
        sm = med.rolling(3, center=True, min_periods=1).mean()
        wf[(t, sea)] = (sm / np.exp(np.mean(np.log(sm)))).to_dict()
    allf = {}
    for sea, g in s.groupby("season"):
        med = g.groupby("tweek")["c_lift"].agg(lambda x: np.exp(np.nanmean(np.clip(np.log(x), -1.5, 1.5)))).reindex(range(1, 19)).interpolate(limit_direction="both")
        sm = med.rolling(3, center=True, min_periods=1).mean()
        allf[sea] = (sm / np.exp(np.mean(np.log(sm)))).to_dict()
    f = np.ones(n)
    for i, (t, sea, w) in enumerate(zip(ev["type"], ev["season"], ev["tweek"])):
        if sea in ("fall", "spring") and w == w and w <= 18:
            f[i] = wf.get((t, sea), allf.get(sea, {})).get(int(w), 1.0)
    ev["wfactor"] = f
    ev["adj"] = ev["lift"] / f

    # how much each center uses sign-ups at all
    rel = {}
    recent = ev[ev["ran"] & ev["type"].isin(EVENT_TYPES)]
    for c, g in recent.groupby("center"):
        rel[c] = {"ran": int(len(g)), "measured": int(g["measured"].sum()), "share": float(g["measured"].mean()),
                  "median": float(g.loc[g["measured"], "regs"].median()) if g["measured"].any() else None}
    centers["n_ran"] = centers.index.map(lambda c: rel.get(c, {}).get("ran", 0))
    centers["n_measured"] = centers.index.map(lambda c: rel.get(c, {}).get("measured", 0))
    centers["rsvp_share"] = centers.index.map(lambda c: rel.get(c, {}).get("share", 0.0))
    centers["median_regs"] = centers.index.map(lambda c: rel.get(c, {}).get("median"))
    centers["reliable"] = (centers["n_measured"] >= 25) & (centers["rsvp_share"] >= 0.4)
    log("scored events", int(ev["scored"].sum()), "of", int(ev["ran"].sum()), "ran | measured", int(ev["measured"].sum()), "| reliable centers", int(centers["reliable"].sum()), "of", len(centers))
    return ev, centers, {"week_factor": {f"{k[0]}|{k[1]}": {int(w): round(float(v), 3) for w, v in d.items()} for k, d in wf.items()},
                         "week_factor_all": {k: {int(w): round(float(v), 3) for w, v in d.items()} for k, d in allf.items()}}
