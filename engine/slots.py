"""When. Two maps for every kind of event a center can place on the week, kept apart on purpose:

  frequency   the share of that kind's events in each weekday x time block. What the network does.
  effect      how events in a block landed against their own center's usual for that kind, on the percentile scale
              (cpct, 0 to 1 among the center's own events in the window). A block is read only through within-center
              differences, so a big center cannot make Thursday look good by being big.

The effect in a block is shrunk toward zero by how little evidence it has (events over events plus a prior of 30),
and tested by permutation: each event's percentile is redrawn from its own center-kind pool 2,000 times, so the
p-value asks "would this block look this good if the day and hour did not matter at this center".

Per center, the same is done for its own history, blended with the network's map by evidence, so a center with
eleven Tuesday classes sees mostly its own Tuesdays and a center with one sees the network.
"""
import numpy as np
from collections import defaultdict
from .common import r, log

FREE = ["class", "speaker", "cooking", "meal", "social", "israel", "chesed", "ritual", "trip", "outreach"]
FIXED = ["shabbat_dinner", "shabbat_lunch", "holiday_meal", "holiday_event", "holiday_service"]   # the calendar sets the day; the hour is still a choice
KIND_LABEL = {"class": "Classes", "speaker": "Speakers", "cooking": "Cooking and baking", "meal": "Meals and food", "social": "Socials", "israel": "Israel programs",
              "chesed": "Chesed and service", "ritual": "Prayer and mitzvah", "trip": "Trips", "outreach": "Tabling and outreach",
              "shabbat_dinner": "Shabbos dinners", "shabbat_lunch": "Shabbos lunches", "holiday_meal": "Holiday meals", "holiday_event": "Holiday events", "holiday_service": "Holiday services"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BLOCKS = [("morning", 7, 12), ("lunch", 12, 14), ("afternoon", 14, 17), ("early", 17, 19), ("evening", 19, 21), ("late", 21, 24)]
BLOCK_LABEL = {"morning": "Morning", "lunch": "Lunchtime", "afternoon": "Afternoon", "early": "Early evening", "evening": "Evening", "late": "Late night"}
PRIOR = 30.0
DRAWS = 2000
MIN_N, MIN_C = 25, 6
RNG = np.random.default_rng(5)


def block_of(hour):
    for key, lo, hi in BLOCKS:
        if lo <= hour < hi:
            return key
    return None


def _dev(df):
    """Each scored event's percentile minus its center's mean percentile for the kind: the within-center signal."""
    m = df.groupby(["center", "type"])["cpct"].transform("mean")
    return df["cpct"] - m


def _cells(df, pools, label_cols=("wd", "block")):
    sc = df[df["scored"] & df["cpct"].notna()].copy()
    sc["dev"] = _dev(sc) if len(sc) else []
    total = max(1, len(df))
    cnt = df.groupby(list(label_cols)).agg(n=("id", "size"), centers=("center", "nunique"))
    out = []
    for key, row in cnt.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        g = sc[(sc[list(label_cols)] == key).all(axis=1)] if len(sc) else sc
        ns, cs = len(g), int(g["center"].nunique()) if len(g) else 0
        eff, p = None, None
        if ns >= 8:
            raw = float(g["dev"].mean())
            eff = raw * ns / (ns + PRIOR)
            # null: redraw each event's percentile from its own center-kind pool
            acc = np.zeros(DRAWS)
            for ck in g[["center", "type"]].itertuples(index=False):
                pool = pools[(ck.center, ck.type)]
                acc += pool[RNG.integers(0, len(pool), DRAWS)]
            null = acc / ns
            p = float((np.sum(null >= raw) + 1) / (DRAWS + 1)) if raw >= 0 else float((np.sum(null <= raw) + 1) / (DRAWS + 1))
        out.append({**{c: (int(v) if isinstance(v, (int, np.integer)) else v) for c, v in zip(label_cols, key)},
                    "n": int(row["n"]), "centers": int(row["centers"]), "share": r(row["n"] / total, 4), "scored": ns, "scored_centers": cs,
                    "effect": r(eff, 3) if eff is not None else None, "p": r(p, 3) if p is not None else None, "ok": bool(ns >= MIN_N and cs >= MIN_C)})
    return out


def _pools(sc):
    pools = {}
    for key, g in sc.groupby(["center", "type"]):
        pools[key] = (g["cpct"] - g["cpct"].mean()).values.astype(float)
    return pools


def build(events, today):
    e = events[events["ran"] & (events["hour"] >= 7)].copy()
    e = e[(today - e["date"]).map(lambda d: d.days) <= 1095]
    e["block"] = e["hour"].map(block_of)
    e = e[e["block"].notna()]
    sc = e[e["scored"] & e["cpct"].notna()]
    pools = _pools(sc)
    kinds = {}
    for kind in FREE + FIXED:
        d = e[e["type"] == kind]
        if len(d) < 40:
            continue
        kinds[kind] = {"label": KIND_LABEL[kind], "n": int(len(d)), "centers": int(d["center"].nunique()), "scored": int(d["scored"].sum()), "free_day": kind in FREE,
                       "grid": _cells(d, pools), "days": _cells(d, pools, ("wd",)), "blocks": _cells(d, pools, ("block",)), "hours": _cells(d, pools, ("hour",))}
    # ---- whole network, all kinds that a center chooses the day for
    free = e[e["type"].isin(FREE)]
    overall = {"grid": _cells(free, pools), "days": _cells(free, pools, ("wd",)), "blocks": _cells(free, pools, ("block",)), "n": int(len(free))}
    # ---- per center: own history by kind and block/day, as evidence-weighted blends with the network
    per_center = {}
    net_eff = {(k, c["wd"], c["block"]): (c["effect"] or 0.0) for k, v in kinds.items() for c in v["grid"]}
    for c, g in e.groupby("center"):
        rec = {}
        g2 = g[g["scored"] & g["cpct"].notna()].copy()
        if len(g2):
            g2["dev"] = g2["cpct"] - g2.groupby("type")["cpct"].transform("mean")
        for kind, gk in g.groupby("type"):
            if kind not in kinds or len(gk) < 5:
                continue
            cells = []
            s = g2[g2["type"] == kind] if len(g2) else g2
            cnt = gk.groupby(["wd", "block"]).size()
            for (wd, blk), n in cnt.items():
                own = s[(s["wd"] == wd) & (s["block"] == blk)]["dev"] if len(s) else []
                k = len(own)
                own_eff = float(np.mean(own)) if k else 0.0
                net = net_eff.get((kind, int(wd), blk), 0.0)
                blend = (k * own_eff + 6.0 * net) / (k + 6.0)
                cells.append({"wd": int(wd), "block": blk, "n": int(n), "scored": int(k), "effect": r(blend, 3), "own": r(own_eff, 3) if k >= 3 else None})
            rec[kind] = {"n": int(len(gk)), "cells": cells}
        per_center[int(c)] = rec
    log("slots: kinds", len(kinds), "| centers", len(per_center))
    return {"kinds": kinds, "overall": overall, "centers": per_center, "days": DAYS, "blocks": [{"key": k, "label": BLOCK_LABEL[k], "from": lo, "to": hi} for k, lo, hi in BLOCKS],
            "method": {"prior": PRIOR, "draws": DRAWS, "min_n": MIN_N, "min_centers": MIN_C, "scale": "percentile points among the center's own events (x100)"}}
