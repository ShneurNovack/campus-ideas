"""Timing. When centers schedule each kind of event, and when it does best.

Two different questions are kept apart:
  count   how many events (and centers) use a slot. This is what the network does.
  lift    how events in that slot did against the same center's normal for the same type. Because the yardstick
          is each center's own median, a slot only scores above 1 when centers that tried it did better there
          than in their other slots. Size and type differences between centers cancel out.

A slot is shown as reliable only with at least 25 scored events from at least 6 centers.
"""
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from . import calendars as cal
from .common import r, log
from .types import TYPES, EVENT_TYPES

# events a center chooses the day and hour for. Holiday events are left out: the calendar picks their day.
FREE = ["class", "speaker", "cooking", "meal", "social", "israel", "chesed", "ritual", "trip", "outreach"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MIN_N, MIN_C = 25, 6


def _gm(x, prior=4.0):
    x = np.clip(np.log(np.asarray(x, float)), -1.5, 1.5)
    return float(np.exp(x.sum() / (len(x) + prior))) if len(x) else None


def _ref(x):
    """The typical score of the whole group. A median yardstick leaves the average event a little under 1,
    so every slot is divided by this: 1.0 then means 'a typical event of this kind'."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.clip(np.log(x), -1.5, 1.5).mean())) if len(x) >= 20 else 1.0


def _grid(df, value="adj"):
    """weekday x hour table."""
    cells = []
    sc = df[df["scored"] & df[value].notna()]
    cnt = df.groupby(["wd", "hour"]).agg(n=("id", "size"), centers=("center", "nunique"))
    per = sc.groupby(["wd", "hour"]).agg(ns=("id", "size"), cs=("center", "nunique"))
    ref = _ref(sc[value].values)
    lifts = {k: _gm(g[value].values) / ref for k, g in sc.groupby(["wd", "hour"])}
    total = max(1, len(df))
    for (wd, hr), row in cnt.iterrows():
        if hr < 7:
            continue
        p = per.loc[(wd, hr)] if (wd, hr) in per.index else None
        ns, cs = (int(p["ns"]), int(p["cs"])) if p is not None else (0, 0)
        cells.append({"wd": int(wd), "hour": int(hr), "n": int(row["n"]), "centers": int(row["centers"]), "share": r(row["n"] / total, 4),
                      "scored": ns, "scored_centers": cs, "lift": r(lifts.get((wd, hr)), 3) if ns >= 8 else None, "ok": bool(ns >= MIN_N and cs >= MIN_C)})
    return cells


def _by(df, col, keys, value="adj"):
    out = []
    sc = df[df["scored"] & df[value].notna()]
    total = max(1, len(df))
    ref = _ref(sc[value].values)
    for k in keys:
        g, s = df[df[col] == k], sc[sc[col] == k]
        out.append({"key": k if not isinstance(k, (np.integer,)) else int(k), "n": int(len(g)), "centers": int(g["center"].nunique()), "share": r(len(g) / total, 4),
                    "scored": int(len(s)), "lift": r(_gm(s[value].values) / ref, 3) if len(s) >= 8 else None, "ok": bool(len(s) >= MIN_N and s["center"].nunique() >= MIN_C)})
    return out


def _top_slots(cells, k=4):
    ok = [c for c in cells if c["ok"] and c["lift"]]
    best = sorted(ok, key=lambda c: -c["lift"])[:k]
    worst = sorted(ok, key=lambda c: c["lift"])[:k]
    common = sorted(cells, key=lambda c: -c["n"])[:k]
    return best, worst, common


def _fall_spring(df):
    """Same center, same academic year: median sign-ups in spring against fall."""
    ratios = []
    m = df[df["measured"] & df["season"].isin(["fall", "spring"])].copy()
    m["ay"] = m["date"].map(lambda d: d.year if d.month >= 7 else d.year - 1)
    for (c, ay), g in m.groupby(["center", "ay"]):
        f, s = g.loc[g["season"] == "fall", "regs"], g.loc[g["season"] == "spring", "regs"]
        if len(f) >= 3 and len(s) >= 3:
            ratios.append((np.median(s) + 2) / (np.median(f) + 2))
    return {"pairs": len(ratios), "spring_vs_fall": r(_gm(ratios, prior=2.0), 3) if len(ratios) >= 5 else None,
            "fall_n": int((df["season"] == "fall").sum()), "spring_n": int((df["season"] == "spring").sum())}


def _week_curve(df, value="lift"):
    out = {}
    s = df[df["scored"] & df["season"].isin(["fall", "spring"]) & (df["tweek"] <= 16)]
    ref = _ref(s[value].values)
    for sea, g in s.groupby("season"):
        rows = []
        for w in range(1, 17):
            x = g.loc[g["tweek"] == w, value]
            rows.append({"week": w, "n": int(len(x)), "lift": r(_gm(x.values, prior=6.0) / ref, 3) if len(x) >= 12 else None})
        out[sea] = rows
    return out


def build(events, centers, today):
    ev = events[events["ran"] & events["type"].isin(EVENT_TYPES)].copy()
    out = {"days": DAYS, "types": {}, "min_scored": MIN_N, "min_centers": MIN_C}
    free = ev[ev["type"].isin(FREE)]
    best, worst, common = _top_slots(_grid(free))
    out["overall"] = {
        "all_grid": _grid(ev), "free_grid": _grid(free), "free_types": FREE,
        "best": best, "worst": worst, "common": common,
        "weekday": _by(free, "wd", range(7)), "weekday_all": _by(ev, "wd", range(7)),
        "hour": _by(free, "hour", range(8, 24)),
        "fall_spring": _fall_spring(free), "weeks": _week_curve(free),
        "n": int(len(ev)), "n_free": int(len(free)), "scored_free": int(free["scored"].sum()),
    }
    ev["month_"] = ev["dt"].dt.month
    free = ev[ev["type"].isin(FREE)]
    out["overall"]["month"] = _by(free, "month_", range(1, 13))
    out["overall"]["third"] = []
    s = free[free["scored"] & free["tpos"].notna()]
    ref3 = _ref(s["lift"].values)
    for lo, hi, lab in [(0, 0.2, "First fifth of the semester"), (0.2, 0.4, "Second fifth"), (0.4, 0.6, "Middle"), (0.6, 0.8, "Fourth fifth"), (0.8, 1.01, "Last fifth")]:
        g = s[(s["tpos"] >= lo) & (s["tpos"] < hi)]
        out["overall"]["third"].append({"label": lab, "n": int(len(g)), "lift": r(_gm(g["lift"].values) / ref3, 3) if len(g) >= 20 else None})
    for t in EVENT_TYPES:
        g = ev[ev["type"] == t]
        if len(g) < 60:
            continue
        cells = _grid(g)
        best, worst, common = _top_slots(cells, 3)
        hrs = sorted(set(int(h) for h in g["hour"] if h >= 7))
        out["types"][t] = {
            "label": TYPES[t], "n": int(len(g)), "centers": int(g["center"].nunique()), "scored": int(g["scored"].sum()),
            "grid": cells, "best": best, "worst": worst, "common": common,
            "weekday": _by(g, "wd", range(7)), "hour": _by(g, "hour", hrs), "month": _by(g, "month_", range(1, 13)),
            "fall_spring": _fall_spring(g), "weeks": _week_curve(g),
            "median_regs": r(float(g.loc[g["measured"], "regs"].median()), 1) if g["measured"].any() else None,
        }
    # ---- the Jewish and secular calendars
    jew = []
    for key, label, *_ in cal.JEWISH:
        g = ev[(ev["jh"] == key) | (ev["hol"] == key)]
        if len(g) < 20:
            continue
        tm = Counter(g["type"])
        sc = g[g["scored"]]
        jew.append({"key": key, "label": label, "n": int(len(g)), "centers": int(g["center"].nunique()), "per_center_year": r(len(g) / max(1, g.groupby(["center", g["date"].map(lambda d: d.year)]).ngroups), 1),
                    "types": [{"type": t, "n": int(n)} for t, n in tm.most_common(6)], "lift_all": r(_gm(sc["c_lift"].dropna().values), 3) if len(sc) >= 20 else None})
    out["jewish"] = jew
    sec = []
    fr = ev[ev["type"].isin(FREE + ["shabbat_dinner"])]
    ref_all = _ref(fr.loc[fr["scored"], "adj"].dropna().values)
    ref_sd = _ref(fr.loc[fr["scored"] & (fr["type"] == "shabbat_dinner"), "adj"].dropna().values)
    for key, label in cal.SECULAR_LABEL.items():
        g = fr[fr["sec"] == key]
        s = g[g["scored"]]
        if len(g) < 30:
            continue
        sd = g[(g["type"] == "shabbat_dinner") & g["scored"]]
        sec.append({"key": key, "label": label, "n": int(len(g)), "centers": int(g["center"].nunique()), "lift": r(_gm(s["adj"].dropna().values) / ref_all, 3) if len(s) >= 20 else None, "scored": int(len(s)),
                    "dinner_lift": r(_gm(sd["adj"].dropna().values) / ref_sd, 3) if len(sd) >= 15 else None, "dinner_n": int(len(sd))})
    out["secular"] = sec
    hm = []
    ref_free = _ref(ev.loc[ev["scored"] & ev["type"].isin(FREE), "adj"].dropna().values)
    for m_, name in [(7, "Tishrei"), (8, "Cheshvan"), (9, "Kislev"), (10, "Tevet"), (11, "Shvat"), (12, "Adar"), (1, "Nisan"), (2, "Iyar"), (3, "Sivan"), (4, "Tammuz"), (5, "Av"), (6, "Elul")]:
        g = ev[ev["hm"].isin([m_, 13] if m_ == 12 else [m_])]
        s = g[g["scored"] & g["type"].isin(FREE)]
        hm.append({"month": name, "n": int(len(g)), "centers": int(g["center"].nunique()), "lift": r(_gm(s["adj"].dropna().values) / ref_free, 3) if len(s) >= 25 else None, "scored": int(len(s))})
    out["hebrew_months"] = hm
    log("timing: types", len(out["types"]), "| free-choice events", out["overall"]["n_free"], "scored", out["overall"]["scored_free"])
    return out
