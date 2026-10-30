"""Which Shabbos dinner themes are spreading, and when in the year each one belongs.

Trend is counted in centers, never in events: a theme is rising when more distinct centers ran it in the last twelve
months than in the twelve before, with at least three centers now. Counting events would let one center that runs
a theme weekly look like a movement. Adopters are centers whose first run was in the last twelve months.

Seasonality is read on the axis each theme follows (themes.py decides: a holiday theme follows the Jewish calendar,
"Welcome Back" follows the semester, "Super Bowl" follows the civil year). The engine publishes how concentrated
the theme is in its peak window and whether that window is coming up from today.
"""
import datetime as dt
from collections import defaultdict
import numpy as np
from .common import r, log

MIN_CENTERS = 3
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _window(dates, today):
    now = [d for d in dates if 0 <= (today - d).days < 365]
    before = [d for d in dates if 365 <= (today - d).days < 730]
    return now, before


def build(events, theme_blob, lift_res, today):
    sd = events[events["ran"] & (events["type"] == "shabbat_dinner")]
    occ = defaultdict(list)       # key -> [(date, center)]
    for d, c, keys in zip(sd["date"].values, sd["center"].values, sd["tkeys"].values):
        for k in (keys or []):
            occ[k].append((d, int(c)))
    out = []
    for t in theme_blob["themes"]:
        k = t["key"]
        rows = occ.get(k, [])
        if not rows:
            continue
        centers_now = {c for d, c in rows if 0 <= (today - d).days < 365}
        centers_before = {c for d, c in rows if 365 <= (today - d).days < 730}
        first = {}
        for d, c in sorted(rows):
            first.setdefault(c, d)
        adopters = sum(1 for c, d in first.items() if 0 <= (today - d).days < 365)
        all_centers = len(first)
        # season on the theme's own axis
        months = np.zeros(12)
        for d, _ in rows:
            months[d.month - 1] += 1
        share = months / months.sum()
        order = np.argsort(-share)
        top2 = float(share[order[:2]].sum())
        peak = [MONTHS[i] for i in order[:2] if share[i] >= 0.12]
        # is the peak ahead? months until the strongest month, 0..11
        ahead = (int(order[0]) + 1 - today.month) % 12
        if t.get("follows") == "jewish" and len(centers_before) <= 1 and len(centers_now) >= MIN_CENTERS:
            status = "calendar"          # the holiday fell on Shabbos this year and not last: the calendar moved, not the centers
        elif len(centers_now) >= MIN_CENTERS and len(centers_now) >= 1.3 * max(len(centers_before), 2):
            status = "rising"
        elif len(centers_before) >= MIN_CENTERS and len(centers_now) <= 0.7 * len(centers_before):
            status = "fading"
        elif len(centers_now) >= MIN_CENTERS:
            status = "steady"
        else:
            status = "rare"
        lr = lift_res.get(k, {})
        out.append({"key": k, "label": t["label"], "family": t.get("family"), "follows": t.get("follows"), "when": t.get("when"),
                    "centers_now": len(centers_now), "centers_before": len(centers_before), "centers_ever": all_centers, "adopters": adopters, "status": status,
                    "seasonal": top2 >= 0.5 and len(rows) >= 6, "peak": peak, "months_ahead": ahead, "concentration": r(top2, 2), "months": [int(x) for x in months],
                    "tier": lr.get("tier", "none"), "band": lr.get("band"), "hit": r(lr.get("hit"), 2) if lr.get("hit") is not None else None,
                    "names": [n["name"] for n in t.get("names", [])[:4]], "descs": [d["text"] for d in t.get("descs", [])[:3]], "paired": t.get("paired", [])[:4], "related": [x["id"] for x in t.get("related", [])[:4]]})
    out.sort(key=lambda x: (-x["centers_now"], -x["centers_ever"]))
    summary = {"themes": len(out), "rising": sum(1 for x in out if x["status"] == "rising"), "fading": sum(1 for x in out if x["status"] == "fading"), "seasonal": sum(1 for x in out if x["seasonal"])}
    log("trends:", summary)
    return {"themes": out, "summary": summary, "min_centers": MIN_CENTERS}
