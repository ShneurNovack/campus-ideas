"""The front page: what is trending now, what did well lately, what usually works at this point of the year,
and what is coming up across the network.

"Now" is a rolling window, not a calendar month: the last two weeks (events with final numbers) plus the next
three (events centers have already posted). A calendar month would be empty on the 1st.
"""
import datetime as dt
import numpy as np
from collections import Counter, defaultdict
from . import calendars as cal
from .common import r, log
from .types import TYPES, EVENT_TYPES

BACK, AHEAD = 14, 21


def build(events, centers, P, ideas, themes_blob, today):
    ev = events[~events["canceled"] & events["type"].isin(EVENT_TYPES + ["other"])]
    lo, hi = today - dt.timedelta(days=BACK), today + dt.timedelta(days=AHEAD)
    win = ev[(ev["date"] >= lo) & (ev["date"] <= hi)]
    idea_of = P["idea"].to_dict()
    ev_idea = ev["pid"].map(idea_of)
    ideas_by_id = {i["id"]: i for i in ideas}
    th = cal.hdoy_norm(today)

    # same window last year, on the calendar each idea follows
    def window_last_year(idea, rows):
        if idea["status"] == "seasonal":
            return rows[(rows["hdoy_n"] >= th - BACK) & (rows["hdoy_n"] <= th + AHEAD) & (rows["date"] < lo - dt.timedelta(days=200)) & (rows["date"] > lo - dt.timedelta(days=560))]
        a, b = lo - dt.timedelta(days=364), hi - dt.timedelta(days=364)
        return rows[(rows["date"] >= a) & (rows["date"] <= b)]

    act_now = win["center"].nunique()
    ly_all = ev[(ev["date"] >= lo - dt.timedelta(days=364)) & (ev["date"] <= hi - dt.timedelta(days=364))]
    growth = act_now / max(1, ly_all["center"].nunique())
    trending = []
    grouped = {i: g for i, g in ev.groupby(ev_idea) if i == i and i >= 0}
    for iid, rows in grouped.items():
        idea = ideas_by_id.get(int(iid))
        if not idea:
            continue
        now = rows[(rows["date"] >= lo) & (rows["date"] <= hi)]
        cn = now["center"].nunique()
        if cn < 2:
            continue
        ly = window_last_year(idea, rows)
        cl = ly["center"].nunique()
        expected = cl * growth
        ratio = (cn + 1) / (expected + 1)
        staple = idea.get("staple", False)
        sc = now[now["scored"]]
        best = now.sort_values(["regs"], ascending=False).iloc[0]
        trending.append({"idea": int(iid), "label": idea["label"], "type": idea["type"], "centers": int(cn), "last_year": int(cl), "ratio": r(ratio, 2),
                         "upcoming": int(now["upcoming"].sum()), "ran": int(now["ran"].sum()), "staple": bool(staple), "status": idea["status"],
                         "lift_now": r(float(np.exp(np.mean(np.clip(np.log(sc["adj"]), -1.5, 1.5)))), 2) if len(sc) >= 3 else None, "lift": idea["lift"],
                         "img": next((x for x in now.sort_values("regs", ascending=False)["img"] if isinstance(x, str)), None), "top_pid": int(best["pid"])})
    hot = sorted([t for t in trending if not t["staple"]], key=lambda t: (-t["centers"], -t["ratio"]))[:24]
    rising = sorted([t for t in trending if not t["staple"] and t["centers"] >= 3 and t["ratio"] >= 1.3 and t["centers"] >= t["last_year"] + 2], key=lambda t: -t["ratio"] * np.sqrt(t["centers"]))[:16]
    new = []
    for i in ideas:
        if i["first"] and (today - dt.date.fromisoformat(i["first"]["date"])).days <= 150 and i["centers"] >= 2 and i["type"] not in ("holiday_service", "shabbat_services") and not i.get("staple"):
            new.append({"idea": i["id"], "label": i["label"], "type": i["type"], "centers": i["centers"], "first": i["first"], "lift": i["lift"], "img_pid": i["flyers"][0] if i["flyers"] else None})
    new.sort(key=lambda x: (x["first"]["date"], x["centers"]), reverse=True)

    # what did well in the last three weeks
    recent = ev[(ev["date"] >= today - dt.timedelta(days=21)) & ev["ran"] & ev["scored"] & (ev["regs"] >= 15) & (ev["base"].fillna(ev["cbase"]) >= 5)]
    rel = set(centers.index[centers["reliable"]])
    recent = recent[recent["center"].isin(rel)].sort_values("adj", ascending=False)
    wins, seen = [], Counter()
    for _, x in recent.iterrows():
        if seen[x["center"]] >= 2:
            continue
        seen[x["center"]] += 1
        wins.append({"id": int(x["id"]), "pid": int(x["pid"]), "name": x["name"], "center": int(x["center"]), "date": x["date"].isoformat(), "regs": int(x["regs"]),
                     "base": r(x["base"] if x["base"] == x["base"] else x["cbase"], 1), "lift": r(x["adj"], 2), "type": x["type"], "img": x["img"] if isinstance(x["img"], str) else None})
        if len(wins) >= 24:
            break

    # what usually runs at this point of the year (all years), staples aside
    usual = []
    for iid, rows in grouped.items():
        idea = ideas_by_id.get(int(iid))
        if not idea or idea.get("staple"):
            continue
        past = rows[rows["ran"]]
        if len(past) < 4:
            continue
        if idea["status"] == "seasonal":
            near = (past["hdoy_n"] >= th - 3) & (past["hdoy_n"] <= th + 28)
        else:
            d0 = today.timetuple().tm_yday
            gap = (past["doy"] - d0) % 365
            near = gap <= 28
        share = float(near.mean()); cn = past.loc[near, "center"].nunique()
        if cn >= 3 and share >= 0.3:
            usual.append({"idea": int(iid), "label": idea["label"], "type": idea["type"], "centers": int(cn), "share": r(share, 2), "lift": idea["lift"], "when": idea["when"],
                          "img_pid": idea["flyers"][0] if idea["flyers"] else None})
    usual.sort(key=lambda x: -x["centers"] * x["share"])

    up = ev[ev["upcoming"]].sort_values("dt")
    pulse = {"upcoming_events": int(len(up)), "upcoming_centers": int(up["center"].nunique()), "signups_so_far": int(up["regs"].sum()),
             "next7": int((up["date"] <= today + dt.timedelta(days=7)).sum()), "ran_last14": int(((ev["date"] >= lo) & ev["ran"]).sum()),
             "active_centers_now": int(act_now), "active_centers_last_year": int(ly_all["center"].nunique()),
             "by_type": [{"type": t, "n": int(n)} for t, n in Counter(up["type"]).most_common()]}
    hol = [{"key": k, "label": l, "date": s.isoformat(), "days": d} for k, l, s, d in cal.upcoming_jewish(today, 75)]
    sec = []
    for y in (today.year, today.year + 1):
        for key, a, b, day in cal.secular_windows(y):
            if 0 <= (day - today).days <= 60:
                sec.append({"key": key, "label": cal.SECULAR_LABEL[key], "date": day.isoformat(), "days": (day - today).days})
    ctx = cal.context(today)
    blob = {"today": today.isoformat(), "window": {"from": lo.isoformat(), "to": hi.isoformat(), "back": BACK, "ahead": AHEAD}, "pulse": pulse,
            "context": {"parsha": ctx["parsha"], "hebrew": f"{ctx['hd']} {ctx['hmonth']} {ctx['hy']}", "holidays": hol, "secular": sorted(sec, key=lambda x: x["days"])},
            "hot": hot, "rising": rising, "new": new[:24], "wins": wins, "usual": usual[:24],
            "friday": themes_blob["fridays"][0] if themes_blob.get("fridays") else None}
    log("home: hot", len(hot), "rising", len(rising), "new", len(new), "wins", len(wins), "usual", len(usual))
    return blob


def coming_up(events, centers, P, today):
    up = events[events["upcoming"] & ~events["canceled"]].sort_values("dt")
    ran = events[events["ran"]]
    last_by_prog = ran.sort_values("dt").groupby("pid").tail(1).set_index("pid")
    yr = ran[ran["measured"] & (ran["date"] >= today - dt.timedelta(days=365))]
    typ_med = yr.groupby(["center", "type"])["regs"].median().to_dict()
    out = []
    for _, x in up.iterrows():
        prev = last_by_prog.loc[x["pid"]] if x["pid"] in last_by_prog.index else None
        out.append({"id": int(x["id"]), "pid": int(x["pid"]), "center": int(x["center"]), "name": x["name"], "dt": x["dt"].isoformat(timespec="minutes"), "type": x["type"],
                    "hol": x["hol"] if isinstance(x["hol"], str) else None, "regs": int(x["regs"]), "cap": int(x["capacity"]) if x["capacity"] == x["capacity"] and x["capacity"] else None,
                    "img": x["img"] if isinstance(x["img"], str) else None, "themes": list(x["tkeys"]), "short": (x["short"] or x["desc"])[:220],
                    "recurring": bool(x["recurring"]), "offsite": bool(x["offsite"]),
                    "prev": {"date": prev["date"].isoformat(), "regs": int(prev["regs"])} if prev is not None and prev["measured"] else None,
                    "base": r(typ_med.get((x["center"], x["type"])), 1)})
    return {"today": today.isoformat(), "events": out}
