"""Raw API pull -> tidy tables.
events   one row per event occurrence (past, upcoming, canceled), with the three calendars attached
centers  one row per Chabad house that uses the RSVP system, with its primary campus
"""
import re
import datetime as dt
import pickle, os, statistics
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from .common import RAW, WORK, IMG_PREFIX, load, log, clean, norm_name
from . import calendars as cal

TODAY = None  # set from the pull timestamp so a re-run on old data is reproducible


def _img(u):
    if not u:
        return None
    return u[len(IMG_PREFIX):] if u.startswith(IMG_PREFIX) else u


def build_centers():
    houses = {h["id"]: h for h in load(f"{RAW}/houses_public.json")}
    detail = {int(k): v for k, v in load(f"{RAW}/house_detail.json").items()}
    campuses = {int(k): v for k, v in load(f"{RAW}/campus_detail.json").items()}
    events = load(f"{RAW}/events.json")
    enroll = {}
    for eid, rows in events.items():
        for row in rows[:1]:
            enroll[row["chabadHouseID"]] = int(eid)
    out = []
    for hid, d in detail.items():
        pub = houses.get(hid, {})
        links = d.get("campuses") or []
        prim = next((c for c in links if c.get("isPrimary")), None) or (links[0] if links else None)
        cid = (prim or {}).get("campusID") or pub.get("campusID")
        c = campuses.get(cid, {}) if cid else {}
        addr = d.get("address") or pub.get("address") or {}
        loc = pub.get("location") or {}
        ju, jg = c.get("jewishUndergradPopulation"), c.get("jewishGraduatePopulation")
        # every campus the center serves, with the Jewish student count its staff entered for each
        served = []
        for l in links:
            lc = campuses.get(l.get("campusID"), {})
            j = (lc.get("jewishUndergradPopulation") or 0) + (lc.get("jewishGraduatePopulation") or 0)
            tot = (lc.get("undergradPopulation") or 0) + (lc.get("graduatePopulation") or 0)
            if tot and j > tot:
                j = 0
            served.append({"id": l.get("campusID"), "name": clean(lc.get("name") or l.get("campusName")), "jpop": int(j) or None, "primary": bool(l.get("isPrimary")), "full_time": bool(l.get("isFullTime"))})
        out.append({
            "id": hid, "enrollment": enroll.get(hid), "name": clean(d.get("name") or pub.get("name")), "slug": d.get("slug"),
            "region": (d.get("region") or {}).get("name"), "tz": d.get("timezoneID"),
            "city": clean(addr.get("city")), "state": clean(addr.get("state")), "country": clean(addr.get("country")),
            "lat": loc.get("latitude"), "lon": loc.get("longitude"), "website": d.get("websiteURL"),
            "logo": _img(d.get("logoURL")), "building": _img(d.get("buildingImageURL") or pub.get("buildingImageURL")),
            "campus_id": cid, "campus": clean(c.get("name") or pub.get("campusName")), "nick": clean(c.get("nickname")) or None,
            "n_campuses": len(links), "full_time": bool((prim or {}).get("isFullTime")),
            "pop_ug": c.get("undergradPopulation"), "pop_grad": c.get("graduatePopulation"),
            "jpop_ug": ju, "jpop_grad": jg, "jpop": (ju or 0) + (jg or 0) or None,
            "public": c.get("isInstitutionControlPublic"), "residency": c.get("studentResidencyType"),
            "cal_type": c.get("schoolCalendarType"), "prestige": c.get("prestige"), "tuition": c.get("annualTuition"),
            "levels": c.get("programLevels") or [], "served": served,
        })
    df = pd.DataFrame(out).set_index("id", drop=False).sort_index()
    # a nickname is a label, so no stray quotes or commas; a Jewish population above the whole enrollment is a typo
    df["nick"] = df["nick"].map(lambda s: (re.sub(r"^[^\w]+|[^\w)]+$", "", s) or None) if isinstance(s, str) else None)
    tot = df[["pop_ug", "pop_grad"]].sum(axis=1, min_count=1)
    bad = df["jpop"].notna() & tot.notna() & (df["jpop"] > tot)
    df.loc[bad, ["jpop", "jpop_ug", "jpop_grad"]] = None
    # Jewish students across every campus the center serves; the primary campus alone when that is all there is
    df["jpop_served"] = [(sum(x["jpop"] or 0 for x in sv) or None) if sv else j for sv, j in zip(df["served"], df["jpop"])]
    df["jpop_served"] = df["jpop_served"].where(df["jpop"].notna(), None).astype(float)   # no count for the main campus: not listed
    return df


_TEST = re.compile(r"\btest(ing)?\b|\bnot real\b|\bdemo\b|\bfake\b", re.I)


def build_events(pulled_at):
    raw = load(f"{RAW}/events.json")
    today = pulled_at.date()
    rows = []
    for eid, lst in raw.items():
        for e in lst:
            try:
                d = dt.datetime.fromisoformat(e["eventDateTime"])
            except Exception:
                continue
            if d.year < 2019 or d.date() > today + dt.timedelta(days=400):
                continue
            name = clean(e.get("name"))
            if _TEST.search(name) and "test your" not in name.lower():
                continue
            desc, short = clean(e.get("description")), clean(e.get("shortDescription"))
            if short and (short == desc or short in desc):
                short = ""
            rows.append({
                "id": e["eventScheduleID"], "setting": e["eventSettingID"], "enrollment": int(eid), "center": e["chabadHouseID"],
                "name": name, "nname": norm_name(name), "desc": desc, "short": short, "img": _img(e.get("imageURL")),
                "dt": d, "date": d.date(), "tz": e.get("timezoneID"),
                "canceled": e.get("eventState") == "Canceled", "upcoming": e.get("occurrenceStatus") == "Upcoming" and e.get("eventState") != "Canceled",
                "setting_state": e.get("eventSettingState"), "recurring": e.get("scheduleType") == "Recurring",
                "offsite": e.get("location") == "Other", "place": clean(e.get("locationName") or e.get("address1")) or None,
                "capacity": e.get("totalCapacity"), "regs": int(e.get("numberOfRegistrations") or 0),
                "fee": e.get("registrationFeeAmount"), "sms": e.get("primarySmsCode"), "reg_end": e.get("registrationEndDateTime"),
            })
    df = pd.DataFrame(rows).drop_duplicates("id").sort_values(["center", "dt"]).reset_index(drop=True)
    df["wd"] = df["dt"].dt.weekday            # Monday 0 .. Sunday 6
    df["hour"] = df["dt"].dt.hour
    df["ran"] = ~df["canceled"] & ~df["upcoming"]
    # calendars
    ctx = {d: cal.context(d) for d in df["date"].unique()}
    for k in ("hy", "hm", "hd", "hdoy", "hdoy_n", "jh", "jh_runup", "jh_days", "sec", "parsha"):
        df[k] = df["date"].map(lambda d, k=k: ctx[d][k])
    df["doy"] = df["dt"].dt.dayofyear
    return df


def add_terms(df, today):
    """Academic terms. No API field gives term dates, so they are inferred. Step one finds clean terms per center
    (runs of events with no gap over 24 days) and takes the network's median start for each season and year.
    Step two gives every center its own start: the first date within three weeks before to seven weeks after the
    network median where its activity really begins (two events inside ten days). Fall runs to Dec 22, spring to
    May 31; the rest is winter or summer break."""
    live = df[~df["canceled"]]
    terms = []
    for c, g in live.groupby("center"):
        ds = sorted(set(g["date"]))
        run = [ds[0]]
        for a, b in zip(ds, ds[1:]):
            if (b - a).days > 24:
                terms.append((c, run[0], run[-1], len(run))); run = [b]
            else:
                run.append(b)
        terms.append((c, run[0], run[-1], len(run)))
    good = [(c, s, e, n) for c, s, e, n in terms if n >= 5 and 35 <= (e - s).days <= 150]
    by = defaultdict(list)
    for c, s, e, n in good:
        if s.month in (8, 9):
            by[("fall", s.year)].append(s.timetuple().tm_yday)
        elif s.month in (1, 2):
            by[("spring", s.year)].append(s.timetuple().tm_yday)
    med = {}
    for y in range(2019, today.year + 2):
        f, sp = by.get(("fall", y), []), by.get(("spring", y), [])
        med[("fall", y)] = dt.date(y, 1, 1) + dt.timedelta(days=int(statistics.median(f)) - 1) if len(f) >= 5 else dt.date(y, 8, 24)
        med[("spring", y)] = dt.date(y, 1, 1) + dt.timedelta(days=int(statistics.median(sp)) - 1) if len(sp) >= 5 else dt.date(y, 1, 18)

    starts = {}  # (center, season, year) -> start date
    dates_by_center = {c: sorted(set(g["date"])) for c, g in live.groupby("center")}
    import bisect
    def center_start(c, season, y):
        key = (c, season, y)
        if key in starts:
            return starts[key]
        m = med[(season, y)]
        lo, hi = m - dt.timedelta(days=21), m + dt.timedelta(days=50)
        if season == "spring":
            lo = max(lo, dt.date(y, 1, 2))
        ds = dates_by_center.get(c, [])
        i = bisect.bisect_left(ds, lo)
        s = m
        while i < len(ds) and ds[i] <= hi:
            j = bisect.bisect_right(ds, ds[i] + dt.timedelta(days=10))
            if j - i >= 2:
                s = ds[i]; break
            i += 1
        starts[key] = s
        return s

    tid, sea_, tw, tp = [], [], [], []
    for c, d in zip(df["center"], df["date"]):
        y = d.year
        fs, ss = center_start(c, "fall", y), center_start(c, "spring", y)
        if fs <= d <= dt.date(y, 12, 22):
            sea, s, e = "fall", fs, dt.date(y, 12, 15)
        elif ss <= d <= dt.date(y, 5, 31):
            sea, s, e = "spring", ss, dt.date(y, 5, 10)
        elif dt.date(y, 6, 1) <= d < fs:
            sea, s, e = "summer", None, None
        else:
            sea, s, e = "winter", None, None
        if s is None:
            tid.append(None); sea_.append(sea); tw.append(np.nan); tp.append(np.nan); continue
        monday = s - dt.timedelta(days=s.weekday())
        tid.append(f"{c}:{sea}{y}"); sea_.append(sea); tw.append((d - monday).days // 7 + 1)
        tp.append(min(1.0, max(0.0, (d - s).days / max((e - s).days, 1))))
    df["term"], df["season"], df["tweek"], df["tpos"] = tid, sea_, tw, tp
    info = {"median_start": {f"{k[0]} {k[1]}": v.isoformat() for k, v in sorted(med.items(), key=lambda x: x[1]) if k[1] >= 2020},
            "clean_terms_used": len(good), "centers_with_clean_terms": len({c for c, *_ in good})}
    return df, info


def run():
    meta = load(f"{RAW}/pull_meta.json")
    pulled_at = dt.datetime.strptime(meta["pulled_at"], "%Y-%m-%dT%H:%M:%SZ")
    centers = build_centers()
    events = build_events(pulled_at)
    events = events[events["center"].isin(centers.index)].reset_index(drop=True)
    events, term_info = add_terms(events, pulled_at.date())
    log("events", len(events), "ran", int(events["ran"].sum()), "upcoming", int(events["upcoming"].sum()), "canceled", int(events["canceled"].sum()), "centers", len(centers))
    log("season mix", dict(Counter(events["season"])))
    pickle.dump({"events": events, "centers": centers, "pulled_at": pulled_at, "term_info": term_info}, open(f"{WORK}/prepared.pkl", "wb"))
    return events, centers, pulled_at, term_info


if __name__ == "__main__":
    run()
