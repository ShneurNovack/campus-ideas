"""Pull the public Chabad on Campus sign-up data into data/raw.

Only the public, unauthenticated endpoints are read: the event schedules of every center that uses the sign-up
system (past, upcoming and canceled), each center's public profile and each campus's public profile.
Polite: three workers and short pauses, about 600 requests for everything. Raw JSON only. Flyer images are not
downloaded (their URLs are kept). Staff contact details are not stored.

  python -m engine.pull
"""
import json, os, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
A = "https://api.chabadoncampus.org/api/1.0"
from .common import RAW
lock = threading.Lock(); nreq = [0]
def get(path, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(A + path, headers={"User-Agent": "campus-ideas/0.1 (+https://github.com/ShneurNovack/campus-ideas)", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                with lock: nreq[0] += 1
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (404, 401, 400): return {"http": e.code}
            time.sleep(2 + 3 * t)
        except Exception as e:
            time.sleep(2 + 3 * t)
    return {"err": True}
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def pull_enrollment(eid):
    out = []; page = 1; total = None
    while True:
        d = get(f"/RsvpEnrollments/{eid}/EventSchedules?occurrenceStatus=All&page={page}&results=200")
        if "payload" not in d: return eid, (None if page == 1 else out), d
        p = d["payload"]; out += p["results"]; total = p["numberOfRows"]
        if len(out) >= total or not p["results"]: break
        page += 1; time.sleep(0.15)
    res = get(f"/RsvpEnrollments/{eid}/Resources")
    return eid, out, {"total": total, "resources": (res.get("payload") or {}).get("results", [])}

def main():
    houses = get("/ChabadHouses/public")
    json.dump(houses, open(f"{RAW}/houses_public.json", "w"))
    log("houses", len(houses))
    system = get("/system/public"); json.dump(system, open(f"{RAW}/system.json", "w"))
    events = {}; meta = {}
    ids = list(range(1, 261))
    with ThreadPoolExecutor(3) as ex:
        for eid, rows, info in ex.map(pull_enrollment, ids):
            if rows is None: continue
            events[eid] = rows; meta[eid] = {"total": info.get("total"), "resources": info.get("resources")}
            if len(events) % 20 == 0: log("enrollments", len(events), "events", sum(len(v) for v in events.values()), "req", nreq[0])
    mx = max(events) if events else 0
    log("done sweep; enrollments", len(events), "max id", mx, "events", sum(len(v) for v in events.values()))
    # extend sweep if ids near the top of the range exist
    nxt = 261
    while mx > nxt - 60:
        more = list(range(nxt, nxt + 60))
        with ThreadPoolExecutor(3) as ex:
            for eid, rows, info in ex.map(pull_enrollment, more):
                if rows is None: continue
                events[eid] = rows; meta[eid] = {"total": info.get("total"), "resources": info.get("resources")}
        mx = max(events); nxt += 60
    json.dump({str(k): v for k, v in events.items()}, open(f"{RAW}/events.json", "w"))
    json.dump({str(k): v for k, v in meta.items()}, open(f"{RAW}/enrollment_meta.json", "w"))
    house_ids = sorted({r["chabadHouseID"] for v in events.values() for r in v})
    log("houses with events", len(house_ids))
    KEEP = ["id","name","slug","websiteURL","logoURL","buildingImageURL","regionID","region","timezoneID","address","campuses","isRegistered501"]
    def house(h):
        d = get(f"/ChabadHouses/Public/{h}"); time.sleep(0.1)
        if "id" not in d: return h, None
        return h, {k: d.get(k) for k in KEEP}
    hd = {}
    with ThreadPoolExecutor(3) as ex:
        for h, d in ex.map(house, house_ids):
            if d: hd[h] = d
    json.dump({str(k): v for k, v in hd.items()}, open(f"{RAW}/house_detail.json", "w"))
    log("house details", len(hd))
    campus_ids = sorted({c["campusID"] for d in hd.values() for c in (d.get("campuses") or [])} | {h.get("campusID") for h in houses if h.get("id") in hd and h.get("campusID")})
    CK = ["id","name","nickname","isActive","annualTuition","isInstitutionControlPublic","studentResidencyType","schoolCalendarType","prestige","graduatePopulation","undergradPopulation","jewishUndergradPopulation","jewishGraduatePopulation","chabadHouses","studyAreas","programLevels","jewishStudentLife","chabadStudentGroupStatus"]
    def campus(c):
        d = get(f"/Campuses/Public/{c}"); time.sleep(0.1)
        if "id" not in d: return c, None
        return c, {k: d.get(k) for k in CK}
    cd = {}
    with ThreadPoolExecutor(3) as ex:
        for c, d in ex.map(campus, campus_ids):
            if d: cd[c] = d
    json.dump({str(k): v for k, v in cd.items()}, open(f"{RAW}/campus_detail.json", "w"))
    log("campus details", len(cd), "| total requests", nreq[0])
    json.dump({"pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "requests": nreq[0]}, open(f"{RAW}/pull_meta.json", "w"))
    log("pull complete")


if __name__ == "__main__":
    main()
