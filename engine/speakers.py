"""Guests and speakers. Who gets brought to campus, how widely, and how those events land.

There is no speaker field, so names come out of the text. Rules alone are noisy ("with Danow Family", "with Sunday
Services"), so the engine does two passes: a wide regex picks the events that look like they feature a person
(cue words, a speaker-type event, a title such as Rabbi or Dr), and a language model reads only those events and
writes down the real names and a role. The model never sees a sign-up count. Its answers are cached by event id, so
a daily run reads only new events. Names are then folded with fuzzy matching (Rebbetzin / no title, spelling).

A guest is published when at least two centers hosted them, or one center brought them back. Turnout is read through lift.py, the same way as an idea.
"""
import json, os, re, time, urllib.request
from collections import defaultdict
import numpy as np
from rapidfuzz import fuzz, process
from . import embed
from .common import WORK, log, r

CACHE = os.path.join(WORK, "speakers_cache.json")
CUE = re.compile(r"\b(with|featuring|feat\.?|ft\.?|guest|speaker|w/|presents|hosted by|starring|conversation with|evening with|lecture|survivor|hostage|author|idf|soldier|comedian|chef|influencer|professor|dr\.?|rabbi [a-z]+ [a-z]+|rebbetzin [a-z]+ [a-z]+)\b", re.I)
TITLE = re.compile(r"^(rabbi|rebbetzin|rebbitzen|dr\.?|professor|prof\.?|mr\.?|mrs\.?|ms\.?|rav|reb|chef|coach|senator|mayor|major|colonel|col\.?|lt\.?|sgt\.?|captain|general|consul general|judge|dean|president|provost|sir)\s+", re.I)
NOT_PEOPLE = {"shabbat", "shabbos", "dinner", "services", "chabad", "jewish", "family", "club", "night", "party", "sushi", "social", "israel", "israeli", "holocaust", "survivor", "guest", "speaker", "students", "student", "campus", "hillel", "torah", "pizza", "parsha", "challah", "bake", "lunch", "kiddush", "seder", "chanukah", "hanukkah", "purim", "pesach", "passover", "sukkah", "sukkot", "rosh", "hashanah", "yom", "kippur", "mega", "shabbaton", "bbq", "sunday", "monday", "friday", "saturday", "week", "weekend", "soul", "food", "jewfro", "jlearn", "sinai", "scholars"}


def candidates(events):
    e = events[events["ran"]]
    text = (e["name"].fillna("") + " . " + e["short"].fillna("")).str.lower()
    mask = (e["type"] == "speaker") | text.str.contains(CUE)
    return e[mask]


def _post(path, body, tries=4):
    data = json.dumps(body).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(embed.EDGE + path, data=data, headers=embed._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except Exception as e:
            if attempt == tries - 1:
                log("extract failed:", e)
                return None
            time.sleep(2 + 3 * attempt)


def extract(events, batch=20, limit=None):
    """{event id: [{name, role}]} for every candidate event, through the cache."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    cand = candidates(events)
    todo = [row for row in cand.itertuples() if str(row.id) not in cache]
    if limit:
        todo = todo[:limit]
    log("speakers: candidates", len(cand), "| cached", len(cand) - len(todo), "| to read", len(todo))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        res = _post("/analyst/extract", {"events": [{"id": str(x.id), "name": x.name, "blurb": (x.short or x.desc or "")[:400]} for x in chunk]})
        if res is None or "people" not in res:
            continue
        got = defaultdict(list)
        for p in res["people"]:
            got[p["id"]].append({"name": p["name"], "role": p["role"]})
        for x in chunk:
            cache[str(x.id)] = got.get(str(x.id), [])
        if (i // batch) % 10 == 0:
            json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))
    return {int(k): v for k, v in cache.items() if v}


def _clean(name):
    n = re.sub(r"[^\w\s'\-.]", " ", name).strip()
    n = re.sub(r"\s+", " ", n)
    bare = TITLE.sub("", n).strip()
    words = bare.lower().split()
    if len(words) < 2 or len(words) > 4 or any(w.strip(".") in NOT_PEOPLE for w in words) or not all(w[0].isalpha() for w in words):
        return None, None
    return n, bare.title()


def fold(found):
    """Group spellings of one person. Returns {canonical: {events: [...], roles: Counter, names: Counter}}."""
    people = {}
    keys = []
    for eid, lst in found.items():
        for p in lst:
            full, bare = _clean(p["name"])
            if not bare:
                continue
            match = process.extractOne(bare, keys, scorer=fuzz.token_sort_ratio, score_cutoff=88) if keys else None
            key = match[0] if match else bare
            if not match:
                keys.append(bare)
                people[key] = {"events": set(), "roles": defaultdict(int), "names": defaultdict(int)}
            people[key]["events"].add(eid); people[key]["roles"][p["role"]] += 1; people[key]["names"][full] += 1
    return people


def run(events, lift_pools, today, limit=None):
    from . import lift
    found = extract(events, limit=limit)
    people = fold(found)
    by_id = events.set_index("id")
    groups, kind_of, centers_of, meta = {}, {}, {}, {}
    for key, p in people.items():
        ev = by_id.loc[[e for e in p["events"] if e in by_id.index]]
        centers = sorted(set(int(c) for c in ev["center"]))
        if len(centers) < 2 and len(ev) < 2:
            continue
        scored = [int(e) for e, s in zip(ev.index, ev["scored"]) if s]
        groups[key] = scored; kind_of[key] = "guest"; centers_of[key] = len(set(ev.loc[scored, "center"])) if scored else 0
        years = sorted(set(int(d.year if d.month >= 7 else d.year - 1) for d in ev["date"]))
        meta[key] = {"name": max(p["names"].items(), key=lambda kv: kv[1])[0], "role": max(p["roles"].items(), key=lambda kv: kv[1])[0],
                     "centers": len(centers), "events": int(len(ev)), "years": [f"{y}-{str(y + 1)[-2:]}" for y in years],
                     "last": ev["date"].max().isoformat(), "recent": int(((today - ev["date"]).map(lambda d: d.days) <= 540).sum()),
                     "kinds": sorted(set(ev["type"])), "titles": [t for t, _ in sorted(((n, c) for n, c in zip(ev["name"], ev["center"])), key=lambda x: x[0])][:6]}
    res = lift.assess(groups, kind_of, centers_of, lift_pools)
    out = []
    for key in meta:
        v = res[key]
        out.append({**meta[key], "tier": v["tier"], "band": v["band"], "scored": v["n"]})
    out.sort(key=lambda g: (-g["centers"], -g["events"]))
    log("speakers: people", len(people), "| published", len(out), "| at 2+ centers", sum(1 for g in out if g["centers"] >= 2))
    return {"guests": out, "read": len(found), "method": "regex candidates, model reads names, fuzzy fold at 88, lift as for ideas"}
