"""Write everything the app reads: the catalog of programs, the search vectors, the page payloads. Also runs
the engine's checks on itself (how good the search projection is, how well neighbors predict results)."""
import os, json, time, datetime as dt
import urllib.request
import numpy as np
from collections import Counter
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from . import embed
from .common import OUT, WORK, IMG_PREFIX, dump, log, r
from .types import TYPES, EVENT_TYPES

EPOCH = dt.date(2020, 1, 1)
K = 256


def day(d):
    return (d - EPOCH).days


def img_short(s):
    return s if not isinstance(s, str) else s


def _prior_axes(d):
    """The search axes from the last run: from the work folder, or from the published projection file."""
    try:
        a = np.load(f"{WORK}/axes.npy")
        if a.shape == (K, d):
            return a.astype(np.float32)
    except Exception:
        pass
    try:
        edge = os.environ.get("EDGE_URL", "https://campus-ideas.shneur.workers.dev")
        with urllib.request.urlopen(urllib.request.Request(f"{edge}/data/proj.bin", headers={"user-agent": "campus-ideas-engine"}), timeout=60) as resp:
            f = np.frombuffer(resp.read(), dtype=np.float32)
        if int(f[0]) == d and int(f[1]) == K:
            return f[2 + d:2 + d + K * d].reshape(K, d).copy()
    except Exception:
        pass
    return None


def project(V, mu_name):
    """Reduce to K dimensions for the in-browser search. Returns the int8 matrix, the projection file, and recall.

    The program vectors are already centered, so the reduction keeps the origin where it is (no second mean). A
    typed search is then placed in the same space by the Worker with one subtraction, the average event name,
    and one matrix product."""
    Vc = V
    w, U = np.linalg.eigh((V.T @ V).astype(np.float64))
    comp = U[:, np.argsort(-w)[:K]].T.astype(np.float32)
    kept = float(np.sort(w)[::-1][:K].sum() / w.sum())
    # Keep yesterday's axes when they still fit. A fresh fit can rotate the weaker axes from one day to the next, and
    # for a few minutes after a run the Worker and open browser tabs would then disagree about the space.
    prior = _prior_axes(V.shape[1])
    if prior is not None:
        kept_prior = float(((V @ prior.T) ** 2).sum() / (V ** 2).sum())
        if kept_prior >= kept - 0.01:
            comp, kept = prior, kept_prior
    np.save(f"{WORK}/axes.npy", comp)
    X = V @ comp.T
    Xn = embed.unit(X)
    q8 = np.clip(np.round(Xn / np.abs(Xn).max(1, keepdims=True) * 127), -127, 127).astype(np.int8)
    # recall of the small int8 search against exact cosine on the full vectors
    rng = np.random.default_rng(11); qs = rng.choice(len(V), size=min(400, len(V)), replace=False)
    full = embed.unit(Vc)
    exact = np.argsort(-(full[qs] @ full.T), axis=1)[:, 1:11]
    Q = q8.astype(np.float32); Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    approx = np.argsort(-(Xn[qs] @ Qn.T), axis=1)[:, 1:11]
    recall = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(exact, approx)]))
    proj = np.concatenate([[V.shape[1], K], mu_name.astype(np.float32), comp.ravel()]).astype(np.float32)
    return q8, proj, Xn, {"dims": K, "recall_at_10": r(recall, 3), "variance_kept": r(kept, 3)}


def thresholds(P, Xn):
    """What cosine means here: pairs of programs inside the same idea (different centers) against random pairs."""
    rng = np.random.default_rng(5)
    same = []
    for i, g in P[P["idea"] >= 0].groupby("idea"):
        ids = g["pid"].values
        if len(ids) < 2:
            continue
        for _ in range(min(6, len(ids))):
            a, b = rng.choice(ids, 2, replace=False)
            if P.at[a, "center"] != P.at[b, "center"]:
                same.append(float(Xn[a] @ Xn[b]))
    a = rng.integers(0, len(P), 20000); b = rng.integers(0, len(P), 20000)
    rand = np.einsum("ij,ij->i", Xn[a], Xn[b])
    return {"same_p10": r(np.percentile(same, 10), 3), "same_p50": r(np.percentile(same, 50), 3), "random_p50": r(np.percentile(rand, 50), 3),
            "random_p95": r(np.percentile(rand, 95), 3), "random_p99": r(np.percentile(rand, 99), 3)}


def _bare(name, desc, short):
    """A program posted with a name and nothing else. Search treats these with care (see src/lib/search.js)."""
    nm, d, s = (name or "").strip().lower(), (desc or "").strip().lower(), (short or "").strip().lower()
    return (len(d) < 25 or d == nm) and (len(s) < 25 or s == nm)


def catalog(P, events, centers, ideas, theme_ids, q8):
    cidx = {c: i for i, c in enumerate(centers.index)}
    tkeys = list(TYPES)
    ev = events[events["ran"] | events["upcoming"]].sort_values("dt")
    occ = {}
    for pid, g in ev[~events.loc[ev.index, "canceled"]].groupby("pid"):
        occ[int(pid)] = [[day(d), int(rg) if not up else -1, int(round(a * 100)) if (a == a and sc_) else 0] for d, rg, a, up, sc_ in zip(g["date"], g["regs"], g["adj"], g["upcoming"], g["scored"])]
    cs = []
    for c, row in centers.iterrows():
        short = row["nick"] if isinstance(row["nick"], str) and row["nick"] else (row["campus"] or row["name"])
        cs.append({"id": int(c), "name": row["name"], "short": short, "campus": row["campus"], "city": row["city"], "state": row["state"], "country": row["country"],
                   "region": row["region"], "jpop": int(row["jpop_served"]) if row.get("jpop_served") == row.get("jpop_served") and row.get("jpop_served") else None,
                   "pop": int(np.nansum([row["pop_ug"], row["pop_grad"]])) if np.nansum([row["pop_ug"], row["pop_grad"]]) > 0 else None,
                   "logo": row["logo"], "img": row["building"], "slug": row["slug"], "site": row["website"], "lat": r(row["lat"], 3), "lon": r(row["lon"], 3),
                   "cat": {k: row[k] for k in ("size", "typical", "pace", "jewish", "share", "strength", "signups", "stage") if k in row and isinstance(row[k], str)},
                   "usual": r(row["usual"], 1) if row.get("usual") == row.get("usual") else None,
                   "reliable": bool(row["reliable"]), "public": row["public"] if row["public"] == row["public"] else None, "residency": row["residency"]})
    cols = {
        "key": list(P["key"]), "c": [cidx[c] for c in P["center"]], "name": list(P["name"]), "t": [tkeys.index(t) for t in P["type"]],
        "hol": [h if isinstance(h, str) else None for h in P["hol"]], "aud": [a or None for a in P["aud"]],
        "n": [int(x) for x in P["n"]], "up": [int(x) for x in P["n_up"]], "first": [day(d) for d in P["first"]], "last": [day(d) for d in P["last"]],
        "reg": [r(x, 1) for x in P["med_regs"]], "mx": [int(x) for x in P["max_regs"]], "adj": [int(round(x * 100)) if x == x and x is not None else None for x in P["adj"]],
        "ns": [int(x) for x in P["n_scored"]],
        # where its runs landed among all of its center's events (percentile, 0 to 100) and the runs behind that:
        # the neighbor system's raw material
        "pc": [int(round(x * 100)) if x == x and x is not None else None for x in P["pc"]], "nl": [int(x) for x in P["n_pc"]], "img": [x if isinstance(x, str) else None for x in P["img"]], "idea": [int(x) for x in P["idea"]],
        "th": [[theme_ids.get(k, k) for k in ks] or None for ks in P["themes"]], "wd": [int(x) for x in P["wd"]], "hr": [int(x) for x in P["hour"]],
        "fl": [(1 if a else 0) + (2 if b else 0) + (4 if c_ else 0) + (8 if _bare(nm, d_, s_) else 0)
               for a, b, c_, nm, d_, s_ in zip(P["collab"], P["offsite"], P["recurring"], P["name"], P["desc"], P["short"])],
    }
    core = {"epoch": EPOCH.isoformat(), "img_prefix": IMG_PREFIX, "types": TYPES, "type_keys": tkeys, "event_types": EVENT_TYPES, "centers": cs, "programs": cols, "dims": int(q8.shape[1]), "count": int(len(P))}
    text = {"desc": [d[:700] for d in P["desc"]], "short": [s[:200] for s in P["short"]], "occ": [occ.get(int(p), []) for p in P["pid"]]}
    return core, text


def write_all(state):
    P, V, events, centers = state["P"], state["V"], state["events"], state["centers"]
    q8, proj, Xn, search_info = project(V, state["space"]["mu_name"])
    q8.tofile(f"{OUT}/vectors.bin"); proj.tofile(f"{OUT}/proj.bin")
    theme_ids = {t["key"]: t["id"] for t in state["theme_blob"]["themes"]}
    core, text = catalog(P, events, centers, state["ideas"], theme_ids, q8)
    sizes = {}
    sizes["catalog.json"] = dump(core, f"{OUT}/catalog.json")
    sizes["text.json"] = dump(text, f"{OUT}/text.json")
    for name in ("themes", "ideas", "timing", "home", "coming", "niches", "centers"):
        sizes[f"{name}.json"] = dump(state["centers_blob"] if name == "centers" else state[name], f"{OUT}/{name}.json")
    thr = thresholds(P, Xn)
    pred = state.get("check")
    dump({"model": state["space"]["model"], "names": state["space"]["names"], "blurbs": state["space"]["blurbs"]}, f"{OUT}/embed.json")
    ev = events
    engine = {
        "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", "pulled_at": state["pulled_at"].isoformat() + "Z", "today": state["today"].isoformat(),
        "counts": {"events": int(len(ev)), "ran": int(ev["ran"].sum()), "upcoming": int(ev["upcoming"].sum()), "canceled": int(ev["canceled"].sum()), "centers": int(len(centers)),
                   "measured": int(ev["measured"].sum()), "scored": int(ev["scored"].sum()), "zero_signups": int((ev["ran"] & (ev["regs"] == 0)).sum()),
                   "signups": int(ev.loc[ev["ran"], "regs"].sum()), "programs": int(len(P)), "ideas": int(len(state["ideas"])), "themes": int(len(state["theme_blob"]["themes"])),
                   "texts": int(state["T"].shape[0]), "with_flyer": int(ev["img"].notna().sum()), "reliable_centers": int(centers["reliable"].sum()),
                   "first_event": ev["date"].min().isoformat(), "by_year": {int(y): int(n) for y, n in Counter(d.year for d in ev.loc[ev["ran"], "date"]).items()}},
        "types": {"labels": TYPES, "events": {t: int(n) for t, n in Counter(ev.loc[~ev["canceled"], "type"]).most_common()},
                  "by_rules": int((ev["type_src"] == "rules").sum()), "by_neighbors": int((ev["type_src"] == "nearest").sum()), "unplaced": int((ev["type_src"] == "none").sum()), **state["type_info"]},
        "terms": state["term_info"], "score": state["score_info"], "search": {**search_info, **thr}, "predictor": pred, "sizes": sizes,
        "embedding_tests": json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_tests.json"))),
        "model": state["space"]["model"], "texts_embedded": {"names": state["space"]["names"], "blurbs": state["space"]["blurbs"]}, "llm_calls": 0,
    }
    dump(engine, f"{OUT}/engine.json")
    log("export:", {k: f"{v // 1024} KB" for k, v in sizes.items()}, "| vectors", q8.shape, "| search", search_info, "| thresholds", thr)
    log("predictor check:", pred)
    return engine


def push(names=None, prefix="public/"):
    """Upload the output folder to the Worker's bucket."""
    edge = os.environ.get("EDGE_URL", "https://campus-ideas.shneur.workers.dev")
    for name in sorted(os.listdir(OUT)):
        hdr = embed._headers()
        if names and name not in names:
            continue
        path = f"{OUT}/{name}"
        ct = "application/json; charset=utf-8" if name.endswith(".json") else "application/octet-stream"
        data = open(path, "rb").read()
        for attempt in range(4):
            try:
                req = urllib.request.Request(f"{edge}/analyst/put?key={prefix}{name}", data=data, method="PUT", headers={**hdr, "content-type": ct})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    json.load(resp)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(3 + 3 * attempt)
        log("pushed", name, len(data) // 1024, "KB")
