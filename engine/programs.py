"""Programs. An occurrence is one date; a program is everything one center has run under one name
("Sushi in the Sukkah" at Texas A&M, three years running). Programs are what gets embedded, clustered and searched."""
import numpy as np
import pandas as pd
from collections import Counter
from .common import log, sha, r
from . import embed, textprep

NAME_WEIGHT = 0.5      # share of the name in an event's vector when it also has a blurb (picked by test, see docs)


def attach_text(events, centers, names, blurbs):
    """One vector per distinct event text.

    The name and the blurb are cleaned (textprep) and embedded separately. Each set is centered on its own average,
    which removes what every Chabad event post has in common, and the two are then averaged. A post with no blurb
    is its name alone, so it sits in the same space as everything else instead of off to one side.
    Returns the events with `tix`, the matrix, and the space (the name average, which typed searches need).
    """
    own = {c: textprep.center_words(row) for c, row in centers.to_dict("index").items()}
    en = [textprep.name_text(n, own.get(c, ())) for n, c in zip(events["name"], events["center"])]
    eb = [textprep.blurb_text(d, s, n) for d, s, n in zip(events["desc"], events["short"], events["name"])]
    un = sorted(set(en)); ub = sorted(set(b for b in eb if b))
    N = embed.unit(names.get(un)); B = embed.unit(blurbs.get(ub)) if ub else np.zeros((0, N.shape[1]), np.float32)
    mu_n, mu_b = N.mean(0), (B.mean(0) if len(B) else np.zeros(N.shape[1], np.float32))
    Nc, Bc = embed.unit(N - mu_n), embed.unit(B - mu_b)
    npos = {t: i for i, t in enumerate(un)}; bpos = {t: i for i, t in enumerate(ub)}
    pairs = sorted(set(zip(en, eb))); ppos = {p: i for i, p in enumerate(pairs)}
    T = np.stack([Nc[npos[a]] if not b else NAME_WEIGHT * Nc[npos[a]] + (1 - NAME_WEIGHT) * Bc[bpos[b]] for a, b in pairs])
    events["tix"] = [ppos[p] for p in zip(en, eb)]
    events["ename"] = en
    log("texts: distinct names", len(un), "| distinct blurbs", len(ub), "| distinct posts", len(pairs))
    return events, embed.unit(T), pairs, {"mu_name": mu_n.astype(np.float32), "names": len(un), "blurbs": len(ub), "model": names.model}


def knn_types(events, T, k=15, min_sim=0.50):
    """Type the events the rules could not place, from the labelled texts nearest to them. Also returns how well
    the same vote reproduces the rules on a held-out sample of rule-typed texts."""
    by_text = events.groupby("tix")["type"].agg(lambda x: Counter(x).most_common(1)[0][0])
    lab = by_text[~by_text.isin(["other"])]
    unl = by_text[by_text == "other"]
    L = T[lab.index.values]; lt = lab.values
    def vote(Q, exclude=None):
        S = Q @ L.T
        if exclude is not None:
            S[np.arange(len(Q)), exclude] = -1
        top = np.argpartition(-S, k, axis=1)[:, :k]
        out = []
        for i in range(len(Q)):
            idx = top[i][np.argsort(-S[i, top[i]])]
            sims = S[i, idx]
            w = Counter()
            for j, s in zip(idx, sims):
                w[lt[j]] += max(s, 0) ** 4
            best, bw = w.most_common(1)[0]
            out.append((best, float(sims[0]), bw / sum(w.values())))
        return out
    # held-out check
    rng = np.random.default_rng(7); sample = rng.choice(len(L), size=min(2000, len(L)), replace=False)
    chk = vote(L[sample], exclude=sample)
    conf = [(p, s, sh) for p, s, sh in chk]
    ok = [p == lt[i] for (p, s, sh), i in zip(chk, sample) if s >= min_sim and sh >= 0.5]
    acc = float(np.mean(ok)) if ok else None
    cover = len(ok) / len(sample)
    res = vote(T[unl.index.values]) if len(unl) else []
    new = {tix: p for tix, (p, s, sh) in zip(unl.index.values, res) if s >= min_sim and sh >= 0.5}
    m = events["tix"].map(new)
    n_fix = int(m.notna().sum())
    events["type_src"] = np.where(m.notna(), "nearest", np.where(events["type"] == "other", "none", "rules"))
    events.loc[m.notna(), "type"] = m[m.notna()]
    log("types: nearest-neighbor placed", n_fix, "events; held-out agreement with the rules", round(acc, 3) if acc else None, "on", round(cover, 2), "of sample")
    return events, {"placed_events": n_fix, "heldout_agreement": r(acc, 3), "heldout_coverage": r(cover, 3), "still_other": int((events["type"] == "other").sum())}


def build(events, T):
    live = events[~events["canceled"]]
    rows, vecs = [], []
    ev_prog = {}
    for (c, nn), g in live.groupby(["center", "nname"], sort=True):
        ran = g[g["ran"]]
        sc = ran[ran["scored"]]
        meas = ran[ran["measured"]]
        names = Counter(g["name"]); name = names.most_common(1)[0][0]
        descs = Counter(d for d in g["desc"] if d)
        desc = max(descs, key=lambda d: (descs[d], len(d))) if descs else ""
        shorts = Counter(s for s in g["short"] if s)
        img = next((x for x in g.sort_values("dt", ascending=False)["img"] if isinstance(x, str)), None)
        tix = Counter(g["tix"])
        v = np.zeros(T.shape[1], np.float32)
        for t, n in tix.items():
            v += T[t] * np.sqrt(n)
        last = g.sort_values("dt").iloc[-1]
        pid = len(rows)
        for i in g["id"]:
            ev_prog[i] = pid
        themes = Counter(k for ks in g["tkeys"] for k in ks) if "tkeys" in g else Counter()
        rows.append({
            "pid": pid, "key": sha(f"{c}|{nn}")[:10], "center": int(c), "nname": nn, "name": name, "desc": desc, "short": shorts.most_common(1)[0][0] if shorts else "",
            "img": img, "type": Counter(g["type"]).most_common(1)[0][0], "hol": next(iter(Counter(h for h in g["hol"] if isinstance(h, str)).most_common(1)), (None,))[0],
            "aud": sorted({a for l in g["aud"] for a in l}), "collab": bool(g["collab"].any()), "offsite": float(g["offsite"].mean()) >= 0.5,
            "themes": [k for k, _ in themes.most_common(3)], "one_off": bool(g["one_off"].any()) if "one_off" in g else False,
            "n": int(len(ran)), "n_up": int(g["upcoming"].sum()), "n_meas": int(len(meas)), "n_scored": int(len(sc)),
            "first": ran["date"].min() if len(ran) else g["date"].min(), "last": ran["date"].max() if len(ran) else g["date"].max(),
            "years": sorted({d.year for d in ran["date"]}), "recurring": float(g["recurring"].mean()) >= 0.5,
            "med_regs": float(meas["regs"].median()) if len(meas) else None, "max_regs": int(meas["regs"].max()) if len(meas) else 0,
            "adj": float(np.exp(np.mean(np.clip(np.log(sc["adj"]), -1.5, 1.5)))) if len(sc) else None,
            # where its runs landed among all of this center's events (0 bottom, 1 top). What the neighbor system predicts.
            "pc": float(ran["cpct"].mean()) if ran["cpct"].notna().any() else None, "n_pc": int(ran["cpct"].notna().sum()),
            "pct": float(sc["pct"].mean()) if len(sc) and sc["pct"].notna().any() else None,
            "wd": int(Counter(g["wd"]).most_common(1)[0][0]), "hour": int(Counter(g["hour"]).most_common(1)[0][0]),
            "setting": int(last["setting"]), "dropped": bool(last["setting_state"] == "Canceled"),
        })
        vecs.append(v)
    P = pd.DataFrame(rows)
    V = embed.unit(np.stack(vecs))
    events["pid"] = events["id"].map(ev_prog)
    log("programs", len(P), "| with flyer", int(P["img"].notna().sum()), "| ran at least once", int((P["n"] > 0).sum()))
    return P, V, events
