"""Ideas. An idea is the same program run by several centers: "Sushi in the Sukkah", "Pizza & Parsha",
"Paint Night". Found by clustering programs, type by type:

  similarity = 0.65 x embedding (centered on its type, so the shared "come to Chabad" wording drops out)
             + 0.35 x character n-gram TF-IDF of the name (keeps near-identical names together)
  average-linkage clustering at 0.50, then clusters whose centers of mass sit at 0.80 or more are merged
  (spelling variants: hamantash, hamentash, hamantaschen).

An idea needs at least two centers. Everything else stays a one-center program.
"""
import re
import datetime as dt
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from . import embed, textprep, calendars as cal
from .common import r, title_case, log
from .types import TYPES

SKIP_TYPES = {"resource", "togo"}
TAU, MERGE = 0.55, 0.85     # recalibrated for the name-and-blurb vectors: more programs land in an idea, at the same purity


def _shrunk(logs, prior=3.0):
    logs = np.clip(np.asarray(logs, float), -1.5, 1.5)
    return float(np.exp(logs.sum() / (len(logs) + prior))) if len(logs) else None


def _ay(d):
    return d.year if d.month >= 7 else d.year - 1


def cluster(P, V):
    Vc = embed.unit(V - V.mean(0))
    label = np.full(len(P), -1); nxt = 0
    for typ in sorted(set(P["type"])):
        if typ in SKIP_TYPES:
            continue
        idx = np.where((P["type"] == typ).values & (P["n"] + P["n_up"] > 0).values)[0]
        if len(idx) < 2:
            for i in idx:
                label[i] = nxt; nxt += 1
            continue
        if typ == "shabbat_dinner":
            # Shabbat dinners are grouped by their theme (themes.py), the plain weekly dinner being one idea
            m = {}
            for i in idx:
                th = P.at[i, "themes"]
                key = ("theme", th[0]) if th else (("one", i) if P.at[i, "one_off"] else ("plain",))
                if key not in m:
                    m[key] = nxt; nxt += 1
                label[i] = m[key]
            continue
        X = embed.unit(Vc[idx] - Vc[idx].mean(0))
        names = P["nname"].values[idx]
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1).fit_transform(names)
        S = 0.65 * (X @ X.T) + 0.35 * (tf @ tf.T).toarray()
        np.fill_diagonal(S, 1.0)
        lab = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1 - TAU).fit_predict(np.clip(1 - S, 0, 2))
        # merge clusters whose centers of mass are nearly the same
        groups = defaultdict(list)
        for k, l in enumerate(lab):
            groups[l].append(k)
        keys = sorted(groups)
        if len(keys) > 1:
            C = embed.unit(np.stack([X[groups[k]].mean(0) for k in keys]))
            sz = np.array([len(groups[k]) for k in keys])
            big = np.where(sz >= 2)[0]
            if len(big) > 1:
                lab2 = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1 - MERGE).fit_predict(np.clip(1 - C[big] @ C[big].T, 0, 2))
                remap = {keys[b]: f"m{l2}" for b, l2 in zip(big, lab2)}
                lab = np.array([remap.get(l, l) for l in lab], dtype=object)
        m = {}
        for i, l in zip(idx, lab):
            if l not in m:
                m[l] = nxt; nxt += 1
            label[i] = m[l]
    return _merge_same_name(P, label)


_FOLD = [(r"hamm?[ae]nt[ae]s(ch|h)?(en|in)?", "hamantash"), (r"\b(c?h)an+uk+a?h?\b", "chanukah"), (r"\bsuc?k+[ao]h?\b", "sukkah"), (r"\bsuc?k+o[ts]h?\b", "sukkot"),
         (r"hashanah?", "hashana"), (r"\bbaking\b", "bake"), (r"\b(pesach|passover)\b", "passover"), (r"\bshavuos\b", "shavuot"), (r"\bshabbos\b", "shabbat"),
         (r"\bbar[- ]?b[- ]?q(ue)?\b|\bbarbe[cq]ue\b", "bbq"), (r"\bcha[- ]?ba?[- ]?b?bq\b", "bbq"), (r"\b(19|20)\d\d\b|\b\d+(st|nd|rd|th)\b", " ")]
_DROP = {"the", "a", "an", "annual", "chabad", "chabad's", "at", "with", "our", "and", "for", "of", "to", "in", "on", "night", "event", "edition", "pm", "am"}


def _canon(nname):
    """The name with spelling variants, years and filler words folded away, so 'Hamentashen Baking 2024' and
    'Hamantash Bake' count as one name."""
    s = nname
    for pat, rep in _FOLD:
        s = re.sub(pat, rep, s)
    toks = [w.strip("'-+/") for w in re.split(r"[\s\-/+]+", s)]
    toks = [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in toks if w and w not in _DROP]
    return " ".join(toks)


def _merge_same_name(P, label):
    """Clusters are cut on wording, and the same event described two ways can land in two clusters. When two
    clusters of one type go by the same name (their most common name, folded), they are one idea."""
    by = defaultdict(Counter)
    for i in np.where(label >= 0)[0]:
        if P["type"].values[i] == "shabbat_dinner":
            continue
        by[label[i]][_canon(P["nname"].values[i])] += 1
    typ = {l: P["type"].values[np.where(label == l)[0][0]] for l in by}
    first = {}
    remap = {}
    for l in sorted(by, key=lambda l: -sum(by[l].values())):
        top, cnt = by[l].most_common(1)[0]
        if not top or cnt < 0.5 * sum(by[l].values()):
            continue
        key = (typ[l], top)
        if key in first:
            remap[l] = first[key]
        else:
            first[key] = l
    if remap:
        label = np.array([remap.get(l, l) for l in label])
    return label


def build(P, V, events, centers, today):
    P["cluster"] = cluster(P, V)
    Vc = embed.unit(V - V.mean(0))
    ev = events[~events["canceled"]]
    by_pid = {pid: g for pid, g in ev.groupby("pid")}
    cur_ay = _ay(today); span = (today - dt.date(cur_ay, 7, 1)).days
    ideas = []; P["idea"] = -1
    for cl, g in P[P["cluster"] >= 0].groupby("cluster"):
        if g["center"].nunique() < 2:
            continue
        iid = len(ideas)
        P.loc[g.index, "idea"] = iid
        occ = [by_pid[p] for p in g["pid"] if p in by_pid]
        E = __import__("pandas").concat(occ) if occ else None
        ran = E[E["ran"]]
        sc = ran[ran["scored"]]
        nc = Counter()
        for n_, c_ in zip(g["nname"], g["center"]):
            nc[n_] += 1
        top = nc.most_common(1)[0][0]
        disp = Counter(g.loc[g["nname"] == top, "name"]).most_common(1)[0][0]
        # the label is the most common name, without the dates, places and school names one center put in it
        src = g.loc[(g["nname"] == top) & (g["name"] == disp), "center"].iloc[0]
        tidy = textprep.name_text(disp, textprep.center_words(centers.loc[src].to_dict()) if src in centers.index else (), keep_numbers=True)
        disp = tidy if len(tidy) >= 3 else disp
        label = disp if (disp != disp.upper() and disp != disp.lower()) else title_case(disp.lower())
        years = defaultdict(set)
        for d, c in zip(ran["date"], ran["center"]):
            years[_ay(d)].add(c)
        ytd_now = len(years.get(cur_ay, set()))
        ytd_last = len({c for d, c in zip(ran["date"], ran["center"]) if _ay(d) == cur_ay - 1 and (d - dt.date(cur_ay - 1, 7, 1)).days <= span})
        full_last, full_prev = len(years.get(cur_ay - 1, set())), len(years.get(cur_ay - 2, set()))
        first = ran.sort_values("dt").iloc[0] if len(ran) else None
        hol = Counter(h for h in g["hol"] if isinstance(h, str))
        hol_key = hol.most_common(1)[0][0] if hol and hol.most_common(1)[0][1] >= 0.5 * len(g) else None
        jh = Counter(x for x in ran["jh"] if isinstance(x, str)); jr = Counter(x for x in ran["jh_runup"] if isinstance(x, str))
        n = max(1, len(ran))
        seasonal_j = hol_key or (jh and (jh.most_common(1)[0][1] + jr.get(jh.most_common(1)[0][0], 0)) / n >= 0.6 and jh.most_common(1)[0][0]) or None
        status = "steady"
        ncen = g["center"].nunique()
        if seasonal_j:
            status = "seasonal"
        elif first is not None and (today - first["date"]).days <= 550 and ncen >= 3:
            status = "new"
        elif (ytd_now >= ytd_last + 2 and ytd_now >= 1.3 * max(1, ytd_last)) or (full_last >= full_prev + 3 and full_last >= 1.3 * max(1, full_prev) and ytd_now >= ytd_last):
            status = "rising"
        elif (full_last <= full_prev - 3 and full_last <= 0.7 * full_prev) or (ytd_last >= 4 and ytd_now <= 0.5 * ytd_last):
            status = "fading"
        elif len([y for y, v in years.items() if len(v) >= 2]) >= 3 and ncen >= 8:
            status = "evergreen"
        months = [0] * 12
        for d in ran["date"]:
            months[d.month - 1] += 1
        wd = Counter(ran["wd"]); hr = Counter(ran["hour"])
        tw = ran[ran["season"].isin(["fall", "spring"])]
        sec = Counter(x for x in ran["sec"] if isinstance(x, str))
        when = "Any time of year"
        top_m = max(range(12), key=lambda i: months[i])
        conc3 = max(sum(months[(i + j) % 12] for j in range(2)) for i in range(12)) / n
        if seasonal_j:
            when = f"Around {cal.JEWISH_LABEL[seasonal_j]}"
        elif sec and sec.most_common(1)[0][1] / n >= 0.5:
            when = f"Around {cal.SECULAR_LABEL[sec.most_common(1)[0][0]]}"
        elif len(tw) and (tw["tweek"] <= 2).mean() >= 0.6:
            when = "First two weeks of the semester"
        elif len(tw) and (tw["tpos"] >= 0.85).mean() >= 0.6:
            when = "Last weeks of the semester"
        elif conc3 >= 0.65 and n >= 4:
            i0 = max(range(12), key=lambda i: sum(months[(i + j) % 12] for j in range(2)))
            mn = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            when = f"{mn[i0]} and {mn[(i0 + 1) % 12]}"
        by_c = sc.groupby("center")["adj"].agg(lambda x: float(np.mean(np.log(x)))) if len(sc) else []
        fl = g.assign(a=g["adj"].fillna(0.9)); fl = fl[fl["img"].notna()].sort_values(["a", "last"], ascending=False)
        flyers = [int(p) for p in fl.drop_duplicates("center")["pid"].head(12)]
        cen = embed.unit(Vc[g["pid"].values].mean(0))
        rp = g.assign(sim=Vc[g["pid"].values] @ cen); rp = rp[rp["desc"].str.len().between(60, 600)].sort_values("sim", ascending=False)
        rep = [int(p) for p in rp.drop_duplicates("center")["pid"].head(5)]
        aud = Counter(a for l in g["aud"] for a in l)
        ideas.append({
            "id": iid, "label": label, "type": Counter(g["type"]).most_common(1)[0][0], "hol": hol_key,
            "programs": [int(p) for p in g.sort_values("n", ascending=False)["pid"]],
            "centers": int(ncen), "n": int(len(ran)), "n_up": int(E["upcoming"].sum()),
            "n_scored": int(len(sc)), "centers_scored": int(sc["center"].nunique()) if len(sc) else 0,
            "lift": r(_shrunk(np.log(sc["adj"])), 3) if len(sc) else None, "hit": r(float((sc["adj"] >= 1.2).mean()), 3) if len(sc) else None,
            "centers_up": int(sum(1 for v in by_c.values if v > 0)) if len(sc) else 0,
            "confidence": "strong" if len(sc) >= 12 and sc["center"].nunique() >= 6 else ("some" if len(sc) >= 5 and sc["center"].nunique() >= 3 else "thin"),
            "med_regs": r(float(ran.loc[ran["measured"], "regs"].median()), 1) if ran["measured"].any() else None,
            "status": status, "when": when, "months": months,
            "wd": [int(wd.get(i, 0)) for i in range(7)], "hour": int(hr.most_common(1)[0][0]) if hr else None,
            "years": [{"ay": f"{y}-{str(y + 1)[2:]}", "centers": len(years[y])} for y in sorted(years) if y >= 2020],
            "ytd_now": ytd_now, "ytd_last": ytd_last,
            "first": {"date": first["date"].isoformat(), "center": int(first["center"])} if first is not None else None,
            "names": [{"name": title_case(n_), "n": c_} for n_, c_ in nc.most_common(6)],
            "aud": [a for a, c_ in aud.most_common(2) if c_ >= 0.4 * len(g)], "recurring": bool(g["recurring"].mean() >= 0.5),
            "staple": bool(len(ran) >= 12 * ncen),
            "flyers": flyers, "rep": rep,
            "_cen": cen,
        })
    log("ideas", len(ideas), "| programs inside an idea", int((P["idea"] >= 0).sum()), "of", len(P))
    # related ideas and a 2-D map
    if ideas:
        C = np.stack([i["_cen"] for i in ideas]); S = C @ C.T
        for k, i in enumerate(ideas):
            o = [j for j in np.argsort(-S[k])[:8] if j != k and S[k, j] >= 0.45][:5]
            i["related"] = [int(j) for j in o]
        try:
            import umap
            xy = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=7).fit_transform(C)
            xy = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
            for i, p in zip(ideas, xy):
                i["xy"] = [r(p[0], 4), r(p[1], 4)]
        except Exception as e:  # the map is a picture only; never fail the run for it
            log("map skipped:", e)
        for i in ideas:
            del i["_cen"]
    return ideas, P
