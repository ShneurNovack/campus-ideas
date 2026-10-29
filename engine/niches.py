"""Niches and collabs: events aimed at one crowd, events run with another group, and events held away from
the Chabad house. Each is read the same way as everything else: against the same center's normal."""
import re
import numpy as np
from collections import Counter, defaultdict
from .common import r, log
from .types import EVENT_TYPES, TYPES

AUD_LABEL = {"grads": "Grad students", "women": "Women", "men": "Men", "freshmen": "Freshmen and new students", "seniors": "Seniors and graduation",
             "greek": "Greek life", "israelis": "Israelis", "parents": "Parents, families and alumni", "faculty": "Faculty and staff", "athletes": "Athletes", "community": "Wider community"}
PARTNERS = [("Hillel", r"\bhillel\b"), ("AEPi", r"\baepi\b|\bae ?pi\b|alpha epsilon pi"), ("AEPhi", r"\baephi\b|alpha epsilon phi"), ("ZBT", r"\bzbt\b|zeta beta tau"),
            ("SDT", r"\bsdt\b|sigma delta tau"), ("SAM (Sammy)", r"\bsammy\b|sigma alpha mu"), ("Students Supporting Israel", r"\bssi\b|students supporting israel"),
            ("Student government", r"student government|\bsga\b|\basb\b"), ("Jewish law students", r"\bjlsa\b|jewish law student"), ("Jewish med students", r"\bjmsa\b|jewish medical"),
            ("Mishelanu / Israeli students", r"mishelanu|israeli students"), ("Sinai Scholars", r"sinai scholars"), ("Birthright", r"birthright"), ("TAMID", r"\btamid\b"),
            ("Other Chabad houses", r"with chabad (at|of)|joint .* chabad"), ("Interfaith and multicultural", r"interfaith|multicultural|\bmsa\b|\bbsu\b")]
_P = [(n, re.compile(rx, re.I)) for n, rx in PARTNERS]


def _gm(x, prior=3.0):
    x = np.clip(np.log(np.asarray(x, float)), -1.5, 1.5)
    return float(np.exp(x.sum() / (len(x) + prior))) if len(x) else None


def _card(g, P, ideas_of, label, key):
    sc = g[g["scored"]]
    progs = P[P["pid"].isin(set(g["pid"]))]
    ic = Counter()
    for pid in set(g["pid"]):
        i = ideas_of.get(pid, -1)
        if i >= 0:
            ic[i] += 1
    top = progs[progs["n_scored"] >= 1].sort_values("adj", ascending=False).drop_duplicates("center").head(10)
    fl = progs[progs["img"].notna()].sort_values("last", ascending=False).drop_duplicates("center").head(12)
    return {"key": key, "label": label, "n": int(len(g)), "centers": int(g["center"].nunique()), "programs": int(len(progs)),
            "scored": int(len(sc)), "lift": r(_gm(sc["adj"].values), 3) if len(sc) >= 10 else None, "hit": r(float((sc["adj"] >= 1.2).mean()), 3) if len(sc) >= 10 else None,
            "med_regs": r(float(g.loc[g["measured"], "regs"].median()), 1) if g["measured"].any() else None,
            "types": [{"type": t, "n": int(n)} for t, n in Counter(g["type"]).most_common(5)],
            "ideas": [{"idea": int(i), "centers": int(n)} for i, n in ic.most_common(10)],
            "top": [int(p) for p in top["pid"]], "flyers": [int(p) for p in fl["pid"]],
            "by_year": [{"year": int(y), "n": int(len(x)), "centers": int(x["center"].nunique())} for y, x in g.groupby(g["date"].map(lambda d: d.year))]}


def build(events, P, today):
    ev = events[events["ran"] & events["type"].isin(EVENT_TYPES)]
    ideas_of = P.set_index("pid")["idea"].to_dict()
    aud = []
    for key, label in AUD_LABEL.items():
        g = ev[ev["aud"].map(lambda l, key=key: key in l)]
        if len(g) >= 15:
            aud.append(_card(g, P, ideas_of, label, key))
    aud.sort(key=lambda x: -x["centers"])
    col = ev[ev["collab"]]
    partners = []
    text = (ev["nname"] + " " + ev["desc"].str.lower().str.slice(0, 400))
    for name, rx in _P:
        m = text.map(lambda t, rx=rx: bool(rx.search(t)))
        g = ev[m]
        if len(g) >= 8:
            c = _card(g, P, ideas_of, name, name)
            partners.append(c)
    partners.sort(key=lambda x: -x["centers"])
    # paired: at centers that do both, how do collabs / off-site events do against the rest
    def paired(flag):
        ratios = []
        s = ev[ev["scored"]]
        for c, g in s.groupby("center"):
            a, b = g.loc[flag.loc[g.index], "adj"], g.loc[~flag.loc[g.index], "adj"]
            if len(a) >= 3 and len(b) >= 10:
                ratios.append(np.exp(np.mean(np.clip(np.log(a), -1.5, 1.5))) / np.exp(np.mean(np.clip(np.log(b), -1.5, 1.5))))
        return {"centers": len(ratios), "ratio": r(_gm(ratios, prior=2.0), 3) if len(ratios) >= 5 else None}
    off = ev[ev["offsite"]]
    places = Counter()
    for p in off["place"].dropna():
        for w in ("quad", "lawn", "park", "library", "student center", "union", "dorm", "hall", "beach", "plaza", "green", "campus", "sukkah", "restaurant", "bowling", "rooftop", "lounge", "zoom"):
            if w in p.lower():
                places[w] += 1
    blob = {"audiences": aud, "partners": partners,
            "collab": {**_card(col, P, ideas_of, "Events run with another group", "collab"), "paired": paired(ev["collab"])},
            "offsite": {**_card(off, P, ideas_of, "Events away from the Chabad house", "offsite"), "paired": paired(ev["offsite"]), "places": [{"place": k, "n": int(v)} for k, v in places.most_common(12)]},
            "one_time": {"paired": paired(~ev["recurring"]), "n": int((~ev["recurring"]).sum()), "recurring_n": int(ev["recurring"].sum())}}
    log("niches: audiences", len(aud), "partners", len(partners), "collab events", len(col), "off-site", len(off))
    return blob
