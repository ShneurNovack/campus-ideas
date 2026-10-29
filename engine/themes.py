"""Shabbat dinner themes.

A theme is what a center adds to "Shabbat dinner" in the event's name: Italian, Pink, Welcome Back, Under the
Stars, Shabbat 1000. Found without a language model:

 1. residual    strip the words every Shabbat dinner name shares (and the center's own name) and keep what is left
 2. house brand a phrase one center puts on most of its dinners ("Friday Night Live") is its name for the regular
                dinner, not a theme
 3. merge       a phrase is a theme when at least three centers use it on its own. Longer names are split into the
                themes they contain ("Welcome Back Sushi Shabbat" is both). Phrases with no known word join the
                nearest theme by embedding, or stay one-offs
 4. measure     each themed dinner is compared with the same center's plain dinners in the six weeks around it
"""
import re
import datetime as dt
import numpy as np
from collections import Counter, defaultdict
from . import calendars as cal
from .common import r, title_case, IMG_PREFIX, log

GENERIC = set("""shabbat shabbos shabbas shabat shabbes shabos chabbat chabad dinner dinners friday fridays night nights fri kabbalat kabolat kabalat
kab shab kab-shab services service and at the with in on of a an for your our to this week weekly meal meals evening eve fnl oneg student students
jewish community center centre house rohr us join come celebrate edition special event rsvp required free open all are everyone time new note
pm am p m no max annual night's is its it's by hosted feat featuring ft presents x le la du de au des et étudiants etudiants soirée soiree repas
fall spring summer winter semester w regular traditional classic services+dinner dinner+ experience tgis candle lighting followed copy
st nd rd th we you i my style themed theme three four course courses delicious home homemade cooked authentic amazing famous epic best ever
pre post taste food menu favorite favorites party celebration festive holiday kosher guest cuisine buffet second end weekend bowl
only walk-in walk rsvp'd reservations reservation registration register sign up signup needed here please bring friend friends invite
get ready dont don't miss out be there see lets let's go more info details tba tbd davening minyan prayer prayers maariv mincha
immediately after before then plus also or but not just yet still again really very so too every each one once twice last next final""".split())
LITTLE = {"the", "in", "of", "under", "and", "for", "to", "on", "at", "with", "your", "a"}
SYN = {"sukka": "sukkah", "succah": "sukkah", "suka": "sukkah", "sukkot": "sukkah", "sukkos": "sukkah", "succot": "sukkah", "israeli": "israel", "isreali": "israel",
       "italiano": "italian", "italy": "italian", "freshmen": "freshman", "frosh": "freshman", "seniors": "senior", "hannukah": "chanukah", "hanukkah": "chanukah",
       "hanukah": "chanukah", "pesach": "passover", "pessach": "passover", "tacos": "taco", "barbecue": "bbq", "barbeque": "bbq", "girls": "women", "ladies": "women",
       "women's": "women", "guys": "men", "mens": "men", "men's": "men", "welcome": "welcomeback", "back": "welcomeback", "persia": "persian", "mexico": "mexican",
       "fiesta": "mexican", "tex-mex": "mexican", "mexicana": "mexican", "shnitzel": "schnitzel", "sendoff": "send-off", "commencement": "graduation",
       "parents": "familyweekend", "parent's": "familyweekend", "mom's": "familyweekend", "moms": "familyweekend", "dad's": "familyweekend", "dads": "familyweekend",
       "hoco": "homecoming", "asia": "asian", "megashabbat": "mega", "supershabbos": "mega", "outdoors": "outdoor", "outside": "outdoor", "desserts": "dessert",
       "shabbat1000": "bignumber", "1k": "bignumber", "2k": "bignumber"}
for _n in ("100", "180", "200", "250", "300", "360", "400", "500", "600", "613", "700", "750", "800", "1000", "1500", "2000"):
    SYN[_n] = "bignumber"
PHRASES = [(re.compile(p), s) for p, s in [
    (r"\bmove[\s\-]?in\b", "move-in"), (r"\bsuper[\s\-]?bowl\b", "superbowl"), (r"\bwelcome[\s\-]back\b|\bback to school\b|\bback to campus\b", "welcomeback"),
    (r"\bunder the star(t)?s?\b", "understars"), (r"\bto[\s\-]go\b", "togo"), (r"\bsend[\s\-]?off\b", "send-off"), (r"\bfirst[\s\-]years?\b", "freshman"),
    (r"\bfamily weekend\b|\bparents?'?s? weekend\b|\bfamily and friends weekend\b|\bmom'?s weekend\b|\bdad'?s weekend\b", "familyweekend"),
    (r"\bblack[\s\-]tie\b", "blacktie"), (r"\baround the world\b", "aroundtheworld"), (r"\bpoke bowls?\b", "poke"), (r"\btu b'?i?[\s\-]?shvat\b", "tubshvat"),
    (r"\brosh hashanah?\b", "roshhashana"), (r"\blag b'?a?'?omer\b", "lagbaomer"), (r"\bwild west\b", "wildwest"), (r"\bmental health\b", "mentalhealth"),
    (r"\bspeed dating\b", "speeddating"), (r"\bhappy hour\b", "happyhour"), (r"\bred carpet\b", "redcarpet"), (r"\bforget your gpa\b", "forgetgpa"),
    (r"\bchallah back\b", "challahback"), (r"\bbring a friend\b", "bringafriend"), (r"\bsinai scholars?\b", "sinaischolars"), (r"\bgrad(uate)? students?\b|\bj?grads\b", "grad"),
    (r"\bba[rt] mitzvah\b", "barmitzvah"), (r"\bshe?mini atzere[ts]\b", "sheminiatzeret"), (r"\bdessert bar\b", "dessertbar"), (r"\bspring break\b", "springbreak"),
    (r"\bpre[\s\-]", " "), (r"\bpost[\s\-]", " "), (r"\bwine tasting\b", "winetasting"), (r"\bcomfort food\b", "comfortfood"), (r"\bfried chicken\b", "friedchicken"),
]]
PRETTY = {"understars": "Under the Stars", "superbowl": "Super Bowl", "welcomeback": "Welcome Back", "move-in": "Move-In", "togo": "To-Go", "familyweekend": "Family Weekend",
          "blacktie": "Black Tie", "aroundtheworld": "Around the World", "tubshvat": "Tu B'Shvat", "roshhashana": "Rosh Hashana", "lagbaomer": "Lag BaOmer", "wildwest": "Wild West",
          "mentalhealth": "Mental Health", "speeddating": "Speed Dating", "happyhour": "Happy Hour", "redcarpet": "Red Carpet", "forgetgpa": "Forget Your GPA", "challahback": "Challah Back",
          "bringafriend": "Bring a Friend", "sinaischolars": "Sinai Scholars", "send-off": "Send-Off", "aepi": "AEPi", "zbt": "ZBT", "bbq": "BBQ", "grad": "Grad Students",
          "bignumber": "Shabbat 500 / 1000", "sukkah": "In the Sukkah", "barmitzvah": "Bar Mitzvah", "sheminiatzeret": "Shemini Atzeret", "dessertbar": "Dessert Bar",
          "springbreak": "Spring Break", "winetasting": "Wine Tasting", "comfortfood": "Comfort Food", "friedchicken": "Fried Chicken", "mega": "Mega Shabbat"}
NOISE = {"b", "day", "year", "test", "person", "dinner2", "super", "york", "dinner1", "copy", "tba", "will", "early", "first", "live", "social", "club", "private",
         "express", "rabbi", "torah", "kiddush", "light", "hot", "dean", "shift", "break", "together", "toast", "cake", "l'chaim", "lchaim", "speaker", "feast", "candy"}
_TIME = re.compile(r"\b\d{1,2}([: ]\d\d)\s?(pm|am|p\.m\.|a\.m\.)?\b|\b\d{1,2}\s?(pm|am)\b|\b\d{1,2}/\d{1,2}(/\d{2,4})?\b|\b(19|20)\d\d\b|\b57\d\d\b|\b\d{1,2}(st|nd|rd|th)\b")
_MONTH = re.compile(r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|june?|july?|aug(ust)?|sept?(ember)?|oct(ober)?|nov(ember)?|dec(ember)?)\b\.? ?\d{0,2}")

FAMILIES = [
    ("start", "Start of the year", r"welcomeback|move-in|freshman|orientation|kickoff|kick-off|opening"),
    ("end", "End of the year", r"graduation|senior|farewell|send-off|finals|goodbye|closing"),
    ("campus", "Campus calendar", r"familyweekend|^family$|homecoming|alumni|siblings|game ?day|rush|formal|blacktie|gala|birthday|anniversary|barmitzvah|springbreak|tailgate"),
    ("jewish", "Jewish calendar", r"passover|sukkah|chanukah|purim|roshhashana|shavuot|tubshvat|lagbaomer|simchat|sheminiatzeret|shofar"),
    ("secular", "Secular calendar", r"superbowl|thanksgiving|friendsgiving|halloween|valentine|patrick|wonderland|election"),
    ("cause", "Cause and solidarity", r"israel|unity|pink|mentalhealth|solidarity|hostage|bibas|ukraine|memorial|tribute|awareness|orange|yellow|resilience|jewnity|pride|remembrance|love|gratitude|idf|ssi|wellness"),
    ("scale", "Big ones", r"mega|bignumber|grand"),
    ("audience", "For one crowd", r"^grad$|women|^men$|greek|aepi|zbt|sdt|aephi|sammy|^law$|medical|sinaischolars|international|faculty|professor|president|kids|hillel"),
    ("food", "Cuisine and menu", r"italian|sushi|mexican|taco|chinese|asian|thai|persian|moroccan|sephardic|brisket|bbq|schnitzel|shawarma|poke|steak|chocolate|dessert|pasta|pizza|deli|falafel|french|hawaiian|latin|russian|indian|japanese|mediterranean|kugel|cholent|wings|burger|dairy|charcuterie|wine|comfort|southern|cajun|texas|fried|ramen|hibachi|gourmet|soup|challah|breakfast|chicken|potluck|shuk|cocktail"),
    ("vibe", "Setting and vibe", r"understars|outdoor|picnic|rooftop|quad|lawn|beach|tent|garden|togo|redcarpet|wildwest|aroundtheworld|white|chill|happyhour|bringafriend|challahback|forgetgpa|game|dancing|speeddating"),
]
_FAM = [(k, l, re.compile(rx)) for k, l, rx in FAMILIES]
SPECIAL = {"understars": "Shabbat Under the Stars", "sukkah": "Shabbat in the Sukkah", "aroundtheworld": "Shabbat Around the World", "bignumber": "Shabbat 500 / 1000",
           "mega": "Mega Shabbat", "president": "Shabbat with the President", "chill": "Shabbat and Chill", "happyhour": "Shabbat Happy Hour", "women": "Women's Shabbat",
           "men": "Men's Shabbat", "poke": "Poke Bowl Shabbat", "togo": "Shabbat To-Go", "outdoor": "Outdoor Shabbat", "kids": "Kids' Shabbat", "taco": "Taco Shabbat",
           "family": "Family Shabbat", "israel": "Israel Shabbat", "law": "Law Students' Shabbat", "medical": "Med Students' Shabbat", "greek": "Greek Life Shabbat",
           "hillel": "Shabbat with Hillel", "latin": "Latin Shabbat", "dancing": "Dancing Shabbat", "game": "Game Night Shabbat"}
FAMILY_LABEL = {k: l for k, l, _ in FAMILIES} | {"other": "Other"}


def family_of(key):
    for k, _l, rx in _FAM:
        if rx.search(key):
            return k
    return "other"


def center_words(row):
    words = set()
    for f in ("name", "campus", "nick", "city"):
        v = row.get(f)
        if isinstance(v, str):
            words |= {w for w in re.findall(r"[\w']+", v.lower()) if len(w) >= 2}
    words -= {"shabbat", "jewish", "student", "students", "center", "the", "of", "at", "and", "for", "house", "chabad", "rohr", "state", "life", "new", "family", "israel", "grad"}
    return words | {"university", "college", "campus", "uni"}


def residual(nname, cwords=()):
    """What is left of a Shabbat dinner name once the shared words are gone. Returns (key, display)."""
    s = nname
    for rx, sub in PHRASES:
        s = rx.sub(sub, s)
    s = _MONTH.sub(" ", s)
    s = _TIME.sub(" ", s)
    toks = re.findall(r"[^\W\d_][\w'’\-]*|\d+k?", s, flags=re.UNICODE)
    out, disp = [], []
    for t in toks:
        t2 = t.strip("-'’")
        if not t2:
            continue
        if t2.isdigit() or re.fullmatch(r"\dk", t2):
            if t2 in SYN:
                out.append(SYN[t2]); disp.append(t2)
            continue
        if t2 in cwords:
            continue
        if t2 in GENERIC:
            if disp and t2 in LITTLE:
                disp.append(t2)
            continue
        k = SYN.get(t2, t2)
        if k not in out:
            out.append(k)
        disp.append(t2)
    while disp and disp[-1] in LITTLE:
        disp.pop()
    return " ".join(out), " ".join(disp)


def assign(events, centers, cache):
    """Tag every Shabbat dinner with its theme keys."""
    from . import embed
    sd = events[(events["type"] == "shabbat_dinner") & (~events["canceled"])].copy()
    cw = {cid: center_words(row) for cid, row in centers.to_dict("index").items()}
    kd = [residual(n, cw.get(cid, ())) for n, cid in zip(sd["nname"], sd["center"])]
    sd["tkey"] = [k for k, _ in kd]; sd["tdisp"] = [d for _, d in kd]
    tot = sd.groupby("center").size()
    hb = sd[sd["tkey"] != ""].groupby(["center", "tkey"]).size().reset_index(name="n")
    hb["share"] = hb["n"] / hb["center"].map(tot)
    brands = {(c, k) for c, k, n, sh in zip(hb["center"], hb["tkey"], hb["n"], hb["share"]) if n >= 6 and (sh >= 0.25 or n >= 15)}
    sd["brand"] = [(c, k) in brands for c, k in zip(sd["center"], sd["tkey"])]
    live = sd[(sd["tkey"] != "") & ~sd["brand"]]
    g = live.groupby("tkey").agg(n=("id", "size"), centers=("center", "nunique"))
    base = {k for k in g.index if " " not in k and g.loc[k, "centers"] >= 3 and k not in NOISE and len(k) > 2}
    multi = {k for k in g.index if " " in k and g.loc[k, "centers"] >= 3 and not any(t in base for t in k.split())}

    def split(k):
        if k in multi:
            return [k]
        out = []
        for t in k.split():
            if t in base and t not in out:
                out.append(t)
        return out[:3]

    assigned = {k: split(k) for k in g.index}
    atoms = sorted(base | multi)
    loose = [k for k, v in assigned.items() if not v and k not in NOISE and not all(t in NOISE for t in k.split())]
    if atoms and loose:
        disp_of = {k: PRETTY.get(k, k) for k in atoms}
        for k in loose:
            disp_of[k] = Counter(live.loc[live["tkey"] == k, "tdisp"]).most_common(1)[0][0] or k
        pv = embed.unit(cache.get([disp_of[k] for k in atoms + loose]))
        pv = embed.unit(pv - pv.mean(0))
        sim = pv[len(atoms):] @ pv[:len(atoms)].T
        for i, k in enumerate(loose):
            j = int(np.argmax(sim[i]))
            if sim[i, j] >= 0.72:
                assigned[k] = [atoms[j]]
    sd["tkeys"] = [([] if b else assigned.get(k, [])) for k, b in zip(sd["tkey"], sd["brand"])]
    sd["one_off"] = [(not b) and bool(k) and not assigned.get(k) and k not in NOISE and not all(t in NOISE for t in k.split()) for k, b in zip(sd["tkey"], sd["brand"])]
    sd["plain"] = [(not t) and (not o) for t, o in zip(sd["tkeys"], sd["one_off"])]
    return sd, sorted(brands)


def _shrunk(logs, prior=3.0):
    logs = np.clip(np.asarray(logs, float), -1.5, 1.5)
    return float(np.exp(logs.sum() / (len(logs) + prior))) if len(logs) else None


def _ay(d):
    return d.year if d.month >= 7 else d.year - 1


def _ay_label(y):
    return f"{y}-{str(y + 1)[2:]}"


_STOP = set("""the and for you your with our are this that from will have all join come out not but can get who what when where how its it's has was were
be been being more most very just also than then them they their there here about into over under after before every each any some such only own same
plus back behind love make sure fellow meet other know these off always take fun way she her his him big turning around place edition courses campus tribe
bring bringing friends friend everyone welcome welcomes rsvp required sign please free food dinner shabbat shabbos chabad night friday evening student students
jewish great good best amazing delicious special week weekly enjoy enjoying experience celebrate celebrating us we i me my a an of in on at to is as by or if so
one two three four course meal services service followed start starts begin begins pm am time see there's we'll you'll let's lets don't dont miss new""".split())


def _terms(text):
    toks = [t for t in re.findall(r"[a-z][a-z'\-]{2,}", text.lower()) if t not in _STOP]
    return set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}


def build(events, centers, cache, today, T):
    """T: unit text vectors, one row per distinct event text; events["tix"] points into it."""
    from . import embed
    sd, brands = assign(events, centers, cache)
    log("themes: dinners", len(sd), "plain", int(sd["plain"].sum()), "themed", int((sd["tkeys"].map(len) > 0).sum()), "one-off", int(sd["one_off"].sum()), "house-brand", int(sd["brand"].sum()))
    cname = centers["name"].to_dict()
    ordv = sd["dt"].values.astype("datetime64[D]").astype(int)
    sd["ord"] = ordv
    # ---- local baseline: the same center's plain dinners within 45 days
    lift = np.full(len(sd), np.nan); src = np.array([""] * len(sd), dtype=object)
    pos = {i: k for k, i in enumerate(sd.index)}
    for c, g in sd.groupby("center"):
        pl = g[g["plain"] & g["measured"]]
        pd_, pr = pl["ord"].values, pl["regs"].values.astype(float)
        o = np.argsort(pd_); pd_, pr = pd_[o], pr[o]
        for i, d, regs, meas, adj, scored in zip(g.index, g["ord"], g["regs"], g["measured"], g["adj"], g["scored"]):
            if not meas:
                continue
            lo, hi = np.searchsorted(pd_, d - 45), np.searchsorted(pd_, d + 45, "right")
            w = pr[lo:hi][pd_[lo:hi] != d]
            if len(w) >= 3:
                lift[pos[i]] = (regs + 2.0) / (np.median(w) + 2.0); src[pos[i]] = "local"
            elif scored and adj == adj:
                lift[pos[i]] = adj; src[pos[i]] = "year"
    sd["tlift"], sd["tsrc"] = lift, src

    keys = Counter(); kc = defaultdict(set)
    for ks, c, ran in zip(sd["tkeys"], sd["center"], sd["ran"]):
        for k in ks:
            keys[k] += 1; kc[k].add(c)
    all_keys = sorted(keys, key=lambda k: (-len(kc[k]), -keys[k]))

    # text vectors, centered on the average Shabbat dinner, for vibe and for "related"
    ids = list(sd["id"])
    tv = T[sd["tix"].values]; mean_sd = tv.mean(0)
    tvc = embed.unit(tv - mean_sd)
    row_of = {i: k for k, i in enumerate(ids)}

    # document frequencies over distinct descriptions of all Shabbat dinners
    docs = sd.drop_duplicates(["center", "desc"])
    df = Counter(); ndocs = 0
    for d in docs["desc"]:
        if len(d) >= 30:
            ndocs += 1
            for t in _terms(d):
                df[t] += 1

    all_doy = sd.loc[sd["ran"], "doy"].values; all_hd = sd.loc[sd["ran"], "hdoy_n"].values
    n_all = max(1, int(sd["ran"].sum()))
    active_by_ay = sd[sd["ran"]].groupby(sd.loc[sd["ran"], "date"].map(_ay))["center"].agg(lambda x: set(x)).to_dict()
    cur_ay = _ay(today); ay_start = dt.date(cur_ay, 7, 1); span = (today - ay_start).days

    themes = []
    cent = {}
    for k in all_keys:
        rows = sd[sd["tkeys"].map(lambda ks, k=k: k in ks)]
        ran = rows[rows["ran"]]
        if ran["center"].nunique() < 2 and len(ran) < 3:
            continue
        pure = rows[rows["tkeys"].map(len) == 1]
        label = SPECIAL.get(k) or f"{PRETTY.get(k) or title_case(k)} Shabbat"
        fam_k = family_of(k)
        # which calendar the theme follows: the one its past dates bunch up on more tightly
        def bunch(vals, period):
            a = 2 * np.pi * np.asarray(vals, float) / period
            return float(np.hypot(np.cos(a).mean(), np.sin(a).mean())) if len(vals) else 0.0
        b_civ, b_heb = bunch(ran["doy"].values, 365.25), bunch(ran["hdoy_n"].values, 354.4)
        follows = "jewish" if (fam_k == "jewish" or b_heb > b_civ + 0.06) and fam_k not in ("secular", "start", "end") else "civil"
        m = ran[ran["tlift"].notna()]
        logs = np.log(m["tlift"].values) if len(m) else np.array([])
        by_center = m.groupby("center")["tlift"].agg(lambda x: float(np.mean(np.log(x)))) if len(m) else {}
        # adoption by academic year
        years = defaultdict(lambda: [0, set()])
        for d, c in zip(ran["date"], ran["center"]):
            y = _ay(d); years[y][0] += 1; years[y][1].add(c)
        ytd_now = {c for d, c in zip(ran["date"], ran["center"]) if _ay(d) == cur_ay}
        ytd_last = {c for d, c in zip(ran["date"], ran["center"]) if _ay(d) == cur_ay - 1 and (d - dt.date(cur_ay - 1, 7, 1)).days <= span}
        full_last, full_prev = len(years[cur_ay - 1][1]), len(years[cur_ay - 2][1])
        # repeat rate: of centers that ran it in a year and stayed active, how many ran it again the next year
        rep_n = rep_d = 0
        for y in list(years):
            if y >= cur_ay:
                continue
            for c in years[y][1]:
                if c in active_by_ay.get(y + 1, set()) and (y + 1 < cur_ay or True):
                    if y + 1 == cur_ay and span < 200:
                        continue
                    rep_d += 1; rep_n += c in years[y + 1][1]
        first = ran.sort_values("dt").iloc[0] if len(ran) else None
        nyears = sum(1 for y, v in years.items() if len(v[1]) >= 2)
        # status
        status = "steady"
        first_days = (today - first["date"]).days if first is not None else 0
        if fam_k == "jewish":
            status = "seasonal"   # its count in a given year depends on where the holiday falls, so no trend is read
        elif first is not None and first_days <= 550 and ran["center"].nunique() >= 3:
            status = "new"
        elif (len(ytd_now) >= len(ytd_last) + 2 and len(ytd_now) >= 1.3 * max(1, len(ytd_last))) or (full_last >= full_prev + 3 and full_last >= 1.3 * max(1, full_prev) and len(ytd_now) >= len(ytd_last)):
            status = "rising"
        elif (full_last <= full_prev - 3 and full_last <= 0.7 * full_prev) or (len(ytd_last) >= 4 and len(ytd_now) <= 0.5 * len(ytd_last)):
            status = "fading"
        elif nyears >= 3 and ran["center"].nunique() >= 10:
            status = "evergreen"
        # when it is run
        doy = ran["doy"].values
        wk = Counter(int(min(52, (d - 1) // 7 + 1)) for d in doy)
        best_w, best_n = 1, 0
        for w0 in range(1, 53):
            n6 = sum(wk.get((w0 + j - 1) % 52 + 1, 0) for j in range(6))
            if n6 > best_n:
                best_w, best_n = w0, n6
        conc = best_n / max(1, len(ran))
        jh = Counter(x for x in ran["jh"] if isinstance(x, str)); jr = Counter(x for x in ran["jh_runup"] if isinstance(x, str))
        sec = Counter(x for x in ran["sec"] if isinstance(x, str))
        tw = ran[ran["season"].isin(["fall", "spring"])]
        early = float((tw["tweek"] <= 2).mean()) if len(tw) else 0.0
        late = float((tw["tpos"] >= 0.85).mean()) if len(tw) else 0.0
        when = None
        top_jh = jh.most_common(1)[0] if jh else (None, 0); top_sec = sec.most_common(1)[0] if sec else (None, 0)
        n = max(1, len(ran))
        if top_jh[1] / n >= 0.5:
            when = f"During {cal.JEWISH_LABEL[top_jh[0]]}"
        elif top_sec[1] / n >= 0.5:
            when = f"Around {cal.SECULAR_LABEL[top_sec[0]]}"
        elif early >= 0.6:
            when = "First two weeks of the semester"
        elif late >= 0.6:
            when = "Last weeks of the semester"
        elif conc >= 0.6:
            a = dt.date(2025, 1, 1) + dt.timedelta(days=(best_w - 1) * 7); b = a + dt.timedelta(days=41)
            when = f"{a.strftime('%b %-d')} to {b.strftime('%b %-d')}"
        elif jh and (top_jh[1] + jr.get(top_jh[0], 0)) / n >= 0.5:
            when = f"Around {cal.JEWISH_LABEL[top_jh[0]]}"
        else:
            when = "Any time of year"
        if when.startswith(("During", "Around")) and top_jh[0] and (top_jh[1] + jr.get(top_jh[0], 0)) / n >= 0.5:
            follows = "jewish"
        months = [0] * 12
        for d in ran["date"]:
            months[d.month - 1] += 1
        # vibe
        tdocs = rows.drop_duplicates(["center", "desc"])
        tf = Counter(); tcen = defaultdict(set); nd = 0
        for d, c in zip(tdocs["desc"], tdocs["center"]):
            if len(d) >= 30:
                nd += 1
                for t in _terms(d):
                    tf[t] += 1; tcen[t].add(c)
        key_words = set(re.findall(r"[a-z]+", (PRETTY.get(k, k) + " " + label).lower()))
        need = max(2, int(np.ceil(0.08 * nd)))
        scored = sorted((((cnt / nd) / ((1 + df[t]) / ndocs) * np.sqrt(cnt), t) for t, cnt in tf.items()
                         if cnt >= need and len(tcen[t]) >= 2 and (cnt / nd) / ((1 + df[t]) / ndocs) >= 2.5 and not (set(t.split()) & key_words)), reverse=True) if nd else []
        terms, used = [], set()
        for _s, t in scored:
            ws = set(t.split())
            if ws & used and " " not in t:
                continue
            terms.append(t); used |= ws
            if len(terms) >= 10:
                break
        idx = [row_of[i] for i in rows["id"]]
        centroid = embed.unit(tvc[idx].mean(0)); cent[k] = centroid
        sims = tvc[idx] @ centroid
        cand = rows.assign(sim=sims).drop_duplicates(["center", "desc"])
        cand = cand[cand["desc"].str.len().between(70, 520)].sort_values("sim", ascending=False)
        descs, seen_c = [], set()
        for _, x in cand.iterrows():
            if x["center"] in seen_c:
                continue
            seen_c.add(x["center"])
            descs.append({"text": x["desc"], "name": x["name"], "center": int(x["center"]), "date": x["date"].isoformat(), "regs": int(x["regs"])})
            if len(descs) >= 5:
                break
        fl = rows.assign(l=rows["tlift"].fillna(0.9)); fl = fl[fl["img"].notna()].sort_values(["l", "dt"], ascending=False)
        flyers, seen_c, seen_i = [], Counter(), set()
        for _, x in fl.iterrows():
            if seen_c[x["center"]] >= 2 or x["img"] in seen_i:
                continue
            seen_c[x["center"]] += 1; seen_i.add(x["img"])
            flyers.append({"img": x["img"], "name": x["name"], "center": int(x["center"]), "date": x["date"].isoformat(), "regs": int(x["regs"]), "lift": r(x["tlift"], 2), "id": int(x["id"])})
            if len(flyers) >= 14:
                break
        occ = [[x["date"].isoformat(), int(x["center"]), int(x["regs"]), r(x["tlift"], 2), int(x["id"]), 1 if x["upcoming"] else 0] for _, x in rows.sort_values("dt").iterrows()]
        first_run = m.sort_values("dt").assign(nth=m.sort_values("dt").groupby("center").cumcount()) if len(m) else m
        themes.append({
            "id": re.sub(r"[^a-z0-9]+", "-", k).strip("-"), "key": k, "label": label, "family": fam_k, "follows": follows,
            "date_bound": fam_k in ("jewish", "secular", "start", "end") or k in ("familyweekend", "homecoming", "springbreak"),
            "n": int(len(ran)), "centers": int(ran["center"].nunique()), "n_measured": int(len(m)), "centers_measured": int(m["center"].nunique()) if len(m) else 0,
            "lift": r(_shrunk(logs), 3) if len(m) else None, "lift_median": r(float(np.median(m["tlift"])), 3) if len(m) else None,
            "hit": r(float((m["tlift"] >= 1.2).mean()), 3) if len(m) else None,
            "centers_up": int(sum(1 for v in by_center.values if v > 0)) if len(m) else 0,
            "first_time_lift": r(_shrunk(np.log(first_run.loc[first_run["nth"] == 0, "tlift"])), 3) if len(m) and (first_run["nth"] == 0).sum() >= 4 else None,
            "repeat_lift": r(_shrunk(np.log(first_run.loc[first_run["nth"] >= 1, "tlift"])), 3) if len(m) and (first_run["nth"] >= 1).sum() >= 4 else None,
            "confidence": "strong" if len(m) >= 12 and m["center"].nunique() >= 6 else ("some" if len(m) >= 5 and m["center"].nunique() >= 3 else "thin"),
            "status": status, "when": when, "months": months, "early": r(early, 2), "late": r(late, 2),
            "years": [{"ay": _ay_label(y), "events": years[y][0], "centers": len(years[y][1])} for y in sorted(years) if y >= 2020],
            "ytd_now": len(ytd_now), "ytd_last": len(ytd_last), "repeat": r(rep_n / rep_d, 2) if rep_d >= 4 else None, "repeat_n": rep_d,
            "first": {"date": first["date"].isoformat(), "center": int(first["center"])} if first is not None else None,
            "names": [{"name": title_case(n_), "n": c_} for n_, c_ in Counter(rows["nname"]).most_common(6)],
            "terms": terms, "descs": descs, "flyers": flyers, "occ": occ,
            "jh": dict(jh.most_common(3)), "sec": dict(sec.most_common(3)),
            "upcoming": int(rows["upcoming"].sum()),
        })
    ids_ = [t["key"] for t in themes]
    C = np.stack([cent[k] for k in ids_]) if ids_ else np.zeros((0, 1024))
    S = C @ C.T if len(ids_) else C
    co = defaultdict(Counter)
    for ks in sd["tkeys"]:
        for a in ks:
            for b in ks:
                if a != b:
                    co[a][b] += 1
    for i, t in enumerate(themes):
        o = [j for j in np.argsort(-S[i]) if j != i][:5]
        t["related"] = [{"id": themes[j]["id"], "label": themes[j]["label"], "sim": r(S[i, j], 2)} for j in o if S[i, j] >= 0.35][:4]
        t["paired"] = [{"id": next((x["id"] for x in themes if x["key"] == b), None), "n": n} for b, n in co[t["key"]].most_common(3) if n >= 2]

    # ---- does theming work at all, and how often can you do it
    th = sd[sd["ran"] & (sd["tkeys"].map(len) > 0) & sd["tlift"].notna()]
    overall = {"themed_measured": int(len(th)), "lift": r(_shrunk(np.log(th["tlift"]), 0), 3) if len(th) else None, "hit": r(float((th["tlift"] >= 1.2).mean()), 3) if len(th) else None}
    fam = []
    key_fam = {t["key"]: t["family"] for t in themes}
    for fk, fl_, _ in FAMILIES:
        g = th[th["tkeys"].map(lambda ks: any(key_fam.get(k) == fk for k in ks))]
        if len(g) >= 8:
            fam.append({"family": fk, "label": fl_, "n": int(len(g)), "centers": int(g["center"].nunique()), "lift": r(_shrunk(np.log(g["tlift"])), 3), "hit": r(float((g["tlift"] >= 1.2).mean()), 3)})
    # frequency: share of a center's dinners that were themed in the 60 days around each themed dinner
    freq = []
    themed_flag = (sd["tkeys"].map(len) > 0) | sd["one_off"]
    shares = np.full(len(sd), np.nan)
    for c, g in sd[sd["ran"]].groupby("center"):
        d = g["ord"].values; f = themed_flag.loc[g.index].values.astype(float)
        o = np.argsort(d); d, f, gi = d[o], f[o], g.index.values[o]
        cs = np.concatenate([[0], np.cumsum(f)])
        lo, hi = np.searchsorted(d, d - 60), np.searchsorted(d, d + 60, "right")
        tot = hi - lo
        sh = (cs[hi] - cs[lo]) / np.maximum(tot, 1)
        for i, s_, t_ in zip(gi, sh, tot):
            if t_ >= 6:
                shares[pos[i]] = s_
    sd["tshare"] = shares
    th = sd[sd["ran"] & (sd["tkeys"].map(len) > 0) & sd["tlift"].notna() & sd["tshare"].notna()]
    for lo_, hi_, lab in [(0, 0.15, "Rarely (under 1 in 7 dinners)"), (0.15, 0.3, "Sometimes (1 in 7 to 1 in 3)"), (0.3, 0.6, "Often (1 in 3 to 3 in 5)"), (0.6, 1.01, "Almost every week")]:
        g = th[(th["tshare"] >= lo_) & (th["tshare"] < hi_)]
        if len(g) >= 15:
            freq.append({"label": lab, "n": int(len(g)), "centers": int(g["center"].nunique()), "lift": r(_shrunk(np.log(g["tlift"])), 3), "hit": r(float((g["tlift"] >= 1.2).mean()), 3)})
    by_year = []
    for y, g in sd[sd["ran"]].groupby(sd.loc[sd["ran"], "date"].map(_ay)):
        if y >= 2021:
            by_year.append({"ay": _ay_label(y), "dinners": int(len(g)), "themed": int((themed_flag.loc[g.index]).sum()), "centers": int(g["center"].nunique())})

    # ---- the next Fridays: what is usually run then, and what centers have posted
    fridays = []
    f0 = today + dt.timedelta(days=(4 - today.weekday()) % 7)
    up = sd[sd["upcoming"]]
    hd_all = sd.loc[sd["ran"], "hdoy_n"].values
    for wk_i in range(6):
        F = f0 + dt.timedelta(days=7 * wk_i)
        ctx = cal.context(F); fdoy = F.timetuple().tm_yday; fh = cal.hdoy_norm(F)
        base_c = float((np.minimum(np.abs(all_doy - fdoy), 365 - np.abs(all_doy - fdoy)) <= 10).mean())
        base_h = float((np.abs(hd_all - fh) <= 8).mean())
        usual = []
        for t in themes:
            k = t["key"]
            rows = sd[sd["ran"] & sd["tkeys"].map(lambda ks, k=k: k in ks)]
            if len(rows) < 3:
                continue
            dd = rows["doy"].values; gap = np.minimum(np.abs(dd - fdoy), 365 - np.abs(dd - fdoy))
            near_c = gap <= 10
            near_h = np.abs(rows["hdoy_n"].values - fh) <= 8
            use_h = t["follows"] == "jewish"
            near = near_h if use_h else near_c
            ncen = rows.loc[near, "center"].nunique()
            if near.sum() < 2 or ncen < 2:
                continue
            share = float(near.mean()); ratio = share / max(base_h if use_h else base_c, 1e-6)
            if ratio < 1.5 and t["when"] != "Any time of year":
                continue
            ly = rows.loc[near & (rows["date"].map(lambda d: (today - d).days <= 800)), "center"].nunique()
            score = ncen * min(ratio, 8) ** 0.5 * (t["lift"] or 1.0)
            usual.append({"id": t["id"], "label": t["label"], "near": int(near.sum()), "of": int(len(rows)), "centers": int(ncen), "recent_centers": int(ly), "share": r(share, 2), "ratio": r(ratio, 1),
                          "calendar": "Jewish" if use_h else "civil", "lift": t["lift"], "status": t["status"], "score": r(score, 2)})
        usual.sort(key=lambda x: -x["score"])
        posted = up[(up["date"] >= F - dt.timedelta(days=1)) & (up["date"] <= F + dt.timedelta(days=1))]
        live = []
        for _, x in posted.iterrows():
            live.append({"id": int(x["id"]), "center": int(x["center"]), "name": x["name"], "regs": int(x["regs"]), "img": x["img"], "themes": [re.sub(r"[^a-z0-9]+", "-", k).strip("-") for k in x["tkeys"]],
                         "one_off": bool(x["one_off"]), "phrase": x["tdisp"] if (x["one_off"] or x["tkeys"]) else None})
        fridays.append({"date": F.isoformat(), "parsha": ctx["parsha"], "jh": ctx["jh"], "jh_runup": ctx["jh_runup"], "jh_days": ctx["jh_days"], "sec": ctx["sec"],
                        "usual": usual[:14], "posted": len(live), "live": sorted(live, key=lambda x: (-len(x["themes"]) - x["one_off"], -x["regs"]))})

    # recent one-offs: ideas only one or two centers have tried
    oo = sd[sd["one_off"] & sd["ran"]].sort_values("dt", ascending=False).drop_duplicates(["center", "tkey"]).head(400)
    one_offs = [{"name": x["name"], "phrase": x["tdisp"], "center": int(x["center"]), "date": x["date"].isoformat(), "regs": int(x["regs"]), "lift": r(x["tlift"], 2), "img": x["img"], "id": int(x["id"])} for _, x in oo.iterrows()]
    blob = {"today": today.isoformat(), "totals": {"dinners": int(sd["ran"].sum()), "themed": int((sd["ran"] & themed_flag).sum()), "centers": int(sd.loc[sd["ran"], "center"].nunique()),
                                                 "measured": int((sd["ran"] & sd["measured"]).sum()), "themes": len(themes)},
            "overall": overall, "families": fam, "frequency": freq, "by_year": by_year, "family_labels": FAMILY_LABEL,
            "themes": themes, "fridays": fridays, "one_offs": one_offs,
            "house_brands": [{"center": int(c), "phrase": k} for c, k in brands]}
    return blob, sd
