"""Event types without a language model.
The API has no category field, so every event is typed from its name, its description and when it happens.
Rules cover the bulk (most events have plain names). What the rules cannot place is typed later from its
embedding, by the labelled events nearest to it (see embed.knn_types).

Three separate readings per event:
  type      what kind of event it is (Shabbat dinner, class, social ...)
  hol       the Jewish holiday its name ties it to, if any
  aud       who it is for, when the name says so (grads, women, freshmen, Greek life ...)
  collab    whether the text says it is run with another group
"""
import re
import numpy as np

TYPES = {
    "shabbat_dinner": "Shabbat dinner",
    "shabbat_lunch": "Shabbat lunch",
    "shabbat_services": "Shabbat services",
    "shabbat_other": "Other Shabbat",
    "holiday_meal": "Holiday meal",
    "holiday_service": "Holiday services",
    "holiday_event": "Holiday event",
    "class": "Class or learning",
    "speaker": "Speaker or panel",
    "cooking": "Baking and cooking",
    "meal": "Food and hangout",
    "social": "Social and fun",
    "israel": "Israel and remembrance",
    "chesed": "Volunteering and chesed",
    "ritual": "Tefillin, prayer, mitzvah",
    "trip": "Trips, Shabbatons and outings",
    "outreach": "Tabling and student board",
    "togo": "To-go and deliveries",
    "resource": "Sign-ups and standing offers",
    "other": "Other",
}
# types whose sign-up counts mean "people came to an event" and can be compared with each other
EVENT_TYPES = [t for t in TYPES if t not in ("togo", "resource", "other")]

R = lambda p: re.compile(p, re.I)
SHAB = R(r"seudah?t? shlishi|shalosh seud|melave malka|\bshab+[aoe]?[ts]\b|shabbat|shabbos|shabbas|\bshabat|kab[\s\-]?shab|friday night|\bfnl\b|\boneg\b|\bchabbat\b")
HOL = [
    ("rosh_hashana", R(r"rosh hash|\brh ?[12]?\b|shofar|tashlich|new year'?s? (dinner|eve)? ?\(?jewish|apples? (and|n|&) honey|honey cake|high holi")),
    ("yom_kippur", R(r"yom kip|kol nidre|neilah|ne'?ila\b|break[\s\-](the[\s\-])?fast|pre[\s\-]?fast|yizkor|kapparot|kaparot")),
    ("simchat_torah", R(r"simchat? torah|simchas torah|hakafot|hakafos|shemini atzeret|shmini atzeres")),
    ("sukkot", R(r"suk+[ao]t|suc+[ao][ts]|sukkos|suk+ah?\b|succah?\b|in the hut|\bhut\b|lulav|etrog|esrog|hoshana")),
    ("chanukah", R(r"c?hanuk+ah?|chanukkah|hanukah|menorah|latke|dreidel|sufganiy|donut.*(night|light)|8 nights|eight nights|festival of lights")),
    ("tu_bshvat", R(r"tu b'?i?shvat|tu b'?shevat|new year for trees|tu bi shvat")),
    ("purim", R(r"purim|megil+ah?|hamantas|hamentas|mishloach|shalach manos|masquerade")),
    ("pesach", R(r"pesach|passover|\bseder|\bmatz[ao]h?s?\b(?! ?ball)|chametz|pessach|moshiach'?s? (seudah|meal|feast)|seudat moshiach|machiach")),
    ("lag_baomer", R(r"lag b'?a?'?omer|lag ba omer|lagbaomer")),
    ("shavuot", R(r"shavuot|shavuos|shavous|cheesecake|ten commandments|10 commandments")),
    ("yom_haatzmaut", R(r"yom ha'?atz|israel independence|yom hazikaron|israel'?s? (birthday|\d+)")),
    ("yom_hashoah", R(r"yom hashoah|holocaust remembrance")),
    ("tisha_bav", R(r"tisha b'?av|9 av|nine days")),
]
AUD = [
    ("grads", R(r"\bj?grads?\b|graduate student|grad student|\blaw (students?|school)\b|\bmed (students?|school)\b|medical students?|\bmba\b|\bphd\b|young professional|\byjp\b|post[\s\-]?docs?|\bdental\b|jlsa|jmsa")),
    ("women", R(r"\bladies\b|\bwomen'?s?\b|\bgirls?'?s?\b|sisterhood|rosh chodesh|galentine|\bgals\b|for her\b|challah (and|n|&) chat")),
    ("men", R(r"\bguys'?\b|\bmen'?s\b|brotherhood|\bbros\b|boys night|whiskey (and|&) wisdom|scotch")),
    ("freshmen", R(r"freshm[ae]n|first[\s\-]?years?|new students?|class of 20\d\d|\bfrosh\b|move[\s\-]?in|orientation|welcome week|transfer students?")),
    ("seniors", R(r"\bseniors? (send|night|shabbat|dinner|farewell|toast|week|brunch)|senior send[\s\-]?off|graduat(ion|ing)|commencement|4th year|fourth year")),
    ("greek", R(r"\baepi\b|aephi|\bae ?pi\b|\bzbt\b|\bsammy\b|\bsdt\b|\bsae\b|\btep\b|\bakpsi\b|\bdphie\b|d ?phi ?e|sigma|\bfraternit|\bsororit|greek (life|shabbat|night)|\bfrat\b|chai beta|alpha epsilon|zeta beta|\brush\b|panhel|\bifc\b")),
    ("israelis", R(r"israeli (students?|night|community)|hebrew speakers?|\bivrit\b|mishelanu|\bisraelis\b|ישראל")),
    ("parents", R(r"\bparents?'?\b|famil(y|ies) weekend|family (shabbat|dinner|brunch|bbq)|homecoming|\balumni\b|alums?\b|siblings")),
    ("faculty", R(r"\bfaculty\b|professors?\b|\bstaff\b")),
    ("athletes", R(r"\bathletes?\b|student[\s\-]athlete")),
    ("community", R(r"community (shabbat|dinner|wide|event|seder|bbq)|young (families|couples)|\bkids\b|children|\btots?\b|hebrew school|\bcteen\b|\bteens?\b|smile on seniors|\bseniors? (lunch|club|center)")),
]
COLLAB = R(r"co[\s\-]?host|co[\s\-]?sponsor|in partnership|partner(ed|ing|ship)? with|together with|\bjoint\b|\bcollab|sponsored by|hosted by .* and|\bx hillel|with hillel|\bhillel\b|\bw/ ?[a-z]{2,}|\bx\b [a-z]{2,6}\b$|student government|\bsga\b|interfaith|multicultural|\bbsu\b|\bmsa\b")

RESOURCE = R(r"help ?line|hotline|general sign|registration (is )?open|sign[\s\-]?up (form|sheet|here)|\bmeal plan|\bmembership\b|\bdues\b|\bwaiver\b|order (your|form|here)|pre[\s\-]?order|mezuzah (for|loan|rental|bank|gemach|check)|^mezuzah!?$|^tefillin!?$|free mezuzah|get a mezuzah|shmurah? matzah|test event|^test\b|^copy of|\bapplication\b|\bapply\b|scholarship|\bsurvey\b|\bform\b|kosher for passover (food|meals)$|sell (your )?chametz|lulav (and|&|n) e[st]rog (set|order)|\bsponsor(ship)?\b a|\bdonate\b|\bdonation|fundrais|\bcharidy|giving day|kosher meal (request|sign)")
NOTICE = R(r"^no (shabbat|shabbos|dinner|events?|class|lunch|services)|\bcancel+ed\b|\bpostponed\b|\bclosed\b|no .* this week|winter break$|spring break$|thanksgiving break$|^break$")
TOGO = R(r"to[\s\-]?go\b|\b2[\s\-]?go\b|pick[\s\-]?up|take[\s\-]?out|takeaway|deliver|care package|\bkits?\b|soup (delivery|deliveries|for the sick)|get well|in a box|(gift|shabbat|holiday|seder) box(es)?|(shabbat|goodie|care|gift|exam|finals|to go) bags?|grab (and|n|&) go|drive[\s\-]?(by|thru|through)")
TRIP = R(r"shabbaton|\bretreat\b|pegisha|birthright|\bgetaway\b|\btrip\b|\btour\b|\bohel\b|road trip|camping|\bski(ing)?\b|israel trip|\bmission\b|\bexpedition|weekend (away|in)|crown heights|national jewish retreat|encounter")
HOL_SERVICE = R(r"(rosh hash|yom kip|high holi|\brh\b|\byk\b|simchat? torah|simchas torah|sukkot|sukkos|pesach|passover|shavuot|shavuos|purim|chanukah|holiday|yom tov|shemini|tisha b)[^,]*\b(service|services|minyan|prayers?|davening|shacharit|mincha|maariv|musaf|hakafot|hakafos|reading)\b|kol nidre|neilah|ne'?ila\b|yizkor|megil+ah? reading|megil+ah?$|megil+ah? (and|&|\+)|shofar (blowing|service|services|in the park|on campus|on the quad)|tashlich|selichot|slichot|hakafot|hakafos|eicha")
HOL_MEAL = R(r"\bseder ?\d?\b|sedarim|break[\s\-](the[\s\-])?fast|pre[\s\-]?fast|yom tov (meal|dinner|lunch)|(rosh hash\w*|\brh\b|sukkot|sukkos|pesach|passover|shavuot|shavuos|simchat? torah|simchas torah|shemini atzeret|purim|holiday|yom tov|chag)[^,]*\b(dinners?|lunch(es|eon)?|meals?|feast|seudah?|nights? [12]|eve|brisket|kiddush)\b|(dinners?|lunch|meals?|feast|seudah?)[^,]*(rosh hash|sukkot|sukkos|pesach|passover|shavuot|shavuos|yom tov|holiday)|\brh ?[12]\b|rosh hashana (night|eve|day)|moshiach'?s? (seudah?|meal|feast)|purim seudah?|dairy (lunch|dinner|feast)")
COOK = R(r"\bbak(e|ing)\b|challah (bake|baking|making|braid|workshop)|cooking|\bcook[\s\-]?off|kosher cooking|\bchef\b|(making|rolling|decorating) (workshop|class|night)|(sushi|pizza|hamantas\w*|latke|pasta|dumpling|pretzel|cookie|cupcake|cake|pickle|babka|rugelach|kugel|cholent|matzah?|challah|donut|sufganiy\w*|bagel) (making|rolling|decorating|workshop|bake|baking)|make your own|decorat(e|ing)|mega challah|loaves of love")
CLASS = R(r"lunch (and|n|'n'?|&) learn|lunch n' learn|\blearn(ing)?\b|\bclass(es)?\b|\bpars?hah?\b|\bparashat?\b|\btorah\b|talmud|\btanya\b|kab+al+ah?|mishmar|\brabbi\b|rebbetzin|\bask\b|your jewish questions|jewish ?u\b|jewishu|sinai scholars|\byeshiva|chavrusa|chavruta|\bstudy\b(?! break)|\bjli\b|\bcourse\b|\bdvar\b|mystic|philosoph|\bethics\b|hebrew (class|reading|lessons?|school)|ask the rabbi|rabbi'?s? (class|table|roundtable)|farbrengen|\bkolel\b|beit midrash|bais medrash|\bwisdom\b|crash course|book club|\bdiscussion\b|\bdebate\b|pirkei avot|chassid|chasid|\bgemara\b|\bmishna|\bhalacha|judaism 101|jewish (thought|law|history|wisdom|mysticism|learning)|soul|meaningful monday|torah tuesday|talmud tuesday|wisdom wednesday|thinkers|think tank|big questions|\bq ?(and|&) ?a\b|deep (talk|dive|convers)|\bseminar\b|\bworkshop\b(?! .*(challah|cook|bak))|tea (and|&) torah|torah (and|&) tea|coffee (and|&) kabbalah|bagels (and|&) (torah|talmud)|pizza (and|&) (parsha|parasha|torah|talmud)|mitzvah (class|101)|marriage|relationship|dating|love and|kosher sex|intimacy")
SPEAKER = R(r"\bspeaker\b|guest (speaker|lecture)|\blecture\b|\bpanel\b|fireside|survivor|\bauthor\b|keynote|\btalk\b(?! torah)|presents?:|featuring|an evening with|live interview|conversation with|hear from|\bsymposium|my story|his story|her story|story of")
ISRAEL = R(r"\bisrael\b(?!i (students|night|community))|\bidf\b|hostage|oct(ober)?\.? ?7|\bvigil\b|\bmemorial\b|\brally\b|solidarity|zionis|antisemit|anti[\s\-]semit|holocaust|\bshoah\b|kristallnacht|yom hazikaron|stand with|bring them home|am yisrael|\bnova\b|remembrance|never again|\bmarch\b for|unity (rally|gathering|event)")
CHESED = R(r"volunteer|\bchesed\b|\bcharity\b|tzedakah|smile on seniors|hospital|nursing home|soup kitchen|food (drive|bank|pantry)|toy drive|blood drive|friendship circle|community service|mitzvah (day|project|tank|marathon)|give back|giving back|(packing|pack) (party|event|night|care|shabbat|food|lunch)|cards for|letters (to|for)|kindness|good deeds?|bikur cholim|visit (the )?(sick|elderly|seniors)|\bmivtzo|sandwich(es)? for|feed the|homeless|shelter|clothing drive")
RITUAL = R(r"tefil+[ie]n|shach?a?rit|\bmitzvah\b|\bminyan\b|shacharit|shachris|\bmincha\b|\bmaariv\b|davening|\bprayers?\b|\bservices?\b|mikvah|candle lighting|light (shabbat )?candles|\bbris\b|bar mitzvah|bat mitzvah|upsherin|\bkaddish\b|\byahrzeit|\bsiyum\b|\bdaven\b|hachnasat|torah (dedication|writing)|letter in the torah|\bwrap\b|kiddush levana|kosher (your )?kitchen|kashering")
OUTREACH = R(r"board (meetings?|elections?|dinner|retreat|members?)|chaboard|(^|student |chabad |general |executive |exec )board$|\bmeeting\b|committee|leadership|\bintern(s|ship)?\b|e[\s\-]?board|info(rmation)? session|interest meeting|\btabling\b|\btable (on|at)\b|welcome table|\w+ table$|club fair|involvement fair|org(anization)? fair|activities fair|student (leaders|council)|exec(utive)?\b|\binterview")
MEAL = R(r"\bbbq\b|barbe[cq]ue|chabb?q|chaba[\s\-]?b[\s\-]?q|brunch|\bbagels?\b|\blox\b|\bblt\b|\bb l t\b|\bkosher\b|caf[eé]|latemeal|late meal|breakfast|\blunch\b|\bdinner\b|\bsupper\b|\bpizza\b|\bsushi\b|falafel|shawarma|\btacos?\b|\bcafe\b|\bcoffee\b|ice cream|\bdessert|\bwings\b|burgers?\b|\bpasta\b|\bsoup\b|shakshuka|schnitzel|kosher (lunch|dinner|food|meal|bbq|deli|kitchen|eats)|food truck|picnic|potluck|\bdeli\b|\bgrill|cholent|\bfood\b|\beats?\b|\bfeast\b|\bmeal\b|hot dogs?|\bsubs?\b|sandwich|smoothie|bubble tea|\bboba\b|waffles?|pancakes?|\bfries\b|\bnosh|snacks?|treats?\b|\btea\b|hot chocolate|cocoa|donuts?|doughnuts?|cookies|kugel|poke|ramen|wraps?\b|bowls?\b|buffet|fleishig|milchig|chicken|steak|brisket|matzah? ball|study break|midnight breakfast|late night|munchies")
SOCIAL = R(r"havdal+ah?|s[c]?hmooze|sisterhood|game night|\bgames?\b|trivia|\bmovie|\bfilm\b|bowling|\bpaint|karaoke|\bparty\b|\bmixer\b|\bsocial\b|hangout|hang out|kickback|kick[\s\-]?off|bonfire|kumzit[sz]|open house|\bwelcome\b|meet (and|&|n) greet|happy hour|l'?chaim|mocktail|cocktail|\bwine\b|\bbeer\b|whiskey|poker|casino|comedy|comedian|concert|\bjam\b|open mic|\bsports?\b|basketball|soccer|football|tailgate|super ?bowl|watch party|escape room|laser tag|top ?golf|mini[\s\-]?golf|\bgolf\b|\bhike\b|hiking|kayak|\bbeach\b|\bpool\b|skating|arcade|carnival|festival|\bfair\b|talent|fashion|\bgala\b|\bformal\b|\bball\b|\bdance\b|dancing|\bcrafts?\b|\bart\b|pottery|tie[\s\-]?dye|candle making|\bdiy\b|\bspa\b|\byoga\b|fitness|workout|meditat|wellness|self[\s\-]?care|de[\s\-]?stress|\bfinals\b|\bfun\b|\bnight\b|\bcelebrat|\bbash\b|\bfest\b|\blounge\b|chill|vibes|birthday|reunion|scavenger|\bbingo\b|pickleball|volleyball|\bgym\b|paintball|go[\s\-]?kart|axe throwing|\bzoo\b|aquarium|apple picking|pumpkin|ice cream social|olympics|tournament|\bcup\b|color war|magic show|mentalist|hypnotist|open bar|mingle|speed (dating|friending)|\bsingles\b|photo|flower|succulent|terrarium|tote|jewelry|bracelet|mezuzah (making|painting|workshop)|build a|sip (and|&|n)")


def name_holiday(n, d=""):
    for key, rx in HOL:
        if rx.search(n):
            return key
    return None


def classify(n, desc, wd, hour, jh=None):
    """Return (type, activity, holiday key or None). n is the normalized name."""
    d = (desc or "").lower()
    fri_eve = wd == 4 and hour >= 15
    sat_day = wd == 5 and 8 <= hour < 17
    sat_eve = wd == 5 and hour >= 17
    hol = name_holiday(n)
    if NOTICE.search(n):
        return "resource", "notice", hol
    if TOGO.search(n):
        return "togo", "togo", hol
    if RESOURCE.search(n):
        return "resource", "resource", hol
    if TRIP.search(n) and not re.search(r"dinner|lunch", n):
        return "trip", "trip", hol
    shab = bool(SHAB.search(n))
    if HOL_SERVICE.search(n) and not re.search(r"dinner|lunch|meal|\band kiddush", n):
        return "holiday_service", "service", hol or jh
    if HOL_MEAL.search(n) and not shab:
        return "holiday_meal", "meal", hol or jh
    if shab:
        if re.search(r"havdala|melave|seudah?t? shlishi|shalosh seud|third meal|motzei", n):
            return "shabbat_other", "shabbat", hol
        lunchy = re.search(r"\blunch|kiddush|\bday\b|morning|shacharit|cholent|\bbrunch\b", n)
        dinner = re.search(r"dinner|friday|\bnight\b|\bmeal\b|kab[\s\-]?shab|\beve\b|evening|\bfnl\b|oneg", n)
        servicey = re.search(r"services?|kabbalat|kabolat|kabalat|minyan|davening|prayer|shacharit|mincha|maariv", n)
        if lunchy and not dinner:
            if servicey and not re.search(r"lunch|kiddush|cholent|brunch", n):
                return "shabbat_services", "service", hol
            return "shabbat_lunch", "meal", hol
        if servicey and not re.search(r"dinner|\bmeal\b|oneg", n):
            if fri_eve and re.search(r"dinner|meal|food|eat|feast|delicious|courses?", d):
                return "shabbat_dinner", "meal", hol
            return ("shabbat_services" if (fri_eve or sat_day or wd in (4, 5)) else "shabbat_other"), "service", hol
        if dinner or fri_eve or (wd == 4 and hour >= 12):
            return "shabbat_dinner", "meal", hol
        if sat_day:
            return "shabbat_lunch", "meal", hol
        if COOK.search(n):
            return "cooking", "cooking", hol
        return "shabbat_other", "shabbat", hol
    if fri_eve and re.search(r"\bdinner\b|kiddush|l'?chaim|candle|lighting|\bmeal\b", n) and not hol:
        return "shabbat_dinner", "meal", hol
    if sat_day and re.search(r"kiddush|cholent|\blunch\b", n) and not hol:
        return "shabbat_lunch", "meal", hol
    act = None
    m = n
    if hol:
        for key, rx in HOL:
            if key == hol:
                m = rx.sub(" ", n).strip()
    n_act = m
    if re.search(r"tefil+[ie]n", n): act = "ritual"
    elif COOK.search(n): act = "cooking"
    elif SPEAKER.search(n_act): act = "speaker"
    elif CLASS.search(n_act): act = "class"
    elif ISRAEL.search(n_act): act = "israel"
    elif CHESED.search(n_act): act = "chesed"
    elif RITUAL.search(n_act): act = "ritual"
    elif OUTREACH.search(n_act): act = "outreach"
    elif MEAL.search(n_act): act = "meal"
    elif SOCIAL.search(n_act): act = "social"
    if hol and act not in ("class", "speaker", "ritual", "chesed", "israel", "outreach"):
        return "holiday_event", act or "social", hol
    if act:
        return act, act, hol
    return "other", None, hol


def audiences(n, desc=""):
    return [k for k, rx in AUD if rx.search(n)]


def is_collab(n, desc=""):
    return bool(COLLAB.search(n) or COLLAB.search((desc or "").lower()[:400]))


def run(events):
    t, a, h = [], [], []
    cache = {}
    for n, d, wd, hr, jh in zip(events["nname"], events["desc"], events["wd"], events["hour"], events["jh"]):
        key = (n, wd, hr >= 15, 8 <= hr < 17, jh, d[:80])
        if key not in cache:
            cache[key] = classify(n, d, wd, hr, jh if isinstance(jh, str) else None)
        x = cache[key]; t.append(x[0]); a.append(x[1]); h.append(x[2])
    events["type"], events["act"], events["hol"] = t, a, h
    events["aud"] = [audiences(n) for n in events["nname"]]
    events["collab"] = [is_collab(n, d) for n, d in zip(events["nname"], events["desc"])]
    return events
