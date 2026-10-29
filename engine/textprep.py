"""Text preparation for embeddings.

An event post is a name plus, sometimes, a blurb. Both carry things that say nothing about what the event is:
dates, times, prices, the center's own name, "RSVP by Thursday". Left in, they pull two posts of the same event
apart and pull unrelated posts from one center together. These functions strip them, so the vector is about the
event itself. The name and the blurb are embedded separately and combined afterwards (see embed.py), which keeps a
post with no blurb comparable with one that has a long one.
"""
import re
import unicodedata

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️←-⇿⬀-⯿•■-◿]")
_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+|\b\S+\.(com|org|net|edu)\S*", re.I)
_PHONE = re.compile(r"\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b")
_TIME = re.compile(r"\b\d{1,2}(:\d{2})?\s?(a\.?m\.?|p\.?m\.?)\b|\b\d{1,2}:\d{2}\b", re.I)
_MONTH = r"(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sept?(ember)?|oct(ober)?|nov(ember)?|dec(ember)?)"
_DAY = r"(mon|tues?|wed(nes)?|thur?s?|fri|sat(ur)?|sun)(day)?"
_DATE = re.compile(rf"\b({_DAY},?\s+)?{_MONTH}\.?\s+\d{{1,2}}(st|nd|rd|th)?(,?\s+\d{{4}})?\b|\b\d{{1,2}}[/.]\d{{1,2}}([/.]\d{{2,4}})?\b|\b(19|20)\d\d\b|\b57\d\d\b", re.I)
_PRICE = re.compile(r"[$€£]\s?\d+(\.\d\d)?|\b\d+\s?(dollars|usd|eur|€|£)\b", re.I)
_NUM = re.compile(r"#\s?\d+|\b\d+(st|nd|rd|th)\b|\b\d{1,2}\b")      # "Lunch 2", "#4"; "Shabbat 500" keeps its number
_PLACE = re.compile(r"\s*(@|\bat\b|\bwith\b)\s+(the\s+)?(chabad( house| center| jewish (student )?cent(er|re))?|beit chabad|our house|the rabbi'?s? house)\b", re.I)
_LOGISTICS = re.compile(r"[^.!?\n]*\b(rsvp|r\.s\.v\.p|sign ?up|register|registration|reply\b|text\b.{0,12}\bto\b|click|link in bio|limited (space|spots|seating)|space is limited|doors open|"
                        r"candle lighting|suggested donation|sponsor(ed|ship)?|venmo|zelle|cashapp|free of charge|no charge|bs\"?d|b\"h|will update soon)\b[^.!?\n]*[.!?]?", re.I)
_WS = re.compile(r"\s+")
_FILLER = re.compile(r"^\s*(join us|come( and)?( join)?( us)?|please join( us)?|you('| a)re invited( to)?|we invite you( to)?)\s+(for|to|as we|at|on|this)?\s*", re.I)


def _base(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = _EMOJI.sub(" ", s).replace("’", "'").replace("“", '"').replace("”", '"')
    return s


def _case(s):
    """Shouted names read differently to a model than the same words typed normally."""
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8 and len(letters) > 4:
        return " ".join(w if (len(w) <= 4 and w.isalpha() and w.isupper() and not re.search(r"[AEIOU]{1}", w[1:-1] or "x")) else w.capitalize() for w in s.split())
    return s


def center_words(center_row):
    """Words that name the center or its school, to take out of that center's event names ("UCF", "Gators")."""
    out = set()
    for k in ("nick",):
        v = center_row.get(k)
        if isinstance(v, str):
            for w in re.split(r"[\s/,\-]+", v):
                if 2 <= len(w) <= 12 and w.lower() not in {"the", "of", "at", "university", "college", "state", "school", "uni", "u"}:
                    out.add(w.lower())
    return out


def name_text(name, own=(), keep_numbers=False):
    """The event's name with dates, times, places and the center's own name taken out. `keep_numbers` is for
    display labels, where "Seder 2" should stay "Seder 2"."""
    s = _base(name)
    s = _URL.sub(" ", s)
    s = re.sub(r"\([^)]*\d[^)]*\)", " ", s)             # "(Early, 5:30pm)"
    s = _TIME.sub(" ", s); s = _DATE.sub(" ", s); s = _PRICE.sub(" ", s)
    if not keep_numbers:
        s = _NUM.sub(" ", s)
    s = _PLACE.sub(" ", s)
    if own:
        s = " ".join(w for w in s.split() if w.strip(".,!?:;'\"()@#").lower() not in own)
    s = re.sub(r"\bchabad('s)?\b", " ", s, flags=re.I) if len(re.sub(r"\bchabad('s)?\b", "", s, flags=re.I).strip(" -:!@&")) >= 4 else s
    s = re.sub(r"[!?*~_|]+", " ", s)
    s = re.sub(r"\s*[-:–@&,/]+\s*$", "", _WS.sub(" ", s)).strip(" -:–@&,/.")
    s = re.sub(r"(\s+(at|with|for|of|in|on|the|and|x|w/|@))+$", "", s, flags=re.I).strip(" -:–@&,/.")     # "Seder 1 with Chabad at UVM" -> "Seder 1"
    s = _case(s)
    return s if len(s) >= 2 else _WS.sub(" ", _base(name)).strip() or "event"


def blurb_text(desc, short="", name="", limit=320):
    """The blurb with dates, times, prices, links and sign-up instructions removed. Empty when nothing useful is left."""
    parts = []
    for x in (short, desc):
        x = _base(x).strip()
        if x and x.lower() != (name or "").strip().lower() and all(x.lower() not in p.lower() for p in parts):
            # the tagline is often the first line of the blurb, cut short
            parts = [p for p in parts if p.lower() not in x.lower() and p.lower()[:40] not in x.lower()] + [x]
    s = " ".join(parts)
    if not s:
        return ""
    s = _URL.sub(" ", s); s = _PHONE.sub(" ", s)
    s = _LOGISTICS.sub(" ", s)
    s = _TIME.sub(" ", s); s = _DATE.sub(" ", s); s = _PRICE.sub(" ", s)
    s = re.sub(r"\b(this|every|next|on)\s+" + _DAY + r"\b( night| evening| morning| afternoon)?", " ", s, flags=re.I)
    s = _PLACE.sub(" ", s)
    s = _FILLER.sub("", _WS.sub(" ", s)).strip(" -:|,.;")
    s = _case(s)
    if len(s) > limit:
        cut = s[:limit]
        stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        s = cut[:stop + 1] if stop > limit * 0.5 else cut.rsplit(" ", 1)[0]
    words = [w for w in re.findall(r"[^\W\d_]{3,}", s)]
    if len(words) < 4 or s.lower().strip(" .!") == (name or "").lower().strip(" .!"):
        return ""
    return s
