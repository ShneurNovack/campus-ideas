"""Shared paths and small helpers for the Campus Playbook engine."""
import json, os, re, hashlib, time, unicodedata

DATA = os.environ.get("CP_DATA") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW = os.path.join(DATA, "raw")
WORK = os.path.join(DATA, "work")
OUT = os.path.join(DATA, "out")
for _d in (RAW, WORK, OUT):
    os.makedirs(_d, exist_ok=True)

IMG_PREFIX = "https://cocistorage.blob.core.windows.net/prod/"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _san(o):
    """JSON has no NaN. Turn NaN and numpy scalars into plain values, all the way down."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_san(v) for v in (sorted(o) if isinstance(o, (set, frozenset)) else o)]
    if isinstance(o, np.ndarray):
        return [_san(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if o != o or o in (float("inf"), float("-inf")) else float(o)
    return o


def dump(obj, path, compact=True):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_san(obj), f, ensure_ascii=False, separators=(",", ":") if compact else None, allow_nan=False)
    return os.path.getsize(path)


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️←-⇿⬀-⯿]")
_WS = re.compile(r"\s+")


def clean(s):
    """Tidy text for display: collapse whitespace, keep case and punctuation."""
    return _WS.sub(" ", (s or "").replace(" ", " ")).strip()


def norm_name(s):
    """Key for 'the same event name': lower case, no emoji, no punctuation runs, single spaces."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _EMOJI.sub(" ", s)
    s = s.replace("&", " and ").replace("@", " at ").replace("’", "'").replace("`", "'")
    s = re.sub(r"[^\w\s'\-+/]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\b(copy of|new)\b", " ", s) if s.startswith("copy of") else s
    return _WS.sub(" ", s).strip(" -'/")


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def r(x, n=3):
    try:
        if x is None or x != x:
            return None
        return round(float(x), n)
    except Exception:
        return None


def title_case(s):
    small = {"and", "or", "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "n"}
    out = []
    for i, w in enumerate(s.split()):
        if w.lower() in small and i:
            out.append(w.lower())
        elif w.isupper() and len(w) <= 5:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)
