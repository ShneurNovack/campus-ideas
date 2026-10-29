"""Embeddings. Every distinct event text (name, tagline, description) is embedded once with bge-m3 through the
Worker (Workers AI) and cached on disk, so a daily run only embeds what is new."""
import os, json, time, threading
import numpy as np
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from .common import WORK, log, sha

EDGE = os.environ.get("EDGE_URL", "https://campus-ideas.shneur.workers.dev")
DIM = 1024
# Chosen by test (experiments in docs/ENGINE.md): bge-m3 against qwen3-embedding-0.6b and embeddinggemma-300m on
# three jobs built from this data. It won or tied each one, so the model stayed and the text preparation changed.
MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
_lock = threading.Lock()


_oidc = {"at": 0.0, "token": None}
_oidc_lock = threading.Lock()


def _github_token():
    """Inside GitHub Actions the job carries no secret. It asks GitHub for a short-lived signed token that names
    this repository and workflow, and the Worker checks the signature (worker/oidc.js). Tokens last a few minutes,
    so a fresh one is requested when the last is older than three."""
    url, bearer = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL"), os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not url or not bearer:
        return None
    with _oidc_lock:
        if _oidc["token"] and time.time() - _oidc["at"] < 180:
            return _oidc["token"]
        req = urllib.request.Request(url + "&audience=campus-ideas-analyst", headers={"authorization": "Bearer " + bearer, "user-agent": "campus-ideas-engine"})
        with urllib.request.urlopen(req, timeout=30) as r:
            _oidc["token"], _oidc["at"] = json.load(r)["value"], time.time()
        return _oidc["token"]


def _headers():
    h = {"content-type": "application/json", "user-agent": "campus-ideas-engine"}
    if os.environ.get("ANALYST_KEY"):
        h["x-analyst-key"] = os.environ["ANALYST_KEY"]
    elif os.environ.get("ANALYST_TOKEN"):
        h["authorization"] = "Bearer " + os.environ["ANALYST_TOKEN"]
    else:
        tok = _github_token()
        if tok:
            h["authorization"] = "Bearer " + tok
    return h


def _call(texts, tries=5, model="bge-m3"):
    body = json.dumps({"texts": texts, "model": model}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(EDGE + "/analyst/embed", data=body, headers=_headers(), method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                v = np.frombuffer(r.read(), dtype=np.float32).reshape(-1, int(r.headers.get("x-dims") or DIM))
                if len(v) == len(texts):
                    return v
        except Exception as e:
            err = e
        time.sleep(2 + 4 * t)
    raise RuntimeError(f"embedding failed: {err}")


def event_text(name, short, desc):
    parts = [name.strip()]
    for x in (short, desc):
        x = (x or "").strip()
        if x and x.lower() != name.strip().lower() and x not in parts:
            parts.append(x)
    return ". ".join(parts)[:700]


class Cache:
    """text -> vector, stored as a float16 matrix plus a key list."""
    def __init__(self, name="emb", model="bge-m3"):
        self.model = model
        self.kp, self.vp = f"{WORK}/{name}_keys.json", f"{WORK}/{name}_vecs.npy"
        if os.path.exists(self.kp) and os.path.exists(self.vp):
            self.keys = json.load(open(self.kp)); self.vecs = np.load(self.vp)
        else:
            self.keys, self.vecs = [], None
        self.pos = {k: i for i, k in enumerate(self.keys)}

    def get(self, texts, batch=48, workers=4):
        ks = [sha(t) for t in texts]
        missing = {}
        for k, t in zip(ks, texts):
            if k not in self.pos and k not in missing:
                missing[k] = t
        if missing:
            mk = list(missing); log("embedding", len(mk), "new texts")
            chunks = [mk[i:i + batch] for i in range(0, len(mk), batch)]
            out = [None] * len(chunks); done = [0]
            def work(i):
                out[i] = _call([missing[k] for k in chunks[i]], model=self.model)
                with _lock:
                    done[0] += 1
                    if done[0] % 40 == 0: log("  embedded", done[0] * batch, "/", len(mk))
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(work, range(len(chunks))))
            new = np.concatenate(out).astype(np.float16)
            self.vecs = new if self.vecs is None else np.concatenate([self.vecs, new]); self.keys += mk
            self.pos = {k: i for i, k in enumerate(self.keys)}
            np.save(self.vp, self.vecs); json.dump(self.keys, open(self.kp, "w"))
        return self.vecs[[self.pos[k] for k in ks]].astype(np.float32)


def unit(m):
    m = np.asarray(m, np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)
