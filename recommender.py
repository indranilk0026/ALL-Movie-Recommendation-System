"""Runtime recommender shared by the notebook (evaluation) and the Streamlit app.

It only needs numpy, pandas and scipy, so the deployed app stays light.
Everything is computed on the fly for ONE query movie, so no N x N similarity
matrix has to be stored (the old project pickled a ~90 MB matrix).
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

METHODS = ["quality", "count", "tfidf", "embed", "cf", "hybrid"]
DEFAULT_WEIGHTS = {"tfidf": 0.25, "embed": 0.35, "cf": 0.30, "quality": 0.10}
LIST_COLS = ["genres_l", "keywords_l", "cast_l"]


def _minmax(v, mask):
    """Scale scores to 0-1 using only the candidate movies (mask == True)."""
    if not mask.any():
        return np.zeros_like(v, dtype=float)
    x = np.where(mask, v, np.nan)
    lo, hi = np.nanmin(x), np.nanmax(x)
    out = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v, dtype=float)
    return np.where(mask, out, 0.0)


def top_n(scores, n):
    """Indices of the n best scores (descending); -inf entries are dropped."""
    n = max(1, min(n, len(scores) - 1))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return idx[np.isfinite(scores[idx])]


class Recommender:
    def __init__(self, meta, count, tfidf, emb, cf, has_cf, weights=None):
        self.meta = meta.reset_index(drop=True)
        self.count = sp.csr_matrix(count)      # L2-normalised Bag-of-Words
        self.tfidf = sp.csr_matrix(tfidf)      # L2-normalised TF-IDF
        self.emb = np.asarray(emb, dtype=np.float32)   # L2-normalised sentence embeddings
        self.cf = np.asarray(cf, dtype=np.float32)     # L2-normalised item factors (0 rows = no ratings)
        self.has_cf = np.asarray(has_cf, dtype=bool)
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.quality = self.meta["quality"].to_numpy(float)
        self.labels = self.meta["label"].tolist()
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.n = len(self.meta)

    # ---------- save / load ----------
    def save(self, art_dir="artifacts"):
        os.makedirs(art_dir, exist_ok=True)
        m = self.meta.copy()
        for c in LIST_COLS:                      # lists -> "a|b|c" (robust across pandas versions)
            m[c] = m[c].apply(lambda xs: "|".join(xs))
        m.to_csv(f"{art_dir}/meta.csv.gz", index=False)
        sp.save_npz(f"{art_dir}/count.npz", self.count)
        sp.save_npz(f"{art_dir}/tfidf.npz", self.tfidf)
        np.save(f"{art_dir}/emb.npy", self.emb)
        np.save(f"{art_dir}/cf.npy", self.cf)
        np.save(f"{art_dir}/has_cf.npy", self.has_cf)
        with open(f"{art_dir}/weights.json", "w") as f:
            json.dump(self.weights, f, indent=2)

    @classmethod
    def load(cls, art_dir="artifacts"):
        meta = pd.read_csv(f"{art_dir}/meta.csv.gz")
        for c in LIST_COLS:
            meta[c] = meta[c].fillna("").apply(lambda s: [x for x in s.split("|") if x])
        meta["overview"] = meta["overview"].fillna("")
        meta["director"] = meta["director"].fillna("")
        weights = None
        if os.path.exists(f"{art_dir}/weights.json"):
            with open(f"{art_dir}/weights.json") as f:
                weights = json.load(f)
        return cls(meta, sp.load_npz(f"{art_dir}/count.npz"), sp.load_npz(f"{art_dir}/tfidf.npz"),
                   np.load(f"{art_dir}/emb.npy"), np.load(f"{art_dir}/cf.npy"),
                   np.load(f"{art_dir}/has_cf.npy"), weights)

    # ---------- scoring ----------
    def components(self, i):
        """Raw similarity of every movie to movie i, for each signal."""
        return {
            "count": (self.count @ self.count[i].T).toarray().ravel(),
            "tfidf": (self.tfidf @ self.tfidf[i].T).toarray().ravel(),
            "embed": self.emb @ self.emb[i],
            "cf": self.cf @ self.cf[i],
        }

    def filter_mask(self, genres=None, min_rating=0.0, language=None, year_range=None, pool=None):
        m = np.ones(self.n, dtype=bool) if pool is None else pool.copy()
        if genres:
            wanted = set(genres)
            m &= self.meta["genres_l"].apply(lambda g: bool(wanted.intersection(g))).to_numpy()
        if min_rating:
            m &= (self.meta["vote_average"] >= min_rating).to_numpy()
        if language and language != "Any":
            m &= (self.meta["original_language"] == language).to_numpy()
        if year_range:
            m &= self.meta["year"].between(year_range[0], year_range[1]).to_numpy()
        return m

    def score(self, i, method="hybrid", weights=None, mask=None, comps=None):
        mask = np.ones(self.n, dtype=bool) if mask is None else mask.copy()
        mask[i] = False                                  # never recommend the query itself
        if method == "quality":
            return np.where(mask, self.quality, -np.inf)
        comps = comps or self.components(i)
        if method != "hybrid":
            return np.where(mask, comps[method], -np.inf)
        w = dict(weights or self.weights)
        if not self.has_cf[i]:                           # query has no rating data -> drop the CF signal
            w["cf"] = 0.0
        total = sum(w.values())
        if total <= 0:
            return np.where(mask, self.quality, -np.inf)
        s = np.zeros(self.n)
        for k in ("tfidf", "embed", "cf"):
            if w.get(k, 0) > 0:
                v = np.clip(comps[k], 0, None) if k == "cf" else comps[k]
                s += (w[k] / total) * _minmax(v, mask)
        s += (w.get("quality", 0) / total) * self.quality
        return np.where(mask, s, -np.inf)

    def recommend(self, movie, n=10, method="hybrid", weights=None, **filters):
        """movie = label string or integer index. filters: genres, min_rating, language, year_range."""
        i = self.label_to_idx[movie] if isinstance(movie, str) else int(movie)
        comps = self.components(i)
        mask = self.filter_mask(**filters)
        s = self.score(i, method, weights, mask, comps)
        top = top_n(s, n)
        out = self.meta.iloc[top].copy()
        out["score"] = s[top]
        for k, v in comps.items():
            out[f"s_{k}"] = v[top]
        return out
