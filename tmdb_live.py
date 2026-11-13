"""Live recommendations for ANY movie in the TMDB database (~1 million titles).

The curated catalog (4,800 movies) is covered by our own ML models. For every other movie we ask
TMDB for its related movies and re-rank them with a transparent scoring rule:

    score = 0.45 * TMDB relevance rank   (TMDB's own collaborative/similarity signal)
          + 0.30 * genre overlap         (Jaccard similarity of genres with the query movie)
          + 0.15 * quality               (vote-count-aware rating, 0-1)
          + 0.10 * same original language

These weights are sensible defaults, not tuned or evaluated like the catalog hybrid model.
If TMDB returns too few related movies, we top up with /discover (same genres, well rated).
Needs only the `requests` package and a TMDB API key.
"""
import random
from functools import lru_cache

import pandas as pd
import requests

API = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p/w342"
COLUMNS = ["id", "title", "year", "genres_l", "genre_ids", "director", "overview", "vote_average",
           "vote_count", "popularity", "original_language", "poster_path"]


@lru_cache(maxsize=1024)
def _cached_get(path, key, params):
    r = requests.get(API + path, params={"api_key": key, **dict(params)}, timeout=10)
    r.raise_for_status()
    return r.json()


def _get(path, key, **params):
    return _cached_get(path, key, tuple(sorted(params.items())))


def genre_map(key):
    return {g["id"]: g["name"] for g in _get("/genre/movie/list", key, language="en-US")["genres"]}


def _year(date):
    return int(date[:4]) if date and date[:4].isdigit() else 0


def _norm(m, gmap):
    """TMDB movie JSON (list or detail format) -> one flat dict with our column names."""
    if m.get("genres"):
        ids = [g["id"] for g in m["genres"]]
        names = [g["name"] for g in m["genres"]]
    else:
        ids = list(m.get("genre_ids") or [])
        names = [gmap[i] for i in ids if i in gmap]
    return {"id": m["id"], "title": m.get("title") or m.get("original_title") or "Untitled",
            "year": _year(m.get("release_date")), "genres_l": names, "genre_ids": ids, "director": "",
            "overview": m.get("overview") or "", "vote_average": float(m.get("vote_average") or 0),
            "vote_count": int(m.get("vote_count") or 0), "popularity": float(m.get("popularity") or 0),
            "original_language": m.get("original_language") or "", "poster_path": m.get("poster_path")}


def search_movies(query, key, gmap, limit=8):
    data = _get("/search/movie", key, query=query.strip(), include_adult="false")
    return [_norm(m, gmap) for m in data.get("results", [])[:limit]]


def get_movie(movie_id, key, gmap):
    """Returns (movie, related) where related = TMDB 'recommendations' first, then 'similar'."""
    d = _get(f"/movie/{int(movie_id)}", key, append_to_response="credits,recommendations,similar")
    movie = _norm(d, gmap)
    movie["director"] = next((c["name"] for c in d.get("credits", {}).get("crew", []) if c.get("job") == "Director"), "")
    related = [(_norm(x, gmap), "TMDB recommendation") for x in d.get("recommendations", {}).get("results", [])]
    try:   # one extra page of recommendations for a bigger candidate pool
        more = _get(f"/movie/{int(movie_id)}/recommendations", key, page=2).get("results", [])
        related += [(_norm(x, gmap), "TMDB recommendation") for x in more]
    except requests.RequestException:
        pass
    related += [(_norm(x, gmap), "TMDB similar") for x in d.get("similar", {}).get("results", [])]
    return movie, related


def random_popular(key, gmap):
    page = random.randint(1, 10)
    results = _get("/movie/popular", key, page=page).get("results", [])
    return _norm(random.choice(results), gmap)


def _passes(c, genres, min_rating, language, year_range, min_votes=5):
    if c["vote_count"] < min_votes:
        return False
    if genres and not set(genres) & set(c["genres_l"]):
        return False
    if min_rating and c["vote_average"] < min_rating:
        return False
    if language and language != "Any" and c["original_language"] != language:
        return False
    if year_range and not (year_range[0] <= c["year"] <= year_range[1]):
        return False
    return True


def _quality(c, global_mean=6.5, m=300):
    v, r = c["vote_count"], c["vote_average"]
    return ((v / (v + m)) * r + (m / (v + m)) * global_mean) / 10.0


def _discover(movie, key, gmap, genres, min_rating, language, year_range, exclude):
    params = {"sort_by": "vote_average.desc", "vote_count.gte": 300, "include_adult": "false"}
    ids = movie["genre_ids"][:2]
    if genres:                                         # user-picked genres override the query's genres
        by_name = {v: k for k, v in gmap.items()}
        ids = [by_name[g] for g in genres if g in by_name][:3]
    if ids:
        params["with_genres"] = ",".join(map(str, ids)) if not genres else "|".join(map(str, ids))
    if language and language != "Any":
        params["with_original_language"] = language
    if min_rating:
        params["vote_average.gte"] = min_rating
    if year_range:
        params["primary_release_date.gte"] = f"{year_range[0]}-01-01"
        params["primary_release_date.lte"] = f"{year_range[1]}-12-31"
    out = []
    for c in _get("/discover/movie", key, **params).get("results", []):
        c = _norm(c, gmap)
        if c["id"] not in exclude and c["id"] != movie["id"]:
            out.append(c)
    return out


def recommend_live(movie, related, key, gmap, n=6, genres=None, min_rating=0.0, language=None, year_range=None):
    """Filter + re-rank TMDB's related movies; top up with /discover when there are too few."""
    seen, pool = set(), []
    for c, src in related:                              # dedupe, keep TMDB's order (recommendations first)
        if c["id"] not in seen and c["id"] != movie["id"]:
            seen.add(c["id"]); pool.append((c, src))
    qgen = set(movie["genres_l"])
    rows = []
    for pos, (c, src) in enumerate(pool):
        if not _passes(c, genres, min_rating, language, year_range):
            continue
        g = set(c["genres_l"])
        jac = len(qgen & g) / len(qgen | g) if (qgen | g) else 0.0
        score = (0.45 * (1 - pos / max(len(pool), 1)) + 0.30 * jac + 0.15 * _quality(c)
                 + 0.10 * (c["original_language"] == movie["original_language"]))
        rows.append({**c, "score": score, "source": src})
    rows.sort(key=lambda r: r["score"], reverse=True)
    rows = rows[:n]
    if len(rows) < n:                                   # not enough related movies -> discover fill
        have = {r["id"] for r in rows} | seen
        for c in _discover(movie, key, gmap, genres, min_rating, language, year_range, have):
            if len(rows) >= n:
                break
            rows.append({**c, "score": 0.0, "source": "Top-rated in similar genres"})
    return pd.DataFrame(rows, columns=COLUMNS + ["score", "source"])
