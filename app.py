import datetime
import os
import random

import requests
import streamlit as st

import tmdb_live as tl
from recommender import DEFAULT_WEIGHTS, Recommender

st.set_page_config(page_title="Smart Movie Recommender", page_icon="🎬", layout="wide")

ENGINES = {
    "Hybrid (best)": "hybrid",
    "Semantic embeddings": "embed",
    "TF-IDF keywords": "tfidf",
    "Bag-of-Words (original)": "count",
    "Collaborative (user ratings)": "cf",
}
LANGS = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "ja": "Japanese",
         "zh": "Chinese", "ko": "Korean", "hi": "Hindi", "ru": "Russian", "pt": "Portuguese", "cn": "Cantonese",
         "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "tr": "Turkish", "sv": "Swedish"}
WORLD, CATALOG = "🌍 All movies in the world (live TMDB)", "📚 Curated catalog (my ML models)"
THIS_YEAR = datetime.date.today().year


# ---------------- data / API helpers ----------------
@st.cache_resource
def load_recommender():
    try:
        return Recommender.load("artifacts")
    except FileNotFoundError:
        return None


def tmdb_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.environ.get("TMDB_API_KEY")


@st.cache_data(show_spinner=False, ttl=86400)
def poster_url(movie_id, key):
    """Poster for catalog movies (world-mode rows already carry their poster_path)."""
    if not key:
        return None
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{int(movie_id)}", params={"api_key": key}, timeout=8)
        r.raise_for_status()
        path = r.json().get("poster_path")
        return f"{tl.IMG}{path}" if path else None
    except requests.RequestException:
        return None


@st.cache_data(show_spinner=False, ttl=86400)
def get_gmap(key):
    return tl.genre_map(key)


def card(row, key, show_scores=False, caption=None):
    path = row.get("poster_path")
    url = f"{tl.IMG}{path}" if isinstance(path, str) and path else poster_url(row["id"], key)
    if url:
        st.image(url)
    else:
        st.markdown("🎞️ *No poster available*")
    year = f" ({row['year']})" if row["year"] else ""
    st.markdown(f"**{row['title']}**{year}")
    st.caption(f"⭐ {row['vote_average']:.1f}  ·  {', '.join(row['genres_l'][:3])}")
    if row.get("director"):
        st.caption(f"🎬 {row['director']}")
    ov = row["overview"]
    st.write(ov[:180] + ("…" if len(ov) > 180 else ""))
    if caption:
        st.caption(caption)
    if show_scores:
        with st.expander("Why this movie?"):
            st.write(f"Keyword similarity (TF-IDF): {row['s_tfidf']:.2f}")
            st.write(f"Semantic similarity (embeddings): {row['s_embed']:.2f}")
            st.write(f"Similar viewers' taste (ratings): {max(row['s_cf'], 0):.2f}")


rec = load_recommender()
key = tmdb_key()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    modes = ([WORLD] if key else []) + ([CATALOG] if rec else [])
    if not modes:
        st.error("No TMDB_API_KEY and no artifacts/ folder found. Add one of them to run the app.")
        st.stop()
    scope = st.radio("Search in", modes)
    world = scope == WORLD
    n_recs = st.slider("Number of recommendations", 3, 12, 6)

    method, weights = "hybrid", (dict(rec.weights) if rec else {})
    if not world:
        method = ENGINES[st.selectbox("Recommendation engine", list(ENGINES), index=0)]
        if method == "hybrid":
            with st.expander("Hybrid weights (advanced)"):
                for k, label in [("embed", "Semantic embeddings"), ("tfidf", "TF-IDF keywords"),
                                 ("cf", "User-rating signal"), ("quality", "Rating / popularity prior")]:
                    weights[k] = st.slider(label, 0.0, 1.0, float(weights.get(k, DEFAULT_WEIGHTS[k])), 0.05)

    st.subheader("Filters")
    if world:
        all_genres = sorted(get_gmap(key).values())
        langs = sorted(LANGS, key=lambda c: LANGS[c])
        y_min, y_max = 1900, THIS_YEAR
    else:
        all_genres = sorted({g for gs in rec.meta["genres_l"] for g in gs})
        langs = sorted(rec.meta["original_language"].dropna().unique(), key=lambda c: (c not in LANGS, LANGS.get(c, c)))
        y_min, y_max = int(rec.meta["year"].min()), int(rec.meta["year"].max())
    genres = st.multiselect("Genres (any of)", all_genres)
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, 0.5)
    language = st.selectbox("Original language", ["Any"] + langs, format_func=lambda c: LANGS.get(c, c))
    year_range = st.slider("Release year", y_min, y_max, (y_min, y_max))
    full_years = year_range == (y_min, y_max)

# ---------------- main page ----------------
st.title("🎬 Smart Movie Recommender")
if world:
    st.write("Search **any movie** (the whole TMDB database, over a million titles) and get similar films.")
else:
    st.write("Pick a movie from the curated catalog and get recommendations from my trained ML models.")
if not key:
    st.caption("Add a TMDB_API_KEY in Streamlit secrets to show posters and unlock worldwide search.")

for k in ("query", "tmdb_id"):
    st.session_state.setdefault(k, None)

go = surprise = False
picked = None
if world:
    text = st.text_input("Search any movie", placeholder="e.g. A Beautiful Mind")
    hits = []
    if text.strip():
        try:
            hits = tl.search_movies(text, key, get_gmap(key))
        except requests.RequestException:
            st.error("Could not reach TMDB. Check your API key and internet connection.")
    labels = {}
    for h in hits:
        lab = f"{h['title']} ({h['year'] or '?'})  ·  ⭐ {h['vote_average']:.1f}"
        labels[lab if lab not in labels else f"{lab}  [id {h['id']}]"] = h
    if text.strip() and not hits:
        st.warning("No movie found for that search. Try another spelling.")
    if labels:
        picked = st.selectbox("Pick the exact movie", list(labels), index=None, placeholder="Select a movie...")
    c1, c2 = st.columns(2)
    go, surprise = c1.button("🔎 Show recommendations"), c2.button("🎲 Surprise me")
    if go:
        if picked is None:
            st.info("Search for a movie and pick it from the list first.")
        else:
            st.session_state.tmdb_id = labels[picked]["id"]
    if surprise:
        try:
            st.session_state.tmdb_id = tl.random_popular(key, get_gmap(key))["id"]
        except requests.RequestException:
            st.error("Could not reach TMDB.")
else:
    selected = st.selectbox("Choose a movie (type to search)", rec.labels, index=None, placeholder="Select a movie...")
    c1, c2 = st.columns(2)
    if c1.button("🔎 Show recommendations"):
        if selected is None:
            st.info("Please choose a movie first.")
        else:
            st.session_state.query = rec.label_to_idx[selected]
    if c2.button("🎲 Surprise me"):
        st.session_state.query = random.randrange(rec.n)

filters = dict(genres=genres, min_rating=min_rating, language=language, year_range=year_range)

if world:
    if st.session_state.tmdb_id is None:
        st.info("Search for a movie, pick it, and press the button.")
    else:
        try:
            movie, related = tl.get_movie(st.session_state.tmdb_id, key, get_gmap(key))
            recs = tl.recommend_live(movie, related, key, get_gmap(key), n=n_recs, genres=genres,
                                     min_rating=min_rating, language=language,
                                     year_range=None if full_years else year_range)
        except requests.RequestException:
            st.error("Could not load data from TMDB. Check your API key and connection.")
            st.stop()
        st.subheader("Because you picked")
        left, _ = st.columns([1, 3])
        with left:
            card(movie, key)
        st.subheader("You might also like")
        if recs.empty:
            st.info("No movies match these filters. Try relaxing them.")
        else:
            for start in range(0, len(recs), 3):
                cols = st.columns(3)
                for col, (_, row) in zip(cols, recs.iloc[start:start + 3].iterrows()):
                    with col:
                        card(row, key, caption=f"Match score: {row['score']:.2f} · {row['source']}")
        if rec is not None and movie["id"] in set(rec.meta["id"]):       # bonus: also in the curated catalog
            with st.expander("🧠 This movie is also in the curated catalog: see what my ML models suggest"):
                idx = int(rec.meta.index[rec.meta["id"] == movie["id"]][0])
                ml = rec.recommend(idx, n=n_recs, method="hybrid", weights=weights, **filters)
                for start in range(0, len(ml), 3):
                    cols = st.columns(3)
                    for col, (_, row) in zip(cols, ml.iloc[start:start + 3].iterrows()):
                        with col:
                            card(row, key, show_scores=True, caption=f"Match score: {row['score']:.2f}")
else:
    q = st.session_state.query
    if q is None:
        st.info("Choose a movie and press the button to get recommendations.")
    else:
        use_method = method
        if method == "cf" and not rec.has_cf[q]:
            st.warning("This movie has too few user ratings for the collaborative engine; using Hybrid instead.")
            use_method = "hybrid"
        recs = rec.recommend(q, n=n_recs, method=use_method, weights=weights, **filters)
        st.subheader("Because you picked")
        left, _ = st.columns([1, 3])
        with left:
            card(rec.meta.iloc[q], key)
        st.subheader("You might also like")
        if recs.empty:
            st.info("No movies match these filters. Try relaxing them.")
        else:
            for start in range(0, len(recs), 3):
                cols = st.columns(3)
                for col, (_, row) in zip(cols, recs.iloc[start:start + 3].iterrows()):
                    with col:
                        card(row, key, show_scores=True, caption=f"Match score: {row['score']:.2f}")
