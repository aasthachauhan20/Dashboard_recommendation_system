import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="DSA Question Recommendation System",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 DSA Question Recommendation System")
st.write(
    "Machine Learning-based recommendation system using TF-IDF Vectorization and KNN"
)

# ----------------------------------
# LOAD DATASET
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("leetcode_dataset - lc.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ----------------------------------
# SHOW COLUMNS
# ----------------------------------
st.sidebar.header("Dataset Information")
st.sidebar.write("Columns Found:")
st.sidebar.write(list(df.columns))

# ----------------------------------
# AUTO-DETECT COLUMNS
# ----------------------------------
title_candidates = [
    "title",
    "Title",
    "question",
    "Question",
    "name",
    "Name"
]

topic_candidates = [
    "related_topics",
    "topics",
    "topic_tags",
    "tags",
    "Related Topics",
    "Topics"
]

difficulty_candidates = [
    "difficulty",
    "Difficulty"
]

TITLE_COL = None
TOPIC_COL = None
DIFFICULTY_COL = None

for col in title_candidates:
    if col in df.columns:
        TITLE_COL = col
        break

for col in topic_candidates:
    if col in df.columns:
        TOPIC_COL = col
        break

for col in difficulty_candidates:
    if col in df.columns:
        DIFFICULTY_COL = col
        break

if TITLE_COL is None:
    st.error(
        "Title column not found. Please check your dataset."
    )
    st.stop()

if TOPIC_COL is None:
    st.error(
        "Topics column not found. Please check your dataset."
    )
    st.stop()

if DIFFICULTY_COL is None:
    st.error(
        "Difficulty column not found. Please check your dataset."
    )
    st.stop()

# ----------------------------------
# PREPROCESSING
# ----------------------------------
df[TOPIC_COL] = df[TOPIC_COL].fillna("").astype(str)
df[DIFFICULTY_COL] = df[DIFFICULTY_COL].fillna("").astype(str)

df["features"] = (
    df[TOPIC_COL] +
    " " +
    df[DIFFICULTY_COL]
)

# ----------------------------------
# TF-IDF VECTORIZATION
# ----------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(
    df["features"]
)

# ----------------------------------
# KNN MODEL
# ----------------------------------
model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model.fit(X)

# ----------------------------------
# SIDEBAR STATS
# ----------------------------------
st.sidebar.header("Statistics")

st.sidebar.metric(
    "Total Questions",
    len(df)
)

easy_count = (
    df[DIFFICULTY_COL]
    .str.lower()
    .eq("easy")
    .sum()
)

medium_count = (
    df[DIFFICULTY_COL]
    .str.lower()
    .eq("medium")
    .sum()
)

hard_count = (
    df[DIFFICULTY_COL]
    .str.lower()
    .eq("hard")
    .sum()
)

st.sidebar.metric("Easy", easy_count)
st.sidebar.metric("Medium", medium_count)
st.sidebar.metric("Hard", hard_count)

# ----------------------------------
# RECOMMENDATION FUNCTION
# ----------------------------------
def recommend(question_name, n=5):

    match = df[
        df[TITLE_COL]
        .astype(str)
        .str.lower()
        ==
        question_name.lower()
    ]

    if len(match) == 0:
        return None

    idx = match.index[0]

    distances, indices = model.kneighbors(
        X[idx],
        n_neighbors=n + 1
    )

    recommendations = []

    for i in range(1, len(indices[0])):

        rec_idx = indices[0][i]

        recommendations.append({
            "Question":
                df.iloc[rec_idx][TITLE_COL],

            "Difficulty":
                df.iloc[rec_idx][DIFFICULTY_COL],

            "Similarity (%)":
                round(
                    (1 - distances[0][i]) * 100,
                    2
                )
        })

    return pd.DataFrame(
        recommendations
    )

# ----------------------------------
# MAIN UI
# ----------------------------------
st.subheader("Select a Question")

question = st.selectbox(
    "Question Name",
    sorted(
        df[TITLE_COL]
        .dropna()
        .astype(str)
        .unique()
    )
)

num_recommendations = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Get Recommendations"):

    result = recommend(
        question,
        num_recommendations
    )

    if result is not None:

        st.success(
            "Recommendations Generated!"
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.subheader(
            "Recommended Questions"
        )

        for _, row in result.iterrows():

            st.markdown(
                f"""
### 📌 {row['Question']}

**Difficulty:** {row['Difficulty']}

**Similarity Score:** {row['Similarity (%)']}%
"""
            )

# ----------------------------------
# DATASET PREVIEW
# ----------------------------------
with st.expander("View Dataset"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )
