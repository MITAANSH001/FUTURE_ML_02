import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


st.set_page_config(page_title="InsightDesk AI — Dashboard", layout="wide")


@st.cache_data
def load_dataset():
    # try common filenames
    candidates = [
        "customer_support_tickets.csv",
        "all_tickets_processed_improved_v3.csv",
        os.path.join("data", "customer_support_tickets.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                try:
                    return pd.read_csv(p, encoding="utf-8", engine="python")
                except Exception:
                    pass
    return None


def sidebar_controls():
    st.sidebar.title("Controls")
    show_raw = st.sidebar.checkbox("Show raw data", value=False)
    sample_n = st.sidebar.slider("Sample rows to show", 5, 200, 25)
    return show_raw, sample_n


def train_model(df, text_col="text", label_col="category"):
    X = df[text_col].fillna("")
    y = df[label_col].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return pipe, acc


def main():
    st.title("InsightDesk AI — Interactive Dashboard")

    df = load_dataset()
    if df is None:
        st.warning("No dataset found in repo root. Upload a CSV or place your dataset at `customer_support_tickets.csv`.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
    if df is None:
        return

    show_raw, sample_n = sidebar_controls()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Dataset sample")
        st.dataframe(df.head(sample_n))
        if show_raw:
            st.write(df)

    with col2:
        st.subheader("Quick stats")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

    # Attempt to find categorical columns
    possible_text = None
    for c in df.columns:
        if c.lower() in ("text", "message", "description", "body", "subject"):
            possible_text = c
            break
    if possible_text is None:
        # fallback to first string column
        for c in df.columns:
            if df[c].dtype == object:
                possible_text = c
                break

    colA, colB = st.columns(2)
    with colA:
        if "category" in df.columns:
            cat_counts = df["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.bar(cat_counts, x="category", y="count", title="Category Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        if "priority" in df.columns:
            pri_counts = df["priority"].value_counts().reset_index()
            pri_counts.columns = ["priority", "count"]
            fig2 = px.bar(pri_counts, x="priority", y="count", title="Priority Distribution")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Model training / prediction
    st.header("Train a simple Category Model (optional)")
    if possible_text is None or "category" not in df.columns:
        st.info("Dataset does not contain both text and category columns; training is disabled.")
    else:
        st.write(f"Text column: **{possible_text}**, Label column: **category**")
        if st.button("Train model (quick)"):
            with st.spinner("Training model..."):
                try:
                    model, acc = train_model(df, text_col=possible_text, label_col="category")
                    os.makedirs("models", exist_ok=True)
                    joblib.dump(model, "models/model.joblib")
                    st.success(f"Model trained and saved — accuracy on test set: {acc:.3f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")
    st.header("Single Prediction")
    model_path = "models/model.joblib"
    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            model = None

    if model is None:
        st.info("No trained model found. Train a model above or upload a pre-trained `models/model.joblib` file.")
    text_input = st.text_area("Enter text to predict category", height=120)
    if st.button("Predict"):
        if model is None:
            st.warning("No model available")
        else:
            pred = model.predict([text_input])[0]
            st.success(f"Predicted category: {pred}")

    st.markdown("---")
    st.header("Batch Prediction")
    uploaded_batch = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
    if uploaded_batch is not None and model is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch)
            textcol = possible_text if possible_text in batch_df.columns else st.selectbox("Select text column", [c for c in batch_df.columns if batch_df[c].dtype == object])
            batch_df["predicted_category"] = model.predict(batch_df[textcol].fillna(""))
            st.dataframe(batch_df.head(200))
            st.download_button("Download predictions CSV", batch_df.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


if __name__ == "__main__":
    main()
