"""
Streamlit app for Customer Segmentation using K-Means.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation (K-Means)", layout="wide")

st.title("🛍️ Customer Segmentation using K-Means")
st.write(
    "Upload your own customer CSV, or use the built-in sample data. "
    "Pick the features to cluster on and the number of segments (k), "
    "and explore the results interactively."
)

# ---------- Load data ----------
st.sidebar.header("1. Data")
uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/customers.csv")
    st.sidebar.caption("Using built-in sample dataset (data/customers.csv)")

st.subheader("Preview of data")
st.dataframe(df.head(10), use_container_width=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns to cluster on.")
    st.stop()

# ---------- Feature selection ----------
st.sidebar.header("2. Features")
default_feats = [c for c in ["Age", "AnnualIncome_k$", "SpendingScore"] if c in numeric_cols]
features = st.sidebar.multiselect(
    "Select features for clustering",
    options=numeric_cols,
    default=default_feats if default_feats else numeric_cols[:2],
)

if len(features) < 2:
    st.warning("Select at least 2 features to continue.")
    st.stop()

X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- k selection ----------
st.sidebar.header("3. Number of clusters (k)")
max_k = min(10, len(X) - 1)
k = st.sidebar.slider("k", min_value=2, max_value=max_k, value=min(4, max_k))

show_elbow = st.sidebar.checkbox("Show elbow / silhouette chart", value=True)

# ---------- Elbow & Silhouette ----------
if show_elbow:
    k_range = range(2, max_k + 1)
    inertias, sil_scores = [], []
    for kk in k_range:
        km_tmp = KMeans(n_clusters=kk, random_state=42, n_init=10)
        labels_tmp = km_tmp.fit_predict(X_scaled)
        inertias.append(km_tmp.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels_tmp))

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(list(k_range), inertias, marker="o")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia (WCSS)")
        ax1.set_title("Elbow Method")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(list(k_range), sil_scores, marker="s", color="tab:red")
        ax2.set_xlabel("k")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score")
        st.pyplot(fig2)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    st.info(f"Silhouette-suggested optimal k: **{best_k}**")

# ---------- Fit K-Means ----------
km = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

result_df = df.loc[X.index].copy()
result_df["Cluster"] = labels

st.subheader(f"Segmentation result (k={k})")

# ---------- Scatter plot ----------
col_a, col_b = st.columns(2)
with col_a:
    x_axis = st.selectbox("X-axis", features, index=0)
with col_b:
    y_axis = st.selectbox("Y-axis", features, index=min(1, len(features) - 1))

fig3, ax3 = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=result_df, x=x_axis, y=y_axis, hue="Cluster", palette="tab10", s=70, ax=ax3
)
ax3.set_title(f"Clusters: {x_axis} vs {y_axis}")
st.pyplot(fig3)

# ---------- Cluster profile ----------
st.subheader("Cluster profile (average values)")
profile = result_df.groupby("Cluster")[features].mean().round(1)
profile["CustomerCount"] = result_df.groupby("Cluster").size()
st.dataframe(profile, use_container_width=True)

# ---------- Download ----------
st.subheader("Download segmented data")
csv_bytes = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download segmented_customers.csv",
    data=csv_bytes,
    file_name="segmented_customers.csv",
    mime="text/csv",
)
