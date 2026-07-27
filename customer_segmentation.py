"""
Customer Segmentation using K-Means Clustering
------------------------------------------------
Steps:
1. Load customer data (data/customers.csv)
2. Explore & preprocess (scale numeric features)
3. Find optimal number of clusters (Elbow Method + Silhouette Score)
4. Fit K-Means and assign cluster labels
5. Visualize clusters
6. Save labeled data + charts to outputs/

Usage:
    python customer_segmentation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import os

DATA_PATH = "data/customers.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Features used for clustering — change these to match your own dataset's columns
FEATURES = ["Age", "AnnualIncome_k$", "SpendingScore"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print("Loaded data:", df.shape)
    print(df.head())
    return df


def preprocess(df, features=FEATURES):
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Elbow plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(list(k_range), inertias, marker="o", color="tab:blue")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(list(k_range), sil_scores, marker="s", color="tab:red")
    ax2.set_ylabel("Silhouette Score", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Elbow Method & Silhouette Score vs k")
    fig.tight_layout()
    plt.savefig(f"{OUT_DIR}/elbow_silhouette.png", dpi=150)
    plt.close()

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"Silhouette-suggested k: {best_k}")
    return best_k, inertias, sil_scores


def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def visualize_clusters(df, labels, k):
    df = df.copy()
    df["Cluster"] = labels

    # Pairwise scatter: Income vs Spending Score (classic mall-customer view)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="AnnualIncome_k$", y="SpendingScore",
        hue="Cluster", palette="tab10", s=70
    )
    plt.title(f"Customer Segments (k={k}): Income vs Spending Score")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clusters_income_vs_spending.png", dpi=150)
    plt.close()

    # Age vs Spending Score
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="Age", y="SpendingScore",
        hue="Cluster", palette="tab10", s=70
    )
    plt.title(f"Customer Segments (k={k}): Age vs Spending Score")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clusters_age_vs_spending.png", dpi=150)
    plt.close()

    # Cluster profile summary (mean of each feature per cluster)
    profile = df.groupby("Cluster")[FEATURES].mean().round(1)
    profile["CustomerCount"] = df.groupby("Cluster").size()
    profile.to_csv(f"{OUT_DIR}/cluster_profiles.csv")
    print("\nCluster profile summary:")
    print(profile)

    return df


def main():
    df = load_data()
    X_scaled, scaler = preprocess(df)
    best_k, inertias, sil_scores = find_optimal_k(X_scaled)

    # You can override best_k manually below if you want a specific number of segments
    k = best_k
    km, labels = run_kmeans(X_scaled, k)
    labeled_df = visualize_clusters(df, labels, k)

    labeled_df.to_csv(f"{OUT_DIR}/segmented_customers.csv", index=False)
    print(f"\nDone. Outputs saved in '{OUT_DIR}/':")
    print(" - elbow_silhouette.png")
    print(" - clusters_income_vs_spending.png")
    print(" - clusters_age_vs_spending.png")
    print(" - cluster_profiles.csv")
    print(" - segmented_customers.csv")


if __name__ == "__main__":
    main()
