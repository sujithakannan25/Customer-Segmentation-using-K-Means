# Customer Segmentation using K-Means

A ready-to-run project that segments customers into groups based on
**Age**, **Annual Income**, and **Spending Score** using K-Means clustering.

## Folder structure
```
customer_segmentation_kmeans/
├── data/
│   └── customers.csv          # sample dataset (200 synthetic customers)
├── outputs/                   # generated after running the script
│   ├── elbow_silhouette.png
│   ├── clusters_income_vs_spending.png
│   ├── clusters_age_vs_spending.png
│   ├── cluster_profiles.csv
│   └── segmented_customers.csv
├── generate_data.py           # (optional) regenerates the sample dataset
├── customer_segmentation.py   # main script — run this
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python customer_segmentation.py
```

This will:
1. Load `data/customers.csv`
2. Scale the features (`StandardScaler`)
3. Test k = 2 to 10 clusters, plotting the **Elbow curve** and
   **Silhouette Score** to `outputs/elbow_silhouette.png`
4. Automatically pick the k with the best silhouette score, fit K-Means
5. Save 2D visualizations (Income vs Spending, Age vs Spending)
6. Save `outputs/cluster_profiles.csv` (average feature values per segment)
7. Save `outputs/segmented_customers.csv` (original data + `Cluster` column)

## Using your own data
Replace `data/customers.csv` with your real customer data. Keep the same
column names, or edit the `FEATURES` list near the top of
`customer_segmentation.py` to match your columns (e.g. `RecencyDays`,
`Frequency`, `MonetaryValue` for RFM segmentation).

## Choosing the number of clusters manually
By default the script auto-selects k using silhouette score. To force a
specific number of segments (e.g. 4, a common choice for
"budget / average / premium / VIP" segments), open
`customer_segmentation.py` and change this line in `main()`:
```python
k = best_k        # <- change to k = 4 (or any number you want)
```

## Requirements
See `requirements.txt` — pandas, numpy, scikit-learn, matplotlib, seaborn.

## 👤 Author:

[sujitha kannan] 

📧 Email: k.sujithakannane2006@gmail.com 

🔗 LinkedIn: www.linkedin.com/in/sujithakannan25

