"""
Generates a synthetic 'Mall Customers'-style dataset for segmentation.
Replace data/customers.csv with your own real dataset (same column names)
if you have one — the main script will work unchanged.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

genders = np.random.choice(["Male", "Female"], size=n, p=[0.45, 0.55])
age = np.random.randint(18, 70, size=n)

# Create a few natural-looking customer archetypes so clusters are meaningful
income = []
spending = []
for a in age:
    if a < 30:
        inc = np.random.normal(40, 15)
        spd = np.random.normal(70, 15)
    elif a < 45:
        inc = np.random.normal(75, 20)
        spd = np.random.normal(50, 20)
    else:
        inc = np.random.normal(60, 18)
        spd = np.random.normal(35, 15)
    income.append(max(15, inc))
    spending.append(min(100, max(1, spd)))

df = pd.DataFrame({
    "CustomerID": range(1, n + 1),
    "Gender": genders,
    "Age": age,
    "AnnualIncome_k$": np.round(income, 1),
    "SpendingScore": np.round(spending, 0).astype(int)
})

df.to_csv("data/customers.csv", index=False)
print("Saved data/customers.csv with", len(df), "rows")
