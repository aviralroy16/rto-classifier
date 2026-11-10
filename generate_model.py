import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Generate synthetic data matching your pipeline
rng = np.random.default_rng(42)
n = 5000
pin_list = np.arange(110001, 110061)
pin_risk = dict(zip(pin_list, rng.uniform(0.0, 0.25, len(pin_list))))

start = pd.Timestamp("2025-01-01")
dates = start + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="m")

df = pd.DataFrame({
    "order_date": np.sort(dates),
    "order_value": rng.gamma(3, 300, n).round(0).clip(199, 5000),
    "payment_mode": rng.choice(["COD", "Prepaid"], n, p=[0.65, 0.35]),
    "pincode": rng.choice(pin_list, n),
    "is_repeat_customer": rng.choice([0, 1], n, p=[0.75, 0.25]),
    "address_length": rng.integers(10, 120, n),
    "discount_used": rng.choice([0, 1], n, p=[0.6, 0.4]),
})
df["order_hour"] = df["order_date"].dt.hour

logit = (
    -4.0
    + 2.0 * (df["payment_mode"] == "COD")
    + 0.0006 * (df["order_value"] - 800)
    - 1.0 * df["is_repeat_customer"]
    + 0.8 * (df["address_length"] < 25)
    + 6 * df["pincode"].map(pin_risk)
    + 0.3 * (df["order_hour"] <= 5)
)
prob = 1 / (1 + np.exp(-logit))
df["rto"] = (rng.random(n) < prob).astype(int)

# 2. Feature Engineering
df["addr_len"] = df["address_length"]
df["addr_words"] = 5
df["has_missing_number"] = 0
df["has_filler"] = 0
stats = df.groupby("pincode")["rto"].mean()
global_rate = df["rto"].mean()
df["pin_rto_rate"] = df["pincode"].map(stats).fillna(global_rate)

features = [
    "order_amount" if "order_amount" in df.columns else "order_value", # handled below
    "order_hour",
    "addr_len",
    "addr_words",
    "has_missing_number",
    "has_filler",
    "pin_rto_rate",
]

# Quick adjustment to match payload key name exactly
df["order_amount"] = df["order_value"]
features[0] = "order_amount"

X = df[features]
y = df["rto"]

# 3. Train Model
model = LogisticRegression()
model.fit(X, y)

# 4. Save Artifacts
joblib.dump(model, "rto_logreg.joblib")
joblib.dump(
    {
        "pin_table": stats.to_dict(),
        "global_rate": float(global_rate),
        "features": features,
    },
    "rto_meta.joblib",
)
print("Model artifacts generated successfully!")
