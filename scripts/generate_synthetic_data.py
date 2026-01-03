"""
Synthetic Netflix Dataset Generator

Generates a synthetic dataset for analyzing revenue, subscriptions,marketing spend, and predicting stock performance.

Data is artificially generated for educational and portfolio use.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

rows = 1000
dates = pd.date_range(start="2015-01-01", periods=rows, freq="M")

df = pd.DataFrame({
    "date": dates,
    "year": dates.year,
    "month": dates.month,
    "revenue_million": np.round(8000 + np.random.normal(0, 300, rows) + np.linspace(0, 4000, rows), 2),
    "subscriptions_million": np.round(75 + np.random.normal(0, 2, rows) + np.linspace(0, 70, rows), 2),
    "marketing_spend_million": np.round(np.random.uniform(250, 600, rows), 2),
    "ad_spend_million": np.round(np.random.uniform(100, 300, rows), 2),
    "avg_watch_time_hours": np.round(np.random.uniform(1.5, 4.5, rows), 2),
    "customer_satisfaction": np.round(np.random.uniform(3.5, 5.0, rows), 2),
    "competitor_price_index": np.round(np.random.uniform(0.9, 1.3, rows), 2),
    "gdp_growth": np.round(np.random.uniform(-1.5, 6.5, rows), 2),
})

df["orders"] = (df["subscriptions_million"] * 1_000_000 / 120).astype(int)
df["average_order_value"] = (df["revenue_million"] * 1_000_000 / df["orders"]).round(2)

df["content_genre"] = np.random.choice(
    ["Movies", "Series", "Documentary", "Kids", "Reality"], size=len(df)
)

df["subscription_type"] = np.random.choice(
    ["Basic", "Standard", "Premium"], size=len(df)
)

df["region"] = np.random.choice(
    ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"],
    size=len(df)
)

# Stock price simulation
stock_price = []
current_price = 350

for i in range(len(df)):
    change = (
        0.0008 * (df["revenue_million"][i] - 8000)
        + 0.5 * (df["subscriptions_million"][i] - 75)
        + 1.2 * (df["customer_satisfaction"][i] - 4.2)
        - 0.003 * df["ad_spend_million"][i]
        - 10 * (df["competitor_price_index"][i] - 1)
        + 2.5 * df["gdp_growth"][i]
        + np.random.normal(0, 3)
    ) / 10

    current_price = max(50, current_price + change)
    stock_price.append(round(current_price, 2))

df["stock_price"] = stock_price
df["stock_price_next_month"] = df["stock_price"].shift(-1)
df["performance_up_next_month"] = (df["stock_price_next_month"] > df["stock_price"]).astype(int)

df = df.iloc[:-1]

df.to_csv("data/netflix_predictive_dataset.csv", index=False)
print("Dataset generated successfully.")
