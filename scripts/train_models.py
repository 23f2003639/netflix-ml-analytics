import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/netflix_predictive_dataset.csv")

features = [
    "revenue_million",
    "subscriptions_million",
    "marketing_spend_million",
    "ad_spend_million",
    "customer_satisfaction",
    "competitor_price_index",
    "gdp_growth"
]

# REGRESSION
X = df[features]
y_reg = df["stock_price_next_month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("=== Regression Metrics ===")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# CLASSIFICATION
y_cls = df["performance_up_next_month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred_cls = clf.predict(X_test)

print("\n=== Classification Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred_cls))
print("Precision:", precision_score(y_test, y_pred_cls))
print("Recall:", recall_score(y_test, y_pred_cls))
print("F1:", f1_score(y_test, y_pred_cls))
