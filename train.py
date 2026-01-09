# ===============================
# EXP-07: Random Forest Regressor (Deeper Trees)
# ===============================

import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load dataset
# -------------------------------
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

# -------------------------------
# 2. Define features and target
# -------------------------------
X = data.drop("quality", axis=1)
y = data["quality"]

# -------------------------------
# 3. Train-test split (80/20)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Train model (Random Forest – Deeper Trees)
# -------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("EXP-07: Random Forest Regressor (100 Trees, Depth 15)")
print("MSE:", mse)
print("R2 Score:", r2)

# -------------------------------
# 6. Save outputs
# -------------------------------
os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/model.pkl")

results = {
    "Experiment": "EXP-07",
    "Model": "Random Forest Regressor",
    "Trees": 100,
    "Max Depth": 15,
    "Train/Test Split": "80/20",
    "MSE": mse,
    "R2_Score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)
