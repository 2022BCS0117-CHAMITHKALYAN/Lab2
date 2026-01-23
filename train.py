import os, json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
import joblib

os.makedirs("output", exist_ok=True)

# Load red wine dataset
df = pd.read_csv("winequality-red.csv", sep=";")

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

# Metrics
f1 = float(f1_score(y_test, pred, average="weighted"))
mse = float(mean_squared_error(y_test, pred))

# Save model
joblib.dump(model, "output/model.pkl")

# Save metrics
metrics = {"f1": round(f1, 4), "mse": round(mse, 4)}
with open("output/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Training complete.")
print("F1:", f1, "MSE:", mse)
