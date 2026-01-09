import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("dataset/winequality-red.csv", sep=";")
X = data.drop("quality", axis=1)
y = data["quality"]


os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/model.pkl")

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)







from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("EXP-02 Ridge Regression + Standardization")
print("MSE:", mse)
print("R2:", r2)

results = {
    "Experiment": "EXP-02",
    "Model": "Ridge Regression",
    "Alpha": 1.0,
    "MSE": mse,
    "R2": r2
}
