import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("EXP-01 Metrics")
print("MSE:", mse)
print("R2:", r2)

os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/model.pkl")

with open("output/results.json", "w") as f:
    json.dump({"MSE": mse, "R2": r2}, f, indent=4)

