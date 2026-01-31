import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Load dataset (change file name if different)
df = pd.read_csv("Advertising.csv")

# ✅ Remove unwanted index column if exists
df.columns = df.columns.str.strip()
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

print("✅ Shape:", df.shape)
print("\n✅ Columns:", df.columns.tolist())
print("\n✅ First 5 rows:\n", df.head())

# ✅ Features & Target
X = df.drop("Sales", axis=1)
y = df["Sales"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Predict
y_pred = model.predict(X_test)

# ✅ Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Performance:")
print("MAE  =", mae)
print("RMSE =", rmse)
print("R²   =", r2)

# ✅ Plot: Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

# ✅ Custom Prediction
print("\n✅ Custom Sales Prediction")
tv = float(input("Enter TV advertising budget: "))
radio = float(input("Enter Radio advertising budget: "))
news = float(input("Enter Newspaper advertising budget: "))

custom = np.array([[tv, radio, news]])
pred_sales = model.predict(custom)[0]

print("\n📌 Predicted Sales:", pred_sales)
