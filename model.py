import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel('personal finance data 2.xlsx')

# convert date to string
df['Date'] = df['Date / Time'].dt.to_period('M').astype(str)

# one hot encoding of categorical cols
df = pd.get_dummies(df, columns=['Category', 'Sub category', 'Mode', 'Income/Expense'])

# drop date cols
df.drop(columns=['Date / Time', 'Date'], inplace=True)

# features and target
X = df.drop(columns=['Debit/Credit'])
y = df['Debit/Credit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Random forest regressor model ----
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# example inp for pred
example_in = X_test.iloc[2]
pred_expense = model.predict([example_in])

print(f"\nPredicted expense for example input: {pred_expense[0]:.2f}")