import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from splitcopy import custom_train_test_split

# Load dataset
df = pd.read_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/encoded_coffee_data_cleaned.xlsx")

# Ensure correct types
df.replace({'TRUE': 1, 'FALSE': 0}, inplace=True)

# Split data
X_train, X_test, y_train, y_test = custom_train_test_split(df, 'Total.Cup.Points', test_size=0.2, seed=1898)

# Convert to numpy arrays
X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

# Standardize for SVR and Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}  MSE: {mse:.4f}, R²: {r2:.4f}")

# 1. Linear Regression (Baseline)
linear_model = LinearRegression()
evaluate(linear_model, X_train, X_test, y_train, y_test, "Linear Regression")

# 2. Support Vector Regressor (SVR with RBF kernel)
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
evaluate(svr_model, X_train_scaled, X_test_scaled, y_train, y_test, "SVR (RBF Kernel)")

# 3. Neural Network (MLP Regressor)
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
evaluate(mlp_model, X_train_scaled, X_test_scaled, y_train, y_test, "Neural Network (MLP)")