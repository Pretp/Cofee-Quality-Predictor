import numpy as np
from randomforest import RandomForestRegressor
from train_test_split import custom_train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



# Load and preprocess the dataset
df = pd.read_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/encoded_coffee_data_cleaned.xlsx")
df.replace({'TRUE': 1, 'FALSE': 0}, inplace=True)

best_mse = float('inf')
best_seed = None

    
# Split the dataset using the current seed
X_train, X_test, y_train, y_test = custom_train_test_split(df, 'Total.Cup.Points', test_size=0.2, seed=1898)
    
# Ensure boolean values are replaced
X_train = X_train.replace({True: 1, False: 0})
X_test = X_test.replace({True: 1, False: 0})


# Function to group rare categories in one-hot encoded columns
def group_rare_onehot(df, prefix, threshold=0.05):
    # Find all columns that match the given prefix
    matching_cols = [col for col in df.columns if col.startswith(prefix)]

    # Calculate frequency of each category (sum of 1s across rows)
    category_counts = df[matching_cols].sum()

    # Identify rare categories (appear in less than 'threshold' proportion)
    rare_categories = category_counts[category_counts < threshold * len(df)].index.tolist()

    if rare_categories:
        # Create a new "Other" category by summing rare categories
        df[f"{prefix}_Other"] = df[rare_categories].sum(axis=1)

        # Drop the original rare columns
        df.drop(columns=rare_categories, inplace=True)

    print(f"Grouped {len(rare_categories)} rare categories for {prefix}")

# Apply to Country.of.Origin and Variety one-hot encoded columns
group_rare_onehot(X_train, 'Country.of.Origin')
group_rare_onehot(X_test, 'Country.of.Origin')  # Ensure train-test consistency

group_rare_onehot(X_train, 'Variety')
group_rare_onehot(X_test, 'Variety')  # Ensure train-test consistency

# Verify changes
print(X_train.head())


# Extract feature names and convert to numpy arrays
feature_names = X_train.columns
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(n_base_learner=140, max_depth=10, min_mse_reduction=0.001, min_samples_leaf=2)

# Train the model
rf.train(X_train, y_train)

# Predict on both training and test sets
y_pred = rf.predict(X_test)
y_pred_train = rf.predict(X_train)

    # Evaluate model performance
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)




print(f"MSE : {mse_test:.4f}")
print(f"R²: {r2_test:.4f}")
print(f"mse (train): {mse_train:.4f}")
print(f"R² Score (train): {r2_train:.4f}")

