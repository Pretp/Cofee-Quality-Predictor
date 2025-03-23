#This is the same custom split function as the one in model_training

import pandas as pd
import random

def custom_train_test_split(df, target_column, test_size=0.2, seed=None):
    if seed is not None:
        random.seed(seed)

    # Shuffle dataset
    indices = list(df.index)
    random.shuffle(indices)

    # Calculate split point
    split_point = int(len(df) * (1 - test_size))

    # Split indices
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    # Create train and test sets
    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    # Separate features and targets
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test

# Usage example
df = pd.read_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/encoded_coffee_data_cleaned.xlsx")
X_train, X_test, y_train, y_test = custom_train_test_split(df, 'Total.Cup.Points', test_size=0.2, seed=42)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
