import pandas as pd

df = pd.read_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/cleaned_coffee_data.xlsx")

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Display categorical columns
print("Categorical columns:", categorical_columns.tolist())

# Apply one-hot encoding
encoded_df = pd.get_dummies(df, columns=categorical_columns)

# Save the encoded DataFrame to a new spreadsheet
encoded_df.to_excel("encoded_coffee_data.xlsx", index=False)

print("Encoding complete. New columns:", encoded_df.columns)