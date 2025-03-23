import pandas as pd

df = pd.read_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/pre-processed.xlsx")# Drop rows with missing values
cleaned_df = df.dropna()

# Save the cleaned DataFrame to a new spreadsheet
cleaned_df.to_excel("cleaned_coffee_data.xlsx", index=False)

print(f"Rows after dropping missing values: {cleaned_df.shape[0]}")