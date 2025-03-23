import pandas as pd

df = pd.read_csv("/Users/prabhpreet16/Documents/IML-Project/merged_data_cleaned.csv")

# Assuming your DataFrame is named df
columns_to_drop = [
    'Unnamed: 0', 'Owner', 'Farm.Name', 'Lot.Number', 'Mill', 'ICO.Number', 'Company', 'Altitude','Producer',
    'Number.of.Bags','Region', 'Bag.Weight', 'In.Country.Partner', 'Harvest.Year', 'Grading.Date', 'Owner.1',
    'Expiration', 'Certification.Body', 'Certification.Address', 'Certification.Contact', 'unit_of_measurement', 
    'altitude_low_meters', 'altitude_high_meters',
    
    # Redundant numeric features contributing to Total.Cup.Points
    'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance',
    'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points'
]

# Drop columns from the DataFrame
processed_df = df.drop(columns=columns_to_drop)

# Save the processed DataFrame to a new spreadsheet
processed_df.to_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/pre-processed.xlsx", index=False)

print("Remaining columns:", processed_df.columns)