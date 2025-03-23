import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("/Users/prabhpreet16/Documents/IML-Project/merged_data_cleaned.csv")

df['Harvest.Year'] = pd.to_numeric(df['Harvest.Year'], errors='coerce')
numeric_df = df.select_dtypes(include='number')

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Set up the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('Correlation Heatmap of Coffee Quality Dataset')
plt.show()

# Correlation of all features with Total.Cup.Points
print(corr_matrix['Total.Cup.Points'].sort_values(ascending=False))

"""Total.Cup.Points        1.000000
Flavor                  0.874279
Aftertaste              0.860656
Balance                 0.828502
Acidity                 0.797024
Aroma                   0.791627
Cupper.Points           0.790217
Body                    0.757165
Clean.Cup               0.658859
Uniformity              0.656454
Sweetness               0.554029
Quakers                 0.013263
altitude_mean_meters   -0.013771
Moisture               -0.117722
Category.One.Defects   -0.130009
Category.Two.Defects   -0.211085"""