import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/encoded_coffee_data.xlsx")

altitude = df["altitude_mean_meters"]
altitudenp = altitude.to_numpy()
estimateMean= np.sum(altitude) / np.size(altitudenp) #Using MLE
estimateSTD = np.sqrt(np.sum(np.square(altitudenp - estimateMean)) / np.size(altitudenp)) 
plt.hist(altitude,bins=50,color='green', edgecolor='black', density=True)

plt.xlabel('Altitude (meters)')
plt.ylabel("Density")
plt.title("Altitude Distribution")
plt.show()

z_scores = (altitudenp - estimateMean) / estimateSTD
outliers = altitudenp[np.abs(z_scores) > 3]
print("Outliers detected:", outliers)

df = df[df["altitude_mean_meters"] <= 10000]
#Outliers detected: [190164. 110000. 190164.] (BIG OOF)

print (estimateMean)
print (estimateSTD)

plt.hist(df["altitude_mean_meters"], bins=50, color='blue', edgecolor='black', density=True)
plt.xlabel('Altitude (meters)')
plt.ylabel("Density")
plt.title("Altitude Distribution (Post-Outlier Handling)")
plt.show()

df.to_excel("/Users/prabhpreet16/Documents/IML-Project/Preprocess/encoded_coffee_data_cleaned.xlsx", index=False)

print("Outliers removed. Updated dataset saved!")