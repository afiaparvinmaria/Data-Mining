import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

data = pd.read_csv('/content/diabetes.csv')



z_scores = data.select_dtypes(include=[np.number]).apply(zscore)
outliers_z = data[(z_scores < -3.5).any(axis=1) | (z_scores > 4).any(axis=1)]

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = data[((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.figure(figsize=(15, 8))
plt.boxplot([data[col] for col in data.select_dtypes(include=[np.number]).columns], vert=False)
plt.yticks(range(1, len(data.select_dtypes(include=[np.number]).columns) + 1), data.select_dtypes(include=[np.number]).columns)
plt.title('Boxplot for Outlier Detection')
plt.show()

print(f'Z-Score Method Outliers: {outliers_z.shape[0]}')
print(f'Boxplot (IQR) Method Outliers: {outliers_iqr.shape[0]}')
