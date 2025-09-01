"""labtask3ID509


Original file is located at
    https://colab.research.google.com/drive/1ecXFnhFkK66chPRm2pBT6IMVkbwe8zuA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace filename with your dataset)
df = pd.read_csv("/content/AirQualityUCI.csv",sep=";")

# Show first 5 rows
df.head()

print("Missing values before cleaning:")
print(df.isnull().sum())

# Fix column names (RH = col 13, AH = col 14)
df.rename(columns={df.columns[13]: "RH", df.columns[14]: "AH"}, inplace=True)

# If there are extra useless columns (like Unnamed: 15, 16), drop them
cols_to_drop = [col for col in df.columns if "Unnamed" in col]
df = df.drop(cols_to_drop, axis=1)

# Convert columns that may contain commas as decimal separators
cols_to_convert = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']  # adjust if dataset has more
for col in cols_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)  # replace ',' with '.'
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert to numbers

# Fill missing values in numeric columns with column mean
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nColumns after cleaning & renaming:")
print(df.columns)

"""***Choosing the target column***"""

X = df.drop(columns=['RH'])   # features = everything except RH
y = df['RH']                  # target = RH


print("Features shape:", X.shape)
print("Target shape:", y.shape)

"""***Correlation Analysis***"""

import matplotlib.pyplot as plt
import seaborn as sns

# Exclude non-numeric columns like 'Date' and 'Time' for correlation
df_numeric = df.drop(columns=['Date', 'Time'])

# Correlation matrix of all numeric features
plt.figure(figsize=(12,8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of All Features")
plt.show()

# Correlation of features with the target column
correlation_with_target = df_numeric.corr()[y.name].sort_values(ascending=False)
print(f"\nCorrelation of features with {y.name}:\n")
print(correlation_with_target)

# Identify highly positive and highly negative features
highly_positive = correlation_with_target[correlation_with_target > 0.5]
highly_negative = correlation_with_target[correlation_with_target < -0.5]

print(f"\nHighly positive features with {y.name}:\n", highly_positive)
print(f"\nHighly negative features with {y.name}:\n", highly_negative)

"""***Train Test split & Linear Regression Model***"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#  Use numeric features only (exclude 'Date' and 'Time')
X_numeric = df_numeric.drop(columns=[y.name])
y_numeric = df_numeric[y.name]

#  Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y_numeric, test_size=0.2, random_state=42
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#  Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#  Evaluate performance
print(f"Train Mean Squared Error (MSE): {mean_squared_error(y_train, y_train_pred):.2f}")
print(f"Test Mean Squared Error (MSE): {mean_squared_error(y_test, y_test_pred):.2f}")
print(f"Train R^2 Score: {r2_score(y_train, y_train_pred):.2f}")
print(f"Test R^2 Score: {r2_score(y_test, y_test_pred):.2f}")

"""***Residual Analysis to check how far off predictions are from the actual values***"""

import matplotlib.pyplot as plt

# Step 6: Residual Analysis

plt.figure(figsize=(10,6))

# Plot residuals for training data
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')

# Plot residuals for testing data
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')

# Draw horizontal line at 0 (perfect prediction)
plt.axhline(y=0, color='k', linewidth=2)

plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Predicted - Actual)')
plt.title(f'Residual Analysis for {y.name}')
plt.legend()
plt.show()

"""***Features VS Target Visualization***"""

import seaborn as sns
import matplotlib.pyplot as plt

# Step 7: Simple plots for each feature vs target
for col in X_numeric.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df_numeric[col], y=y_numeric)
    sns.regplot(x=df_numeric[col], y=y_numeric, scatter=False, color='red')  # regression line
    plt.xlabel(col)
    plt.ylabel(y.name)
    plt.title(f"{col} vs {y.name}")
    plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', lw=2, label='Perfect prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual RH')
plt.legend()
plt.show()
