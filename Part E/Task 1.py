import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

# Load the dataset
file_path = 'Data\FuelConsumption.csv'
data = pd.read_csv(file_path)

# Drop specific columns; ensure these columns exist to avoid KeyErrors
columns_to_drop = ['Year', 'MODEL', 'VEHICLE CLASS', 'TRANSMISSION', 'FUEL']
data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)

# Display basic information about the dataset
print("Basic Dataset Information:")
print(data.info())

# Describe the dataset
print("\nDataset Description:")
print(data.describe())

# Check for missing values
print("\nMissing Values in Each Column:")
missing_values = data.isnull().sum()
print(missing_values)

# Handle missing values by filling them with the median (for numerical data)
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column] = data[column].fillna(data[column].median())

# Check for duplicates
print("\nDuplicate Rows in the Dataset:")
print(data.duplicated().sum())

# Remove duplicates
data = data.drop_duplicates()

# Feature selection and correlation analysis
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualizations
numeric_data.hist(figsize=(20, 15))
plt.suptitle('Histograms of Numerical Features')
plt.show()

numeric_data.plot(kind='box', subplots=True, layout=(ceil(len(numeric_data.columns) / 3), 2), figsize=(20, 20))
plt.suptitle('Boxplot of Numerical Features')
plt.show()

# Scatter plot analysis
if 'ENGINE SIZE' in data.columns and 'FUEL CONSUMPTION' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='ENGINE SIZE', y='FUEL CONSUMPTION', data=data)
    plt.title('ENGINE SIZE vs FUEL CONSUMPTION')
    plt.xlabel('ENGINE SIZE (litres)')
    plt.ylabel('FUEL CONSUMPTION (litres/100km)')
    plt.show()

# Visualizing the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix Heatmap')
plt.show()