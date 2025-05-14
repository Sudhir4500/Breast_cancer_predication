import pandas as pd

# Load the dataset
df = pd.read_csv('data/data.csv')
print("Columns:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)
print("\nRows with any NaN values:")
print(df[df.isnull().any(axis=1)])