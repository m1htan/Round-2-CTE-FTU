import pandas as pd

df = pd.read_csv('../../data/cleaned_all_stocks.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(df.head())

print(df.shape)

print(df.info())

missing_summary = df.isna().sum
print("Số giá trị thiếu:\n", missing_summary)
print(missing_summary)

# Data Formatting - Check the data types
print("\nKiểu dữ liệu:\n", df.dtypes)

# Descriptive statistics
print("Thống kê mô tả:")
print(df.describe())