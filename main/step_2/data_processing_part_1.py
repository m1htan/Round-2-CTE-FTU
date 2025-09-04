import pandas as pd

df = pd.read_csv('../../data/step_2/HNX_ohlcv_with_fundamentals.csv')

cols_to_drop = df.columns[df.isna().all()].tolist()
print("Các cột sẽ bị xóa:", cols_to_drop)

# Xóa các cột null (null toàn bộ mới xóa)
df = df.dropna(axis=1, how='all')

# xóa 200 dòng đầu tiên (trừ header)
df = df.iloc[200:].reset_index(drop=True)

print(df.head())

df.to_csv('../../data/step_2/part_1_cleaned_ohlcv_with_fundamentals_and_technical.csv')
print("Đã lưu file csv thành công!")