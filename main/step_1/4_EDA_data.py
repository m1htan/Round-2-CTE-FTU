import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../../data/step_1/cleaned_stocks.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(df.head())

print(df.shape)

print(df.info())

print(df.shape)
print(df['ticker'].nunique(), "tickers")
print(df['Exchange'].value_counts())
print(df['timestamp'].min(), df['timestamp'].max())

print(df.describe().T)


df.groupby('ticker')['close'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 cổ phiếu giá trung bình cao nhất")
plt.show()


ticker = "VE4"
df_ticker = df[df['ticker']==ticker].set_index('timestamp')
df_ticker['close'].plot(figsize=(12,6), title=f"{ticker} Closing Price")
plt.show()


top10 = df.groupby('ticker')['volume'].sum().nlargest(10).index
pivot = df[df['ticker'].isin(top10)].pivot(index='timestamp', columns='ticker', values='close')
corr = pivot.pct_change().corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# Histogram giá đóng cửa
df['close'].hist(bins=100, figsize=(10,5))
plt.title("Histogram giá đóng cửa toàn bộ cổ phiếu")
plt.show()

# Top 10 cổ phiếu thanh khoản
top10_vol = (df.groupby('ticker')['volume']
             .mean().sort_values(ascending=False).head(10))
top10_vol.plot(kind='bar', title="Top 10 cổ phiếu theo volume trung bình")
plt.show()

# Boxplot giá theo sàn
sns.boxplot(data=df, x='Exchange', y='close')
plt.ylim(0, 200000)  # giới hạn để tránh outlier quá lớn
plt.show()

# Ma trận tương quan cho 10 mã thanh khoản nhất
tickers_top = top10_vol.index
pivot = df[df['ticker'].isin(tickers_top)].pivot(index='timestamp', columns='ticker', values='close')
returns = pivot.pct_change()
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation giữa các cổ phiếu top liquidity")
plt.show()
