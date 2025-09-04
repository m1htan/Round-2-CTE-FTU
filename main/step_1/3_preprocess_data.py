import pandas as pd
import numpy as np

def clean_trading_data(df: pd.DataFrame) -> pd.DataFrame:

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Drop OHLCV bất hợp lý
    mask_valid = (
        (df['open'] >= 0) &
        (df['high'] >= 0) &
        (df['low'] >= 0) &
        (df['close'] >= 0) &
        (df['volume'] >= 0) &
        (df['high'] >= df['low']) &
        (df['open'] >= df['low']) & (df['open'] <= df['high']) &
        (df['close'] >= df['low']) & (df['close'] <= df['high'])
    )
    df = df[mask_valid].copy()

    # Drop duplicate ticker + timestamp
    df = df.drop_duplicates(subset=['ticker', 'timestamp'])

    # Điền giá trị thiếu bằng nội suy
    cleaned_dfs = []
    for ticker, g in df.groupby('ticker'):
        g = g.set_index('timestamp').sort_index()

        # Tạo dải ngày liên tục
        full_range = pd.date_range(g.index.min().floor('D'),
                                   g.index.max().ceil('D'),
                                   freq='1D')

        g = g.reindex(full_range)
        g['ticker'] = ticker

        # Xác định cột numeric
        num_cols = [c for c in ['open', 'high', 'low', 'close', 'volume', 'bu', 'sd', 'fn', 'fs', 'fb'] if
                    c in g.columns]

        # Nội suy giá, nhưng volume thì giữ nguyên (không interpolate)
        price_cols = [c for c in ['open', 'high', 'low', 'close', 'bu', 'sd', 'fn', 'fs', 'fb'] if c in g.columns]
        if price_cols:
            g[price_cols] = g[price_cols].interpolate(method='linear', limit_direction='both')

        # Điền volume = 0 nếu NaN
        if 'volume' in g.columns:
            g['volume'] = g['volume'].fillna(0)

        g = g.reset_index().rename(columns={'index': 'timestamp'})
        cleaned_dfs.append(g)

    df_clean = pd.concat(cleaned_dfs, ignore_index=True)
    return df_clean


df = pd.read_csv('../../data/step_1/raw_stocks.csv')
df_clean = clean_trading_data(df)

out_path = "../../data/step_1/cleaned_stocks.csv"
df_clean.to_csv(out_path, index=False)
print(f"Đã lưu file cleaned: {out_path}, shape={df_clean.shape}")
