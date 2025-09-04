import pandas as pd

df = pd.read_csv('../../data/step_1/raw_stocks.csv')

def validate_data(df, check_missing_for_ticker=None):
    report = {}

    # 1. Phạm vi thời gian
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    report['date_range'] = (min_date, max_date)
    print(f"[INFO] Dữ liệu từ {min_date.date()} đến {max_date.date()}")

    # 2. Kiểm tra giá trị hợp lý cho OHLCV
    invalid = df[
        (df["open"] <= 0) |
        (df["high"] <= 0) |
        (df["low"] <= 0) |
        (df["close"] <= 0) |
        (df["low"] > df["high"]) |
        (df["open"] > df["high"]) |
        (df["close"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df["close"] < df["low"]) |
        (df["volume"] < 0)
        ]
    report['invalid_rows'] = len(invalid)
    print(f"[CHECK] Số dòng giá trị OHLCV bất hợp lý: {len(invalid)}")
    print(invalid.head(10))

    # 3. Trùng lặp
    dupes = df[df.duplicated(subset=["ticker", "timestamp"], keep=False)]
    report['duplicates'] = len(dupes)
    print(f"[CHECK] Số dòng trùng lặp ticker+timestamp: {len(dupes)}")
    print(dupes.head(20))

    # 4. Missing dates (cho 1 ticker nếu chỉ định)
    missing = []
    if check_missing_for_ticker:
        df_t = df[df["ticker"] == check_missing_for_ticker].copy()
        if not df_t.empty:
            start = df_t["timestamp"].min()
            end = df_t["timestamp"].max()
            all_days = pd.date_range(start=start, end=end, freq="B")
            existing_days = pd.to_datetime(df_t["timestamp"].dt.date.unique())
            missing = sorted(set(all_days.date) - set(existing_days))
            print(f"[CHECK] Missing {len(missing)} ngày cho {check_missing_for_ticker}")
        else:
            print(f"[WARN] Không tìm thấy dữ liệu cho {check_missing_for_ticker}")
    report['missing_dates'] = missing

    # 5. Kiểm tra bu, sd, fb, fs, fn
    orderflow_invalid = df[
        (df["fb"] < 0) | (df["fs"] < 0) | (df["fn"].isna()) |
        (df["fb"] > df["volume"]) |
        (df["fs"] > df["volume"]) |
        (abs(df["fn"] - (df["fb"] - df["fs"])) > df["volume"]*0.05)  # lệch quá 5%
    ]
    report['orderflow_invalid'] = len(orderflow_invalid)
    print(f"[CHECK] Số dòng orderflow bất hợp lý: {len(orderflow_invalid)}")
    print(orderflow_invalid.head())

    # 6. Thống kê mô tả
    stats = df[['open', 'high', 'low', 'close', 'volume']].describe()
    report['stats'] = stats
    print("[INFO] Thống kê tổng quan:")
    print(stats)

    return report

report = validate_data(df, check_missing_for_ticker="VNM")

# Xem chi tiết
print(report['date_range'])
print("Invalid rows:", report['invalid_rows'])
print("Duplicates:", report['duplicates'])
print("Missing dates:", report['missing_dates'][:10])
