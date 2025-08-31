from datetime import datetime, timedelta
import time
import pandas as pd
from dotenv import load_dotenv
import os

from FiinQuantX import FiinSession

# Load config
load_dotenv(dotenv_path='../../config/.env')

USERNAME = os.getenv("FIINQUANT_USERNAME")
PASSWORD = os.getenv("FIINQUANT_PASSWORD")

client = FiinSession(username=USERNAME, password=PASSWORD).login()

# Params
fields = ['open', 'high', 'low', 'close', 'volume', 'bu', 'sd', 'fn', 'fs', 'fb']
years_back = 3
to_date   = "2025-08-30"
from_date = "2022-01-01"

def fetch_ohlcv_by_exchange(tickers, from_date, to_date=None, fields=None,
                            adjusted=True, by="1d", batch_size=80, pause=0.8):
    if fields is None:
        fields = ['open', 'high', 'low', 'close', 'volume', 'bu', 'sd', 'fn', 'fs', 'fb']
    all_parts = []
    n = len(tickers)
    for i in range(0, n, batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = client.Fetch_Trading_Data(
                realtime=False,
                tickers=batch,
                fields=fields,
                adjusted=adjusted,
                by=by,
                from_date=from_date,
                to_date=to_date
            ).get_data()
            if isinstance(df, pd.DataFrame) and not df.empty:
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                all_parts.append(df)
        except Exception as e:
            print(f"[WARN] Batch {i//batch_size+1}: lỗi {e} với {batch[:3]}... ({len(batch)} mã)")
        time.sleep(pause)
    if all_parts:
        out = pd.concat(all_parts, ignore_index=True)
        out = out.sort_values(['ticker','timestamp']).reset_index(drop=True)
        return out
    return pd.DataFrame()

# Crawl
all_data = []
for index in ["HNXIndex"]:
    try:
        tickers = client.TickerList(ticker=index)
        tickers = sorted(set(tickers))
        print(f"{index}: {len(tickers)} mã")
    except Exception as e:
        print(f"Lỗi khi lấy danh sách ticker cho {index}: {e}")
        tickers = []

    df_ex = fetch_ohlcv_by_exchange(tickers, from_date, to_date, fields=fields)
    if not df_ex.empty:
        df_ex["Exchange"] = index.replace("Index", "")
        all_data.append(df_ex)

# Gộp toàn bộ df
if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    print("Shape final:", df_all.shape)
    print(df_all.head())

    # Export CSV
    out_path = f"../../data/raw_stocks.csv"
    df_all.to_csv(out_path, index=False)
    print(f"Đã lưu file: {out_path}")
else:
    print("Không lấy được dữ liệu nào.")