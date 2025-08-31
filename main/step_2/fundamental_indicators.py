import os
import time
import math
import sys
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from FiinQuantX import FiinSession

# Ghi Log
LOGGER = logging.getLogger("funda")

def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    LOGGER.setLevel(lvl)
    LOGGER.info("Logging initialized. Level=%s", level)

def log_df(df: pd.DataFrame, name: str):
    try:
        LOGGER.info("%s: shape=%s, cols=%d", name, tuple(df.shape), len(df.columns))
    except Exception:
        LOGGER.info("%s: shape=? (object has no shape)", name)

# CONFIG
DEFAULT_BATCH_SIZE = 50
SLEEP_BETWEEN_BATCH = 0.2
RATIOS_NUM_PERIODS = 8
RATIOS_LATEST_YEAR = pd.Timestamp.today().year
RATIOS_TIMEFILTER = "Quarterly"   # hoặc "Yearly"
RATIOS_CONSOLIDATED = True
MARKETCAP_TIMEFILTER = "Daily"
FOREIGN_TIMEFILTER = "Daily"

def init_client(username: str, password: str):
    return FiinSession(username=username, password=password).login()

def _unique_tickers_from_prices(df_prices: pd.DataFrame) -> List[str]:
    u = (df_prices['ticker']
         .dropna()
         .astype(str)
         .str.upper()
         .unique()
         .tolist())
    return u

def _chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _quarter_end_date(year: int, q: int) -> pd.Timestamp:
    month = 3 * q
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

def _normalize_ratios_to_frame(ticker: str, raw: Dict) -> pd.DataFrame:
    df = pd.json_normalize(raw)
    if df.empty:
        df['ticker'] = ticker
        return df
    df['ticker'] = ticker
    year_col = 'year' if 'year' in df.columns else 'Year' if 'Year' in df.columns else None
    quarter_col = 'quarter' if 'quarter' in df.columns else 'Quarter' if 'Quarter' in df.columns else None
    if quarter_col and year_col:
        df['period_end_date'] = [
            _quarter_end_date(int(y), int(q))
            for y, q in zip(df[year_col], df[quarter_col])
        ]
    elif year_col:
        df['period_end_date'] = pd.to_datetime(df[year_col].astype(int).astype(str) + "-12-31")
    else:
        if 'timestamp' in df.columns:
            df['period_end_date'] = pd.to_datetime(df['timestamp'])
        else:
            df['period_end_date'] = pd.NaT
    return df

def _left_asof_per_ticker(df_left: pd.DataFrame, df_right: pd.DataFrame,
                          left_on: str, right_on: str) -> pd.DataFrame:
    if df_right.empty:
        LOGGER.warning("[ASOF] right DF trống, trả lại left DF.")
        return df_left.copy()
    out = []
    for tkr, g in df_left.groupby('ticker', sort=False):
        r = df_right[df_right['ticker'] == tkr]
        if r.empty:
            out.append(g.assign(_no_funda=True))
            continue
        tmp = pd.merge_asof(
            g.sort_values(left_on),
            r.sort_values(right_on),
            left_on=left_on,
            right_on=right_on,
            direction='backward'
        )
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

def _extract_year_quarter_cols(df):
    year_col = None
    quarter_col = None
    for cand in ['year', 'Year']:
        if cand in df.columns:
            year_col = cand
            break
    for cand in ['quarter', 'Quarter']:
        if cand in df.columns:
            quarter_col = cand
            break
    return year_col, quarter_col

def _normalize_ratios_obj_to_df(obj) -> pd.DataFrame:
    df = pd.json_normalize(obj)
    if 'ticker' not in df.columns:
        for k in ['Ticker', 'meta.ticker', 'Meta.Ticker']:
            if k in df.columns:
                df['ticker'] = df[k]
                break
    if 'ticker' not in df.columns and isinstance(obj, dict):
        if 'Ticker' in obj:
            df['ticker'] = obj['Ticker']
        elif 'ticker' in obj:
            df['ticker'] = obj['ticker']
    return df

def _add_period_end_date(df: pd.DataFrame) -> pd.DataFrame:
    ycol, qcol = _extract_year_quarter_cols(df)
    if qcol and ycol:
        df['period_end_date'] = [
            _quarter_end_date(int(y), int(q))
            for y, q in zip(df[ycol], df[qcol])
        ]
    elif ycol:
        df['period_end_date'] = pd.to_datetime(df[ycol].astype(int).astype(str) + "-12-31")
    else:
        for cand in ['timestamp', 'periodEndDate', 'PeriodEndDate', 'reportDate', 'ReportDate']:
            if cand in df.columns:
                df['period_end_date'] = pd.to_datetime(df[cand])
                break
        if 'period_end_date' not in df.columns:
            df['period_end_date'] = pd.NaT
    return df

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(LOGGER, logging.WARNING),
)
def _api_get_ratios(client, tickers: List[str], timefilter: str, latest_year: int, num_periods: int, consolidated: bool):
    return client.FundamentalAnalysis().get_ratios(
        tickers=tickers,
        TimeFilter=timefilter,
        LatestYear=latest_year,
        NumberOfPeriod=num_periods,
        Consolidated=consolidated
    )

def _collect_ratios_raw_into_frames(frames: list, raw, batch: List[str], bi: int):
    """
    Gom dữ liệu từ get_ratios (dict hoặc list) vào frames.
    """
    if isinstance(raw, dict):
        LOGGER.info("[RATIOS] Batch %d: %d mục (dict)", bi, len(raw))
        for tkr, obj in raw.items():
            df_one = _normalize_ratios_obj_to_df(obj)
            if 'ticker' not in df_one.columns:
                df_one['ticker'] = tkr
            df_one = _add_period_end_date(df_one)
            frames.append(df_one)

    elif isinstance(raw, list):
        LOGGER.info("[RATIOS] Batch %d: %d mục (list)", bi, len(raw))
        for obj in raw:
            df_one = _normalize_ratios_obj_to_df(obj)
            if 'ticker' not in df_one.columns and len(batch) == 1:
                df_one['ticker'] = batch[0]
            df_one = _add_period_end_date(df_one)
            frames.append(df_one)

    else:
        LOGGER.warning("[RATIOS] Batch %d: kiểu trả về lạ: %s", bi, type(raw).__name__)

def fetch_ratios_all_tickers(
    client,
    tickers: List[str],
    timefilter: str = RATIOS_TIMEFILTER,
    latest_year: int = RATIOS_LATEST_YEAR,
    num_periods: int = RATIOS_NUM_PERIODS,
    consolidated: bool = RATIOS_CONSOLIDATED,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    LOGGER.info("[RATIOS] Bắt đầu fetch cho %d tickers | timefilter=%s latest_year=%s periods=%s consolidated=%s",
                len(tickers), timefilter, latest_year, num_periods, consolidated)
    n_batches = math.ceil(len(tickers) / batch_size)
    frames = []
    skipped = []   # tickers bị lỗi khi gọi đơn lẻ
    t0 = time.time()

    for bi, batch in enumerate(_chunk(tickers, batch_size), start=1):
        bstart = time.time()
        LOGGER.info("[RATIOS] Batch %d/%d | batch_size=%d | ví dụ tickers=%s",
                    bi, n_batches, len(batch), batch[:5])

        try:
            raw = _api_get_ratios(client, batch, timefilter, latest_year, num_periods, consolidated)
            LOGGER.info("[RATIOS] Batch %d nhận kiểu=%s", bi, type(raw).__name__)
            _collect_ratios_raw_into_frames(frames, raw, batch, bi)

        except Exception as e:
            # Batch fail -> fallback từng ticker để cô lập mã lỗi
            LOGGER.warning("[RATIOS] Batch %d FAILED (%s). Fallback per-ticker.", bi, repr(e))
            for tkr in batch:
                try:
                    raw_one = _api_get_ratios(client, [tkr], timefilter, latest_year, num_periods, consolidated)
                    _collect_ratios_raw_into_frames(frames, raw_one, [tkr], bi)
                except Exception as ee:
                    LOGGER.error("[RATIOS] Ticker %s lỗi: %s -> SKIP", tkr, repr(ee))
                    skipped.append(tkr)

        elapsed = time.time() - bstart
        LOGGER.info("[RATIOS] Batch %d hoàn tất trong %.2fs", bi, elapsed)
        time.sleep(SLEEP_BETWEEN_BATCH)

    if skipped:
        LOGGER.warning("[RATIOS] Skipped %d tickers do lỗi API: %s%s",
                       len(skipped),
                       ", ".join(skipped[:15]),
                       " ..." if len(skipped) > 15 else "")

    if not frames:
        LOGGER.warning("[RATIOS] Không nhận được dữ liệu nào.")
        return pd.DataFrame(columns=['ticker','period_end_date'])

    df_rat = pd.concat(frames, ignore_index=True)
    if 'ticker' in df_rat.columns:
        df_rat['ticker'] = df_rat['ticker'].astype(str).str.upper()
    df_rat['period_end_date'] = pd.to_datetime(df_rat['period_end_date'], errors='coerce')
    df_rat = df_rat.drop_duplicates(subset=['ticker','period_end_date'])

    LOGGER.info("[RATIOS] Tổng thời gian: %.2fs", time.time() - t0)
    log_df(df_rat, "df_ratios")
    if 'period_end_date' in df_rat.columns and not df_rat['period_end_date'].isna().all():
        LOGGER.info("[RATIOS] period_end_date range: %s -> %s",
                    df_rat['period_end_date'].min(), df_rat['period_end_date'].max())
    return df_rat


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(LOGGER, logging.WARNING),
)
def _api_get_overview(client, tickers: List[str], timefilter: str, start_date: str, end_date: Optional[str]):
    return client.PriceStatistics().get_Overview(
        tickers=tickers,
        TimeFilter=timefilter,
        StartDate=start_date,
        EndDate=end_date
    ).get_data()

def fetch_marketcap_all_tickers(
    client,
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    timefilter: str = MARKETCAP_TIMEFILTER,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    LOGGER.info("[MCAP] Fetch MarketCap %d tickers | %s .. %s | tf=%s",
                len(tickers), start_date, end_date, timefilter)
    n_batches = math.ceil(len(tickers) / batch_size)
    frames = []
    t0 = time.time()

    for bi, batch in enumerate(_chunk(tickers, batch_size), start=1):
        bstart = time.time()
        LOGGER.info("[MCAP] Batch %d/%d | size=%d | ví dụ=%s", bi, n_batches, len(batch), batch[:5])
        data = _api_get_overview(client, batch, timefilter, start_date, end_date)
        df = pd.DataFrame(data)
        frames.append(df)
        LOGGER.info("[MCAP] Batch %d rows=%d", bi, len(df))
        time.sleep(SLEEP_BETWEEN_BATCH)
        LOGGER.info("[MCAP] Batch %d hoàn tất trong %.2fs", bi, time.time() - bstart)

    if not frames:
        LOGGER.warning("[MCAP] Không có dữ liệu.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    log_df(df, "df_marketcap")
    LOGGER.info("[MCAP] Tổng thời gian: %.2fs", time.time() - t0)
    return df

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(LOGGER, logging.WARNING),
)
def _api_get_foreign(client, tickers: List[str], timefilter: str, start_date: str, end_date: Optional[str]):
    return client.PriceStatistics().get_Foreign(
        tickers=tickers,
        TimeFilter=timefilter,
        StartDate=start_date,
        EndDate=end_date
    ).get_data()

def fetch_foreign_all_tickers(
    client,
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    timefilter: str = FOREIGN_TIMEFILTER,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    LOGGER.info("[FOREIGN] Fetch %d tickers | %s .. %s | tf=%s",
                len(tickers), start_date, end_date, timefilter)
    n_batches = math.ceil(len(tickers) / batch_size)
    frames = []
    t0 = time.time()

    for bi, batch in enumerate(_chunk(tickers, batch_size), start=1):
        bstart = time.time()
        LOGGER.info("[FOREIGN] Batch %d/%d | size=%d | ví dụ=%s", bi, n_batches, len(batch), batch[:5])
        data = _api_get_foreign(client, batch, timefilter, start_date, end_date)
        df = pd.DataFrame(data)
        frames.append(df)
        LOGGER.info("[FOREIGN] Batch %d rows=%d", bi, len(df))
        time.sleep(SLEEP_BETWEEN_BATCH)
        LOGGER.info("[FOREIGN] Batch %d hoàn tất trong %.2fs", bi, time.time() - bstart)

    if not frames:
        LOGGER.warning("[FOREIGN] Không có dữ liệu.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    log_df(df, "df_foreign")
    LOGGER.info("[FOREIGN] Tổng thời gian: %.2fs", time.time() - t0)
    return df

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(LOGGER, logging.WARNING),
)
def _api_get_freefloat(client, tickers: List[str], start_date: str, end_date: Optional[str]):
    return client.PriceStatistics().get_Freefloat(
        tickers=tickers,
        StartDate=start_date,
        EndDate=end_date
    ).get_data()

def fetch_freefloat_all_tickers(
    client,
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    LOGGER.info("[FREEFLOAT] Fetch %d tickers | %s .. %s", len(tickers), start_date, end_date)
    n_batches = math.ceil(len(tickers) / batch_size)
    frames = []
    t0 = time.time()

    for bi, batch in enumerate(_chunk(tickers, batch_size), start=1):
        bstart = time.time()
        LOGGER.info("[FREEFLOAT] Batch %d/%d | size=%d | ví dụ=%s", bi, n_batches, len(batch), batch[:5])
        data = _api_get_freefloat(client, batch, start_date, end_date)
        df = pd.DataFrame(data)
        frames.append(df)
        LOGGER.info("[FREEFLOAT] Batch %d rows=%d", bi, len(df))
        time.sleep(SLEEP_BETWEEN_BATCH)
        LOGGER.info("[FREEFLOAT] Batch %d hoàn tất trong %.2fs", bi, time.time() - bstart)

    if not frames:
        LOGGER.warning("[FREEFLOAT] Không có dữ liệu.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    log_df(df, "df_freefloat")
    LOGGER.info("[FREEFLOAT] Tổng thời gian: %.2fs", time.time() - t0)
    return df


def align_ratios_to_prices(df_prices: pd.DataFrame, df_ratios: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("[ALIGN] ASOF ratios -> prices")
    left = df_prices.copy()
    left['timestamp'] = pd.to_datetime(left['timestamp'])
    right = df_ratios.copy()

    if 'period_end_date' not in right.columns:
        if 'timestamp' in right.columns:
            right = right.rename(columns={'timestamp': 'period_end_date'})
        else:
            LOGGER.warning("[ALIGN] df_ratios không có period_end_date/timestamp. Sẽ điền NaT.")
            right['period_end_date'] = pd.NaT

    right['period_end_date'] = pd.to_datetime(right['period_end_date'])
    out = _left_asof_per_ticker(
        df_left=left,
        df_right=right,
        left_on='timestamp',
        right_on='period_end_date'
    )
    log_df(out, "df_after_asof")
    return out

def align_timeseries_to_prices(df_prices: pd.DataFrame, df_ts: pd.DataFrame, name: str = "ts") -> pd.DataFrame:
    LOGGER.info("[ALIGN] Join %s (daily) -> prices", name)
    left = df_prices.copy()
    left['timestamp'] = pd.to_datetime(left['timestamp'])
    right = df_ts.copy()
    if 'timestamp' in right.columns:
        right['timestamp'] = pd.to_datetime(right['timestamp'])
    out = pd.merge(left, right, on=['ticker','timestamp'], how='left')
    log_df(out, f"df_after_join_{name}")
    return out


def build_fundamental_features_for_all(
    client,
    df_prices: pd.DataFrame,
    start_date: str,
    end_date: Optional[str] = None,
    ratios_timefilter: str = RATIOS_TIMEFILTER,
    ratios_latest_year: int = RATIOS_LATEST_YEAR,
    ratios_num_periods: int = RATIOS_NUM_PERIODS,
    ratios_consolidated: bool = RATIOS_CONSOLIDATED,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    LOGGER.info(
        "[START] Build fundamentals | tickers=%d | price_range=%s..%s",
        df_prices['ticker'].nunique(),
        df_prices['timestamp'].min().date(),
        df_prices['timestamp'].max().date()
    )
    LOGGER.info(
        "[CONFIG] batch_size=%d, ratios_timefilter=%s, latest_year=%s, periods=%s, consolidated=%s",
        batch_size, ratios_timefilter, ratios_latest_year, ratios_num_periods, ratios_consolidated
    )

    tickers = _unique_tickers_from_prices(df_prices)
    LOGGER.info("[TICKERS] %d unique tickers", len(tickers))

    # 1) RATIOS
    df_rat = fetch_ratios_all_tickers(
        client, tickers,
        timefilter=ratios_timefilter,
        latest_year=ratios_latest_year,
        num_periods=ratios_num_periods,
        consolidated=ratios_consolidated,
        batch_size=batch_size
    )

    # 2) MARKETCAP
    df_cap = fetch_marketcap_all_tickers(
        client, tickers,
        start_date=start_date, end_date=end_date,
        timefilter=MARKETCAP_TIMEFILTER,
        batch_size=batch_size
    )

    # 3) FOREIGN
    df_foreign = fetch_foreign_all_tickers(
        client, tickers,
        start_date=start_date, end_date=end_date,
        timefilter=FOREIGN_TIMEFILTER,
        batch_size=batch_size
    )

    # 4) FREEFLOAT
    df_free = fetch_freefloat_all_tickers(
        client, tickers,
        start_date=start_date, end_date=end_date,
        batch_size=batch_size
    )

    # Align / merge
    df_rat_aligned = align_ratios_to_prices(df_prices, df_rat)
    df_cap_merged = align_timeseries_to_prices(df_rat_aligned, df_cap, name="marketcap")
    df_foreign_merged = align_timeseries_to_prices(df_cap_merged, df_foreign, name="foreign")
    df_final = align_timeseries_to_prices(df_foreign_merged, df_free, name="freefloat")

    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    log_df(df_final, "df_final")

    return df_final


if __name__ == "__main__":
    # 0) Logging
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    # 1) Load GIÁ
    df_prices = pd.read_csv("../../data/cleaned_all_stocks.csv")
    df_prices["timestamp"] = pd.to_datetime(df_prices["timestamp"], errors="coerce")
    df_prices["ticker"] = df_prices["ticker"].astype(str).str.upper()
    log_df(df_prices, "df_prices")

    # 2) Login FiinQuantX
    load_dotenv(dotenv_path='../../config/.env')
    USERNAME = os.getenv("FIINQUANT_USERNAME")
    PASSWORD = os.getenv("FIINQUANT_PASSWORD")
    if not USERNAME or not PASSWORD:
        LOGGER.error("Thiếu FIINQUANT_USERNAME / FIINQUANT_PASSWORD trong .env")
        sys.exit(1)
    client = FiinSession(username=USERNAME, password=PASSWORD).login()
    LOGGER.info("Đăng nhập FiinQuantX thành công.")

    # 3) Date range cho PriceStatistics
    start_date = df_prices['timestamp'].min().strftime("%Y-%m-%d")
    end_date   = df_prices['timestamp'].max().strftime("%Y-%m-%d")
    LOGGER.info("Date range: %s .. %s", start_date, end_date)

    # 4) Build fundamentals (ALL)
    df_funda_all = build_fundamental_features_for_all(
        client=client,
        df_prices=df_prices[['timestamp','ticker','close','volume']].copy(),
        start_date=start_date,
        end_date=end_date,
        ratios_timefilter="Quarterly",
        ratios_latest_year=pd.Timestamp.now().year,
        ratios_num_periods=8,
        ratios_consolidated=True,
        batch_size=50
    )

    # 5) Save
    os.makedirs("../../data", exist_ok=True)
    out_path = "../../data/fundamentals_indicators.csv"
    df_funda_all.to_csv(out_path, index=False)
    LOGGER.info("Saved to %s | shape=%s", out_path, tuple(df_funda_all.shape))
    print("Done:", df_funda_all.shape)