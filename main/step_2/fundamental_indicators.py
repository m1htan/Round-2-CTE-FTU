import os
from FiinQuantX import FiinSession
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import time
import re
import json
import random
from dotenv import load_dotenv

load_dotenv(dotenv_path='../../config/.env')

USERNAME = os.getenv("FIINQUANT_USERNAME")
PASSWORD = os.getenv("FIINQUANT_PASSWORD")

client = FiinSession(username=USERNAME, password=PASSWORD).login()
# Khoảng thời gian như bạn yêu cầu
FROM_DATE = pd.Timestamp("2022-01-01")
TO_DATE   = pd.Timestamp("2025-08-30")

# Type theo yêu cầu
FIN_TYPE = "consolidated"

# ------------------------------------------------

# Batching khi gọi API get_ratios
BATCH_SIZE = 10
BATCH_SLEEP_SEC = 0.2  # nghỉ nhẹ giữa các batch để tránh rate limit

# ---------- Helper: thời gian/quarter ----------
def quarter_of_month(m: int) -> int:
    return (m - 1) // 3 + 1

def quarter_end_date(year: int, q: int) -> pd.Timestamp:
    if q == 1: return pd.Timestamp(year=year, month=3, day=31)
    if q == 2: return pd.Timestamp(year=year, month=6, day=30)
    if q == 3: return pd.Timestamp(year=year, month=9, day=30)
    if q == 4: return pd.Timestamp(year=year, month=12, day=31)
    raise ValueError("quarter must be 1..4")

def years_and_quarters_in_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Tuple[List[int], List[int]]:
    """Trả về YEARS = [start_year..end_year], QUARTERS = [1,2,3,4].
       Lý do: API get_ratios nhận list năm/quý. Ta sẽ lọc theo period_end_date sau."""
    years = list(range(start_ts.year, end_ts.year + 1))
    quarters = [1, 2, 3, 4]
    return years, quarters

# ---------- Helper: xử lý OHLCV ----------
def add_year_quarter_from_timestamp(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out["year"] = out[ts_col].dt.year
    out["quarter"] = ((out[ts_col].dt.month - 1) // 3 + 1).astype(int)
    # Chuẩn ticker viết hoa để đồng nhất
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out

def extract_hnx_tickers(ohlcv: pd.DataFrame) -> List[str]:
    # Nếu file chỉ chứa HNX thì đơn giản:
    # Nếu có cột "Exchange", có thể lọc:
    #   ohlcv = ohlcv.loc[ohlcv["Exchange"].str.upper().eq("HNX")]
    return pd.Index(ohlcv["ticker"].astype(str).str.upper().unique()).tolist()

# ---------- Helper: tìm keys an toàn ----------
def normalize_key(k: str) -> str:
    # Bỏ non-alnum, upper-case để so sánh an toàn: "P/E" -> "PE", "BookValuePerShare" -> "BOOKVALUEPERSHARE"
    return re.sub(r"[^A-Za-z0-9]+", "", str(k)).upper()

CANONICAL_KEYS = {"PE", "PB", "ROE", "EPS", "BVPS"}

# Một số synonym phổ biến (đề phòng vendor đặt tên hơi khác)
SYNONYM_TO_CANON = {
    "PBR": "PB",
    "PRICETOBOOK": "PB",
    "PRICEBOOK": "PB",
    "BOOKVALUEPERSHARE": "BVPS",
    "BVPERPSHARE": "BVPS",
    "BVPS": "BVPS",
    "EPSBASIC": "EPS",
    "RETURNONEQUITY": "ROE",
    "ROEA": "ROE",
    "PRICEEARNING": "PE",
    "PRICEEARNINGRATIO": "PE",
    "PERATIO": "PE",
}

SYNONYM_TO_CANON.update({
    "EARNINGSPERSHARE": "EPS",
    "EARNINGSPERSHAREBASIC": "EPS",
    "BASICEPS": "EPS",
    "DILUTEDEPS": "EPS",
    "EPSDILUTED": "EPS",
    "EPSADJUSTED": "EPS",
    "BVPERSHARE": "BVPS",
    "BOOKVALUEPS": "BVPS",
    "BOOKVALUESHARE": "BVPS",
})

def canonical_of(key: str) -> Optional[str]:
    k = normalize_key(key)
    if k in CANONICAL_KEYS:
        return k
    return SYNONYM_TO_CANON.get(k)

def deep_find_ratios(obj: Any, targets: set) -> Dict[str, Any]:
    """Duyệt sâu dict/list; tìm các key phù hợp PE, PB, ROE, EPS, BVPS (kể cả synonym)."""
    found: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            canon = canonical_of(k)
            if canon in targets and canon not in found:
                # Ưu tiên key đầu tiên tìm thấy; nếu cần ưu tiên TTM hay Basic có thể chỉnh logic ở đây
                found[canon] = v
            # Duyệt sâu
            child_found = deep_find_ratios(v, targets)
            for ck, cv in child_found.items():
                if ck not in found:
                    found[ck] = cv
    elif isinstance(obj, list):
        for v in obj:
            child_found = deep_find_ratios(v, targets)
            for ck, cv in child_found.items():
                if ck not in found:
                    found[ck] = cv
    return found

def call_with_retry(fn, max_retries=5, base_sleep=0.5, max_sleep=8.0, *args, **kwargs):
    """Gọi fn(*args, **kwargs) với retry/backoff. Ném exception cuối nếu thất bại."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1))) * (1 + 0.2 * random.random())
            print(f"[WARN] Retry {attempt}/{max_retries} after error: {e}. Sleeping ~{sleep_s:.2f}s")
            time.sleep(sleep_s)

def iter_quarter_after(year: int, q: int) -> tuple[int, int]:
    q += 1
    if q > 4:
        return year + 1, 1
    return year, q

def compute_quarter_params(from_date: pd.Timestamp, to_date: pd.Timestamp) -> tuple[int, int]:
    """
    Trả về (LatestYear, NumberOfPeriod) cho TimeFilter='Quarterly'
    sao cho cover các quý có period_end_date trong [from_date, to_date].
    """
    # Xác định quý bắt đầu (kết thúc quý >= from_date)
    y, q = int(from_date.year), int((from_date.month - 1)//3 + 1)
    start_end = quarter_end_date(y, q)
    while start_end < from_date:
        y, q = iter_quarter_after(y, q)
        start_end = quarter_end_date(y, q)

    # Lặp đến quý cuối có end_date <= to_date
    quarters = []
    cy, cq = y, q
    while True:
        endd = quarter_end_date(cy, cq)
        if endd > to_date:
            break
        quarters.append((cy, cq))
        cy, cq = iter_quarter_after(cy, cq)

    if not quarters:
        # Không có quý nào trọn vẹn trong khoảng → vẫn trả về quý của to_date để lib trả dữ liệu,
        # rồi ta sẽ lọc sau (ratios_df sẽ rỗng nếu không khớp).
        last_year = to_date.year
        return last_year, 1

    last_year = quarters[-1][0]
    num_periods = len(quarters)
    return last_year, num_periods


def _get_any(d: Dict[str, Any], names: list[str]):
    for n in names:
        if isinstance(d, dict) and n in d:
            return d[n]
    return None

def _parse_quarter(val: Any) -> Optional[int]:
    """Hỗ trợ '1', 1, 'Q1', 'Quý 1'… Trả 1..4 hoặc None."""
    if val is None:
        return None
    if isinstance(val, (int, float)) and 1 <= int(val) <= 4:
        return int(val)
    m = re.search(r'([1-4])', str(val))
    return int(m.group(1)) if m else None

_TKR_CANDIDATES = ["ticker", "Ticker", "SYMBOL", "Symbol", "stockcode", "StockCode", "Code"]
_YR_CANDIDATES  = ["year", "Year", "ReportYear"]
_Q_CANDIDATES   = ["quarter", "Quarter", "ReportQuarter", "Q"]

def _build_row_from_item(item: Dict[str, Any], fallback_ticker: Optional[str], frequency: str) -> Optional[Dict[str, Any]]:
    # ticker: ưu tiên field trong item, nếu không có dùng key cha (fallback_ticker)
    tkr = None
    for k in _TKR_CANDIDATES:
        v = item.get(k)
        if v:
            tkr = v
            break
    if not tkr:
        tkr = fallback_ticker
    if not tkr:
        return None

    year = None
    for k in _YR_CANDIDATES:
        v = item.get(k)
        if v is not None:
            year = pd.to_numeric(v, errors="coerce")
            break

    quarter_raw = None
    for k in _Q_CANDIDATES:
        v = item.get(k)
        if v is not None:
            quarter_raw = v
            break
    quarter = _parse_quarter(quarter_raw)

    # Ratios root: có thể là 'Ratios', 'FinancialRatios', hoặc trả phẳng
    ratios_root = item.get("Ratios", item)
    if isinstance(ratios_root, dict) and "FinancialRatios" in ratios_root:
        ratios_root = ratios_root["FinancialRatios"]

    found = deep_find_ratios(ratios_root, CANONICAL_KEYS)

    rec = {
        "ticker": str(tkr).upper(),
        "year": year,
    }
    if frequency == "quarterly":
        rec["quarter"] = quarter

    rec["PE"] = _to_scalar(found.get("PE"))
    rec["PB"] = _to_scalar(found.get("PB"))
    rec["ROE"] = _to_scalar(found.get("ROE"))
    rec["EPS"] = _to_scalar(found.get("EPS"))
    rec["BVPS"] = _to_scalar(found.get("BVPS"))

    return rec

def _to_number(x):
    if isinstance(x, str):
        xs = x.strip()
        neg = False
        if xs.startswith("(") and xs.endswith(")"):
            xs = xs[1:-1]
            neg = True
        if xs.endswith("%"):
            try:
                val = float(xs[:-1].replace(",", "")) / 100.0
                return -val if neg else val
            except:
                return pd.to_numeric(xs, errors="coerce")
        xs = xs.replace(",", "")
        val = pd.to_numeric(xs, errors="coerce")
        return -val if (neg and pd.notna(val)) else val
    return pd.to_numeric(x, errors="coerce")

def _to_scalar(x):
    # nếu vendor bọc dưới {'Value': 0.123} hoặc {'TTM':..., 'QoQ':..., 'Value':...}
    if isinstance(x, dict):
        for k in ["Value", "value", "VAL", "val"]:
            if k in x:
                return _to_number(x[k])
        # fallback: nếu dict có đúng 1 phần tử thì lấy giá trị
        if len(x) == 1:
            return _to_number(next(iter(x.values())))
        return np.nan
    return _to_number(x)

def _extract_records_container(obj: Dict[str, Any]) -> Optional[list]:
    """Nếu top-level là dict kiểu {'Data': [...]} thì lấy ra list đó (case-insensitive)."""
    for k in ["Data", "data", "Items", "items", "Rows", "rows", "Records", "records", "Result", "result"]:
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], list):
            return obj[k]
    return None

def flatten_ratios_payload(fi_dict: Dict[str, Any], frequency: str = "quarterly") -> pd.DataFrame:
    """
    Hỗ trợ các dạng payload:
      A) { 'HPG': [ {...}, ...], 'AAA': [...] }
      B) [ {...}, {...}, ... ]
      C) { 'Data': [ {...}, ... ] } / {'Items': [...]} ...
    """
    rows = []

    if isinstance(fi_dict, dict):
        container = _extract_records_container(fi_dict)
        if container is not None:
            # Trường hợp C
            for it in container:
                if isinstance(it, dict):
                    rec = _build_row_from_item(it, fallback_ticker=None, frequency=frequency)
                    if rec: rows.append(rec)
        else:
            # Trường hợp A
            for key, val in fi_dict.items():
                if isinstance(val, list):
                    for it in val:
                        if isinstance(it, dict):
                            rec = _build_row_from_item(it, fallback_ticker=key, frequency=frequency)
                            if rec: rows.append(rec)
                elif isinstance(val, dict):
                    rec = _build_row_from_item(val, fallback_ticker=key, frequency=frequency)
                    if rec: rows.append(rec)

    elif isinstance(fi_dict, list):
        # Trường hợp B
        for it in fi_dict:
            if isinstance(it, dict):
                rec = _build_row_from_item(it, fallback_ticker=None, frequency=frequency)
                if rec: rows.append(rec)

    # Tạo DataFrame với schema mong đợi
    expected_cols = ["ticker", "year"] + (["quarter"] if frequency == "quarterly" else []) + ["PE","PB","ROE","EPS","BVPS"]
    df = pd.DataFrame(rows, columns=expected_cols)

    if df.empty:
        # vẫn trả về các cột cần thiết để merge không sập
        if "quarter" in expected_cols and "quarter" not in df.columns:
            df["quarter"] = pd.Series(dtype="float64")
        return df

    # Lọc quarter hợp lệ
    if "quarter" in df.columns:
        df = df[df["quarter"].isin([1,2,3,4])].copy()

    # đảm bảo có year/quarter hợp lệ
    if "year" in df.columns:
        df = df.dropna(subset=["year"])
    if "quarter" in df.columns:
        df = df.dropna(subset=["quarter"])

    # Thêm period_end_date
    if "quarter" in df.columns:
        df["period_end_date"] = df.apply(lambda r: quarter_end_date(int(r["year"]), int(r["quarter"])), axis=1)
    else:
        df["period_end_date"] = pd.to_datetime(dict(year=df["year"], month=12, day=31))

    # Khử trùng lặp
    df = df.drop_duplicates(subset=[c for c in ["ticker","year","quarter"] if c in df.columns])
    return df

def build_quarter_end_price(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Lấy giá 'close' của NGÀY GIAO DỊCH CUỐI CÙNG trong mỗi (ticker,year,quarter).
    Trả: [ticker, year, quarter, price_eoq, price_eoq_ts]
    """
    o = add_year_quarter_from_timestamp(ohlcv, "timestamp").copy()
    o = o.sort_values(["ticker", "year", "quarter", "timestamp"])

    # Lấy dòng cuối mỗi nhóm bằng tail(1) (ổn định hơn idxmax + .loc)
    last = (
        o.groupby(["ticker", "year", "quarter"], as_index=False)
         .tail(1)[["ticker", "year", "quarter", "close", "timestamp"]]
         .rename(columns={"close": "price_eoq", "timestamp": "price_eoq_ts"})
         .reset_index(drop=True)
    )

    # Chuẩn kiểu
    last["price_eoq"] = pd.to_numeric(last["price_eoq"], errors="coerce")
    return last


def compute_eps_ttm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính EPS_TTM = tổng EPS của 4 quý gần nhất cho từng ticker.
    Yêu cầu đủ 4 quý (min_periods=4); nếu thiếu → NaN.
    """
    if df.empty:
        out = df.copy()
        out["EPS_TTM"] = np.nan
        return out
    out = df.sort_values(["ticker", "year", "quarter"]).copy()
    g = out.groupby("ticker", group_keys=False)
    out["EPS_TTM"] = g["EPS"].rolling(window=4, min_periods=4).sum().reset_index(level=0, drop=True)
    return out


def attach_pe_ttm(ratios_df: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Gắn price_eoq, tính PE_TTM = price_eoq / EPS_TTM (chỉ khi EPS_TTM>0).
    Tạo PE_filled: dùng PE nếu có, nếu NaN thì dùng PE_TTM.
    """
    if ratios_df.empty:
        out = ratios_df.copy()
        for c in ["EPS_TTM", "price_eoq", "price_eoq_ts", "PE_TTM", "PE_filled"]:
            out[c] = np.nan
        return out

    # 1) EPS_TTM
    out = compute_eps_ttm(ratios_df)

    # 2) Giá cuối quý
    price_q = build_quarter_end_price(ohlcv)
    out = out.merge(price_q, on=["ticker", "year", "quarter"], how="left")

    # 3) PE_TTM
    cond_valid = (out["EPS_TTM"] > 0) & (out["price_eoq"] > 0)
    out["PE_TTM"] = np.where(cond_valid, out["price_eoq"] / out["EPS_TTM"], np.nan)

    # 4) PE_filled
    out["PE_filled"] = out["PE"]
    out.loc[out["PE_filled"].isna(), "PE_filled"] = out["PE_TTM"]
    return out

def attach_pb_ttm(ratios_df: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo PB_TTM bằng BVPS_TTM_base (rolling mean 4 quý; fallback = ffill BVPS),
    dùng giá cuối quý (price_eoq). Sinh thêm PB_filled (fallback khi PB vendor bị NaN).
    """
    if ratios_df.empty:
        out = ratios_df.copy()
        for c in ["BVPS_TTM_mean", "BVPS_TTM_base", "PB_TTM", "PB_filled"]:
            out[c] = np.nan
        return out

    out = ratios_df.sort_values(["ticker", "year", "quarter"]).copy()

    # Bổ sung giá cuối quý nếu chưa có (tận dụng helper bạn đã có)
    if "price_eoq" not in out.columns:
        price_q = build_quarter_end_price(ohlcv)
        out = out.merge(price_q, on=["ticker", "year", "quarter"], how="left")

    g = out.groupby("ticker", group_keys=False)
    # Trung bình BVPS 4 quý gần nhất (yêu cầu >=2 điểm để đáng tin)
    out["BVPS_TTM_mean"] = (
        g["BVPS"].rolling(window=4, min_periods=2).mean().reset_index(level=0, drop=True)
    )
    # Fallback: BVPS gần nhất (ffill theo thời gian)
    out["BVPS_ffill"] = g["BVPS"].ffill()
    # Cơ sở BVPS để chia: mean nếu có, ngược lại dùng ffill
    out["BVPS_TTM_base"] = out["BVPS_TTM_mean"].where(out["BVPS_TTM_mean"].notna(), out["BVPS_ffill"])

    # Tính PB_TTM khi hợp lệ
    cond = (out["BVPS_TTM_base"] > 0) & (out["price_eoq"] > 0)
    out["PB_TTM"] = np.where(cond, out["price_eoq"] / out["BVPS_TTM_base"], np.nan)

    # PB_filled: ưu tiên PB vendor, nếu trống dùng PB_TTM
    out["PB_filled"] = out["PB"]
    out.loc[out["PB_filled"].isna(), "PB_filled"] = out["PB_TTM"]

    # Dọn cột phụ nếu muốn
    out.drop(columns=["BVPS_ffill"], inplace=True)
    return out


def attach_eps_ttm_yoy(ratios_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo EPS_TTM_yoy = pct_change của EPS_TTM với lag = 4 quý (TTM vs TTM của 1 năm trước).
    """
    if ratios_df.empty:
        out = ratios_df.copy()
        out["EPS_TTM_yoy"] = np.nan
        return out
    out = ratios_df.sort_values(["ticker", "year", "quarter"]).copy()
    if "EPS_TTM" not in out.columns:
        # đảm bảo đã chạy attach_pe_ttm (nơi tạo EPS_TTM). Nếu chưa có thì khởi tạo NaN.
        out["EPS_TTM"] = np.nan
    out["EPS_TTM_yoy"] = out.groupby("ticker", group_keys=False)["EPS_TTM"].pct_change(4)
    out["EPS_TTM_yoy"] = out["EPS_TTM_yoy"].replace([np.inf, -np.inf], np.nan)
    return out

def attach_preferred_valuation(ratios_df: pd.DataFrame, prefer: str = "PE_over_PB") -> pd.DataFrame:
    """
    Tạo 2 cột:
      - valuation_pref: hệ số định giá ưu tiên (ưu tiên PE_filled, fallback PB_filled)
      - valuation_pref_metric: dùng "PE" hay "PB"
    Quy tắc hợp lệ: chỉ dùng giá trị > 0 (loại bỏ âm/0/inf/NaN).
    """
    out = ratios_df.copy()

    for c in ["PE_filled", "PB_filled"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    pe_valid = out["PE_filled"].gt(0)
    pb_valid = out["PB_filled"].gt(0)

    out["valuation_pref"] = np.nan
    out["valuation_pref_metric"] = np.nan

    if prefer == "PE_over_PB":
        # Ưu tiên PE_filled
        out.loc[pe_valid, "valuation_pref"] = out.loc[pe_valid, "PE_filled"]
        out.loc[pe_valid, "valuation_pref_metric"] = "PE"
        need_fill = out["valuation_pref"].isna() & pb_valid
        out.loc[need_fill, "valuation_pref"] = out.loc[need_fill, "PB_filled"]
        out.loc[need_fill, "valuation_pref_metric"] = "PB"
    elif prefer == "PB_over_PE":
        # Ưu tiên PB_filled
        out.loc[pb_valid, "valuation_pref"] = out.loc[pb_valid, "PB_filled"]
        out.loc[pb_valid, "valuation_pref_metric"] = "PB"
        need_fill = out["valuation_pref"].isna() & pe_valid
        out.loc[need_fill, "valuation_pref"] = out.loc[need_fill, "PE_filled"]
        out.loc[need_fill, "valuation_pref_metric"] = "PE"
    else:
        raise ValueError("prefer must be 'PE_over_PB' or 'PB_over_PE'")

    return out

# ---------- Tính tăng trưởng EPS ----------
def compute_eps_growth(df: pd.DataFrame, frequency: str = "quarterly") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.sort_values(["ticker", "year"] + (["quarter"] if frequency == "quarterly" else []))
    if "EPS" not in out.columns:
        out["EPS"] = np.nan

    if frequency == "quarterly":
        out["EPS_g_qoq"] = out.groupby("ticker", group_keys=False)["EPS"].pct_change()
        out["EPS_g_yoy"] = out.groupby("ticker", group_keys=False)["EPS"].pct_change(4)
    else:
        out["EPS_g_yoy"] = out.groupby("ticker", group_keys=False)["EPS"].pct_change()

    for c in ["EPS_g_qoq", "EPS_g_yoy"]:
        if c in out.columns:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


# ---------- Merge vào OHLCV ----------
def merge_ratios_into_ohlcv(ohlcv: pd.DataFrame, ratios: pd.DataFrame, frequency: str = "quarterly") -> pd.DataFrame:
    o = add_year_quarter_from_timestamp(ohlcv, "timestamp").copy()

    # ép kiểu an toàn
    o["year"] = o["year"].astype("int64")
    o["quarter"] = o["quarter"].astype("int64")

    r = ratios.copy()
    for k in ["year", "quarter"]:
        if k in r.columns:
            r[k] = pd.to_numeric(r[k], errors="coerce").astype("Int64")
            r = r.dropna(subset=[k])
            r[k] = r[k].astype("int64")

    keys = ["ticker", "year"] + (["quarter"] if frequency == "quarterly" else [])

    # đảm bảo ratios có đủ cột khóa
    for k in keys:
        if k not in r.columns:
            r[k] = pd.Series(dtype=o[k].dtype if k in o.columns else "float64")

    merged = o.merge(
        r.drop(columns=["period_end_date"], errors="ignore"),
        on=keys, how="left", validate="m:1"
    )
    return merged

# ---------- Gọi API theo batch ----------
def chunked(lst: List[str], size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def _merge_ratios_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    if not isinstance(src, dict):
        return
    for t, items in src.items():
        if t not in dst:
            dst[t] = []
        if isinstance(items, list):
            dst[t].extend(items)
        elif isinstance(items, dict):
            dst[t].append(items)
        # nếu None hoặc kiểu lạ thì bỏ qua

def get_ratios_batched(client,
                       tickers: List[str],
                       latest_year: int,
                       number_of_period: int,
                       consolidated: bool = True) -> Dict[str, Any] | List[Dict[str, Any]]:
    fa = client.FundamentalAnalysis()
    out_dict: Dict[str, Any] = {}
    out_list: List[Dict[str, Any]] = []

    for i, batch in enumerate(chunked(tickers, BATCH_SIZE), start=1):
        def _once():
            return fa.get_ratios(
                tickers=batch,
                TimeFilter="Quarterly",
                NumberOfPeriod=number_of_period,
                LatestYear=latest_year,
                Consolidated=consolidated,
                Fields=None  # lấy full để tự dò key
            )
        fi_dict = call_with_retry(_once, max_retries=5, base_sleep=0.6, max_sleep=6.0)

        if isinstance(fi_dict, dict):
            # gộp dạng dict (key = ticker)
            for t, items in fi_dict.items():
                out_dict.setdefault(t, [])
                if isinstance(items, list):
                    out_dict[t].extend(items)
                elif isinstance(items, dict):
                    out_dict[t].append(items)
        elif isinstance(fi_dict, list):
            # gộp dạng list record
            out_list.extend([x for x in fi_dict if isinstance(x, dict)])

        time.sleep(BATCH_SLEEP_SEC)

    # Trả về theo format mà flatten xử lý được
    if out_list and not out_dict:
        return out_list
    if out_list and out_dict:
        return {"Data": out_list, **out_dict}
    return out_dict


# ---------- Pipeline chính ----------
def build_ratios_dataframe_for_hnx(
    ohlcv: pd.DataFrame,
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
    type_: str = "consolidated"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tickers = extract_hnx_tickers(ohlcv)
    if not tickers:
        raise ValueError("Không tìm thấy mã nào trong OHLCV.")

    latest_year, number_of_period = compute_quarter_params(from_date, to_date)
    consolidated_flag = (str(type_).lower() == "consolidated")

    client = FiinSession(username=USERNAME, password=PASSWORD).login()

    ratios_raw = get_ratios_batched(
        client=client,
        tickers=tickers,
        latest_year=latest_year,
        number_of_period=number_of_period,
        consolidated=consolidated_flag
    )

    print(f"[INFO] LatestYear={latest_year}, NumberOfPeriod={number_of_period}, tickers={len(tickers)}")

    # Peek top-level
    print("[INFO] ratios_raw type:", type(ratios_raw))
    if isinstance(ratios_raw, dict):
        print("[INFO] top keys sample:", list(ratios_raw.keys())[:5])
        container = _extract_records_container(ratios_raw)
        if container is not None and isinstance(container, list) and container:
            print("[INFO] container first record keys:", list(container[0].keys())[:12])
        else:
            first_key = next(iter(ratios_raw), None)
            if first_key:
                v = ratios_raw[first_key]
                if isinstance(v, list) and v:
                    print("[INFO] first list record keys:", list(v[0].keys())[:12])
                elif isinstance(v, dict):
                    print("[INFO] first dict record keys:", list(v.keys())[:12])
    elif isinstance(ratios_raw, list) and ratios_raw:
        print("[INFO] first list record keys:", list(ratios_raw[0].keys())[:12])

    ratios_df = flatten_ratios_payload(ratios_raw, frequency="quarterly")

    print("[INFO] ratios_df shape:", ratios_df.shape)
    print("[INFO] ratios_df columns:", list(ratios_df.columns))
    print("[INFO] ratios_df head:\n", ratios_df.head(3))

    if not ratios_df.empty:
        ratios_df = ratios_df[
            (ratios_df["period_end_date"] >= from_date) &
            (ratios_df["period_end_date"] <= to_date)
        ].reset_index(drop=True)

    ratios_df = compute_eps_growth(ratios_df, frequency="quarterly")

    ratios_df = attach_pe_ttm(ratios_df, ohlcv)

    ratios_df = attach_pb_ttm(ratios_df, ohlcv)  # <-- thêm dòng này
    ratios_df = attach_eps_ttm_yoy(ratios_df)  # <-- và dòng này

    ratios_df = attach_preferred_valuation(ratios_df, prefer="PE_over_PB")
    print("[INFO] valuation_pref null ratio:", ratios_df["valuation_pref"].isna().mean())

    # rank riêng trong từng metric (rẻ = rank nhỏ)
    ratios_df["valuation_rank_in_metric"] = ratios_df.groupby(
        ["year", "quarter", "valuation_pref_metric"]
    )["valuation_pref"].rank(method="average", ascending=True, na_option="keep")

    print("[INFO] extra cols:", ["EPS_TTM", "price_eoq", "price_eoq_ts", "PE_TTM", "PE_filled"],
          "-> null ratio:", ratios_df[["EPS_TTM", "price_eoq", "PE_TTM", "PE_filled"]].isna().mean().to_dict())

    merged_df = merge_ratios_into_ohlcv(ohlcv, ratios_df, frequency="quarterly")
    return ratios_df, merged_df

# ---------- Ví dụ sử dụng ----------
ohlcv = pd.read_csv("../../data/cleaned_stocks.csv")
ratios_df, merged_df = build_ratios_dataframe_for_hnx(
    ohlcv=ohlcv,
    from_date=FROM_DATE,
    to_date=TO_DATE,
    type_=FIN_TYPE
)

ratios_df.to_csv("../../data/HNX_fundamental_ratios_quarterly.csv", index=False)
merged_df.to_csv("../../data/HNX_ohlcv_with_fundamentals.csv", index=False)