import os
import numpy as np
import pandas as pd
from typing import Dict
from FiinQuantX import FiinSession
from dotenv import load_dotenv

INPUT_FA = "../../data/step_2/HNX_ohlcv_with_fundamentals.csv"
OUTPUT_TA_ONLY = "../../data/step_2/HNX_technical_indicators.csv"
OUTPUT_FA_MERGED = "../../data/step_2/HNX_ohlcv_with_fundamentals.csv"

USE_FIIN = True
FIIN_FLAGS = {
    "psar": True,
    "ichimoku": True,
    "supertrend": True,
    "aroon": True,
    "zigzag": True,
    "smc_fvg": True,
    "smc_swing": True,
    "smc_bos_choch": True,
    "smc_ob": True,
    "smc_liquidity": True,
}

PARAMS = {
    "sma": [20, 50, 200],
    "ema": [12, 26],
    "wma": [20],
    "vma_sma": [20],      # VMA = Volume Moving Average (SMA)
    "vma_ema": [20],      # VMA = Volume Moving Average (EMA)
    "rsi": 14,
    "macd": (12, 26, 9),
    "bb": (20, 2),
    "atr": 14,
    "stoch": (14, 3),
    "mfi": 14,
    "vwap": 14,           # rolling VWAP
    "adx": 14,
}

REQUIRED_OHLCV = ["timestamp","ticker","open","high","low","close","volume"]

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if miss:
        raise ValueError(f"Thiếu cột bắt buộc: {miss}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["ticker"] = out["ticker"].astype(str).str.upper()
    for c in ["open","high","low","close","volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values(["ticker","timestamp"]).drop_duplicates(["ticker","timestamp"])
    return out

def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()

def ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

def wma(s: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window+1, dtype=float)
    def _w(x): return np.dot(x, weights) / weights.sum()
    return s.rolling(window, min_periods=window).apply(_w, raw=True)

def rsi_wilder(close: pd.Series, window=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_gain = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_core(s: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = macd_line.ewm(alpha=1/signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close: pd.Series, window=20, k=2):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mid, mid + k*std, mid - k*std

def true_range(high, low, close):
    prev_close = close.shift(1)
    return pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)

def atr_wilder(high, low, close, window=14):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

def stoch_osc(high, low, close, window=14, smooth=3):
    ll = low.rolling(window, min_periods=window).min()
    hh = high.rolling(window, min_periods=window).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(smooth, min_periods=smooth).mean()
    return k, d

def mfi_core(high, low, close, volume, window=14):
    tp = (high + low + close) / 3.0
    mf = tp * volume
    direction = np.sign(tp.diff())
    pos = mf.where(direction > 0, 0.0).rolling(window, min_periods=window).sum()
    neg = mf.where(direction < 0, 0.0).rolling(window, min_periods=window).sum()
    mfr = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))

def obv_core(close, volume):
    sign = np.sign(close.diff())
    flow = np.where(sign > 0, volume, np.where(sign < 0, -volume, 0.0))
    return pd.Series(flow, index=close.index).cumsum()

def vwap_rolling(high, low, close, volume, window=14):
    tp = (high + low + close) / 3.0
    pv = tp * volume
    return pv.rolling(window, min_periods=window).sum() / volume.rolling(window, min_periods=window).sum().replace(0, np.nan)

def adx_wilder(high, low, close, window=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr = atr_wilder(high, low, close, window)
    plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return adx, plus_di, minus_di

def add_core_indicators(ohlcv: pd.DataFrame, params: Dict) -> pd.DataFrame:
    o = _ensure_schema(ohlcv).copy()
    g = o.groupby("ticker", group_keys=False)

    # MAs on price
    for w in params["sma"]:
        o[f"sma_{w}"] = g["close"].transform(lambda s: sma(s, w))
    for w in params["ema"]:
        o[f"ema_{w}"] = g["close"].transform(lambda s: ema(s, w))
    for w in params["wma"]:
        o[f"wma_{w}"] = g["close"].transform(lambda s: wma(s, w))

    # VMA = Volume moving averages
    for w in params["vma_sma"]:
        o[f"vma_sma_{w}"] = g["volume"].transform(lambda s: sma(s, w))
    for w in params["vma_ema"]:
        o[f"vma_ema_{w}"] = g["volume"].transform(lambda s: ema(s, w))

    # RSI
    o[f"rsi_{params['rsi']}"] = g["close"].transform(lambda s: rsi_wilder(s, params["rsi"]))

    # MACD
    f, s, sig = params["macd"]
    macd_df = g["close"].apply(lambda x: pd.DataFrame({
        f"macd_{f}_{s}": macd_core(x, f, s, sig)[0],
        f"macd_signal_{f}_{s}_{sig}": macd_core(x, f, s, sig)[1],
        f"macd_hist_{f}_{s}_{sig}": macd_core(x, f, s, sig)[2],
    }))
    o = o.join(macd_df[[f"macd_{f}_{s}", f"macd_signal_{f}_{s}_{sig}", f"macd_hist_{f}_{s}_{sig}"]])

    # Bollinger
    bb_w, bb_k = params["bb"]
    bb = g["close"].apply(lambda x: pd.DataFrame({
        f"bb_mid_{bb_w}": bollinger(x, bb_w, bb_k)[0],
        f"bb_up_{bb_w}_{bb_k}": bollinger(x, bb_w, bb_k)[1],
        f"bb_low_{bb_w}_{bb_k}": bollinger(x, bb_w, bb_k)[2],
    }))
    o = o.join(bb[[f"bb_mid_{bb_w}", f"bb_up_{bb_w}_{bb_k}", f"bb_low_{bb_w}_{bb_k}"]])

    # ATR
    o[f"atr_{params['atr']}"] = g.apply(lambda d: atr_wilder(d["high"], d["low"], d["close"], params["atr"]))

    # Stochastic
    k_w, d_w = params["stoch"]
    stoch = g.apply(lambda d: pd.DataFrame({
        f"stoch_k_{k_w}": stoch_osc(d["high"], d["low"], d["close"], k_w, d_w)[0],
        f"stoch_d_{k_w}_{d_w}": stoch_osc(d["high"], d["low"], d["close"], k_w, d_w)[1],
    }))
    o = o.join(stoch[[f"stoch_k_{k_w}", f"stoch_d_{k_w}_{d_w}"]])

    # MFI
    o[f"mfi_{params['mfi']}"] = g.apply(lambda d: mfi_core(d["high"], d["low"], d["close"], d["volume"], params["mfi"]))

    # OBV
    o["obv"] = g.apply(lambda d: obv_core(d["close"], d["volume"]))

    # Rolling VWAP
    o[f"vwap_{params['vwap']}"] = g.apply(lambda d: vwap_rolling(d["high"], d["low"], d["close"], d["volume"], params["vwap"]))

    # ADX + DI
    adx_w = params["adx"]
    adx_df = g.apply(lambda d: pd.DataFrame({
        f"adx_{adx_w}": adx_wilder(d["high"], d["low"], d["close"], adx_w)[0],
        f"di_plus_{adx_w}": adx_wilder(d["high"], d["low"], d["close"], adx_w)[1],
        f"di_minus_{adx_w}": adx_wilder(d["high"], d["low"], d["close"], adx_w)[2],
    }))
    o = o.join(adx_df[[f"adx_{adx_w}", f"di_plus_{adx_w}", f"di_minus_{adx_w}"]])

    return o

def add_fiin_indicators(df: pd.DataFrame,
                        client: FiinSession,
                        flags: Dict[str, bool]) -> pd.DataFrame:
    fi = client.FiinIndicator()
    out = df.copy()
    g = out.groupby("ticker", group_keys=False)

    # --- helpers giữ nguyên như bạn đã có ---
    def _as_series_like(group_df: pd.DataFrame, values) -> pd.Series:
        arr = np.asarray(values)
        n = min(len(group_df.index), arr.shape[0])
        s = pd.Series(np.nan, index=group_df.index, dtype=float)
        if n > 0:
            s.iloc[:n] = arr[:n]
        return s

    def _as_frame_like(group_df: pd.DataFrame, cols: Dict[str, any]) -> pd.DataFrame:
        data = {name: _as_series_like(group_df, vals) for name, vals in cols.items()}
        return pd.DataFrame(data, index=group_df.index)

    def _cols_ri(d: pd.DataFrame):
        # trả về các Series đã reset về RangeIndex để fi.* không lỗi
        o = d["open"].reset_index(drop=True)
        h = d["high"].reset_index(drop=True)
        l = d["low"].reset_index(drop=True)
        c = d["close"].reset_index(drop=True)
        v = d["volume"].reset_index(drop=True) if "volume" in d else None
        return o, h, l, c, v

    # PSAR
    if flags.get("psar"):
        out["psar"] = g.apply(lambda d: _as_series_like(
            d, fi.psar(_cols_ri(d)[1], _cols_ri(d)[2], _cols_ri(d)[3], step=0.02, max_step=0.2)
        ))

    # Supertrend (+ bands)  <-- chỗ bạn đang lỗi
    if flags.get("supertrend"):
        def _supertrend_block(d):
            o, h, l, c, v = _cols_ri(d)
            return _as_frame_like(d, {
                "supertrend":       fi.supertrend(h, l, c, window=14, multiplier=3.0),
                "supertrend_hband": fi.supertrend_hband(h, l, c, window=14, multiplier=3.0),
                "supertrend_lband": fi.supertrend_lband(h, l, c, window=14, multiplier=3.0),
            })
        sup_df = g.apply(_supertrend_block)
        out = out.join(sup_df[["supertrend","supertrend_hband","supertrend_lband"]])

    # Ichimoku
    if flags.get("ichimoku"):
        def _ichi_block(d):
            o, h, l, c, v = _cols_ri(d)
            return _as_frame_like(d, {
                "ichimoku_a":  fi.ichimoku_a(h, l, c),
                "ichimoku_b":  fi.ichimoku_b(h, l, c),
                "kijun_sen":   fi.ichimoku_base_line(h, l, c),
                "tenkan_sen":  fi.ichimoku_conversion_line(h, l, c),
                "chikou_span": fi.ichimoku_lagging_line(h, l, c),
            })
        ichi_df = g.apply(_ichi_block)
        out = out.join(ichi_df)

    # Aroon
    if flags.get("aroon"):
        aro_df = g.apply(lambda d: _as_frame_like(d, {
            "aroon":      fi.aroon(_cols_ri(d)[1], _cols_ri(d)[2]),
            "aroon_up":   fi.aroon_up(_cols_ri(d)[1], _cols_ri(d)[2]),
            "aroon_down": fi.aroon_down(_cols_ri(d)[1], _cols_ri(d)[2]),
        }))
        out = out.join(aro_df)

    # ZigZag
    if flags.get("zigzag"):
        out["zigzag"] = g.apply(lambda d: _as_series_like(
            d, fi.zigzag(_cols_ri(d)[1], _cols_ri(d)[2], dev_threshold=5.0, depth=10)
        ))

    # SMC: FVG / Swing / BOS & CHoCH / OB / Liquidity
    if flags.get("smc_fvg"):
        out["fvg"] = g.apply(lambda d: _as_series_like(
            d, fi.fvg(_cols_ri(d)[0], _cols_ri(d)[1], _cols_ri(d)[2], _cols_ri(d)[3], join_consecutive=True)
        ))
    if flags.get("smc_swing"):
        out["swing_HL"] = g.apply(lambda d: _as_series_like(
            d, fi.swing_HL(_cols_ri(d)[0], _cols_ri(d)[1], _cols_ri(d)[2], _cols_ri(d)[3], swing_length=50)
        ))
    if flags.get("smc_bos_choch"):
        out["bos"] = g.apply(lambda d: _as_series_like(
            d, fi.break_of_structure(_cols_ri(d)[0], _cols_ri(d)[1], _cols_ri(d)[2], _cols_ri(d)[3], close_break=True, swing_length=50)
        ))
        out["choch"] = g.apply(lambda d: _as_series_like(
            d, fi.chage_of_charactor(_cols_ri(d)[0], _cols_ri(d)[1], _cols_ri(d)[2], _cols_ri(d)[3], close_break=True, swing_length=50)
        ))
    if flags.get("smc_ob"):
        out["ob"] = g.apply(lambda d: _as_series_like(
            d, fi.ob(_cols_ri(d)[0], _cols_ri(d)[1], _cols_ri(d)[2], _cols_ri(d)[3], _cols_ri(d)[4], close_mitigation=False, swing_length=50)
        ))
    if flags.get("smc_liquidity"):
        out["liquidity"] = g.apply(lambda d: _as_series_like(
            d, fi.liquidity(_cols_ri(d)[0], _cols_ri(d)[1], _cols_ri(d)[2], _cols_ri(d)[3], range_percent=0.01, swing_length=50)
        ))

    return out

if __name__ == "__main__":
    # 1) đọc dataset FA (daily)
    base = pd.read_csv(INPUT_FA)
    base = _ensure_schema(base)

    # 2) tính TA core
    ta_all = add_core_indicators(base[REQUIRED_OHLCV + ["Exchange"] if "Exchange" in base.columns else REQUIRED_OHLCV], PARAMS)

    # 3)thêm chỉ báo nâng cao từ FiinQuantX
    if USE_FIIN:
        load_dotenv(dotenv_path='../../config/.env')
        USERNAME = os.getenv("FIINQUANT_USERNAME")
        PASSWORD = os.getenv("FIINQUANT_PASSWORD")
        client = FiinSession(username=USERNAME, password=PASSWORD).login()
        ta_all = add_fiin_indicators(ta_all, client, FIIN_FLAGS)

    # 4) Forward-fill theo ticker cho các cột TA
    ta_cols = [c for c in ta_all.columns if c not in base.columns]  # các cột mới sinh
    ta_all[ta_cols] = ta_all.groupby("ticker")[ta_cols].ffill()

    # 5) Drop warm-up: giữ từ khi có sma_200
    warmup_col = "sma_200" if "sma_200" in ta_all.columns else None
    if warmup_col:
        ta_all = ta_all[~ta_all[warmup_col].isna()].copy()

    # 6) Ghi file TA-only
    ta_only = ta_all[["timestamp","ticker"] + ta_cols].copy()
    ta_only.to_csv(OUTPUT_TA_ONLY, index=False)
    print(f"[OK] Saved TA-only -> {OUTPUT_TA_ONLY} (rows={len(ta_only):,})")

    # 7) Merge vào file FA rồi ghi đè
    #    (chỉ merge cột TA để không đụng các cột FA sẵn có)
    merged = base.merge(ta_only, on=["timestamp","ticker"], how="left")
    # ffill lần nữa cho các ticker trong merged (đảm bảo tính liên tục sau merge)
    merged[ta_cols] = merged.groupby("ticker")[ta_cols].ffill()
    merged.to_csv(OUTPUT_FA_MERGED, index=False)
    print(f"[OK] Merged back -> {OUTPUT_FA_MERGED} (rows={len(merged):,})")
