import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH     = "../../data/step_2/part_2_cleaned_ohlcv_with_fundamentals_and_technical.csv"
OUT_SIG     = "../../data/step_3/HNX_signals_daily.csv"
OUT_PICKS   = "../../data/step_3/HNX_picks_daily.csv"
OUT_MERGED  = "../../data/step_3/step_3_merged.csv"

EXCHANGE_FILTER = "HNX"
TOP_N           = 20
MISSING_POLICY  = "ffill_then_drop"

# helpers
def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    prev = (a.shift(1) <= b.shift(1))
    now  = (a > b)
    return (prev & now).astype(int)

def pct_rank_xs_in_universe(s: pd.Series, by_date: pd.Series, universe_mask: pd.Series) -> pd.Series:
    """Rank phần trăm theo NGÀY nhưng chỉ tính trong universe (ví dụ HNX). Ngoài universe -> NaN."""
    out = pd.Series(np.nan, index=s.index, dtype=float)
    s_u = s[universe_mask]
    by_u = by_date[universe_mask]
    out.loc[universe_mask] = s_u.groupby(by_u).rank(pct=True)
    return out

def safe_ratio(a, b):
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

def ensure_cols(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Thiếu cột: {miss}")

# load
df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
df = df.sort_values(["timestamp","ticker"]).reset_index(drop=True)

# missing policy
if MISSING_POLICY == "ffill_then_drop":
    df = df.groupby("ticker", group_keys=False).apply(lambda d: d.ffill()).reset_index(drop=True)
    need_cols = [
        "close","sma_50","sma_200","ema_12","ema_26","macd_hist_12_26_9",
        "rsi_14","adx_14","supertrend","psar",
        "valuation_pref","EPS_TTM_yoy","ROE","volume","Exchange"
    ]
    df = df.dropna(subset=need_cols)

# Rule-based signal (3A)
need_cols_rule = [
    "close","sma_50","sma_200","ema_12","ema_26","macd_hist_12_26_9",
    "rsi_14","adx_14","supertrend","psar",
    "valuation_pref","EPS_TTM_yoy","ROE","volume","Exchange"
]
ensure_cols(df, need_cols_rule)

is_hnx = df["Exchange"].astype(str).str.upper().eq(EXCHANGE_FILTER)

# rank định giá rẻ trong HNX (<= 40% rẻ nhất)
val_rank_hnx = pct_rank_xs_in_universe(df["valuation_pref"], df["timestamp"], is_hnx)
df["val_pct_rank_hnx"] = val_rank_hnx

rule = (
    (df["close"] > df["sma_50"]) &
    (df["sma_50"] > df["sma_200"]) &
    (df["ema_12"] > df["ema_26"]) &
    (df["macd_hist_12_26_9"] > 0) &
    (df["rsi_14"] >= 50) &
    (df["adx_14"] >= 20) &
    (df["close"] > df["supertrend"]) &
    (df["psar"] < df["close"]) &
    (df["val_pct_rank_hnx"] <= 0.40) &
    (df["EPS_TTM_yoy"] > 0) &
    (df["ROE"] > 0) &
    (df["volume"] > 0)
).astype(int)
df["signal_rule_trend_value"] = rule

# Cross signals (3C)
df["signal_cross_gc"] = (
    df.groupby("ticker", group_keys=False)
      .apply(lambda d: cross_up(d["sma_50"], d["sma_200"]))
)

df["signal_cross_macd"] = (
    df.groupby("ticker", group_keys=False)
      .apply(lambda d: cross_up(d["macd_12_26"], d["macd_signal_12_26_9"]))
)

if {"stoch_k_14","stoch_d_14_3"} <= set(df.columns):
    df["signal_cross_stoch"] = (
        df.groupby("ticker", group_keys=False)
          .apply(lambda d: (cross_up(d["stoch_k_14"], d["stoch_d_14_3"]) & (d["stoch_k_14"] < 30)).astype(int))
    )
else:
    df["signal_cross_stoch"] = np.nan

# Composite score (3B)
# tất cả percentile rank thực hiện trong HNX để nhất quán universe
df["score_mom"]   = pct_rank_xs_in_universe(df["macd_hist_12_26_9"].clip(-3, 3), df["timestamp"], is_hnx)
df["score_trend"] = pct_rank_xs_in_universe(safe_ratio(df["close"], df["sma_200"]), df["timestamp"], is_hnx)
df["score_adx"]   = pct_rank_xs_in_universe(df["adx_14"].clip(0, 40), df["timestamp"], is_hnx)
df["score_aroon"] = pct_rank_xs_in_universe((df.get("aroon_up", np.nan) - df.get("aroon_down", np.nan)), df["timestamp"], is_hnx)
# valuation: rẻ hơn -> điểm cao -> 1 - pct_rank
df["score_val"]   = 1 - pct_rank_xs_in_universe(df["valuation_pref"], df["timestamp"], is_hnx)
df["score_eps"]   = pct_rank_xs_in_universe(df["EPS_TTM_yoy"], df["timestamp"], is_hnx)
df["score_roe"]   = pct_rank_xs_in_universe(df["ROE"], df["timestamp"], is_hnx)

w_mom, w_trend, w_val, w_eps, w_roe, w_adx, w_aroon = 0.15, 0.15, 0.30, 0.20, 0.10, 0.05, 0.05
df["score_composite"] = (
    w_mom*df["score_mom"] + w_trend*df["score_trend"] + w_val*df["score_val"]
  + w_eps*df["score_eps"] + w_roe*df["score_roe"] + w_adx*df["score_adx"] + w_aroon*df["score_aroon"]
)

# Rank & Picks Top-N (HNX & pass Rule)
eligible = is_hnx & (df["signal_rule_trend_value"] == 1)

df["rank_composite"] = np.nan
df.loc[eligible, "rank_composite"] = (
    df.loc[eligible].groupby("timestamp")["score_composite"].rank(ascending=False, method="first")
)

df["pick_topN_composite"] = 0
df.loc[eligible & (df["rank_composite"] <= TOP_N), "pick_topN_composite"] = 1
# (nếu < TOP_N mã đủ điều kiện trong ngày, kết quả sẽ ít hơn 20 — đúng yêu cầu)

# Labels cho Step 4 (k = 10/20, excess vs median trong HNX)
for k in (10, 20):
    retk = df.groupby("ticker")["close"].shift(-k) / df["close"] - 1.0
    df[f"fwd_ret_{k}"] = retk

    # median cross-section theo NGÀY nhưng chỉ HNX
    medk_hnx = (
        df.loc[is_hnx, f"fwd_ret_{k}"]
          .groupby(df.loc[is_hnx, "timestamp"])
          .transform("median")
    )
    medk_hnx = medk_hnx.reindex(df.index)
    df[f"label_{k}"] = (retk > medk_hnx).astype(float)

# Xuất file
sig_cols = [
    "timestamp","ticker",
    "signal_rule_trend_value","signal_cross_gc","signal_cross_macd","signal_cross_stoch",
    "score_composite","rank_composite","pick_topN_composite",
    "fwd_ret_10","label_10","fwd_ret_20","label_20"
]
Path(OUT_SIG).parent.mkdir(parents=True, exist_ok=True)
df[sig_cols].to_csv(OUT_SIG, index=False)

# Picks hằng ngày (chỉ HNX & pass Rule & Top-N)
picks = df.loc[df["pick_topN_composite"]==1, ["timestamp","ticker","score_composite","rank_composite","Exchange"]]
picks.to_csv(OUT_PICKS, index=False)

# Merge vào file gốc
base = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
merged = base.merge(df[["timestamp","ticker"] + [c for c in sig_cols if c not in {"timestamp","ticker"}]],
                    on=["timestamp","ticker"], how="left")
merged.to_csv(OUT_MERGED, index=False)

print("Step 3 done.")
print("Signals :", OUT_SIG)
print("Picks   :", OUT_PICKS)
print("Merged  :", OUT_MERGED)
