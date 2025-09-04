import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IN_PRICES      = "../../data/step_2/part_2_cleaned_ohlcv_with_fundamentals_and_technical.csv"
PICKS_STEP3    = "../../data/step_3/HNX_picks_daily.csv"
PICKS_STEP4    = "../../data/step_4/HNX_ml_picks_daily.csv"   # horizon column + proba column
BENCH_MODE    = "FILE"
BENCH_PATH    = "../../data/step_1/cleaned_stocks.csv"
OUT_DIR        = "../../output_backtest"

SIGNAL_SOURCE  = "auto"
K_HORIZON      = 10
TOP_N          = 20
TC_BPS         = 10
START_DATE     = None
END_DATE       = None

# Helpers
def ensure_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Thiếu cột: {miss}")

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity/roll_max - 1.0
    return dd.min()

def annualize_ret(daily_ret: pd.Series, freq=252) -> float:
    r = (1 + daily_ret.dropna()).prod()
    n = daily_ret.dropna().shape[0]
    return r**(freq/max(n,1)) - 1 if n > 0 else np.nan

def annualize_vol(daily_ret: pd.Series, freq=252) -> float:
    return daily_ret.dropna().std() * np.sqrt(freq)

def sharpe(daily_ret: pd.Series, rf=0.0, freq=252) -> float:
    mu = daily_ret.dropna().mean()*freq
    vol = annualize_vol(daily_ret, freq)
    return mu/vol if vol and not np.isnan(vol) and vol > 0 else np.nan

def to_equity(daily_ret: pd.Series, start_val=1.0) -> pd.Series:
    return start_val * (1 + daily_ret.fillna(0)).cumprod()

def read_benchmark(path: str, dates: pd.DatetimeIndex) -> pd.Series:
    try:
        b = pd.read_csv(path, parse_dates=["timestamp"])
        ensure_cols(b, ["timestamp","close"])
        b = b.sort_values("timestamp")
        # Lợi suất t→t+1, gán về t để đồng bộ với trọng số
        bench_ret = b.set_index("timestamp")["close"].pct_change().shift(-1)
        # reindex theo tập ngày backtest (không forward-fill lợi suất)
        return bench_ret.reindex(dates)
    except Exception:
        # Proxy: Equal-Weight HNX
        return None

def build_weight_matrix_from_step3(picks: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    # picks: timestamp,ticker,score_composite,rank_composite,Exchange ; chỉ dùng những hàng pick_topN_composite==1 nếu có
    if "pick_topN_composite" in picks.columns:
        P = picks[picks["pick_topN_composite"]==1].copy()
    else:
        # nếu file picks đã là Top-N sẵn thì giữ nguyên
        P = picks.copy()
    P = P[["timestamp","ticker"]].dropna()
    W = pd.DataFrame(0.0, index=dates, columns=tickers)
    for day, g in P.groupby("timestamp"):
        codes = [c for c in g["ticker"].unique() if c in W.columns]
        if len(codes) == 0:
            continue
        w = 1.0/len(codes)
        W.loc[day, codes] = w
    return W

def build_weight_matrix_from_step4(picks: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index, k: int, top_n: int) -> pd.DataFrame:
    # picks: timestamp,horizon,rank,ticker,proba_en|proba_ens
    proba_col = "proba_ens"
    if proba_col not in picks.columns:
        # Một số file đặt tên 'proba_en'
        if "proba_en" in picks.columns:
            proba_col = "proba_en"
        # Hoặc picks đã là top-N theo rank => không cần proba
    P = picks.copy()
    if "horizon" in P.columns:
        P = P[P["horizon"] == k]
    P = P.sort_values(["timestamp", proba_col], ascending=[True, False]) if proba_col in P.columns else P
    W = pd.DataFrame(0.0, index=dates, columns=tickers)
    for day, g in P.groupby("timestamp"):
        gg = g[g["ticker"].isin(tickers)]
        if proba_col in gg.columns:
            gg = gg.head(top_n)
        codes = gg["ticker"].tolist()
        if len(codes) == 0:
            continue
        w = 1.0/len(codes)
        W.loc[day, codes] = w
    return W

def build_trade_log(W: pd.DataFrame, ret_fwd: pd.DataFrame) -> pd.DataFrame:
    """Mỗi 'trade' = 1 chu kỳ nắm giữ liên tục của 1 mã (w>0). PnL = tích (1+r) trong chu kỳ - 1."""
    holds = (W > 0).astype(int)
    trades = []
    for col in holds.columns:
        s = holds[col]
        # điểm vào khi 0->1, điểm ra ngay trước khi 1->0
        entries = (s.shift(1, fill_value=0) == 0) & (s == 1)
        exits   = (s.shift(-1, fill_value=0) == 0) & (s == 1)
        entry_idx = list(s.index[entries])
        exit_idx  = list(s.index[exits])
        # khớp số lượng
        n = min(len(entry_idx), len(exit_idx))
        for i in range(n):
            t_in, t_out = entry_idx[i], exit_idx[i]
            # return trong khoảng [t_in, t_out] sử dụng ret_fwd (r_{t} là lợi suất t→t+1)
            rr = ret_fwd.loc[t_in:t_out, col].dropna()
            if rr.empty:
                pnl = np.nan
                days = 0
            else:
                pnl = (1 + rr).prod() - 1.0
                days = rr.shape[0]
            trades.append({"ticker": col, "entry": t_in, "exit": t_out, "days": days, "pnl": pnl})
    return pd.DataFrame(trades)

# Load data
prices = pd.read_csv(IN_PRICES, parse_dates=["timestamp"])
prices = prices[prices["Exchange"].astype(str).str.upper().eq("HNX")].copy()
ensure_cols(prices, ["timestamp","ticker","close"])

if START_DATE:
    prices = prices[prices["timestamp"] >= pd.Timestamp(START_DATE)]
if END_DATE:
    prices = prices[prices["timestamp"] <= pd.Timestamp(END_DATE)]

prices = prices.sort_values(["timestamp","ticker"]).reset_index(drop=True)

# Pivot close (ngày x mã)
close_px = prices.pivot(index="timestamp", columns="ticker", values="close").sort_index()
dates = close_px.index
tickers = close_px.columns

# Lợi suất t→t+1 gán về t (để nhân với trọng số tại t)
ret_fwd = close_px.pct_change().shift(-1)

# Read picks (Step4 ưu tiên, rồi Step3)
use_src = SIGNAL_SOURCE
if use_src == "auto":
    use_src = "step4" if Path(PICKS_STEP4).exists() else "step3"

if use_src == "step4" and Path(PICKS_STEP4).exists():
    picks4 = pd.read_csv(PICKS_STEP4, parse_dates=["timestamp"])
    W0 = build_weight_matrix_from_step4(picks4, dates, tickers, k=K_HORIZON, top_n=TOP_N)
else:
    picks3 = pd.read_csv(PICKS_STEP3, parse_dates=["timestamp"])
    W0 = build_weight_matrix_from_step3(picks3, dates, tickers)

# Chỉ giữ ngày có trong ret_fwd
W0 = W0.reindex(ret_fwd.index).fillna(0.0)

# Nếu một số mã dừng giao dịch (NaN return), phân bổ lại đều cho các mã còn lại trong ngày
mask_valid = ret_fwd.notna().astype(float)
W_eff = (W0 * mask_valid)
row_sum = W_eff.sum(axis=1).replace(0, np.nan)
W = W_eff.div(row_sum, axis=0).fillna(0.0)

# Portfolio returns (net of cost)
gross_ret = (W * ret_fwd).sum(axis=1)

# Turnover & cost (chi phí trên tổng |Δw|)
delta_w = W.diff().abs().sum(axis=1)  # sum |Δw_i|
TC_RATE = TC_BPS / 10000.0
cost = TC_RATE * delta_w
net_ret = gross_ret - cost

# Benchmark
bench_ret = read_benchmark(BENCH_PATH, net_ret.index)
bench_name = "VN-Index"
if bench_ret is None:
    # proxy: EW HNX (không tính phí)
    bench_ret = ret_fwd.mean(axis=1)
    bench_name = "HNX Equal-Weight (proxy)"

# Equity & metrics
eq_port = to_equity(net_ret)
eq_bench = to_equity(bench_ret)

def summarize(label, r):
    return {
        "label": label,
        "CAGR": annualize_ret(r),
        "Vol": annualize_vol(r),
        "Sharpe": sharpe(r),
        "MaxDD": max_drawdown(to_equity(r)),
        "Days": r.dropna().shape[0],
        "HitRate_Daily": (r.dropna() > 0).mean()
    }

sum_port  = summarize("Portfolio (net)", net_ret)
sum_bench = summarize(bench_name, bench_ret)
summary_df = pd.DataFrame([sum_port, sum_bench])

# Trade log & trade metrics
trades = build_trade_log(W, ret_fwd)
if not trades.empty:
    trades["win"] = trades["pnl"] > 0
    trade_summary = {
        "Trades": len(trades),
        "WinRate": trades["win"].mean() if len(trades) else np.nan,
        "MedianPnL": trades["pnl"].median() if len(trades) else np.nan,
        "AvgDays": trades["days"].mean() if len(trades) else np.nan,
    }
else:
    trade_summary = {"Trades": 0, "WinRate": np.nan, "MedianPnL": np.nan, "AvgDays": np.nan}

# Save outputs
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 1) Daily series
out_daily = pd.DataFrame({
    "timestamp": net_ret.index,
    "port_gross_ret": gross_ret.values,
    "turnover": delta_w.values,
    "cost": cost.values,
    "port_net_ret": net_ret.values,
    "bench_ret": bench_ret.values
})
out_daily.to_csv(f"{OUT_DIR}/HNX_backtest_daily.csv", index=False)

# 2) Equity curves
eq_df = pd.DataFrame({"timestamp": eq_port.index, "portfolio": eq_port.values, "benchmark": eq_bench.values})
eq_df.to_csv(f"{OUT_DIR}/HNX_equity_curve.csv", index=False)

# 3) Summary tables
summary_df.to_csv(f"{OUT_DIR}/HNX_backtest_summary.csv", index=False)
pd.DataFrame([trade_summary]).to_csv(f"{OUT_DIR}/HNX_trade_summary.csv", index=False)
trades.to_csv(f"{OUT_DIR}/HNX_trades.csv", index=False)

# 4) Charts
plt.figure()
plt.plot(eq_port.index, eq_port.values, label="Portfolio (net)")
plt.plot(eq_bench.index, eq_bench.values, label=bench_name)
plt.legend(); plt.title("Equity Curve"); plt.xlabel("Date"); plt.ylabel("Equity")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/HNX_equity_curve.png"); plt.close()

dd = eq_port/eq_port.cummax() - 1.0
plt.figure()
plt.plot(dd.index, dd.values)
plt.title("Portfolio Drawdown"); plt.xlabel("Date"); plt.ylabel("Drawdown")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/HNX_drawdown.png"); plt.close()

plt.figure()
net_ret.dropna().hist(bins=50)
plt.title("Distribution of Daily Returns (Net)"); plt.xlabel("Daily return"); plt.ylabel("Freq")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/HNX_daily_return_hist.png"); plt.close()

print("Step 5 done.")
print(f"- Daily series:  {OUT_DIR}/HNX_backtest_daily.csv")
print(f"- Equity curve:  {OUT_DIR}/HNX_equity_curve.csv / .png")
print(f"- Summary:       {OUT_DIR}/HNX_backtest_summary.csv")
print(f"- Trade summary: {OUT_DIR}/HNX_trade_summary.csv")
print(f"- Trades log:    {OUT_DIR}/HNX_trades.csv")
