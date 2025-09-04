import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# CONFIG
IN_BASE        = "../../data/step_2/part_2_cleaned_ohlcv_with_fundamentals_and_technical.csv"
PICKS_MODE     = "RULE"
PICKS_PATH     = "../../data/step_3/HNX_picks_daily.csv"
ML_HORIZON     = 10
EXCHANGE       = "HNX"

TOP_N          = 20
SHIFT_PICK_D   = 1
COST_BPS       = 0
SLIPPAGE_BPS   = 0

BENCH_MODE     = "FILE"
BENCH_PATH     = "../../data/step_1/cleaned_stocks.csv"

# phát hiện stress windows:
STRESS_METHOD  = "BOTH"
RET_WIN_DAYS   = 21
RET_DROP_TH    = -0.10
VOL_WIN_DAYS   = 21
VOL_Q          = 0.90
MAX_WINDOWS    = 5

OUT_DIR        = "../../output_stress_test"

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def read_prices_filter_hnx(path: str, exchange: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp","ticker"])
    df = df[df["Exchange"].astype(str).str.upper().eq(exchange)].copy()
    # dùng close & volume
    keep = ["timestamp","ticker","close","volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong base file: {missing}")
    return df[keep]

def pivot_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = (prices.pivot_table(index="timestamp", columns="ticker", values="close", aggfunc="last")
                 .sort_index())
    vol = (prices.pivot_table(index="timestamp", columns="ticker", values="volume", aggfunc="sum")
                  .sort_index())
    # returns close->close ngày T+1, KHÔNG fill NA (để tránh warning & behaviour không mong muốn)
    ret = px.pct_change(fill_method=None).shift(-1)
    return ret, vol

def build_benchmark(ret: pd.DataFrame, vol: pd.DataFrame) -> tuple[pd.Series, str]:
    mode = BENCH_MODE.upper()
    if mode == "FILE":
        b = pd.read_csv(BENCH_PATH, parse_dates=["timestamp"])

        # sanity checks
        if not {"timestamp", "close"} <= set(b.columns):
            raise ValueError("Benchmark file phải có cột 'timestamp' và 'close'.")

        # sắp xếp & loại bỏ trùng timestamp (giữ bản ghi cuối cùng)
        b = b.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        dups = b["timestamp"].duplicated(keep="last").sum()
        if dups:
            print(f"[WARN] Benchmark có {dups} timestamp bị trùng -> keep='last'.")

        b = b.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")
        # đảm bảo index thực sự unique
        b = b[~b.index.duplicated(keep="last")]

        bench = b["close"].pct_change(fill_method=None).shift(-1)

        # căn ngày theo universe (ret.index đã unique do pivot_table)
        bench = bench.reindex(ret.index)
        return bench, "HNX-Index (file)"

    if mode == "HNX_EW":
        return ret.mean(axis=1), "HNX Equal-Weight (proxy)"
    if mode == "HNX_MEDIAN":
        return ret.median(axis=1), "HNX Median (proxy)"
    if mode == "HNX_VOLW":
        w = vol.div(vol.sum(axis=1), axis=0).fillna(0.0)
        return (w * ret).sum(axis=1), "HNX Volume-Weight (proxy)"

    # mặc định
    return ret.mean(axis=1), "HNX Equal-Weight (proxy)"

def read_picks() -> pd.DataFrame:
    pk = pd.read_csv(PICKS_PATH, parse_dates=["timestamp"])
    if PICKS_MODE.upper() == "ML":
        need = {"timestamp","horizon","rank","ticker"}
        miss = need - set(pk.columns)
        if miss:
            raise ValueError(f"Thiếu cột trong ML picks: {miss}")
        pk = pk[pk["horizon"].astype(int).eq(ML_HORIZON)].copy()
        pk = pk.sort_values(["timestamp","rank"])
    else:
        need = {"timestamp","ticker"}
        miss = need - set(pk.columns)
        if miss:
            raise ValueError(f"Thiếu cột trong RULE picks: {miss}")
        if "rank_composite" in pk.columns:
            pk = pk.sort_values(["timestamp","rank_composite"])
        else:
            pk = pk.sort_values(["timestamp","ticker"])
    return pk

def holdings_daily(picks: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    # danh sách top-N theo ngày
    top_by_day = {}
    for day, g in picks.groupby("timestamp"):
        tickers = list(g["ticker"].head(TOP_N))
        top_by_day[day] = tickers

    days = ret.index
    tickers_all = ret.columns
    W = pd.DataFrame(0.0, index=days, columns=tickers_all)

    # áp dụng SHIFT_PICK_D: dùng picks của ngày t-1 cho ngày t
    shifted_days = list(days)
    for i, day in enumerate(shifted_days):
        src_i = i - SHIFT_PICK_D
        if src_i < 0:
            continue
        src_day = shifted_days[src_i]
        if src_day not in top_by_day:
            continue
        sel = [t for t in top_by_day[src_day] if t in tickers_all]
        if len(sel) == 0:
            continue
        w = 1.0 / len(sel)
        W.loc[day, sel] = w
    return W

def portfolio_returns(W: pd.DataFrame, ret: pd.DataFrame, cost_bps=0, slip_bps=0) -> tuple[pd.Series, pd.Series]:
    ret_aligned = ret.reindex_like(W)
    gross = (W * ret_aligned).sum(axis=1)

    dW = (W - W.shift(1)).abs().sum(axis=1)
    # chi phí per unit turnover
    cost_perc = (cost_bps + slip_bps) / 1e4
    costs = dW * cost_perc

    net = gross - costs.fillna(0.0)
    return net.fillna(0.0), dW.fillna(0.0)

def rolling_vol(x: pd.Series, win: int) -> pd.Series:
    # annualized (xấp xỉ) để so sánh mức cao/thấp
    return x.rolling(win).std() * np.sqrt(252)

def find_stress_windows(bench: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str, float]]:
    s = bench.copy().dropna()
    if s.empty:
        return []
    tags = pd.Series("", index=s.index)

    # điều kiện 1: rơi mạnh trong RET_WIN_DAYS
    ret_win = s.rolling(RET_WIN_DAYS).apply(lambda a: np.prod(1+a) - 1, raw=False)
    cond_drop = ret_win <= RET_DROP_TH

    # điều kiện 2: vol spike
    vol = rolling_vol(s, VOL_WIN_DAYS)
    th = vol.quantile(VOL_Q)
    cond_vol = vol >= th

    sel = None
    if STRESS_METHOD == "RET_DROP":
        sel = cond_drop
        tags[sel] = "RET_DROP"
    elif STRESS_METHOD == "VOL_SPIKE":
        sel = cond_vol
        tags[sel] = "VOL_SPIKE"
    else:
        sel = cond_drop | cond_vol
        tags[cond_drop & ~cond_vol] = "RET_DROP"
        tags[cond_vol & ~cond_drop] = "VOL_SPIKE"
        tags[cond_drop & cond_vol]  = "BOTH"

    # gom thành đoạn liên tục
    sel_idx = s.index[sel.fillna(False)]
    windows = []
    if len(sel_idx) == 0:
        return windows

    start = sel_idx[0]
    prev  = sel_idx[0]
    for t in sel_idx[1:]:
        if (t - prev).days > 3:  # gap lớn -> kết thúc đoạn
            end = prev
            tag = tags.loc[start:end].mode().iloc[0] if not tags.loc[start:end].empty else ""
            # severity = hiệu suất benchmark của đoạn
            sev = (1 + s.loc[start:end]).prod() - 1
            windows.append((start, end, tag, sev))
            start = t
        prev = t
    # đoạn cuối
    end = prev
    tag = tags.loc[start:end].mode().iloc[0] if not tags.loc[start:end].empty else ""
    sev = (1 + s.loc[start:end]).prod() - 1
    windows.append((start, end, tag, sev))

    # chọn MAX_WINDOWS nghiêm trọng nhất (sev nhỏ nhất)
    windows = sorted(windows, key=lambda x: x[3])[:MAX_WINDOWS]
    return windows

def section_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return dict(cumret=np.nan, annret=np.nan, vol=np.nan, sharpe=np.nan,
                    mdd=np.nan, winrate=np.nan, n_days=0)
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq/peak - 1
    mdd = dd.min()
    ann = (eq.iloc[-1])**(252/max(1,len(r))) - 1 if len(r)>0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = ann/vol if vol>0 else np.nan
    winrate = (r > 0).mean()
    return dict(cumret=eq.iloc[-1]-1, annret=ann, vol=vol, sharpe=sharpe, mdd=mdd,
                winrate=winrate, n_days=len(r))

def summarize_trading(W: pd.DataFrame, turnover: pd.Series, ret: pd.Series) -> dict:
    # số giao dịch xấp xỉ = tổng số lần thêm mã (số weight từ 0 -> >0)
    adds = ((W > 0) & (W.shift(1) == 0)).sum(axis=1).sum()
    # tail loss (p5)
    p5 = ret.quantile(0.05)
    return dict(trades=int(adds), avg_turnover=turnover.mean(), p5_daily=p5)

def plot_equity(bench, strat, out_png):
    eq_b = (1 + bench.fillna(0)).cumprod()
    eq_s = (1 + strat.fillna(0)).cumprod()
    dd_s = eq_s/eq_s.cummax() - 1

    fig, ax = plt.subplots(2, 1, figsize=(10,7), sharex=True)
    ax[0].plot(eq_b.index, eq_b.values, label="Benchmark", linewidth=1.3)
    ax[0].plot(eq_s.index, eq_s.values, label="Strategy", linewidth=1.6)
    ax[0].legend(); ax[0].set_title("Equity Curve")
    ax[1].plot(dd_s.index, dd_s.values)
    ax[1].set_title("Strategy Drawdown")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def main():
    ensure_dir(OUT_DIR)

    # 1) data
    prices = read_prices_filter_hnx(IN_BASE, EXCHANGE)
    ret, vol = pivot_returns(prices)
    bench, bench_name = build_benchmark(ret, vol)

    # 2) picks -> holdings -> portfolio
    picks = read_picks()
    W     = holdings_daily(picks, ret)
    strat, turnover = portfolio_returns(W, ret, cost_bps=COST_BPS, slip_bps=SLIPPAGE_BPS)

    # 3) stress windows
    wins = find_stress_windows(bench)
    rows = []
    for (s, e, tag, sev) in wins:
        rs = strat.loc[s:e]
        rb = bench.loc[s:e]
        m_s = section_metrics(rs)
        m_b = section_metrics(rb)
        trade_stat = summarize_trading(W.loc[s:e], turnover.loc[s:e], rs)
        rows.append({
            "start": s, "end": e, "tag": tag, "bench_cumret": (1+rb.dropna()).prod()-1,
            "strat_cumret": (1+rs.dropna()).prod()-1,
            "strat_mdd": m_s["mdd"], "strat_winrate": m_s["winrate"], "strat_sharpe": m_s["sharpe"],
            "trades": trade_stat["trades"], "avg_turnover": trade_stat["avg_turnover"], "p5_daily": trade_stat["p5_daily"]
        })
    win_df = pd.DataFrame(rows).sort_values("strat_cumret")
    win_df.to_csv(f"{OUT_DIR}/HNX_stress_windows.csv", index=False)

    # 4) full-period summary & lưu daily
    full_s = section_metrics(strat)
    full_b = section_metrics(bench)
    daily = pd.DataFrame({
        "bench_ret": bench,
        "strat_ret": strat
    })
    # lưu equity & drawdown để xem trên Excel
    daily["bench_eq"] = (1 + daily["bench_ret"].fillna(0)).cumprod()
    daily["strat_eq"] = (1 + daily["strat_ret"].fillna(0)).cumprod()
    daily["strat_dd"] = daily["strat_eq"]/daily["strat_eq"].cummax() - 1
    daily.to_csv(f"{OUT_DIR}/HNX_stress_daily_returns.csv", index=True)

    # 5) plot
    plot_equity(bench, strat, f"{OUT_DIR}/HNX_stress_plot.png")

    # 6) report
    with open(f"{OUT_DIR}/HNX_stress_report.txt","w",encoding="utf-8") as f:
        f.write(f"Benchmark: {bench_name}\n")
        f.write(f"Mode picks: {PICKS_MODE} | TOP_N={TOP_N} | SHIFT={SHIFT_PICK_D} | COST={COST_BPS}bp | SLIP={SLIPPAGE_BPS}bp\n")
        f.write("=== Full period ===\n")
        f.write(f"Strategy: cum={full_s['cumret']:.2%}, ann={full_s['annret']:.2%}, vol={full_s['vol']:.2%}, sharpe={full_s['sharpe']:.2f}, mdd={full_s['mdd']:.2%}, winrate={full_s['winrate']:.2%}\n")
        f.write(f"Benchmark: cum={full_b['cumret']:.2%}, ann={full_b['annret']:.2%}, vol={full_b['vol']:.2%}, sharpe={full_b['sharpe']:.2f}, mdd={full_b['mdd']:.2%}, winrate={full_b['winrate']:.2%}\n\n")
        f.write("=== Stress windows ===\n")
        if win_df.empty:
            f.write("Không phát hiện stress window theo tiêu chí hiện tại.\n")
        else:
            f.write(win_df.to_string(index=False))
            f.write("\n")

    print("Step 6 done.")
    print(f"- Windows  : {OUT_DIR}/HNX_stress_windows.csv")
    print(f"- Daily    : {OUT_DIR}/HNX_stress_daily_returns.csv")
    print(f"- Plot     : {OUT_DIR}/HNX_stress_plot.png")
    print(f"- Report   : {OUT_DIR}/HNX_stress_report.txt")
    print(f"Benchmark  : {bench_name}")

if __name__ == "__main__":
    main()
