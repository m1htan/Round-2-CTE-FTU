import sys, time, logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Optional GBMs
HAS_XGB = False
HAS_LGBM = False
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier  # type: ignore
    HAS_LGBM = True
except Exception:
    pass

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Paths
IN_PATH     = "../../data/step_2/part_2_cleaned_ohlcv_with_fundamentals_and_technical.csv"
OUT_ML      = "../../data/step_4/HNX_ml_scores_daily.csv"
OUT_PICKS   = "../../data/step_4/HNX_ml_picks_daily.csv"
OUT_MERGED  = "../../data/step_4/step_4_merged.csv"
OUT_IMP     = "../../data/step_4/HNX_ml_feature_importance_daily.csv"

Path("../../data/step_4").mkdir(parents=True, exist_ok=True)

# FAST MODE knobs
FAST_MODE = True
EXCHANGE_FILTER   = "HNX"
HORIZONS          = (10, 20)
MIN_TRAIN_DAYS    = 126                 # ~6 tháng
RETRAIN_EVERY_N_D = 10                  # retrain mỗi 10 ngày
TOP_N             = 20                  # chấp nhận < TOP_N nếu không đủ mã
VAL_DAYS_FOR_IMPORTANCE = 0             # 0 = tắt permutation-importance
CHKPT_EVERY       = 50                  # checkpoint mỗi 50 ngày
LOG_EVERY         = 10                  # in log mỗi 10 ngày
RANDOM_SEED       = 2024

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../../logs/step4_progress.log", mode="w")
    ],
)
logger = logging.getLogger("step4")

# Load & basic prep
df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
df = df[df["Exchange"].astype(str).str.upper().eq(EXCHANGE_FILTER)].copy()
df = df.groupby("ticker", group_keys=False).apply(lambda d: d.ffill()).reset_index(drop=True)

# Features
FEATS = [
    # Trend / Momentum
    "macd_hist_12_26_9","macd_12_26","macd_signal_12_26_9","rsi_14","adx_14",
    "aroon_up","aroon_down","stoch_k_14","stoch_d_14_3",
    "close","sma_20","sma_50","sma_200","ema_12","ema_26","wma_20",
    "bb_mid_20","bb_up_20_2","bb_low_20_2",
    # Volatility / Volume
    "atr_14","obv","vwap_14","mfi_14","volume","vma_sma_20","vma_ema_20",
    # Value / Quality
    "valuation_pref","EPS_TTM_yoy","ROE","PB_filled","PE_filled",
]

def safe_ratio(a, b):
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)

if {"close","sma_200"} <= set(df.columns):
    df["feat_close_div_sma200"] = safe_ratio(df["close"], df["sma_200"])
    FEATS.append("feat_close_div_sma200")

FEATS = [c for c in FEATS if c in df.columns]
if not FEATS:
    raise ValueError("Không có feature nào. Hãy chắc chắn đã chạy Step 2 & 3 trước.")

# Labels (excess vs median cross-section HNX)
for k in HORIZONS:
    fwd = df.groupby("ticker")["close"].shift(-k) / df["close"] - 1.0
    med = fwd.groupby(df["timestamp"]).transform("median")
    df[f"fwd_ret_{k}"] = fwd
    df[f"label_{k}"]   = (fwd > med).astype(int)

# Clean ML frame
df_ml = df.copy()
df_ml[FEATS] = df_ml[FEATS].replace([np.inf, -np.inf], np.nan)
df_ml = df_ml.dropna(subset=FEATS + [f"label_{k}" for k in HORIZONS])

# ép dtype cho nhanh & tiết kiệm RAM
for c in FEATS:
    df_ml[c] = pd.to_numeric(df_ml[c], errors="coerce").astype("float32")
df_ml["ticker"] = df_ml["ticker"].astype("category")

logger.info(f"Data rows={len(df_ml):,} | tickers={df_ml['ticker'].nunique()} | days={df_ml['timestamp'].nunique()} | feats={len(FEATS)}")

# Model zoo (FAST)
def build_models() -> List[Tuple[str, Any, bool]]:
    models: List[Tuple[str, Any, bool]] = []

    # 1) Logistic Regression (cần scaling)
    models.append(("lr", LogisticRegression(max_iter=250, class_weight="balanced",
                                            random_state=RANDOM_SEED), True))

    # 2) Random Forest (không cần scaling)
    models.append(("rf", RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_SEED, n_jobs=-1
    ), False))

    # 3) (tuỳ chọn) LGBM / XGB (nếu sẵn lib)
    if HAS_LGBM:
        models.append(("lgbm", LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=RANDOM_SEED, n_jobs=-1
        ), False))
    if HAS_XGB:
        models.append(("xgb", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=RANDOM_SEED, tree_method="hist",
            eval_metric="logloss", n_jobs=-1
        ), False))
    return models

# Fit+Predict one model
def fit_predict_one(
    name: str, est: Any, needs_scaling: bool,
    train_df: pd.DataFrame, test_df: pd.DataFrame, feats: List[str], label_col: str
) -> Tuple[pd.Series, Dict[str, Any]]:
    X_train = train_df[feats].values
    y_train = train_df[label_col].values.astype(int)
    X_test  = test_df[feats].values

    scaler = None
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    est.fit(X_train, y_train)
    proba = est.predict_proba(X_test)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X_test)
    proba_s = pd.Series(proba, index=test_df.index)

    # explainability: LR coef / tree FI
    expl: Dict[str, Any] = {}
    if name == "lr" and hasattr(est, "coef_"):
        coef = est.coef_.ravel()
        expl_df = pd.DataFrame({"feature": feats, "importance": coef})
        expl = {"type": "coef_std", "values": expl_df}

    if name in {"rf","xgb","lgbm"} and hasattr(est, "feature_importances_"):
        fi = est.feature_importances_.ravel()
        expl_df = pd.DataFrame({"feature": feats, "importance": fi})
        expl = {"type": "gain_importance", "values": expl_df}

    return proba_s, expl

# Walk-forward train & predict
all_dates = np.sort(df_ml["timestamp"].unique())
logger.info(f"Total days: {len(all_dates)} | MIN_TRAIN_DAYS={MIN_TRAIN_DAYS} | RETRAIN_EVERY={RETRAIN_EVERY_N_D}")

proba_cols_all: List[str] = []
imp_rows: List[Dict[str, Any]] = []

for k in HORIZONS:
    label_col = f"label_{k}"
    model_specs = build_models()
    model_names = [m[0] for m in model_specs]
    logger.info(f"[H{k}] models={model_names}")

    # holder
    proba_holder = {name: pd.Series(index=df_ml.index, dtype=float) for name in model_names}

    # mốc bắt đầu
    start_idx = MIN_TRAIN_DAYS if len(all_dates) >= MIN_TRAIN_DAYS else len(all_dates)
    last_fit_i = -10**9
    cached_train_df = None

    for i in range(start_idx, len(all_dates)):
        day = all_dates[i]
        train_days = all_dates[:i]
        test_day   = day

        if (i - start_idx) % LOG_EVERY == 0 or i == len(all_dates) - 1:
            logger.info(f"[H{k}] Day {i}/{len(all_dates)-1} -> {pd.Timestamp(day).date()} | train_days={i} | test_rows={df_ml[df_ml['timestamp'].eq(test_day)].shape[0]}")

        # (re)fit
        if (i - last_fit_i) >= RETRAIN_EVERY_N_D:
            t0 = time.time()
            cached_train_df = df_ml[df_ml["timestamp"].isin(train_days)].dropna(subset=[label_col])
            last_fit_i = i
            logger.info(f"[H{k}] Refit @ {pd.Timestamp(day).date()} | train_rows={len(cached_train_df):,} | elapsed={time.time()-t0:.2f}s")

        test_df = df_ml[df_ml["timestamp"].eq(test_day)]
        if cached_train_df is None or cached_train_df.empty or test_df.empty:
            continue
        if cached_train_df[label_col].nunique() < 2:
            continue

        for (name, est, needs_scaling) in model_specs:
            try:
                t1 = time.time()
                proba_s, expl = fit_predict_one(name, est, needs_scaling, cached_train_df, test_df, FEATS, label_col)
                proba_holder[name].loc[proba_s.index] = proba_s.values
                logger.info(f"[H{k}]  - {name} fit+predict: {time.time()-t1:.2f}s | pred_rows={len(proba_s)}")

                if expl and "values" in expl:
                    df_ex = expl["values"].copy()
                    df_ex["train_day"] = pd.Timestamp(train_days[-1])
                    df_ex["model"] = name
                    df_ex["horizon"] = k
                    df_ex["importance_type"] = expl.get("type", "unknown")
                    imp_rows.extend(df_ex.to_dict(orient="records"))
            except Exception as e:
                logger.warning(f"[H{k}]  - {name} ERROR: {e}")
                continue

        # checkpoint
        if (i - start_idx) % CHKPT_EVERY == 0 and i > start_idx:
            tmp_cols = [f"proba_{n}_{k}" for n in model_names]
            tmp = pd.DataFrame({
                "timestamp": df_ml["timestamp"],
                "ticker"   : df_ml["ticker"]
            })
            for n in model_names:
                tmp[f"proba_{n}_{k}"] = proba_holder[n]
            tmp[f"proba_ens_{k}"] = tmp[[f"proba_{n}_{k}" for n in model_names]].mean(axis=1, skipna=True)
            tmp = tmp[tmp["timestamp"] <= day]
            ckpt_path = OUT_ML.replace(".csv", f".ckpt_H{k}.csv")
            tmp.to_csv(ckpt_path, index=False)
            logger.info(f"[H{k}] Checkpoint saved up to {pd.Timestamp(day).date()} -> {ckpt_path}")

    # ghép proba theo model
    for name in model_names:
        col = f"proba_{name}_{k}"
        df_ml[col] = proba_holder[name]
        proba_cols_all.append(col)

    # ensemble = mean (equal-weights)
    ens_col = f"proba_ens_{k}"
    df_ml[ens_col] = df_ml[[f"proba_{n}_{k}" for n in model_names]].mean(axis=1, skipna=True)
    proba_cols_all.append(ens_col)

# Xuất scores
score_cols = ["timestamp","ticker"] + proba_cols_all + [f"label_{k}" for k in HORIZONS] + [f"fwd_ret_{k}" for k in HORIZONS]
df_ml[score_cols].to_csv(OUT_ML, index=False)

# Picks theo ensemble (Top-N; chấp nhận < TOP_N)
picks_list = []
for day, g in df_ml.groupby("timestamp"):
    for k in HORIZONS:
        col = f"proba_ens_{k}"
        gg = g[["ticker", col]].dropna().sort_values(col, ascending=False).head(TOP_N)
        for rank, (_, r) in enumerate(gg.iterrows(), start=1):
            picks_list.append({
                "timestamp": day, "horizon": k, "rank": rank,
                "ticker": r["ticker"], "proba_en": r[col],
            })
picks_df = pd.DataFrame(picks_list).sort_values(["timestamp","horizon","rank"])
picks_df.to_csv(OUT_PICKS, index=False)

# Explainability log
if imp_rows:
    imp_df = pd.DataFrame(imp_rows)
    cols = ["train_day","model","horizon","importance_type","feature","importance"]
    for c in cols:
        if c not in imp_df.columns:
            imp_df[c] = np.nan
    imp_df = imp_df[cols].sort_values(["train_day","horizon","model","importance"], ascending=[True, True, True, False])
    imp_df.to_csv(OUT_IMP, index=False)
else:
    pd.DataFrame(columns=["train_day","model","horizon","importance_type","feature","importance"]).to_csv(OUT_IMP, index=False)

# Merge vào file gốc
base = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
merged = base.merge(df_ml[["timestamp","ticker"] + proba_cols_all], on=["timestamp","ticker"], how="left")
merged.to_csv(OUT_MERGED, index=False)

logger.info("Step 4 FAST MODE done.")
logger.info(f"Scores : {OUT_ML}")
logger.info(f"Picks  : {OUT_PICKS}")
logger.info(f"Merged : {OUT_MERGED}")
logger.info(f"Explains: {OUT_IMP}")
logger.info(f"Models used: {[m[0] for m in build_models()]}")