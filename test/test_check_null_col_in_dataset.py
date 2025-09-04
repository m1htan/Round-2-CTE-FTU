import pandas as pd

df = pd.read_csv('../main/step_2/cleaned_ohlcv_with_fundamentals_and_technical.csv')

# nhóm cột
not_null_cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]

fundamental_cols = [
    "PB", "ROE", "EPS", "EPS_g_qoq", "EPS_g_yoy", "EPS_TTM", "PE_TTM",
    "PE_filled", "PB_filled", "EPS_TTM_yoy",
    "valuation_pref", "valuation_pref_metric", "valuation_rank_in_metric"
]

technical_cols = [
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "wma_20",
    "vma_sma_20", "vma_ema_20", "rsi_14", "macd_12_26", "macd_signal_12_26_9",
    "macd_hist_12_26_9", "bb_mid_20", "bb_up_20_2", "bb_low_20_2", "atr_14",
    "stoch_k_14", "stoch_d_14_3", "mfi_14", "obv", "vwap_14", "adx_14",
    "di_plus_14", "di_minus_14", "psar", "supertrend", "supertrend_hband",
    "supertrend_lband", "ichimoku_a", "ichimoku_b", "kijun_sen", "tenkan_sen",
    "chikou_span", "aroon", "aroon_up", "aroon_down", "zigzag", "fvg",
    "swing_HL", "ob", "liquidity"
]

# check null
must_not_null = df[not_null_cols].isnull().any()
fundamental_null = df[fundamental_cols].isnull().any()
technical_null = df[technical_cols].isnull().any()

# summary
cols_error = must_not_null[must_not_null].index.tolist()
fundamental_with_null = fundamental_null[fundamental_null].index.tolist()
technical_with_null = technical_null[technical_null].index.tolist()

print("===== 📋 BÁO CÁO TÌNH TRẠNG NULL =====")

if cols_error:
    print("\n❌ Các cột BẮT BUỘC phải có dữ liệu nhưng đang bị thiếu:")
    for col in cols_error:
        missing_count = df[col].isnull().sum()
        print(f"- {col}: {missing_count} giá trị null")
else:
    print("\n✅ Nhóm cột bắt buộc (timestamp, ticker, OHLCV) đầy đủ, không có null.")

if fundamental_with_null:
    print("\nℹ️ Các cột Fundamental có null (chấp nhận được, cần theo dõi):")
    for col in fundamental_with_null:
        missing_count = df[col].isnull().sum()
        print(f"- {col}: {missing_count} giá trị null")
else:
    print("\n✅ Nhóm Fundamental không có null.")

if technical_with_null:
    print("\nℹ️ Các cột Technical có null (bình thường do warm-up hoặc không có tín hiệu):")
    for col in technical_with_null:
        missing_count = df[col].isnull().sum()
        print(f"- {col}: {missing_count} giá trị null")
else:
    print("\n✅ Nhóm Technical không có null.")
