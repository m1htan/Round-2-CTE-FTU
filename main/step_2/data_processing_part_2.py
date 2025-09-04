import pandas as pd
from typing import Dict, List, Tuple

class NullCleaner:
    def __init__(
        self,
        not_null_cols: List[str] = None,
        fundamental_cols: List[str] = None,
        technical_signal_cols: List[str] = None,   # tín hiệu rời rạc: ob/liquidity/bos/choch/zigzag/fvg...
        technical_indicator_cols: List[str] = None,# chỉ báo liên tục: MA/MACD/RSI/BB/ATR/Ichimoku...
        price_action_cols: List[str] = None,       # zigzag/fvg/swing_HL/ob/liquidity...
        max_lookback: int = 200,                   # số phiên warm-up tối đa cần để tính indicators
        timestamp_col: str = "timestamp",
        ticker_col: str = "ticker"
    ):
        self.timestamp_col = timestamp_col
        self.ticker_col = ticker_col
        self.max_lookback = max_lookback

        self.not_null_cols = not_null_cols or ["timestamp", "ticker", "open", "high", "low", "close", "volume"]

        # nhóm fundamental
        self.fundamental_cols = fundamental_cols or [
    "PB", "ROE", "EPS", "EPS_g_qoq", "EPS_g_yoy", "EPS_TTM", "PE_TTM",
    "PE_filled", "PB_filled", "EPS_TTM_yoy",
    "valuation_pref", "valuation_pref_metric", "valuation_rank_in_metric"
]

        # nhóm technical: tách 2 loại để xử lý khác nhau
        self.technical_signal_cols = technical_signal_cols or [
            # các cột dạng tín hiệu (0/1, sự kiện, vùng): điền 0 khi NaN
            "bos","choch","ob","liquidity"
        ]
        self.technical_indicator_cols = technical_indicator_cols or [
            # các cột cần warm-up (MA/MACD/RSI/BB/ATR/Ichimoku/ADX/ARoon/VWAP/OBV...)
            "sma_20","sma_50","sma_200","ema_12","ema_26","wma_20",
            "vma_sma_20","vma_ema_20","rsi_14",
            "macd_12_26","macd_signal_12_26_9","macd_hist_12_26_9",
            "bb_mid_20","bb_up_20_2","bb_low_20_2","atr_14",
            "stoch_k_14","stoch_d_14_3","mfi_14","obv","vwap_14",
            "adx_14","di_plus_14","di_minus_14",
            "psar","supertrend","supertrend_hband","supertrend_lband",
            "ichimoku_a","ichimoku_b","kijun_sen","tenkan_sen","chikou_span",
            "aroon","aroon_up","aroon_down"
        ]

        # nhóm price action mở rộng
        self.price_action_cols = price_action_cols or [
            "zigzag","fvg","swing_HL"
        ]

        # hợp nhất danh sách để kiểm tra tồn tại cột
        self.all_known_cols = set(
            self.not_null_cols
            + self.fundamental_cols
            + self.technical_signal_cols
            + self.technical_indicator_cols
            + self.price_action_cols
        )

    def _coerce_time_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df = df.copy()
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")
        df = df.sort_values([self.ticker_col, self.timestamp_col]).reset_index(drop=True)
        return df

    def _available(self, df: pd.DataFrame, cols: List[str]) -> List[str]:
        # chỉ dùng những cột thực sự có trong df
        return [c for c in cols if c in df.columns]

    def _report_missing(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        counts = df.isnull().sum()
        total = len(df)
        rep = (
            pd.DataFrame({"missing_count": counts, "percent": (counts / total * 100).round(2)})
            .reset_index().rename(columns={"index": "column"})
        )
        rep["stage"] = label
        return rep.sort_values(["percent","column"], ascending=[False, True])

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self._coerce_time_and_sort(df)

        # Chỉ xét các cột có trong df để tránh lỗi
        not_null_cols = self._available(df, self.not_null_cols)
        fundamental_cols = self._available(df, self.fundamental_cols)
        tech_sig_cols = self._available(df, self.technical_signal_cols)
        tech_ind_cols = self._available(df, self.technical_indicator_cols)
        pa_cols = self._available(df, self.price_action_cols)

        # Báo cáo trước xử lý
        report_before = self._report_missing(df[ list(set(not_null_cols + fundamental_cols + tech_sig_cols + tech_ind_cols + pa_cols)) ], "before")

        # 1) Bảo đảm nhóm bắt buộc (drop dòng lỗi nếu có)
        if not_null_cols:
            # nếu có null ở nhóm bắt buộc → drop hàng tương ứng (giữ dữ liệu sạch cho mọi bước sau)
            before_rows = len(df)
            df = df.dropna(subset=not_null_cols)
            after_rows = len(df)
            # (tuỳ chọn) print(f"Dropped {before_rows - after_rows} rows due to required columns missing.")

        # 2) FUNDAMENTAL: ffill theo ticker → sau đó median theo ngày → sau cùng median toàn bộ
        if fundamental_cols:
            # forward-fill theo ticker (giữ ổn định theo quý)
            df[fundamental_cols] = (
                df.groupby(self.ticker_col, group_keys=False)[fundamental_cols]
                  .apply(lambda g: g.ffill())
            )

            # median theo ngày (cross-sectional): áp dụng cho numeric
            num_fund_cols = [c for c in fundamental_cols if pd.api.types.is_numeric_dtype(df[c])]
            if num_fund_cols:
                # median theo cùng ngày (timestamp) nếu cùng phiên có doanh nghiệp khác
                daily_median = (
                    df.groupby(self.timestamp_col)[num_fund_cols]
                      .transform("median")
                )
                # điền phần còn thiếu bằng daily median
                for c in num_fund_cols:
                    df[c] = df[c].fillna(daily_median[c])

                # fallback: median toàn bộ cột
                for c in num_fund_cols:
                    if df[c].isna().any():
                        df[c] = df[c].fillna(df[c].median())

            # với cột phân loại (vd valuation_pref) → fill "unknown"
            cat_fund_cols = [c for c in fundamental_cols if c not in num_fund_cols]
            for c in cat_fund_cols:
                df[c] = df[c].fillna("unknown")

        # 3) TECHNICAL (tín hiệu rời rạc): tạo cờ null & fill 0 (không có tín hiệu)
        for c in tech_sig_cols:
            flag_col = f"isnull_{c}"
            df[flag_col] = df[c].isna().astype("int8")
            df[c] = df[c].fillna(0)

        # 4) TECHNICAL (chỉ báo liên tục): tạo cờ null, giữ NaN (warm-up), sau đó drop warm-up mỗi ticker
        for c in tech_ind_cols:
            flag_col = f"isnull_{c}"
            df[flag_col] = df[c].isna().astype("int8")

        # Loại bỏ warm-up đầu mỗi ticker: mặc định bỏ tối đa max_lookback dòng đầu
        if tech_ind_cols:
            def drop_warmup(g: pd.DataFrame) -> pd.DataFrame:
                # tìm số NaN "đầu chuỗi" tối đa trong các chỉ báo → cắt bỏ
                # đơn giản hoá: bỏ min(len(g), max_lookback) dòng đầu
                return g.iloc[self.max_lookback:] if len(g) > self.max_lookback else g.iloc[0:0]
            df = (
                df.groupby(self.ticker_col, group_keys=False)
                  .apply(drop_warmup)
                  .reset_index(drop=True)
            )

            # Sau khi bỏ warm-up, phần NaN còn lại (nếu có do lỗ hổng dữ liệu) → fill forward theo ticker
            df[tech_ind_cols] = (
                df.groupby(self.ticker_col, group_keys=False)[tech_ind_cols]
                  .apply(lambda g: g.ffill())
            )

        # 5) PRICE-ACTION: tạo cờ null & fill 0 (ngầm định không có tín hiệu)
        for c in pa_cols:
            flag_col = f"isnull_{c}"
            df[flag_col] = df[c].isna().astype("int8")
            df[c] = df[c].fillna(0)

        # Báo cáo sau xử lý
        consider_cols = list(set(not_null_cols + fundamental_cols + tech_sig_cols + tech_ind_cols + pa_cols))
        report_after = self._report_missing(df[consider_cols], "after")

        # Trả về df sạch + báo cáo gộp
        report = pd.concat([report_before, report_after], axis=0, ignore_index=True)
        return df, report


df = pd.read_csv("../../data/step_2/part_1_cleaned_ohlcv_with_fundamentals_and_technical.csv")
cleaner = NullCleaner(max_lookback=200)
df_clean, null_report = cleaner.clean(df)
print(null_report.head(30))
df_clean.to_csv("../../data/step_2/part_2_cleaned_ohlcv_with_fundamentals_and_technical.csv", index=False)
null_report.to_csv("../../data/step_2/null_report_before_after.csv", index=False)
