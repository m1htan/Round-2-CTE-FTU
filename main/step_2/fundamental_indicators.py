from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union
from FiinQuantX import FiinSession

import pandas as pd
from dotenv import load_dotenv


# ----------------------------
# Utilities
# ----------------------------
def _ensure_list(x: Union[str, Iterable[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)

def _safe_get(d: dict, path: str, default=None):
    """Get nested key with dot-path (e.g., 'ProfitabilityRatio.ROE')."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _normalize_ratios_payload(raw) -> pd.DataFrame:
    """
    Chuẩn hóa phản hồi get_ratios thành DataFrame có cột:
      [Ticker, Period, Values]

    Hỗ trợ 3 dạng payload:
    1) dict: { "HPG": { "2024Q4": {...}, ... }, "VCB": {...} }
    2) dict + list: { "HPG": [ {"Period":"2024Q4","Values":{...}}, ... ] }
    3) list: [ {"Ticker":"HPG","Period":"2024Q4","Values":{...}}, ... ]
    """
    records = []

    if raw is None:
        return pd.DataFrame(columns=["Ticker","Period","Values"])

    # Case 3: top-level list
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            ticker = item.get("Ticker") or item.get("ticker")
            period = item.get("Period") or item.get("period") or ""
            values = item.get("Values") or item.get("values") or {
                k: v for k, v in item.items() if k not in ("Ticker","ticker","Period","period")
            }
            if ticker is None:
                # Thử nội suy từ trường khác nếu vendor không đặt "Ticker" trong item
                ticker = item.get("Symbol") or item.get("symbol")
            records.append({"Ticker": ticker, "Period": period, "Values": values})
        return pd.DataFrame.from_records(records)

    # Case 1/2: top-level dict
    if isinstance(raw, dict):
        for ticker, payload in raw.items():
            # Case 1: dict of periods
            if isinstance(payload, dict):
                for period_key, vals in payload.items():
                    records.append({"Ticker": ticker, "Period": period_key, "Values": vals})
            # Case 2: list of {Period, Values}
            elif isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    period = item.get("Period") or item.get("period") or ""
                    values = item.get("Values") or item.get("values") or item
                    records.append({"Ticker": ticker, "Period": period, "Values": values})
            else:
                # Không rõ cấu trúc -> lưu thô
                records.append({"Ticker": ticker, "Period": "", "Values": payload})
        return pd.DataFrame.from_records(records)

    # Fallback: kiểu không hỗ trợ
    return pd.DataFrame(columns=["Ticker","Period","Values"])


def _normalize_fs_payload(raw) -> pd.DataFrame:
    """
    Chuẩn hóa phản hồi get_financeStatement thành [Ticker, Period, Values],
    hỗ trợ top-level dict hoặc list.
    """
    records = []

    if raw is None:
        return pd.DataFrame(columns=["Ticker","Period","Values"])

    if isinstance(raw, list):
        # list phẳng kiểu [{Ticker,Period,Values}, ...] hoặc item thiếu Ticker
        for item in raw:
            if not isinstance(item, dict):
                continue
            ticker = item.get("Ticker") or item.get("ticker") or item.get("Symbol") or item.get("symbol")
            period = item.get("Period") or item.get("period") or ""
            values = item.get("Values") or item.get("values") or {
                k: v for k, v in item.items() if k not in ("Ticker","ticker","Symbol","symbol","Period","period")
            }
            records.append({"Ticker": ticker, "Period": period, "Values": values})
        return pd.DataFrame.from_records(records)

    if isinstance(raw, dict):
        for ticker, payload in raw.items():
            if isinstance(payload, dict):
                for period_key, vals in payload.items():
                    records.append({"Ticker": ticker, "Period": period_key, "Values": vals})
            elif isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    period = item.get("Period") or item.get("period") or ""
                    values = item.get("Values") or item.get("values") or item
                    records.append({"Ticker": ticker, "Period": period, "Values": values})
            else:
                records.append({"Ticker": ticker, "Period": "", "Values": payload})
        return pd.DataFrame.from_records(records)

    return pd.DataFrame(columns=["Ticker","Period","Values"])


_period_re = re.compile(r"(?P<year>\d{4}).*?(?P<q>[1-4])", re.IGNORECASE)

def _period_sort_key(period: str) -> Tuple[int,int]:
    """
    Best-effort parse Period to (year, quarter). Works with '2024Q4', 'Q4-2024', etc.
    Unknown -> (0,0).
    """
    if not isinstance(period, str):
        return (0,0)
    m = _period_re.search(period)
    if not m:
        # try year only
        y = re.findall(r"\d{4}", period)
        return (int(y[0]) if y else 0, 0)
    return (int(m.group("year")), int(m.group("q")))

def _sum_skip_none(series: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in series if v is not None]
    return sum(vals) if vals else None

def _div_safe(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0):
        return None
    return a / b

# ----------------------------
# Core
# ----------------------------
@dataclass
class FundamentalIndicators:
    client: object  # FiinQuantX session (already logged in)

    # Ratio-level field map (from get_ratios)
    default_field_map: Dict[str, str] = field(default_factory=lambda: {
        # Valuation
        "PE": "ValuationRatio.PE",
        "PB": "ValuationRatio.PB",
        "DividendYield": "ValuationRatio.DividendYield",
        # Some vendors may expose EV/EBITDA directly:
        "EV_EBITDA": "ValuationRatio.EV_EBITDA",  # primary candidate
        # alt fallbacks will be tried at runtime
        # Profitability
        "ROE": "ProfitabilityRatio.ROE",
        "ROA": "ProfitabilityRatio.ROA",
        "NetMargin": "ProfitabilityRatio.NetMargin",
        "GrossMargin": "ProfitabilityRatio.GrossMargin",
        "OperatingMargin": "ProfitabilityRatio.OperatingMargin",
        "EBITDAMargin": "ProfitabilityRatio.EBITDAMargin",
        # Growth
        "SalesGrowth": "GrowthRatio.SalesGrowth",
        "EPSGrowth": "GrowthRatio.EPSGrowth",
        # Efficiency / Leverage / Liquidity
        "AssetTurnover": "EfficiencyRatio.AssetTurnover",
        "DebtToEquity": "LeverageRatio.DebtToEquity",
        "InterestCoverage": "LeverageRatio.InterestCoverage",
        "CurrentRatio": "LiquidityRatio.CurrentRatio",
        "QuickRatio": "LiquidityRatio.QuickRatio",
        # Per-share
        "EPS": "PerShare.EPS",
        "BVPS": "PerShare.BVPS",
        "SPS": "PerShare.SPS",
        # FCF per share (if vendor provides)
        "FCFPS": "PerShare.FCFPS",
    })

    # Finance statement field map (from get_financeStatement)
    # These are *section or dot paths* to extract raw values if ratios don't provide.
    default_fs_field_map: Dict[str, str] = field(default_factory=lambda: {
        "Revenue": "IncomeStatement.Revenue",
        "EBITDA": "IncomeStatement.EBITDA",
        "OperatingCashFlow": "CashFlow.OperatingCashFlow",
        "CapitalExpenditure": "CashFlow.CapitalExpenditure",  # aka CAPEX (usually negative outflow)
        "FreeCashFlow": "CashFlow.FreeCashFlow",
        "NetIncome": "IncomeStatement.NetIncome",
        "SharesOutstanding": "PerShare.SharesOutstanding",
        "TotalDebt": "BalanceSheet.TotalDebt",
        "CashAndCashEquivalents": "BalanceSheet.CashAndCashEquivalents",
    })

    # Alternate ratio paths to try if main key missing
    ev_ebitda_ratio_fallbacks: List[str] = field(default_factory=lambda: [
        "ValuationRatio.EVOverEBITDA",
        "ValuationRatio.EVToEBITDA",
        "ValuationRatio.EV_EBITDA",
    ])

    def fetch_ratios(
        self,
        tickers: Union[str, Iterable[str]],
        timefilter: str,
        latest_year: int,
        n_periods: int,
        consolidated: bool,
        fields: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        tickers = _ensure_list(tickers)
        kwargs = dict(
            tickers=tickers,
            TimeFilter=timefilter,
            LatestYear=latest_year,
            NumberOfPeriod=n_periods,
            Consolidated=consolidated,
        )
        if fields:
            kwargs["Fields"] = list(fields)
        fa = self.client.FundamentalAnalysis()
        raw = fa.get_ratios(**kwargs)
        df = _normalize_ratios_payload(raw)
        df["TimeFilter"] = timefilter
        df["Consolidated"] = consolidated
        return df

    def fetch_finance_statements(
            self,
            tickers: Union[str, Iterable[str]],
            statement: Union[str, Iterable[str]] = "full",
            years: Optional[Iterable[int]] = None,
            quarters: Optional[Iterable[int]] = None,
            audited: bool = True,
            type_: str = "consolidated",
            fields: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Gọi FiinQuantX FundamentalAnalysis().get_financeStatement và chuẩn hóa.
        Lưu ý: SDK thực tế KHÔNG nhận list/tuple cho 'statement', nên:
          - nếu statement là iterable và có >1 phần tử -> dùng 'full'
          - nếu statement là iterable và có 1 phần tử -> lấy phần tử đó (string)
          - nếu là string -> giữ nguyên
        """
        tickers = _ensure_list(tickers)

        # Chuẩn hóa 'statement' thành string
        if isinstance(statement, (list, tuple, set)):
            statement = list(statement)
            if len(statement) == 0:
                statement = "full"
            elif len(statement) == 1:
                statement = str(statement[0])
            else:
                statement = "full"
        elif not isinstance(statement, str) or not statement:
            statement = "full"

        kwargs = dict(
            tickers=tickers,
            statement=statement,  # <-- luôn là string
            years=list(years) if years is not None else None,
            quarters=list(quarters) if quarters is not None else None,
            audited=audited,
            type=type_,
        )
        if fields:
            kwargs["fields"] = list(fields)

        fa = self.client.FundamentalAnalysis()
        raw = fa.get_financeStatement(**kwargs)
        return _normalize_fs_payload(raw)

    # -------- Extraction helpers --------
    def _extract_ratios(self, values_dict: dict, ratio_map: Dict[str,str]) -> Dict[str, Optional[float]]:
        out = {}
        for name, path in ratio_map.items():
            out[name] = _safe_get(values_dict, path, default=None)
        # EV/EBITDA: if primary None, try fallbacks
        if out.get("EV_EBITDA") is None:
            for p in self.ev_ebitda_ratio_fallbacks:
                val = _safe_get(values_dict, p, default=None)
                if val is not None:
                    out["EV_EBITDA"] = val
                    break
        return out

    def _extract_fs_values(self, values_dict: dict, fs_map: Dict[str,str]) -> Dict[str, Optional[float]]:
        out = {}
        for k, path in fs_map.items():
            out[k] = _safe_get(values_dict, path, default=None)
        return out

    # -------- TTM aggregation helpers --------
    def _build_ttm(self, df_q: pd.DataFrame, cols_sum: List[str], cols_ratio_defs: Dict[str, Tuple[str,str]]) -> pd.DataFrame:
        """
        Compute TTM per ticker using last 4 quarterly periods:
        - cols_sum: fields to sum across 4 quarters (e.g., Revenue, EBITDA, OCF, CAPEX, FCF, NetIncome)
        - cols_ratio_defs: mapping {new_col: (num_col, den_col)} -> compute sum(num)/sum(den)
        """
        if df_q.empty:
            return pd.DataFrame(columns=df_q.columns)

        df_q = df_q.copy()
        df_q["_year"], df_q["_q"] = zip(*df_q["Period"].map(_period_sort_key))
        df_q = df_q.sort_values(["Ticker","_year","_q"])

        out_rows = []
        for tkr, g in df_q.groupby("Ticker", sort=False):
            g = g[g["_q"]>0]  # quarterly rows only
            if len(g) < 4:
                continue
            # rolling window of 4
            for i in range(3, len(g)):
                win = g.iloc[i-3:i+1]
                row = {
                    "Ticker": tkr,
                    "Period": f"TTM@{win.iloc[-1]['Period']}",
                    "TimeFilter": "Quarterly",
                    "Consolidated": win.iloc[-1]["Consolidated"],
                }
                # sums
                for c in cols_sum:
                    row[f"TTM:{c}"] = _sum_skip_none(win[c]) if c in win else None
                # ratios as sum(num)/sum(den)
                for newc, (numc, denc) in cols_ratio_defs.items():
                    s_num = _sum_skip_none(win[numc]) if numc in win else None
                    s_den = _sum_skip_none(win[denc]) if denc in win else None
                    row[newc] = _div_safe(s_num, s_den)
                out_rows.append(row)
        if not out_rows:
            return pd.DataFrame(columns=["Ticker","Period","TimeFilter","Consolidated"] + [f"TTM:{c}" for c in cols_sum] + list(cols_ratio_defs.keys()))
        return pd.DataFrame(out_rows)

    # -------- Main API --------
    def build_indicator_frame(
        self,
        tickers: Union[str, Iterable[str]],
        timefilter: str = "Quarterly",
        latest_year: int = 2024,
        n_periods: int = 12,
        consolidated: bool = True,
        include: Optional[Iterable[str]] = None,
        # FS enrichment
        addl_fields_fs: Optional[Iterable[str]] = None,
        fs_statement: Union[str, Iterable[str]] = "full",
        fs_years: Optional[Iterable[int]] = None,
        fs_quarters: Optional[Iterable[int]] = None,
        fs_audited: bool = True,
        fs_type: str = "consolidated",
        # mapping overrides
        field_map_override: Optional[Dict[str, str]] = None,
        fs_field_map_override: Optional[Dict[str, str]] = None,
        # TTM options
        compute_ttm: bool = True,
    ) -> pd.DataFrame:

        # 0) Resolve field maps
        ratio_map = self.default_field_map.copy()
        if field_map_override:
            ratio_map.update(field_map_override)

        fs_map = self.default_fs_field_map.copy()
        if fs_field_map_override:
            fs_map.update(fs_field_map_override)

        # 1) Determine indicators to include
        default_include = [
            # base
            "PE","PB","ROE","ROA","EPS","EPSGrowth","SalesGrowth",
            "NetMargin","GrossMargin","OperatingMargin","EBITDAMargin",
            "AssetTurnover","DebtToEquity","InterestCoverage",
            "CurrentRatio","QuickRatio","DividendYield",
            # added
            "FCFPS", "EV_EBITDA",
            # derived from FS when needed:
            "FCF_per_share", "CAPEX_over_Sales", "OCF_over_NetIncome",
        ]
        include = list(include) if include is not None else default_include

        # 2) Fetch ratios (try to request only needed fields)
        request_ratio_paths = []
        for k in include:
            if k in ratio_map:
                request_ratio_paths.append(ratio_map[k])
        # Ensure we also request EV/EBITDA fallbacks
        request_ratio_paths += [p for p in self.ev_ebitda_ratio_fallbacks if p not in request_ratio_paths]

        df_rat = self.fetch_ratios(
            tickers=tickers,
            timefilter=timefilter,
            latest_year=latest_year,
            n_periods=n_periods,
            consolidated=consolidated,
            fields=request_ratio_paths if request_ratio_paths else None,
        )

        # 3) Flatten ratio indicators from 'Values'
        rows = []
        for _, r in df_rat.iterrows():
            vals = r["Values"] if isinstance(r["Values"], dict) else {}
            flat = self._extract_ratios(vals, ratio_map)
            rows.append({
                "Ticker": r["Ticker"],
                "Period": r["Period"],
                "TimeFilter": r["TimeFilter"],
                "Consolidated": r["Consolidated"],
                **flat,
            })
        out = pd.DataFrame(rows)

        # 4) Optionally fetch finance statements to support derived metrics
        fs_df = pd.DataFrame()
        if addl_fields_fs:
            # derive years/quarters if not provided (SDK requires 'years')
            tfq = str(timefilter).lower().startswith("quarter")
            if fs_years is None:
                need_years = max(1, (n_periods + 3) // 4) if tfq else max(1, n_periods)
                fs_years_eff = [latest_year - i for i in range(need_years)]
            else:
                fs_years_eff = list(fs_years)

            fs_quarters_eff = list(fs_quarters) if fs_quarters is not None else ([1, 2, 3, 4] if tfq else None)

            fs_df = self.fetch_finance_statements(
                tickers=tickers,
                statement=fs_statement,  # nhớ đã vá thành "full" ở hàm fetch_finance_statements
                years=fs_years_eff,  # <-- luôn là list[int], không để None
                quarters=fs_quarters_eff,  # <-- Quarterly thì [1,2,3,4], Yearly thì None
                audited=fs_audited,
                type_=fs_type,
                fields=addl_fields_fs,
            )
            ...

            # Build a flattened FS table with targeted fields (using fs_map paths)
            fs_rows = []
            for _, r in fs_df.iterrows():
                vals = r["Values"] if isinstance(r["Values"], dict) else {}
                flat_fs = self._extract_fs_values(vals, fs_map)
                fs_rows.append({
                    "Ticker": r["Ticker"],
                    "Period": r["Period"],
                    **flat_fs,
                })
            fs_flat = pd.DataFrame(fs_rows)
            out = out.merge(fs_flat, on=["Ticker","Period"], how="left")

        # 5) Derive extra metrics from FS if ratio not present
        # FCF per share
        if "FCF_per_share" in include:
            # prefer ratio FCFPS if available
            if "FCFPS" in out.columns:
                out["FCF_per_share"] = out["FCFPS"]
            else:
                fcf = out.get("FreeCashFlow")
                shs = out.get("SharesOutstanding")
                if fcf is not None and shs is not None:
                    out["FCF_per_share"] = out.apply(lambda x: _div_safe(x.get("FreeCashFlow"), x.get("SharesOutstanding")), axis=1)
                else:
                    out["FCF_per_share"] = None

        # EV/EBITDA: nếu ratios không có EV_EBITDA nhưng có EV và EBITDA (hiếm)
        # Ở đây giả định vendor sẽ trả trực tiếp EV/EBITDA qua ratio; nếu không thì để None.
        # (Tự tính EV đòi hỏi giá thị trường -> MarketCap, không có trong FS chuẩn.)
        if "EV_EBITDA" in include and "EV_EBITDA" not in out.columns:
            out["EV_EBITDA"] = None  # giữ cột để downstream không lỗi

        # EBITDA Margin (nếu ratios không có, tự tính từ FS)
        if "EBITDAMargin" in include:
            if "EBITDAMargin" not in out.columns:
                out["EBITDAMargin"] = out.apply(
                    lambda x: _div_safe(x.get("EBITDA"), x.get("Revenue")), axis=1
                ) if ("EBITDA" in out and "Revenue" in out) else None

        # CAPEX/Sales
        if "CAPEX_over_Sales" in include:
            out["CAPEX_over_Sales"] = out.apply(
                lambda x: _div_safe(x.get("CapitalExpenditure"), x.get("Revenue")), axis=1
            ) if ("CapitalExpenditure" in out and "Revenue" in out) else None

        # OCF/NetIncome
        if "OCF_over_NetIncome" in include:
            out["OCF_over_NetIncome"] = out.apply(
                lambda x: _div_safe(x.get("OperatingCashFlow"), x.get("NetIncome")), axis=1
            ) if ("OperatingCashFlow" in out and "NetIncome" in out) else None

        # 6) Sắp xếp cột
        core_cols = ["Ticker","Period","TimeFilter","Consolidated"]
        metric_cols = [c for c in include if c in out.columns]
        derived_cols = [c for c in ["FCF_per_share","CAPEX_over_Sales","OCF_over_NetIncome"] if c in out.columns]
        fs_cols = [c for c in out.columns if c not in core_cols + metric_cols + derived_cols and c not in ["Values"]]
        # Đảm bảo order: core -> metrics -> derived -> FS (nếu có)
        ordered = core_cols + metric_cols + derived_cols + [c for c in fs_cols if c not in metric_cols+derived_cols]
        out = out[ordered].sort_values(["Ticker"], kind="stable").reset_index(drop=True)

        # 7) TTM (Quarterly only) — cộng dồn 4 quý gần nhất
        if compute_ttm and timefilter.lower().startswith("quarter"):
            # Xác định các trường cần sum để hỗ trợ ratios TTM
            cols_sum = [c for c in ["Revenue","EBITDA","OperatingCashFlow","CapitalExpenditure","FreeCashFlow","NetIncome"] if c in out.columns]
            # Các tỷ lệ TTM cần tính từ tổng (num/den)
            cols_ratio_defs = {}
            if "EBITDAMargin" in include:
                # TTM: sum(EBITDA)/sum(Revenue)
                if ("EBITDA" in out.columns) and ("Revenue" in out.columns):
                    cols_ratio_defs["TTM:EBITDAMargin"] = ("EBITDA","Revenue")
            if "CAPEX_over_Sales" in include and ("CapitalExpenditure" in out.columns) and ("Revenue" in out.columns):
                cols_ratio_defs["TTM:CAPEX_over_Sales"] = ("CapitalExpenditure","Revenue")
            if "OCF_over_NetIncome" in include and ("OperatingCashFlow" in out.columns) and ("NetIncome" in out.columns):
                cols_ratio_defs["TTM:OCF_over_NetIncome"] = ("OperatingCashFlow","NetIncome")

            ttm_df = self._build_ttm(out, cols_sum=cols_sum, cols_ratio_defs=cols_ratio_defs)

            # Ghép các trường TTM về frame chính (left join theo (Ticker, Period=TTM@last_q))
            if not ttm_df.empty:
                out = out.merge(ttm_df, on=["Ticker","Period","TimeFilter","Consolidated"], how="left")

        return out

df_tickers = pd.read_csv("../../data/cleaned_stocks.csv")
load_dotenv(dotenv_path='../../config/.env')

USERNAME = os.getenv("FIINQUANT_USERNAME")
PASSWORD = os.getenv("FIINQUANT_PASSWORD")

client = FiinSession(username=USERNAME, password=PASSWORD).login()
fi = FundamentalIndicators(client)

ticker = df_tickers["ticker"].dropna().unique().tolist()

df = fi.build_indicator_frame(
    tickers=ticker,
    timefilter="Quarterly",
    latest_year=2025,
    n_periods=12,
    consolidated=True,
    addl_fields_fs=["IncomeStatement","CashFlow","BalanceSheet","PerShare"],
    fs_statement="full",
    fs_audited=True,
    fs_type="consolidated",
    compute_ttm=True,)

df.to_csv('../../data/df_all.csv')