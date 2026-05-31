"""
估值数据 Provider — 三源统一入口

优先级: Tushare daily_basic → 本地CSV/DuckDB → AKShare真实值 → FAIL
估算PE/PB 不进入模型，不让Gate通过
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

CSV_DIR = Path(__file__).parent.parent / "data_sources" / "valuation_csv"


def is_valid_valuation(df, min_coverage=0.90):
    """检查估值数据是否有效"""
    if df is None or df.empty:
        return False
    required = ["pe_ttm", "pb", "total_mv"]
    for col in required:
        if col not in df.columns:
            return False
        if df[col].notna().mean() < min_coverage:
            return False
    n_unique_pe = df["pe_ttm"].nunique()
    n_unique_pb = df["pb"].nunique()
    if n_unique_pe < 10 or n_unique_pb < 10:
        return False
    return True


class ValuationProvider:
    """三源估值Provider"""

    def __init__(self, tushare_provider=None, ak_client=None, db=None):
        self.tushare = tushare_provider
        self.ak = ak_client
        self.db = db
        CSV_DIR.mkdir(parents=True, exist_ok=True)

    def get_valuation_daily(self, trade_date, codes):
        """
        获取指定交易日估值数据

        返回 DataFrame 或 None
        """
        # 1. DuckDB 缓存
        if self.db is not None:
            try:
                df = self._load_from_duckdb(trade_date, codes)
                if is_valid_valuation(df):
                    df["valuation_source"] = "duckdb_cache"
                    df["valuation_quality"] = "PASS"
                    return df
            except Exception:
                pass

        # 2. Tushare daily_basic
        df_ts = self._try_tushare(trade_date)
        if is_valid_valuation(df_ts):
            df_ts = df_ts[df_ts["code"].isin(codes)]
            if len(df_ts) > 0:
                df_ts["valuation_source"] = "tushare"
                df_ts["valuation_quality"] = "PASS"
                if self.db is not None:
                    self._save_to_duckdb(df_ts)
                return df_ts

        # 3. 本地 CSV
        df_csv = self._try_csv(trade_date)
        if is_valid_valuation(df_csv):
            df_csv = df_csv[df_csv["code"].isin(codes)]
            if len(df_csv) > 0:
                df_csv["valuation_source"] = "manual_csv"
                df_csv["valuation_quality"] = "PASS"
                if self.db is not None:
                    self._save_to_duckdb(df_csv)
                return df_csv

        # 4. AKShare 真实当前值（仅当日展示，不进历史训练）
        df_ak = self._try_akshare(codes)
        if is_valid_valuation(df_ak, min_coverage=0.50):
            df_ak["valuation_source"] = "akshare_current"
            df_ak["valuation_quality"] = "WARN"
            return df_ak

        # 5. 无有效数据
        return None

    def _load_from_duckdb(self, trade_date, codes):
        from data.duckdb_schema import load_valuation_daily
        return load_valuation_daily(codes, trade_date, trade_date)

    def _save_to_duckdb(self, df):
        from data.duckdb_schema import save_valuation_daily
        save_valuation_daily(df)

    def _try_tushare(self, trade_date):
        if self.tushare is None:
            return None
        try:
            date_str = trade_date if isinstance(trade_date, str) else trade_date.strftime("%Y%m%d")
            df = self.tushare.daily_basic(trade_date=date_str, force=False)
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "ts_code": "code", "trade_date": "trade_date",
                    "pe_ttm": "pe_ttm", "pb": "pb",
                    "ps_ttm": "ps_ttm", "total_mv": "total_mv",
                    "circ_mv": "circ_mv", "turnover_rate": "turnover_rate",
                })
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                return df
        except Exception:
            pass
        return None

    def _try_csv(self, trade_date):
        date_str = trade_date if isinstance(trade_date, str) else trade_date.strftime("%Y-%m-%d")
        path = CSV_DIR / f"valuation_{date_str}.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df
        except Exception:
            return None

    def _try_akshare(self, codes):
        if self.ak is None:
            return None
        try:
            from data.fetcher import is_akshare_available
            if not is_akshare_available():
                return None
            import akshare as ak
            spot = ak.stock_zh_a_spot_em()
            if spot is None or len(spot) < 100:
                return None
            # 字段适配
            code_col = _pick(spot, ["代码", "code"])
            pe_col = _pick(spot, ["市盈率-动态", "市盈率TTM", "市盈率"])
            pb_col = _pick(spot, ["市净率", "市净率PB"])
            mc_col = _pick(spot, ["总市值", "总市值1"])
            if not code_col or not pe_col:
                return None
            df = pd.DataFrame()
            df["code"] = spot[code_col].astype(str).str.zfill(6)
            df["pe_ttm"] = pd.to_numeric(spot[pe_col], errors="coerce")
            df["pb"] = pd.to_numeric(spot[pb_col], errors="coerce") if pb_col else np.nan
            df["total_mv"] = pd.to_numeric(spot[mc_col], errors="coerce") if mc_col else np.nan
            df["circ_mv"] = df["total_mv"] * 0.7
            df["trade_date"] = datetime.now().strftime("%Y-%m-%d")
            df = df[df["code"].isin(codes)]
            df["valuation_source"] = "akshare_current"
            df["valuation_quality"] = "WARN"
            return df
        except Exception:
            return None

    def make_missing_valuation(self, codes, trade_date):
        """生成标记为缺失的估值DataFrame"""
        df = pd.DataFrame({"code": codes})
        df["trade_date"] = trade_date
        df["pe_ttm"] = np.nan
        df["pb"] = np.nan
        df["ps_ttm"] = np.nan
        df["total_mv"] = np.nan
        df["circ_mv"] = np.nan
        df["turnover_rate"] = np.nan
        df["valuation_source"] = "missing"
        df["valuation_quality"] = "FAIL"
        return df

    def get_valuation_coverage_report(self, codes, start_date, end_date):
        """估值覆盖率报告"""
        if self.db is None:
            return {"status": "NO_DB", "coverage": 0}
        from data.duckdb_schema import load_valuation_daily
        df = load_valuation_daily(codes, start_date, end_date)
        if df is None or df.empty:
            return {"status": "NO_DATA", "coverage": 0}
        total_trading_days = 242 * max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days // 365)
        expected = len(codes) * total_trading_days
        actual = len(df)
        coverage = actual / expected if expected > 0 else 0
        pe_coverage = df["pe_ttm"].notna().mean()
        pb_coverage = df["pb"].notna().mean()
        n_unique_pe = df["pe_ttm"].nunique()
        return {
            "status": "PASS" if coverage > 0.90 and pe_coverage > 0.90 else ("WARN" if coverage > 0.50 else "FAIL"),
            "coverage": round(coverage, 3),
            "pe_coverage": round(pe_coverage, 3),
            "pb_coverage": round(pb_coverage, 3),
            "n_unique_pe": n_unique_pe,
            "n_records": actual,
            "expected": expected,
        }


def _pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def generate_csv_template(codes, sample_date="2024-01-02"):
    """生成估值CSV模板（用户填入数据后导入）"""
    rows = []
    for code in sorted(codes):
        rows.append({
            "code": code,
            "trade_date": sample_date,
            "pe_ttm": "",
            "pb": "",
            "ps_ttm": "",
            "total_mv": "",
            "circ_mv": "",
            "turnover_rate": "",
        })
    df = pd.DataFrame(rows)
    path = CSV_DIR / "valuation_template.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"估值CSV模板: {path} ({len(codes)} 只股票)")
    return path
