"""
指数数据多源提供者

优先级: DuckDB本地缓存 → Tushare index_daily → AKShare index_zh_a_hist → fallback
"""
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

INDEX_CODE_MAP = {
    "000300": {"tushare": "000300.SH", "akshare": "000300", "ak_index": "sh000300"},
    "000905": {"tushare": "000905.SH", "akshare": "000905", "ak_index": "sh000905"},
    "000852": {"tushare": "000852.SH", "akshare": "000852", "ak_index": "sh000852"},
    "399006": {"tushare": "399006.SZ", "akshare": "399006", "ak_index": "sz399006"},
    "000922": {"tushare": "000922.CSI", "akshare": "000922", "ak_index": "sh000922"},
}


def get_index_daily(index_code, start_date, end_date,
                    duckdb_conn=None, tushare_provider=None, akshare_client=None):
    """
    多源指数日线获取

    返回: DataFrame (date, open, high, low, close, volume, source)
    """
    mapping = INDEX_CODE_MAP.get(index_code, {})

    # 1. DuckDB 本地缓存
    if duckdb_conn is not None:
        try:
            df = duckdb_conn.execute(
                f"SELECT * FROM index_daily WHERE code='{index_code}' "
                f"AND date BETWEEN '{start_date}' AND '{end_date}' ORDER BY date"
            ).df()
            if df is not None and len(df) > 100:
                df["source"] = "duckdb_cache"
                return df
        except Exception:
            pass

    # 2. Tushare
    if tushare_provider is not None:
        ts_code = mapping.get("tushare")
        if ts_code:
            try:
                df = tushare_provider.pro.index_daily(
                    ts_code=ts_code, start_date=start_date, end_date=end_date
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "trade_date": "date", "vol": "volume", "amount": "amount"
                    })
                    df["date"] = pd.to_datetime(df["date"])
                    df["code"] = index_code
                    df["source"] = "tushare"

                    # 缓存到 DuckDB
                    if duckdb_conn is not None:
                        try:
                            duckdb_conn.execute("CREATE TABLE IF NOT EXISTS index_daily AS SELECT * FROM df LIMIT 0")
                            duckdb_conn.execute("INSERT INTO index_daily SELECT * FROM df")
                        except Exception:
                            pass

                    return df
            except Exception:
                pass

    # 3. AKShare
    if akshare_client is not None:
        import akshare as ak
        ak_symbol = mapping.get("ak_index", mapping.get("akshare"))
        if ak_symbol:
            try:
                df = akshare_client.cached_call(
                    f"index_{ak_symbol}_{start_date}_{end_date}",
                    lambda: ak.stock_zh_index_daily(symbol=ak_symbol),
                    force=True,
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "date": "date", "open": "open", "close": "close",
                        "high": "high", "low": "low", "volume": "volume",
                    })
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    df["code"] = index_code
                    df["source"] = "akshare"

                    if duckdb_conn is not None:
                        try:
                            duckdb_conn.execute("CREATE TABLE IF NOT EXISTS index_daily AS SELECT * FROM df LIMIT 0")
                            duckdb_conn.execute("INSERT INTO index_daily SELECT * FROM df")
                        except Exception:
                            pass

                    return df
            except Exception:
                pass

            # AKShare 兜底: index_zh_a_hist
            ak_sym2 = mapping.get("akshare")
            if ak_sym2:
                try:
                    time.sleep(1.0)
                    df = akshare_client.cached_call(
                        f"index_zh_{ak_sym2}_{start_date}_{end_date}",
                        lambda: ak.index_zh_a_hist(
                            symbol=ak_sym2, period="daily",
                            start_date=start_date, end_date=end_date
                        ),
                        force=True,
                    )
                    if df is not None and not df.empty:
                        col_map = {}
                        for c in df.columns:
                            cl = str(c).strip()
                            if "日期" in cl: col_map[c] = "date"
                            elif "开" in cl: col_map[c] = "open"
                            elif "收" in cl: col_map[c] = "close"
                            elif "高" in cl: col_map[c] = "high"
                            elif "低" in cl: col_map[c] = "low"
                            elif "量" in cl: col_map[c] = "volume"
                        df = df.rename(columns=col_map)
                        df["date"] = pd.to_datetime(df["date"])
                        df["code"] = index_code
                        df["source"] = "akshare_fallback"

                        if duckdb_conn is not None:
                            try:
                                duckdb_conn.execute("CREATE TABLE IF NOT EXISTS index_daily AS SELECT * FROM df LIMIT 0")
                                duckdb_conn.execute("INSERT INTO index_daily SELECT * FROM df")
                            except Exception:
                                pass

                        return df
                except Exception:
                    pass

    # 所有源失败
    raise RuntimeError(f"指数数据不可用: {index_code}")
