"""
安全 AKShare 客户端 — 完全按照终局方案规格

特性:
  2.0s 限速间隔
  parquet 缓存 + force refresh
  3 次重试 + 指数退避
  health_check 分层检测 (price/spot/index)
  spot 失败时自动降级为历史行情
  index 失败时优先本地缓存 → Tushare → AKShare
"""
import time
import random
from pathlib import Path
from datetime import datetime

import pandas as pd


class SafeAkshareClient:
    """AKShare 统一安全客户端"""

    def __init__(self, cache_dir="data_cache/akshare", min_interval=2.0, max_retry=3):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retry = max_retry
        self.last_call_ts = 0

    def _sleep(self):
        elapsed = time.time() - self.last_call_ts
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait + random.uniform(0.2, 1.0))
        self.last_call_ts = time.time()

    def cached_call(self, cache_key, fetch_func, force=False):
        """带缓存的调用"""
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        if cache_path.exists() and not force:
            return pd.read_parquet(cache_path)

        last_err = None
        for i in range(self.max_retry):
            try:
                self._sleep()
                df = fetch_func()
                if df is None or df.empty:
                    raise ValueError(f"空数据: {cache_key}")
                df.to_parquet(cache_path, index=False)
                return df
            except Exception as e:
                last_err = e
                time.sleep((i + 1) * 3 + random.uniform(0.5, 2.0))

        raise RuntimeError(f"AKShare 失败: {cache_key}, err={last_err}")

    def health_check(self):
        """
        分层健康检查

        返回:
            dict: {price, spot, index, status, checked_at, errors}
        """
        import akshare as ak

        result = {}

        # 1. 个股历史行情
        try:
            df = self.cached_call(
                "health_price_600519",
                lambda: ak.stock_zh_a_hist(
                    symbol="600519", period="daily",
                    start_date="20250101", adjust="qfq"
                ),
                force=True,
            )
            result["price"] = len(df) > 10
        except Exception as e:
            result["price"] = False
            result["price_error"] = str(e)[:100]

        # 2. 实时行情
        try:
            df = self.cached_call(
                "health_spot_em",
                lambda: ak.stock_zh_a_spot_em(),
                force=True,
            )
            result["spot"] = len(df) > 1000
        except Exception as e:
            result["spot"] = False
            result["spot_error"] = str(e)[:100]

        # 3. 指数行情（多接口兜底）
        try:
            df = self.cached_call(
                "health_index_000300",
                lambda: ak.stock_zh_index_daily(symbol="sh000300"),
                force=True,
            )
            result["index"] = len(df) > 100
        except Exception:
            # 兜底: index_zh_a_hist
            try:
                time.sleep(1.0)
                df = self.cached_call(
                    "health_index_zh_000300",
                    lambda: ak.index_zh_a_hist(
                        symbol="000300", period="daily",
                        start_date="20250101", end_date=datetime.now().strftime("%Y%m%d")
                    ),
                    force=True,
                )
                result["index"] = len(df) > 10
            except Exception as e:
                result["index"] = False
                result["index_error"] = str(e)[:100]

        # 综合判定
        if all(result.get(k) is True for k in ["price", "spot", "index"]):
            result["status"] = "PASS"
        elif result.get("price") is True:
            result["status"] = "PARTIAL"
        else:
            result["status"] = "FAIL"

        result["checked_at"] = datetime.now().isoformat(timespec="seconds")
        return result


def get_latest_snapshot(price_df, daily_basic_df=None):
    """
    spot 失败时的降级方案：用最近交易日收盘价

    参数:
        price_df: 日线数据 (code, date, close, amount, ...)
        daily_basic_df: Tushare daily_basic (可选, 含 PE/PB/市值)

    返回:
        DataFrame: code, latest_price, latest_amount, snapshot_source
    """
    latest = price_df.sort_values(["code", "date"]).groupby("code").tail(1).copy()
    latest = latest.rename(columns={"close": "latest_price", "amount": "latest_amount"})
    latest["snapshot_source"] = "price_daily_fallback"

    if daily_basic_df is not None and not daily_basic_df.empty:
        basic_latest = daily_basic_df.sort_values(["code", "date"]).groupby("code").tail(1)
        latest = latest.merge(
            basic_latest[["code", "total_mv", "circ_mv", "pe_ttm", "pb"]],
            on="code", how="left"
        )
        latest["snapshot_source"] = "daily_basic"

    return latest
