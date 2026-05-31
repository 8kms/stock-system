"""
Tushare Pro 数据提供者 — 接管财务/PIT 核心数据

按照终局方案规格:
  income / balancesheet / cashflow / fina_indicator / daily_basic / stock_basic
  公告日 (ann_date / f_ann_date) → effective_date
"""
import os
import time
from pathlib import Path

import pandas as pd


class TushareProvider:
    """Tushare Pro 统一客户端"""

    def __init__(self, token=None, cache_dir="data_cache/tushare", sleep_sec=0.35):
        self.token = token or os.getenv("TUSHARE_TOKEN")
        if not self.token:
            raise ValueError("缺少 TUSHARE_TOKEN。请设置环境变量 TUSHARE_TOKEN")

        import tushare as ts
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_sec = sleep_sec

    def _cached(self, api_name, cache_key, force=False, **kwargs):
        """通用缓存调用"""
        path = self.cache_dir / f"{cache_key}.parquet"

        if path.exists() and not force:
            return pd.read_parquet(path)

        time.sleep(self.sleep_sec)

        func = getattr(self.pro, api_name)
        df = func(**kwargs)

        if df is None or df.empty:
            raise ValueError(f"Tushare 空数据: {api_name}, kwargs={kwargs}")

        df.to_parquet(path, index=False)
        return df

    def stock_basic(self, force=False):
        """股票列表 + 行业 + 上市日期"""
        return self._cached(
            "stock_basic", "stock_basic", force=force,
            exchange="", list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date",
        )

    def daily_basic(self, ts_code=None, trade_date=None, start_date=None, end_date=None, force=False):
        """每日估值: PE/PB/市值"""
        fields = "ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv"
        kwargs = {"fields": fields}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if trade_date:
            kwargs["trade_date"] = trade_date
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        key = f"daily_basic_{ts_code or 'all'}_{trade_date or start_date}_{end_date}"
        return self._cached("daily_basic", key, force=force, **kwargs)

    def fina_indicator(self, ts_code, force=False):
        """财务指标: ROE/毛利率/净利率/同比增速/现金流"""
        fields = (
            "ts_code,ann_date,end_date,eps,grossprofit_margin,netprofit_margin,"
            "roe,roe_waa,roa,netprofit_yoy,or_yoy,ocf_yoy,debt_to_assets"
        )
        return self._cached("fina_indicator", f"fina_indicator_{ts_code}",
                            force=force, ts_code=ts_code, fields=fields)

    def income(self, ts_code, force=False):
        """利润表: 营收/净利润/扣非/公告日"""
        fields = (
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
            "total_revenue,revenue,operate_profit,n_income_attr_p"
        )
        return self._cached("income", f"income_{ts_code}",
                            force=force, ts_code=ts_code, fields=fields)

    def balancesheet(self, ts_code, force=False):
        """资产负债表: 总资产/负债/净资产/商誉/应收"""
        fields = (
            "ts_code,ann_date,f_ann_date,end_date,report_type,"
            "total_assets,total_liab,total_hldr_eqy_exc_min_int,"
            "accounts_receiv,goodwill"
        )
        return self._cached("balancesheet", f"balancesheet_{ts_code}",
                            force=force, ts_code=ts_code, fields=fields)

    def cashflow(self, ts_code, force=False):
        """现金流: 经营/投资/筹资"""
        fields = (
            "ts_code,ann_date,f_ann_date,end_date,report_type,"
            "n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act"
        )
        return self._cached("cashflow", f"cashflow_{ts_code}",
                            force=force, ts_code=ts_code, fields=fields)


def build_effective_date(df):
    """构建 PIT effective_date"""
    for col in ["ann_date", "f_ann_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "f_ann_date" in df.columns:
        df["effective_date"] = df["f_ann_date"].fillna(df["ann_date"])
    else:
        df["effective_date"] = df["ann_date"]

    df["report_period"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def build_financial_pit_panel(income_df, balance_df, cashflow_df, indicator_df):
    """
    合并四大报表 → PIT 财务宽表

    关键: effective_date 取各表最晚公告日，避免混入尚未公开字段
    """
    income_df = build_effective_date(income_df)
    balance_df = build_effective_date(balance_df)
    cashflow_df = build_effective_date(cashflow_df)

    indicator_df["ann_date"] = pd.to_datetime(indicator_df["ann_date"], errors="coerce")
    indicator_df["effective_date"] = indicator_df["ann_date"]
    indicator_df["report_period"] = pd.to_datetime(indicator_df["end_date"], errors="coerce")

    keys = ["ts_code", "end_date"]

    df = indicator_df.merge(
        income_df[["ts_code","end_date","total_revenue","n_income_attr_p","effective_date"]],
        on=keys, how="left", suffixes=("", "_income"))

    df = df.merge(
        balance_df[["ts_code","end_date","total_assets","total_liab",
                     "total_hldr_eqy_exc_min_int","accounts_receiv","goodwill","effective_date"]],
        on=keys, how="left", suffixes=("", "_balance"))

    df = df.merge(
        cashflow_df[["ts_code","end_date","n_cashflow_act","effective_date"]],
        on=keys, how="left", suffixes=("", "_cashflow"))

    date_cols = [c for c in df.columns if c.startswith("effective_date")]
    if date_cols:
        df["effective_date_final"] = df[date_cols].max(axis=1)

    df["debt_ratio"] = df["total_liab"] / df["total_assets"]
    df["goodwill_to_equity"] = df["goodwill"] / df["total_hldr_eqy_exc_min_int"]
    df["cfo_to_profit"] = df["n_cashflow_act"] / df["n_income_attr_p"]

    df = df.rename(columns={
        "ts_code": "code",
        "grossprofit_margin": "gross_margin",
        "netprofit_margin": "net_margin",
        "effective_date_final": "effective_date",
    })

    return df
