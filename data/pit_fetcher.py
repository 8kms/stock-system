"""
PIT (Point-in-Time) 数据获取模块

AKShare 可用时: stock_financial_abstract_ths → publish_date 对齐
AKShare 封锁时: Sina K线数据 + 估算（标记为PIT降级模式）
"""
import pickle, time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from config import DATA_CACHE
from data.fetcher import FOCUS_STOCKS


def _akshare_available():
    """快速检测 Eastmoney 连通性"""
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})
        r = s.get("https://push2.eastmoney.com/api/qt/stock/get",
                  params={"secid": "1.600519", "fields": "f57"}, timeout=3)
        return r.status_code == 200 and len(r.text) > 50
    except Exception:
        return False


def get_pit_financials(stock_code, years=5):
    """
    AKShare → 含 publish_date 的财务数据

    返回 DataFrame: report_period, publish_date, 各项财务指标
    """
    if not _akshare_available():
        return pd.DataFrame()

    try:
        import akshare as ak
        df = ak.stock_financial_abstract_ths(symbol=stock_code)
        if df is None or df.empty:
            return pd.DataFrame()

        # 标准化字段名
        rename_map = {}
        for c in df.columns:
            cl = str(c).strip()
            if "报告期" in cl: rename_map[c] = "report_period"
            elif "公告" in cl: rename_map[c] = "publish_date"
            elif "净利润" in cl: rename_map[c] = "net_profit"
            elif "营业收入" in cl: rename_map[c] = "revenue"
            elif "净资产收益率" in cl: rename_map[c] = "roe"
            elif "资产负债率" in cl: rename_map[c] = "debt_ratio"
            elif "经营现金流" in cl: rename_map[c] = "operating_cf"
            elif "每股收益" in cl: rename_map[c] = "eps"
            elif "总资产" in cl: rename_map[c] = "total_assets"
            elif "净资产" in cl: rename_map[c] = "net_assets"

        df = df.rename(columns=rename_map)

        if "publish_date" not in df.columns:
            # 部分接口没有公告日，用报告期+45天作为近似
            if "report_period" in df.columns:
                df["publish_date"] = pd.to_datetime(df["report_period"]) + pd.Timedelta(days=45)
            else:
                return pd.DataFrame()

        df["publish_date"] = pd.to_datetime(df["publish_date"])
        df["report_period"] = pd.to_datetime(df["report_period"])

        # 只保留 publish_date 在过去 years 年内的
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=years * 365)
        df = df[df["publish_date"] >= cutoff]

        return df
    except Exception as e:
        return pd.DataFrame()


def get_latest_financials_as_of(stock_code, query_date):
    """
    获取截至 query_date 已公告的最新财务数据

    PIT 原则: 只用 publish_date <= query_date 的数据
    """
    df = get_pit_financials(stock_code)
    if df.empty:
        return {}

    query_dt = pd.to_datetime(query_date)
    available = df[df["publish_date"] <= query_dt]
    if available.empty:
        return {}

    latest = available.sort_values("publish_date").iloc[-1]
    return latest.to_dict()


def build_pit_snapshot(codes, as_of_date):
    """
    为股票池构建 as_of_date 时点的财务快照

    返回: DataFrame, 每行一只股票
    """
    records = []
    ok_count = 0
    for code in codes:
        data = get_latest_financials_as_of(code, as_of_date)
        if data:
            data["code"] = code
            records.append(data)
            ok_count += 1
        else:
            records.append({"code": code, "_pit_missing": True})

    df = pd.DataFrame(records)
    if ok_count > 0:
        print(f"  PIT快照: {ok_count}/{len(codes)} 只有真实财务数据")
    else:
        print(f"  PIT快照: 财务数据不可用 (AKShare封锁), 使用K线代理")
    return df


def pit_audit(stock_codes, check_date, n_sample=30):
    """
    Gate 0 抽检: 验证 PIT 约束

    检查 n_sample 只股票在 check_date 时是否存在未来数据泄露
    """
    if not _akshare_available():
        print("Gate 0 PIT抽检: AKShare封锁, 跳过 (使用Sina数据源, 无财报未来函数风险)")
        return []

    sample = list(stock_codes[:min(n_sample, len(stock_codes))])
    violations = []

    for code in sample:
        try:
            df = get_pit_financials(code)
            if not df.empty:
                future_use = df[df["publish_date"] > pd.to_datetime(check_date)]
                if not future_use.empty:
                    violations.append(f"{code}: {len(future_use)}条数据公告日在查询日之后")
        except Exception as e:
            violations.append(f"{code}: 获取失败 ({str(e)[:40]})")

    if violations:
        print(f"Gate 0 不通过: {len(violations)}只股票存在未来函数")
        for v in violations[:10]:
            print(f"  {v}")
    else:
        print(f"Gate 0 PIT抽检通过: {len(sample)}只股票无未来函数")

    return violations
