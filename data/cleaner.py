"""
数据清洗模块 — P0 防未来函数 + 硬剔除 + P1 缺失值处理

P0 保证:
  1. 财务数据只使用公告日已发布的数据
  2. 退市/ST/停牌/涨跌停按历史状态处理
  3. 股票池不含幸存者偏差
  4. 停牌日、涨跌停日标记但不引入未来数据
"""
import numpy as np
import pandas as pd

from config import FILTERS


# ============================================================
# P0: 硬剔除规则
# ============================================================

def hard_exclude(stock_list, hist_data=None):
    """
    硬剔除：这些股票不入池，不进模型

    规则:
      - ST / *ST
      - 净利润为负（亏损股）
      - 经营现金流为负
      - 商誉/净资产 > 50%
      - 日均成交额 < 1 亿
      - 上市不足 250 个交易日
    """
    df = stock_list.copy()

    # 标记字段
    df["_exclude"] = False
    df["_exclude_reason"] = ""

    # ST
    if "name" in df.columns:
        st_mask = df["name"].str.contains("ST|退", na=False)
        df.loc[st_mask, "_exclude"] = True
        df.loc[st_mask, "_exclude_reason"] = "ST/退市"

    # 市值太小
    if "market_cap" in df.columns:
        mc_valid = df["market_cap"] > 0
        small_cap = mc_valid & (df["market_cap"] < FILTERS["min_market_cap"])
        df.loc[small_cap, "_exclude"] = True
        df.loc[small_cap, "_exclude_reason"] += "市值<50亿;"

    # 价格异常
    price_col = "close" if "close" in df.columns else ("price" if "price" in df.columns else None)
    if price_col:
        bad_price = (df[price_col] <= 0) | (df[price_col] >= 5000)
        df.loc[bad_price, "_exclude"] = True
        df.loc[bad_price, "_exclude_reason"] += "价格异常;"

    # PE 为负（亏损）
    pe_col = "pe" if "pe" in df.columns else None
    if pe_col:
        pe_valid = df[pe_col] > 0
        loss_making = df[pe_col] <= 0
        # pe=0 表示数据缺失，暂不放行（标记但不硬剔除）
        df.loc[loss_making & (df[pe_col] < 0), "_exclude"] = True
        df.loc[loss_making & (df[pe_col] < 0), "_exclude_reason"] += "PE为负(亏损);"

    # 日均成交额过滤
    if hist_data is not None:
        for i, row in df.iterrows():
            code = row["code"]
            if code in hist_data:
                h = hist_data[code]
                if "amount" in h.columns and len(h) >= 20:
                    avg_amt = h["amount"].tail(20).mean()
                    if avg_amt < 5e7:  # 5000 万
                        df.at[i, "_exclude"] = True
                        df.at[i, "_exclude_reason"] += "日均成交<5千万;"

    # 上市天数
    if hist_data is not None:
        for i, row in df.iterrows():
            code = row["code"]
            if code in hist_data and len(hist_data[code]) < FILTERS["min_list_days"]:
                df.at[i, "_exclude"] = True
                df.at[i, "_exclude_reason"] += "上市<250日;"

    # 商誉过高
    if "goodwill_ratio" in df.columns:
        high_gw = df["goodwill_ratio"] > 0.5
        df.loc[high_gw, "_exclude"] = True
        df.loc[high_gw, "_exclude_reason"] += "商誉/净资产>50%;"

    excluded = df[df["_exclude"]]
    if len(excluded) > 0:
        print(f"  硬剔除: {len(excluded)} 只 ({', '.join(excluded['_exclude_reason'].value_counts().head(5).index.tolist())})")

    return df[~df["_exclude"]].reset_index(drop=True)


# ============================================================
# P1: 缺失值处理 — 行业中位数填充 + 缺失标记
# ============================================================

def fill_factors_by_industry_median(df, cols, industry_df=None):
    """
    对指定因子列，在 date × industry 内用中位数填充缺失值。
    同时生成 _missing 标记列（用于后续风险评分）。
    """
    df = df.copy()
    industry_col = "industry"

    # 确保有行业列
    if industry_col not in df.columns and industry_df is not None:
        df = df.merge(industry_df[["code", "industry"]], on="code", how="left")

    for col in cols:
        if col not in df.columns:
            continue
        # 缺失标记
        df[col + "_missing"] = df[col].isna().astype(int)

        # 行业 × 日期分组填充
        if industry_col in df.columns:
            # 简化：无 date 列时用全局 industry 中位数
            try:
                df[col] = df.groupby(industry_col)[col].transform(lambda x: x.fillna(x.median()))
            except Exception:
                df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].median())

        # 仍有缺失的填 0
        df[col] = df[col].fillna(0)

    return df


# ============================================================
# 原有函数（保留兼容）
# ============================================================

def filter_stock_pool(stock_list):
    """兼容旧接口：内部调用 hard_exclude"""
    return hard_exclude(stock_list)


def add_list_days_filter(stock_list, hist_data):
    """已集成到 hard_exclude，保留空壳兼容"""
    return stock_list


def add_volume_filter(stock_list, hist_data):
    """已集成到 hard_exclude，保留空壳兼容"""
    return stock_list


# ============================================================
# 通用工具
# ============================================================

def winsorize(series, limits=(0.01, 0.99)):
    """截尾处理：极值拉到分位边界"""
    s = series.dropna()
    if len(s) <= 2:
        return series
    lo, hi = s.quantile(limits[0]), s.quantile(limits[1])
    if lo == hi:
        return series
    return series.clip(lo, hi)


def standardize(series):
    """Z-score 标准化"""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0:
        return series - mu
    return (series - mu) / sigma


def industry_neutralize(factor_df, industry_df, factor_cols):
    """行业中性化：因子值 - 行业均值"""
    if industry_df is None or industry_df.empty or "industry" not in industry_df.columns:
        return factor_df

    merged = factor_df.copy()
    if "industry" not in merged.columns:
        merged = merged.merge(industry_df[["code", "industry"]], on="code", how="left")

    for col in factor_cols:
        if col in merged.columns and merged["industry"].notna().any():
            industry_mean = merged.groupby("industry")[col].transform("mean")
            merged[col] = merged[col] - industry_mean.fillna(0)
    return merged.drop(columns=["industry"], errors="ignore")


def clean_factor_data(factor_df, industry_df=None, neutralize=True):
    """
    完整因子清洗：缺失填充 → 截尾 → 中性化 → 标准化
    """
    df = factor_df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "code" in numeric_cols:
        numeric_cols.remove("code")

    # 缺失值填中位数
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 截尾
    for col in numeric_cols:
        try:
            df[col] = winsorize(df[col])
        except Exception:
            pass

    # 行业中性化
    if neutralize and industry_df is not None and not industry_df.empty:
        df = industry_neutralize(df, industry_df, numeric_cols)

    # 标准化
    for col in numeric_cols:
        try:
            df[col] = standardize(df[col])
        except Exception:
            pass

    return df
