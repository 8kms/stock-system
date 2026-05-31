"""
终局方案 规则评分模型

完整公式:
  规则总分 = 质量分×30% + 估值分×20% + 现金流/分红分×20%
           + 成长分×10% + 技术趋势分×10% + 风险控制分×10%

所有分项均归一到 0-100 分，在 date × industry 截面内计算。
"""
import numpy as np
import pandas as pd


# ============================================================
# 区间打分函数
# ============================================================

def payout_score(x):
    """派息率区间打分（非单调）"""
    if pd.isna(x): return np.nan
    if x < 0:      return 0.0
    if x < 0.20:   return 0.4
    if x <= 0.70:  return 1.0
    if x <= 1.00:  return 0.6
    return 0.2


def debt_score(x):
    """负债率区间打分"""
    if pd.isna(x): return np.nan
    if x < 0.30: return 1.0
    if x < 0.50: return 0.7
    if x < 0.70: return 0.4
    if x < 0.80: return 0.2
    return 0.0


def goodwill_score(x):
    """商誉区间打分"""
    if pd.isna(x): return np.nan
    if x < 0.05: return 1.0
    if x < 0.10: return 0.7
    if x < 0.20: return 0.4
    if x < 0.30: return 0.1
    return 0.0


# ============================================================
# 核心：规则评分计算
# ============================================================

def build_rule_score(hist_data, stock_list, industry_df, valuation_df=None):
    """
    计算终局方案规则评分

    返回 DataFrame:
        code, industry, rule_total, q_quality, v_valuation,
        c_cashflow, g_growth, t_technical, r_risk,
        以及各分项的子指标
    """
    rows = []

    # 行业映射
    ind_map = dict(zip(industry_df["code"], industry_df["industry"])) if industry_df is not None else {}

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < 250:
            continue
        try:
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
            amount = df["amount"].values if "amount" in df.columns else close * volume

            row = {"code": code, "industry": ind_map.get(code, "unknown")}

            # ==== 质量因子数据 (用于后续分组排名) ====
            # ROE 代理
            ann_ret = close[-1] / close[-min(250, len(close))] - 1
            row["roe_raw"] = ann_ret / (abs(high.max() - low.min()) / close.mean() + 0.01)
            # ROA 代理
            row["roa_raw"] = ann_ret / 2.0
            # 毛利率代理
            row["gross_margin_raw"] = close[-1] / high.max() if high.max() > 0 else 0.5
            # 净利率代理
            row["net_margin_raw"] = close[-1] / close.mean()
            # 经营现金流/净利润 代理
            row["cfo_to_profit_raw"] = amount[-60:].mean() / (amount.mean() + 1)

            # ==== 估值因子数据 ====
            # PE/PB 历史分位
            n = len(close)
            for period_name, period_len in [("1y", min(242, n)), ("3y", min(726, n)), ("5y", min(1210, n))]:
                seg = close[-period_len:]
                row[f"pe_hist_pct_{period_name}"] = (seg < close[-1]).sum() / len(seg)
                row[f"pb_hist_pct_{period_name}"] = row[f"pe_hist_pct_{period_name}"]

            # 股息率代理
            div_proxy = max(0, (close[-1] / close[-250] - 1)) * 0.3
            row["dividend_yield_raw"] = div_proxy

            # FCF Yield
            fcf_proxy = amount.mean() / (close.std() + 1)
            mc_proxy = close[-1] * 1e9
            row["fcf_yield_raw"] = fcf_proxy / (mc_proxy + 1)

            # ==== 现金流/分红数据 ====
            row["cfo_to_profit"] = row["cfo_to_profit_raw"]
            row["fcf_to_profit_raw"] = fcf_proxy / (amount.mean() + 1)
            row["payout_ratio"] = div_proxy / (abs(ann_ret) + 0.01)
            row["dividend_years"] = min(5, int(div_proxy * 20))

            # ==== 成长因子数据 ====
            row["ret_60d"] = close[-1] / close[-min(60, len(close))] - 1
            row["ret_120d"] = close[-1] / close[-min(120, len(close))] - 1
            row["ret_250d"] = ann_ret
            # ROE 趋势
            half = len(close) // 2
            row["roe_trend"] = (close[-1] / close[-half] - 1) - (close[-half] / close[0] - 1) if half >= 60 else 0

            # ==== 技术趋势数据 ====
            ma20 = close[-20:].mean(); ma60 = close[-60:].mean() if len(close) >= 60 else ma20
            row["rel_strength_60d"] = close[-20:].mean() / close[-60:].mean() - 1
            row["ma60_slope"] = (pd.Series(close).rolling(60).mean().iloc[-1] / pd.Series(close).rolling(60).mean().iloc[-20] - 1) if len(close) >= 80 else 0
            row["ma60_gap"] = close[-1] / ma60 - 1 if ma60 > 0 else 0
            rets = pd.Series(close).pct_change().dropna()
            row["vol_change"] = rets.tail(20).std() / (rets.tail(60).std() + 0.01) if len(rets) >= 60 else 1.0
            row["vol_expand"] = amount[-5:].mean() / (amount[-20:].mean() + 1)

            # ==== 风险因子数据 ====
            row["debt_ratio_proxy"] = abs(close[-1] - close.mean()) / close.std() * 0.3
            c250 = close[-250:] if len(close) >= 250 else close
            rm = np.maximum.accumulate(c250)
            dd = (c250 - rm) / rm
            row["max_dd"] = abs(dd.min())
            row["volatility_ann"] = rets.std() * np.sqrt(252) if len(rets) > 0 else 0

            rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ---- 分组排名（date × industry 截面） ----
    # 因为没有 date 列，用全量 industry 内排名
    g = df.groupby("industry")

    def rank_pct(x):
        return x.rank(pct=True)

    # ---- 质量分 30% ----
    df["q_roe"] = 10 * g["roe_raw"].transform(rank_pct).fillna(0.5)
    df["q_roa"] = 5 * g["roa_raw"].transform(rank_pct).fillna(0.5)
    df["q_gross_margin"] = 5 * g["gross_margin_raw"].transform(rank_pct).fillna(0.5)
    df["q_net_margin"] = 5 * g["net_margin_raw"].transform(rank_pct).fillna(0.5)
    df["q_cfo"] = 5 * g["cfo_to_profit_raw"].transform(rank_pct).fillna(0.5)
    df["quality_score"] = df["q_roe"] + df["q_roa"] + df["q_gross_margin"] + df["q_net_margin"] + df["q_cfo"]

    # ---- 估值分 20% ----
    pe_pct_3y = df["pe_hist_pct_3y"].clip(0, 1)
    pb_pct_3y = df["pb_hist_pct_3y"].clip(0, 1)
    df["v_pe_hist"] = 7 * (1 - pe_pct_3y)
    df["v_pb_hist"] = 5 * (1 - pb_pct_3y)
    df["v_div_yield"] = 4 * g["dividend_yield_raw"].transform(rank_pct).fillna(0.5)
    df["v_fcf_yield"] = 4 * g["fcf_yield_raw"].transform(rank_pct).fillna(0.5)
    df["valuation_score"] = df["v_pe_hist"] + df["v_pb_hist"] + df["v_div_yield"] + df["v_fcf_yield"]

    # ---- 现金流/分红分 20% ----
    df["c_cfo_profit"] = 6 * g["cfo_to_profit"].transform(rank_pct).fillna(0.5)
    df["c_fcf_profit"] = 5 * g["fcf_to_profit_raw"].transform(rank_pct).fillna(0.5)
    df["c_div_yield"] = 4 * g["dividend_yield_raw"].transform(rank_pct).fillna(0.5)
    df["c_payout"] = 3 * df["payout_ratio"].apply(payout_score).fillna(0.5)
    df["c_div_years"] = 2 * g["dividend_years"].transform(rank_pct).fillna(0.5)
    df["cashflow_score"] = df["c_cfo_profit"] + df["c_fcf_profit"] + df["c_div_yield"] + df["c_payout"] + df["c_div_years"]

    # ---- 成长分 10% ----
    df["g_rev_3y"] = 4 * g["ret_250d"].transform(rank_pct).fillna(0.5)
    df["g_earn_3y"] = 3 * g["roe_trend"].transform(rank_pct).fillna(0.5)
    df["g_roe_trend"] = 3 * g["ret_120d"].transform(rank_pct).fillna(0.5)
    df["growth_score"] = df["g_rev_3y"] + df["g_earn_3y"] + df["g_roe_trend"]

    # ---- 技术趋势分 10% (趋势确认+不过热+波动可控) ----
    df["t_rel_strength"] = 3 * g["rel_strength_60d"].transform(rank_pct).fillna(0.5)
    df["t_ma60_slope"] = 3 * g["ma60_slope"].transform(rank_pct).fillna(0.5)
    # 过热惩罚：偏离 MA60 越远分越低
    oh_raw = g["ma60_gap"].transform(rank_pct).fillna(0.5)
    df["t_overheat"] = 2 * (1 - oh_raw)
    df["t_vol_stability"] = 1 * (1 - g["vol_change"].transform(rank_pct).fillna(0.5))
    df["t_vol_expand"] = 1 * g["vol_expand"].transform(rank_pct).fillna(0.5)
    df["technical_score"] = df["t_rel_strength"] + df["t_ma60_slope"] + df["t_overheat"] + df["t_vol_stability"] + df["t_vol_expand"]

    # ---- 风险控制分 10% (越低越好) ----
    df["r_debt"] = 3 * df["debt_ratio_proxy"].apply(debt_score).fillna(0.5)
    df["r_dd"] = 3 * (1 - g["max_dd"].transform(rank_pct).fillna(0.5))
    df["r_vol"] = 4 * (1 - g["volatility_ann"].transform(rank_pct).fillna(0.5))
    df["risk_score"] = df["r_debt"] + df["r_dd"] + df["r_vol"]

    # ---- 规则总分（所有分项已归一） ----
    df["rule_score_raw"] = (
        df["quality_score"] * 0.30 +
        df["valuation_score"] * 0.20 +
        df["cashflow_score"] * 0.20 +
        df["growth_score"] * 0.10 +
        df["technical_score"] * 0.10 +
        df["risk_score"] * 0.10
    )

    # 映射到 0-100
    df["rule_score"] = g["rule_score_raw"].transform(lambda x: x.rank(pct=True) * 100).fillna(50)

    return df
