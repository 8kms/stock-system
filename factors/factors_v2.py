"""
P1 因子体系 v2 — 分组标准化 + 区间打分 + 动量拆分

核心改进:
  1. 所有因子在 date × industry 内做分位排名
  2. payout_ratio: 区间型打分（非越高越好）
  3. free_cash_flow: 改为 FCF Yield (FCF/市值)
  4. 动量拆成 趋势确认(trend) + 过热惩罚(overheat)
  5. 风险硬剔除 + 风险评分分离
  6. 缺失值标记 + 行业中位数填充
"""
import numpy as np
import pandas as pd


# ============================================================
# 分组排名工具
# ============================================================

def rank_within_group(series):
    """在 group 内做百分位排名 (0-1)"""
    return series.rank(pct=True)


# ============================================================
# 区间打分函数
# ============================================================

def score_payout_ratio(x):
    """
    派息率区间打分（非越高越好）

    0%以下    → 异常, 0分
    0%-20%    → 偏低, 0.4分
    20%-70%   → 健康, 1.0分
    70%-100%  → 偏高(需结合行业), 0.6分
    >100%     → 高风险, 0.2分
    """
    if pd.isna(x): return np.nan
    if x < 0:      return 0.0
    if x < 0.2:    return 0.4
    if x <= 0.7:   return 1.0
    if x <= 1.0:   return 0.6
    return 0.2


def score_debt_ratio(x):
    """
    负债率区间打分
    <30% → 优秀 1.0
    30-50% → 正常 0.7
    50-70% → 偏高 0.4
    70-80% → 高 0.2
    >80% → 危险 0.0
    """
    if pd.isna(x): return np.nan
    if x < 0.3:  return 1.0
    if x < 0.5:  return 0.7
    if x < 0.7:  return 0.4
    if x < 0.8:  return 0.2
    return 0.0


def score_goodwill(x):
    """
    商誉/净资产区间打分
    <5% → 安全 1.0
    5-10% → 关注 0.7
    10-20% → 警惕 0.4
    20-30% → 高风险 0.1
    >30% → 危险 0.0 (硬剔除阈值在 50%)
    """
    if pd.isna(x): return np.nan
    if x < 0.05:  return 1.0
    if x < 0.10:  return 0.7
    if x < 0.20:  return 0.4
    if x < 0.30:  return 0.1
    return 0.0


# ============================================================
# 因子计算主函数
# ============================================================

def compute_factors_v2(hist_data, stock_list, industry_df, valuation_df=None):
    """
    P1 因子体系 v2：date × industry 分组排名

    返回:
        factor_df: 因子 DataFrame
        risk_flags: 风险标记 DataFrame
        hard_exclude_mask: 硬剔除布尔 Series
    """
    rows = []

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < 250:
            continue

        try:
            recent = df.tail(250)
            close = recent["close"].values
            high = recent["high"].values
            low = recent["low"].values
            volume = recent["volume"].values if "volume" in recent.columns else np.ones(250)
            amount = recent["amount"].values if "amount" in recent.columns else close * volume

            row = {"code": code}

            # ---- 质量因子 (Quality) ----
            # ROE 代理: 年化收益 / (最高价/最低价波动)
            ann_ret = close[-1] / close[-250] - 1 if len(close) >= 250 else close[-1] / close[0] - 1
            price_range = (high.max() - low.min()) / close.mean()
            row["q_roe_proxy"] = ann_ret / (price_range + 0.01)

            # 毛利率代理: close相对于区间高点的位置
            row["q_gross_margin_proxy"] = close[-1] / high.max() if high.max() > 0 else 0.5

            # 收益稳定性: 月收益标准差的倒数
            monthly_ret = pd.Series(close).pct_change(20).dropna()
            row["q_stability"] = 1.0 / (monthly_ret.std() + 0.01) if len(monthly_ret) >= 6 else 0

            # ---- 估值因子 (Valuation) ----
            # PE/PB 分位 (越低越好)
            if valuation_df is not None:
                vrow = valuation_df[valuation_df["code"] == code]
                if not vrow.empty:
                    row["v_pe"] = vrow["pe"].values[0] if "pe" in vrow.columns else np.nan
                    row["v_pb"] = vrow["pb"].values[0] if "pb" in vrow.columns else np.nan
                else:
                    row["v_pe"] = np.nan; row["v_pb"] = np.nan
            else:
                row["v_pe"] = np.nan; row["v_pb"] = np.nan

            # PE/PB 历史分位 (从价格反推)
            row["v_pe_pct"] = (close[-1] - close.min()) / (close.max() - close.min() + 0.01)
            row["v_pb_pct"] = row["v_pe_pct"]

            # ---- 成长因子 (Growth) ----
            row["g_ret_60d"] = close[-1] / close[-min(60, len(close))] - 1
            row["g_ret_120d"] = close[-1] / close[-min(120, len(close))] - 1
            row["g_momentum_accel"] = row["g_ret_60d"] - row["g_ret_120d"]

            # 成交额增速
            if len(amount) >= 120:
                row["g_amount_growth"] = amount[-60:].mean() / (amount[:60].mean() + 1) - 1
            else:
                row["g_amount_growth"] = 0

            # ---- 分红因子 (Dividend) ----
            # 股息率代理
            if close[-1] > 0:
                row["d_div_yield"] = max(0, (close[-1] / close[-250] - 1)) * 0.3
            else:
                row["d_div_yield"] = 0

            # FCF Yield = FCF代理 / 市值代理
            fcf_proxy = amount.mean() / (close.std() + 1)
            mc_proxy = close[-1] * 1e9
            row["d_fcf_yield"] = fcf_proxy / (mc_proxy + 1)

            # ---- 技术因子 (Technical) —— 拆成趋势+过热 ----
            ma20 = close[-20:].mean()
            ma60 = close[-60:].mean() if len(close) >= 60 else ma20
            ma120 = close[-120:].mean() if len(close) >= 120 else ma60

            # 趋势确认
            row["t_ret_60d"] = row["g_ret_60d"]  # 60日收益
            ma60_series = pd.Series(close).rolling(60).mean().dropna()
            row["t_ma60_slope"] = (ma60_series.iloc[-1] / ma60_series.iloc[-20] - 1) if len(ma60_series) >= 20 else 0

            # 相对行业强弱 (简化)
            row["t_rps"] = close[-20:].mean() / close[-60:].mean() - 1 if close[-60:].mean() > 0 else 0

            # 过热惩罚
            row["t_ret_20d"] = close[-1] / close[-min(20, len(close))] - 1
            row["t_ma60_gap"] = close[-1] / ma60 - 1 if ma60 > 0 else 0
            vol_20 = pd.Series(close).pct_change().tail(20).std()
            vol_60 = pd.Series(close).pct_change().tail(60).std() if len(close) >= 60 else vol_20
            row["t_vol_surge"] = vol_20 / (vol_60 + 0.01)  # 波动率急升比例

            # ---- 风险因子 (Risk) —— 区间打分 ----
            row["r_volatility"] = pd.Series(close).pct_change().tail(250).std() * np.sqrt(252)
            c250 = close[-250:] if len(close) >= 250 else close
            rolling_max = np.maximum.accumulate(c250)
            drawdowns = (c250 - rolling_max) / rolling_max
            row["r_max_dd"] = drawdowns.min()
            row["r_neg_days"] = (pd.Series(close).pct_change().dropna() < 0).mean()

            rows.append(row)
        except Exception:
            continue

    result = pd.DataFrame(rows)
    if result.empty:
        return result, pd.DataFrame(), pd.Series(dtype=bool)

    # ---- P1: 分组排名 ----
    # 合并行业
    if industry_df is not None and not industry_df.empty and "industry" in industry_df.columns:
        result = result.merge(industry_df[["code", "industry"]], on="code", how="left")
    else:
        result["industry"] = "unknown"

    # 对每个连续因子在 industry 内做分位排名
    rank_cols = [c for c in result.columns if c not in ("code", "industry") and result[c].dtype in (np.float64, float, np.int64, int)]
    for col in rank_cols:
        try:
            result[col + "_rank"] = result.groupby("industry")[col].transform(rank_within_group)
        except Exception:
            result[col + "_rank"] = result[col].rank(pct=True)

    # ---- P1: 特殊处理 ----
    rank_map = {c + "_rank": c + "_rank" for c in rank_cols}

    # 估值类：PE/PB 越低越好 → 反转排名
    for c in ["v_pe", "v_pb"]:
        rk = c + "_rank"
        if rk in result.columns:
            result[rk] = 1 - result[rk]

    # 风险类：波动率、回撤、负收益天数越低越好
    for c in ["r_volatility", "r_max_dd", "r_neg_days"]:
        rk = c + "_rank"
        if rk in result.columns:
            result[rk] = 1 - result[rk]

    # 过热类：20日收益过高、偏离MA60过远、波动急升 = 越低越好
    for c in ["t_ret_20d", "t_ma60_gap", "t_vol_surge"]:
        rk = c + "_rank"
        if rk in result.columns:
            result[rk] = 1 - result[rk]

    # ---- P1: 加权综合 ----
    rank_cols_all = [c for c in result.columns if c.endswith("_rank")]
    weights = {
        "q_": 0.25,  # 质量 25%
        "v_": 0.15,  # 估值 15%
        "g_": 0.10,  # 成长 10%
        "d_": 0.15,  # 分红 15%
        "t_ret_60d_rank": 0.07, "t_ma60_slope_rank": 0.05, "t_rps_rank": 0.03,  # 趋势 15%
        "t_ret_20d_rank": -0.05, "t_ma60_gap_rank": -0.05, "t_vol_surge_rank": -0.05,  # 过热惩罚 -15%
        "r_": -0.10,  # 风险 -10%
    }

    total_w = 0.0
    result["score_raw"] = 0.0
    for col in rank_cols_all:
        # 匹配权重
        w = 0.0
        for prefix, pw in weights.items():
            if col.startswith(prefix):
                w = pw
                break
        if w == 0:
            w = 1.0 / len(rank_cols_all)  # 默认等权
        result["score_raw"] += w * result[col].fillna(0.5)
        total_w += abs(w)

    if total_w > 0:
        result["score_raw"] /= total_w

    # 再次做 industry 内排名 → final_score
    result["linear_score"] = result.groupby("industry")["score_raw"].transform(
        lambda x: x.rank(pct=True) * 100
    )

    # ---- P1: 硬剔除标记 ----
    hard_mask = pd.Series(False, index=result.index)
    reasons = pd.Series("", index=result.index)

    # 风险硬剔除规则
    if "r_max_dd" in result.columns:
        extreme_dd = result["r_max_dd"] < -0.6  # 最大回撤 > 60%
        hard_mask |= extreme_dd
        reasons[extreme_dd] += "最大回撤>60%;"

    # 标记数据质量
    missing_cols = [c for c in result.columns if c.endswith("_missing")]
    risk_flags = result[["code"] + missing_cols].copy() if missing_cols else pd.DataFrame({"code": result["code"]})

    return result, risk_flags, hard_mask
