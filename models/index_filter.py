"""
指数环境评分过滤器
5 个指数各 0-2 分，满分 10 分
"""
import numpy as np
import pandas as pd

from config import INDEX_SCORE_RULES, MARKET_STATE, INDEX_NAMES, WATCH_INDICES


def calc_index_score(index_data):
    """
    计算 5 个指数的环境评分
    返回: (总分, 市场状态, 明细 DataFrame)
    """
    if index_data is None or len(index_data) == 0:
        return 5, "震荡", pd.DataFrame()

    details = []

    for idx_code in WATCH_INDICES:
        if idx_code not in index_data:
            continue

        df = index_data[idx_code]
        if df is None or df.empty or len(df) < 120:
            continue

        try:
            close = df["close"]
            volume = df["volume"] if "volume" in df.columns else pd.Series([1] * len(close))

            score = 0.0
            indicators = {}

            # 1. 站上 20 日线
            ma20 = close.rolling(20).mean().iloc[-1]
            above_ma20 = close.iloc[-1] > ma20
            indicators["站上20日线"] = above_ma20
            if above_ma20:
                score += INDEX_SCORE_RULES["above_ma20"]

            # 2. 站上 60 日线
            ma60 = close.rolling(60).mean().iloc[-1]
            above_ma60 = close.iloc[-1] > ma60
            indicators["站上60日线"] = above_ma60
            if above_ma60:
                score += INDEX_SCORE_RULES["above_ma60"]

            # 3. 20 日线向上
            ma20_series = close.rolling(20).mean()
            if len(ma20_series) >= 10:
                ma20_up = ma20_series.iloc[-1] > ma20_series.iloc[-10]
            else:
                ma20_up = False
            indicators["20日线向上"] = ma20_up
            if ma20_up:
                score += INDEX_SCORE_RULES["ma20_up"]

            # 4. 成交额温和放大
            if len(volume) >= 10:
                vol_5 = volume.tail(5).mean()
                vol_20 = volume.tail(20).mean()
                vol_expand = vol_5 > vol_20 * 1.05
            else:
                vol_expand = False
            indicators["成交额放大"] = vol_expand
            if vol_expand:
                score += INDEX_SCORE_RULES["volume_expand"]

            details.append({
                "指数代码": idx_code,
                "指数名称": INDEX_NAMES.get(idx_code, idx_code),
                "当前价格": round(float(close.iloc[-1]), 2),
                "MA20": round(float(ma20), 2),
                "MA60": round(float(ma60), 2),
                "得分": score,
                "站上20日线": "是" if above_ma20 else "否",
                "站上60日线": "是" if above_ma60 else "否",
                "20日线向上": "是" if ma20_up else "否",
                "成交额放大": "是" if vol_expand else "否",
            })
        except Exception:
            continue

    detail_df = pd.DataFrame(details)
    total_score = detail_df["得分"].sum() if not detail_df.empty else 5

    # 计算 20 日线向上的比例
    up_col = "20日线向上"
    if up_col in detail_df.columns:
        up_ratio = (detail_df[up_col] == "是").mean()
    else:
        up_ratio = 0.5

    # 判断市场状态
    state = _classify_market(total_score)

    return total_score, state, detail_df


def calc_market_state_detail(index_data):
    """
    增强版市场状态判断
    返回更详细的状态字典，用于 UI 展示
    """
    total, state, details = calc_index_score(index_data)

    detail_records = details.to_dict("records") if not details.empty else []

    return {
        "total_score": round(total, 1),
        "max_score": 10,
        "state": state,
        "state_cn": {
            "strong": "强势 — 可以正常选股",
            "oscillation": "震荡 — 只买高质量强股",
            "weak": "弱势 — 减少新开仓",
            "risk": "风险区 — 不开新仓，等待",
        }.get(state, "未知"),
        "details": detail_records,
        "suggestion": _get_suggestion(state),
    }


def _classify_market(score):
    for state, (lo, hi) in MARKET_STATE.items():
        if lo <= score <= hi:
            return state
    return "oscillation"


def _get_suggestion(state):
    suggestions = {
        "strong": "市场强势，可以正常执行选股策略。建议保持较高仓位，重点配置得分 85+ 的标的。",
        "oscillation": "市场震荡，注意控制仓位。只买入高质量、低估值、高现金流的股票，回避纯题材股。",
        "weak": "市场弱势，建议降低总仓位。已有持仓做好止损计划，新开仓需格外谨慎。",
        "risk": "市场处于风险区域，建议空仓或极低仓位等待。密切观察指数能否重回均线上方。",
    }
    return suggestions.get(state, "无法判断")
