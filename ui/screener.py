"""
智能选股器 — 策略模板 + 条件筛选
"""
import pandas as pd
import numpy as np

SCREENER_TEMPLATES = {
    "核心候选型": {
        "score_min": {"rule_score": 75, "model_score": 60, "technical_score": 55, "risk_score": 60, "final_action_score_adj": 70},
        "exclude_data_fail": True, "exclude_valuation_fail": True, "exclude_model_disagree": True,
        "stock_pools": ["核心候选池", "模型观察池", "价值观察池"], "sort_col": "final_action_score_adj",
        "desc": "规则、模型、技术、风险均通过的最严格筛选，适合重点研究。"
    },
    "价值低估型": {
        "score_min": {"rule_score": 70, "risk_score": 55},
        "score_max": {"pe_hist_pct_5y": 50, "pb_hist_pct_5y": 60},
        "exclude_data_fail": True, "exclude_valuation_fail": True,
        "valuation_quality_allowed": ["PASS", "WARN"],
        "stock_pools": ["价值观察池", "核心候选池", "普通观察池"], "sort_col": "rule_score",
        "desc": "估值历史偏低 + 规则评分≥70 + 风险可控。用于发现研究对象，不代表立即买入。"
    },
    "价值低估型-严格": {
        "score_min": {"rule_score": 80, "valuation_score": 70, "risk_score": 60},
        "score_max": {"pe_hist_pct_5y": 40, "pb_hist_pct_5y": 50},
        "exclude_data_fail": True, "exclude_valuation_fail": True, "exclude_model_disagree": True,
        "valuation_quality_allowed": ["PASS"],
        "stock_pools": ["价值观察池", "核心候选池"], "sort_col": "rule_score",
        "desc": "严格低估值筛选，只保留估值数据完全通过的高分股票。"
    },
    "质量白马型": {
        "score_min": {"quality_score": 75, "cashflow_score": 65, "risk_score": 65, "rule_score": 75},
        "exclude_data_fail": True,
        "stock_pools": ["价值观察池", "核心候选池", "普通观察池"], "sort_col": "quality_score",
        "desc": "基本面质量高、现金流稳健的白马股。"
    },
    "模型确认型": {
        "score_min": {"model_score": 70, "rule_score": 65, "technical_score": 50},
        "model_consensus": ["一致偏强", "一致中性", "轻度分歧"],
        "exclude_data_fail": True, "exclude_model_disagree": True,
        "stock_pools": ["模型观察池", "核心候选池", "技术强势观察池"], "sort_col": "model_score",
        "desc": "LightGBM/XGBoost 一致看好的股票，模型信号较强。"
    },
    "技术修复型": {
        "score_min": {"technical_score": 70, "rule_score": 60, "model_score": 40},
        "above_ma20": True, "ret20_positive": True,
        "exclude_data_fail": True,
        "stock_pools": ["技术强势观察池", "模型观察池"], "sort_col": "technical_score",
        "desc": "趋势开始修复，站上MA20且近20日收益为正的股票。"
    },
    "防御现金流型": {
        "score_min": {"cashflow_score": 70, "risk_score": 65, "rule_score": 70},
        "industries": ["公用事业", "食品饮料", "银行", "煤炭", "通信", "交通运输"],
        "exclude_data_fail": True,
        "stock_pools": ["价值观察池", "核心候选池", "普通观察池"], "sort_col": "cashflow_score",
        "desc": "现金流充裕、行业偏防御的低波动标的。"
    },
}


def apply_screener_filters(df, filters):
    """应用筛选条件，返回过滤后的 DataFrame"""
    out = df.copy()

    pools = filters.get("stock_pools", [])
    if pools and "全部" not in pools:
        out = out[out["stock_pool"].isin(pools)]

    industries = filters.get("industries", [])
    if industries and "全部" not in industries:
        out = out[out["industry"].isin(industries)]

    for col, min_v in filters.get("score_min", {}).items():
        if col in out.columns:
            out = out[out[col] >= min_v]

    for col, max_v in filters.get("score_max", {}).items():
        if col in out.columns:
            out = out[out[col] <= max_v]

    consensus = filters.get("model_consensus", [])
    if consensus:
        out = out[out["model_consensus"].isin(consensus)]

    if filters.get("exclude_data_fail", True):
        col = "data_quality_flag"
        if col in out.columns:
            out = out[out[col] == "PASS"]
    if filters.get("exclude_valuation_fail", False):
        col = "valuation_quality"
        if col in out.columns:
            allowed = filters.get("valuation_quality_allowed", ["PASS"])
            out = out[out[col].isin(allowed)]
    if filters.get("exclude_model_disagree", False):
        if "model_consensus" in out.columns:
            out = out[~out["model_consensus"].isin(["模型分歧", "严重分歧"])]
    if filters.get("above_ma20", False):
        if "close" in out.columns and "ma20" in out.columns:
            out = out[out["close"] >= out["ma20"]]
    if filters.get("above_ma60", False):
        if "close" in out.columns and "ma60" in out.columns:
            out = out[out["close"] >= out["ma60"]]
    if filters.get("ret20_positive", False):
        if "ret_20d" in out.columns:
            out = out[out["ret_20d"] > 0]

    sort_col = filters.get("sort_col", "final_action_score_adj")
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)

    return out
