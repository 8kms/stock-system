"""
多维度评分体系 — 规则分/模型分/技术分/最终操作分 + 候选池分类

规则评分: 基本面/估值/现金流/风险 — 判断"是否值得研究"
模型评分: LightGBM + XGBoost 排名 — 判断"未来排序是否被模型支持"
技术评分: K线/均线/动量 — 判断"现在是否适合买"
最终操作分: 四维加权 — 决定进入哪个池
"""
import numpy as np
import pandas as pd


def calc_model_score_from_rank(lgb_rank_pct, xgb_rank_pct):
    """排名百分位 → 模型评分 (0-100)"""
    return max(0, min(100, ((1 - lgb_rank_pct) * 0.5 + (1 - xgb_rank_pct) * 0.5) * 100))


def calc_technical_score(close, ma20=None, ma60=None, ret5=0, ret20=0, ret60=0, ret120=0, vol_change=0):
    """技术面评分 (0-100)"""
    score = 100.0
    if ma20 is not None and close < ma20: score -= 15
    if ma60 is not None and close < ma60: score -= 35
    if ret5 < -3: score -= 10
    if ret20 < -5: score -= 20
    if ret60 < 0: score -= 15
    if ret120 < 0: score -= 10
    if vol_change > 0.5: score -= 10
    return max(0, min(100, score))


def calc_final_action_score(rule_score, model_score, technical_score, risk_score=70):
    """四维加权最终操作分 — 模型分提权到35%防止规则分主导"""
    return round(rule_score * 0.35 + model_score * 0.35 + technical_score * 0.20 + risk_score * 0.10, 1)


POOL_PRIORITY = {
    "核心候选池": 1, "价值观察池": 2, "模型观察池": 3,
    "技术强势观察池": 4, "普通观察池": 5, "数据待补池": 6, "剔除池": 7,
}


def apply_score_adjustments(row):
    """模型一致性 + 数据质量 + 估值质量 + 风险全面调整"""
    score = float(row.get("final_action_score", 50) or 0)
    label = str(row.get("model_consensus", ""))
    pool = str(row.get("stock_pool", ""))
    dq = str(row.get("data_quality_flag", ""))
    vq = str(row.get("valuation_quality", ""))
    rr = int(row.get("risk_red_count", 0) or 0)

    if label == "严重分歧": score -= 15
    elif label == "模型分歧": score -= 8
    elif label == "轻度分歧": score -= 4
    elif label == "一致偏弱": score -= 20
    elif label == "一致偏强": score += 5

    if dq != "PASS": score -= 15
    if vq == "FAIL": score -= 20
    elif vq == "WARN": score -= 5
    if rr > 0: score -= min(30, rr * 10)
    if pool == "技术强势观察池": score -= 3

    return max(0, min(100, round(score, 1)))


def calc_industry_confidence(industry_count):
    """行业样本数 → 置信度权重"""
    if industry_count < 10: return 0.5, "LOW"
    if industry_count < 20: return 0.75, "MEDIUM"
    return 1.0, "HIGH"


def build_scores(ranking_df, hist_data):
    """为主 ranking_df 添加 model_score, technical_score, final_action_score"""
    df = ranking_df.copy()
    n_total = len(df)

    # 模型评分
    if "lgb_rank" in df.columns and "xgb_rank" in df.columns:
        lgb_pct = df["lgb_rank"] / n_total
        xgb_pct = df["xgb_rank"] / n_total
        df["model_score"] = df.apply(lambda r: calc_model_score_from_rank(
            r.get("lgb_rank", n_total/2) / n_total,
            r.get("xgb_rank", n_total/2) / n_total), axis=1)
    elif "lgb_score" in df.columns and "xgb_score" in df.columns:
        df["lgb_rank_pct"] = df["lgb_score"].rank(pct=True)
        df["xgb_rank_pct"] = df["xgb_score"].rank(pct=True)
        df["model_score"] = calc_model_score_from_rank(df["lgb_rank_pct"], df["xgb_rank_pct"])
    else:
        df["model_score"] = 50

    df["lgb_rank_pct"] = df["lgb_rank_pct"] if "lgb_rank_pct" in df.columns else 0.5
    df["xgb_rank_pct"] = df["xgb_rank_pct"] if "xgb_rank_pct" in df.columns else 0.5

    # 技术评分
    tech_scores = []
    for _, row in df.iterrows():
        code = row["code"]
        if code in hist_data:
            h = hist_data[code]
            close = h["close"].iloc[-1]
            ma20 = h["close"].rolling(20).mean().iloc[-1] if len(h) >= 20 else None
            ma60 = h["close"].rolling(60).mean().iloc[-1] if len(h) >= 60 else None
            ret5 = (close / h["close"].iloc[-6] - 1) * 100 if len(h) >= 6 else 0
            ret20 = (close / h["close"].iloc[-21] - 1) * 100 if len(h) >= 21 else 0
            ret60 = (close / h["close"].iloc[-61] - 1) * 100 if len(h) >= 61 else 0
            ret120 = (close / h["close"].iloc[-121] - 1) * 100 if len(h) >= 121 else 0
            vol_change = h["close"].pct_change().tail(20).std() / (h["close"].pct_change().tail(60).std() + 0.01) if len(h) >= 60 else 0
            tech_scores.append(calc_technical_score(close, ma20, ma60, ret5, ret20, ret60, ret120, vol_change))
        else:
            tech_scores.append(50)
    df["technical_score"] = tech_scores

    # 最终操作分
    rule_s = df["rule_score"].fillna(50) if "rule_score" in df.columns else 50
    risk_s = df["risk_score"].fillna(70) if "risk_score" in df.columns else 70
    df["final_action_score"] = calc_final_action_score(rule_s, df["model_score"], df["technical_score"], risk_s)

    # 全面调整分
    if "model_consensus" in df.columns:
        df["final_action_score_adj"] = df.apply(apply_score_adjustments, axis=1)
    else:
        df["final_action_score_adj"] = df["final_action_score"]

    return df


# ============================================================
# 候选池分类
# ============================================================

def is_core_candidate(row):
    return (
        row.get("data_quality_flag", "PASS") == "PASS"
        and row.get("valuation_quality", "") in ("PASS", "WARN")
        and row.get("rule_score", 0) >= 75
        and row.get("model_score", 0) >= 60
        and row.get("technical_score", 0) >= 55
        and row.get("risk_score", 0) >= 60
        and row.get("final_action_score_adj", row.get("final_action_score", 0)) >= 70
        and row.get("model_consensus", "") in ("一致偏强", "一致中性", "轻度分歧")
        and int(row.get("risk_red_count", 0) or 0) == 0
    )


def is_value_watchlist(row):
    return (
        row.get("rule_score", 0) >= 80
        and (row.get("model_score", 0) < 60 or row.get("technical_score", 0) < 55)
    )


def is_model_watchlist(row):
    return (
        row.get("model_score", 0) >= 60
        and row.get("technical_score", 0) >= 55
        and row.get("rule_score", 0) >= 65
        and row.get("rule_score", 0) < 75
    )


def is_momentum_watchlist(row):
    return (
        row.get("technical_score", 0) >= 70
        and row.get("model_score", 0) >= 60
        and row.get("rule_score", 0) >= 60
    )


def is_data_pending(row):
    return row.get("data_quality_flag", "PASS") != "PASS"


def is_excluded(row):
    return row.get("rule_score", 0) < 60 or row.get("risk_red_count", 0) > 0


def assign_stock_pool(row):
    if is_data_pending(row): return "数据待补池"
    if is_excluded(row): return "剔除池"
    if is_core_candidate(row): return "核心候选池"
    if is_value_watchlist(row): return "价值观察池"
    if is_model_watchlist(row): return "模型观察池"
    if is_momentum_watchlist(row): return "技术强势观察池"
    return "普通观察池"


def generate_downgrade_reasons(row):
    """基于'为什么不是核心候选'生成降级原因"""
    pool = str(row.get("stock_pool", ""))
    if pool == "核心候选池":
        return "满足核心候选条件"
    reasons = []
    if row.get("data_quality_flag", "PASS") != "PASS":
        reasons.append("数据质量未完全通过")
    vq = str(row.get("valuation_quality", ""))
    if vq == "FAIL": reasons.append("估值数据未通过")
    elif vq == "WARN": reasons.append("估值数据为WARN，需复核来源")
    if row.get("rule_score", 0) < 75:
        reasons.append("规则评分未达到核心阈值")
    if row.get("model_score", 0) < 60:
        reasons.append("模型评分未达到核心阈值")
    if row.get("technical_score", 0) < 55:
        reasons.append("技术评分未达到核心阈值")
    if row.get("risk_score", 0) < 60:
        reasons.append("风险评分偏低")
    if int(row.get("risk_red_count", 0) or 0) > 0:
        reasons.append("存在风险红灯")
    mc = str(row.get("model_consensus", ""))
    if mc == "一致偏弱": reasons.append("LGB/XGBoost一致偏弱")
    if mc == "模型分歧": reasons.append("LGB/XGBoost存在模型分歧")
    if mc == "严重分歧": reasons.append("LGB/XGBoost严重分歧")
    if pool == "价值观察池":
        reasons.append("基本面/估值较好，但模型或技术尚未确认")
    if pool == "模型观察池":
        reasons.append("模型支持度较高，但尚未满足核心候选全部条件")
    if pool == "技术强势观察池":
        reasons.append("技术强，但需复核基本面和估值，避免纯动量")
    if row.get("industry_stock_count", 999) < 15:
        reasons.append("行业样本数偏少，行业排名置信度降低")
    return "；".join(reasons) if reasons else "未进入核心候选池"


def model_consensus_label(row):
    rd = row.get("rank_diff", 0)
    lp = row.get("lgb_rank_pct", 0.5)
    xp = row.get("xgb_rank_pct", 0.5)
    if rd <= 0.10:
        if lp > 0.70 and xp > 0.70: return "一致偏弱"
        if lp < 0.30 and xp < 0.30: return "一致偏强"
        return "一致中性"
    return "模型分歧"


def generate_action_text(row):
    pool = row.get("stock_pool", "普通观察池")
    if pool == "核心候选池":
        if row.get("technical_score", 50) >= 70: return "可重点研究，等待合适买点"
        return "核心候选，但买点一般，等待回踩确认"
    if pool == "价值观察池":
        if row.get("close", 0) < row.get("ma60", row.get("close", 1)):
            return "基本面/估值较好，但跌破60日线，等待重新站上MA60或止跌企稳"
        return "规则评分较高，但模型未确认，继续观察"
    if pool == "技术强势观察池": return "技术强，但需复核基本面和估值，避免纯动量"
    if pool == "数据待补池": return "数据质量不足，暂不判断"
    if pool == "剔除池": return "不符合系统要求，剔除"
    return "观察"
