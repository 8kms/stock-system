"""
Gate 4: 行业暴露控制

约束:
  Top30: 单行业≤5只, 单行业权重≤20%, 前三行业合计≤45%, ≥8个行业
  Top10: 单行业≤2只, 单行业权重≤25%, ≥5个行业
"""
import numpy as np
import pandas as pd


def check_industry_exposure(ranking_df, top_n=30, core_n=10):
    """
    检查行业暴露

    参数:
        ranking_df: 含 code, industry, total_score 的 DataFrame
        top_n: Top N 池规模
        core_n: 核心池规模

    返回:
        dict: { status, warnings, details, industry_report }
    """
    if ranking_df is None or ranking_df.empty or "industry" not in ranking_df.columns:
        return {"status": "SKIP", "warnings": ["无行业数据"], "details": pd.DataFrame()}

    df = ranking_df.dropna(subset=["total_score"]).sort_values("total_score", ascending=False)

    # Top30 检查
    top30 = df.head(top_n).copy()
    top30["weight"] = 1.0 / len(top30)

    ind_exp = top30.groupby("industry").agg(
        count=("code", "count"),
        weight=("weight", "sum"),
        avg_score=("total_score", "mean"),
    ).reset_index().sort_values("weight", ascending=False)

    ind_exp["status"] = "OK"
    warnings = []

    # 单行业 ≤5 只
    over5 = ind_exp[ind_exp["count"] > 5]
    if len(over5) > 0:
        for _, r in over5.iterrows():
            warnings.append(f"行业'{r['industry']}' {int(r['count'])}只 > 5只上限")
            ind_exp.loc[ind_exp["industry"] == r["industry"], "status"] = "WARN_count"

    # 单行业权重 ≤20%
    over20 = ind_exp[ind_exp["weight"] > 0.20]
    if len(over20) > 0:
        for _, r in over20.iterrows():
            warnings.append(f"行业'{r['industry']}' 权重{r['weight']:.1%} > 20%")
            ind_exp.loc[ind_exp["industry"] == r["industry"], "status"] = "WARN_weight"

    # 前三行业合计 ≤45%
    top3_weight = ind_exp.head(3)["weight"].sum()
    if top3_weight > 0.45:
        warnings.append(f"前三行业合计权重 {top3_weight:.1%} > 45%")

    # 行业数量 ≥8
    n_industries = ind_exp["count"].gt(0).sum()
    if n_industries < 8:
        warnings.append(f"行业数量 {n_industries} < 8")

    # Top10 核心池检查
    top10 = df.head(core_n).copy()
    top10["weight"] = 1.0 / len(top10)
    core_exp = top10.groupby("industry").agg(
        count=("code", "count"),
        weight=("weight", "sum"),
    ).reset_index()

    core_warnings = []
    core_over2 = core_exp[core_exp["count"] > 2]
    if len(core_over2) > 0:
        for _, r in core_over2.iterrows():
            core_warnings.append(f"核心池 行业'{r['industry']}' {int(r['count'])}只 > 2只")
    core_over25 = core_exp[core_exp["weight"] > 0.25]
    if len(core_over25) > 0:
        for _, r in core_over25.iterrows():
            core_warnings.append(f"核心池 行业'{r['industry']}' 权重{r['weight']:.1%} > 25%")
    core_n_ind = core_exp["count"].gt(0).sum()
    if core_n_ind < 5:
        core_warnings.append(f"核心池行业数量 {core_n_ind} < 5")

    all_warnings = warnings + [f"[核心池] {w}" for w in core_warnings]

    passed = len(all_warnings) == 0

    # 行业调整后收益（简化：等权行业组合 vs 原始 Top30）
    if len(ind_exp) >= 3:
        sector_neutral_ret = ind_exp["avg_score"].mean()
        original_ret = top30["total_score"].mean()
        sector_adj_diff = sector_neutral_ret - original_ret
        if sector_adj_diff < -2:
            all_warnings.append(f"行业调整后得分下降 {abs(sector_adj_diff):.1f} 分")
            passed = False
    else:
        sector_adj_diff = 0

    return {
        "status": "PASS" if passed else "WARN",
        "passed": passed,
        "warnings": all_warnings,
        "top30_industries": n_industries,
        "top3_weight": round(top3_weight, 3),
        "sector_adj_diff": round(sector_adj_diff, 1) if sector_adj_diff else 0,
        "industry_report": ind_exp,
        "core_warnings": core_warnings,
    }


def apply_industry_constraints(ranking_df, top_n=30):
    """
    应用行业约束：超出限制的降级到观察池

    返回:
        core_df: 满足约束的股票
        watch_df: 被降级的股票
    """
    if ranking_df is None or ranking_df.empty:
        return ranking_df, pd.DataFrame()

    df = ranking_df.sort_values("total_score", ascending=False)
    core = []
    watch = []
    ind_count = {}

    for _, row in df.iterrows():
        ind = row.get("industry", "unknown")
        cnt = ind_count.get(ind, 0)

        if cnt < 5 and len(core) < top_n:
            core.append(row)
            ind_count[ind] = cnt + 1
        else:
            watch.append(row)

    core_df = pd.DataFrame(core) if core else pd.DataFrame()
    watch_df = pd.DataFrame(watch) if watch else pd.DataFrame()

    # 显示降级信息
    if len(watch_df) > 0:
        over_inds = {k: v for k, v in ind_count.items() if v >= 5}
        if over_inds:
            print(f"  行业约束: {len(over_inds)} 个行业达到上限, {len(watch_df)} 只降级到观察池")

    return core_df, watch_df
