"""
Top 股票页 — 四维评分 + 六池分类
"""
import streamlit as st
import pandas as pd
import numpy as np
from ui.components import ranking_bar_chart, factor_importance_chart
from models.score_builder import POOL_PRIORITY

POOL_COLORS = {
    "核心候选池": "#d4edda", "价值观察池": "#fff3cd", "模型观察池": "#d1ecf1",
    "技术强势观察池": "#eadcf8", "普通观察池": "#f8f9fa", "数据待补池": "#e2e3e5",
    "剔除池": "#f8d7da",
}

def render(ranking_df, importance_df, industry_df=None):
    st.title("候选池")

    if ranking_df is None or ranking_df.empty:
        st.warning("暂无排名数据")
        return

    # Ensure score columns exist
    if "stock_pool" not in ranking_df.columns:
        from models.score_builder import build_scores, assign_stock_pool
        ranking_df = build_scores(ranking_df, {})
        ranking_df["stock_pool"] = ranking_df.apply(assign_stock_pool, axis=1)

    # Pool filter — default to 核心候选池 if exists, else 价值观察池
    pool_list = ["全部"] + sorted([p for p in ranking_df["stock_pool"].unique()], key=lambda p: POOL_PRIORITY.get(p, 99))
    default_pool = "核心候选池" if "核心候选池" in ranking_df["stock_pool"].values else ("价值观察池" if "价值观察池" in ranking_df["stock_pool"].values else "全部")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_pool = st.selectbox("股票池", pool_list, index=pool_list.index(default_pool) if default_pool in pool_list else 0)
    with col2:
        sort_by = st.selectbox("排序", ["最终操作分(调整)", "最终操作分", "规则评分", "模型评分", "技术评分"])
    with col3:
        n_show = st.selectbox("显示", [10, 20, 30, 50], index=1)

    sort_map = {"最终操作分(调整)": "final_action_score_adj", "最终操作分": "final_action_score",
                "规则评分": "rule_score", "模型评分": "model_score", "技术评分": "technical_score"}
    sort_col = sort_map.get(sort_by, "final_action_score_adj")

    filtered = ranking_df.copy()
    if selected_pool != "全部":
        filtered = filtered[filtered["stock_pool"] == selected_pool]
    # 全部视图: pool_priority 优先, 再按调整分
    if selected_pool == "全部" and "pool_priority" in filtered.columns:
        filtered = filtered.sort_values(["pool_priority", sort_col], ascending=[True, False])
    else:
        filtered = filtered.sort_values(sort_col, ascending=False)

    # Pool summary — st.metric layout
    st.markdown("### 池分布")
    pool_counts = ranking_df["stock_pool"].value_counts()
    ordered_pools = ["核心候选池", "价值观察池", "模型观察池", "技术强势观察池", "普通观察池", "数据待补池", "剔除池"]
    cols = st.columns(7)
    for i, pool in enumerate(ordered_pools):
        cnt = int(pool_counts.get(pool, 0))
        cols[i].metric(pool, cnt)
    if pool_counts.get("核心候选池", 0) == 0:
        st.info("核心候选池为空是正常情况，说明当前没有规则、模型、技术、风险同时通过的标的。默认展示价值观察池。")

    # Table
    st.markdown(f"### {selected_pool} ({len(filtered)}只)")
    display = filtered.head(n_show)
    dcols = ["code", "name", "stock_pool", "final_action_score_adj", "final_action_score",
             "rule_score", "model_score", "technical_score", "risk_score",
             "model_consensus", "action_text", "downgrade_reasons", "industry"]
    dcols = [c for c in dcols if c in display.columns]
    display_df = display[dcols].copy()

    # Color rows by pool
    def color_row(r):
        pool = r.get("stock_pool", "")
        c = POOL_COLORS.get(pool, "#fff")
        return [f"background-color: {c}22"] * len(r)

    st.dataframe(display_df.style.apply(color_row, axis=1), use_container_width=True, hide_index=True)

    # Bar chart — "全部"视图只显示前4个优先池
    chart_data = filtered.copy()
    if selected_pool == "全部" and "pool_priority" in chart_data.columns:
        chart_data = chart_data[chart_data["pool_priority"] <= 4]
    bar_col = "final_action_score_adj" if "final_action_score_adj" in chart_data.columns else "final_action_score"
    chart_data = chart_data.sort_values(bar_col, ascending=True).tail(15)
    fig = ranking_bar_chart(chart_data, n=15)
    st.plotly_chart(fig, use_container_width=True)

    if importance_df is not None and not importance_df.empty:
        st.markdown("### 因子重要性")
        st.plotly_chart(factor_importance_chart(importance_df), use_container_width=True)
