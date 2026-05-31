"""
Gate 控制台 — 每周操作系统主页
显示 Gate 0-8 状态、候选池概览、风险预警
"""
import streamlit as st
import pandas as pd
import numpy as np


def render(gates, exposure_report, decay_report, ranking_df, divergence_stats):
    st.title("本周总览 — Gate 控制台")

    if gates is None:
        st.warning("请先运行 run_weekly.py 生成Gate检查数据")
        return

    # ---- Gate 状态总览 ----
    st.markdown("## Gate 0-8 状态")

    cols = st.columns(4)
    for i, g in enumerate(gates):
        with cols[i % 4]:
            status_color = {"PASS": "green", "WARN": "orange", "FAIL": "red", "INFO": "blue"}.get(g["status"], "gray")
            st.markdown(
                f"<div style='padding:8px;border-left:4px solid {status_color};margin:4px 0;background:#f8f9fa;border-radius:4px'>"
                f"<b>Gate {g['gate']}</b>: {g['name']}<br>"
                f"<span style='color:{status_color};font-weight:bold'>{g['status']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # 总体判定
    passed_gates = sum(1 for g in gates if g["passed"])
    if passed_gates >= 6:
        st.success(f"通过 {passed_gates}/9 道门 — 系统可用")
    elif passed_gates >= 4:
        st.warning(f"通过 {passed_gates}/9 道门 — 部分预警，谨慎使用")
    else:
        st.error(f"仅通过 {passed_gates}/9 道门 — 系统不可用，请修复后再选股")

    # ---- Gate 详情 ----
    st.markdown("---")
    st.markdown("## Gate 详情")
    for g in gates:
        status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}.get(g["status"], "❓")
        expander = st.expander(f"{status_icon} Gate {g['gate']}: {g['name']} — {g['status']}", expanded=(g["status"] != "PASS"))
        with expander:
            st.write(f"**详情:** {g['details']}")
            if g["recommendations"]:
                st.info(g["recommendations"])

    # ---- 本周候选池 ----
    st.markdown("---")
    st.markdown("## 本周候选池概览")

    if ranking_df is not None and not ranking_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        total = len(ranking_df)
        score_col = "total_score" if "total_score" in ranking_df.columns else "model_score"
        above85 = (ranking_df[score_col] >= 85).sum() if score_col in ranking_df.columns else 0
        above75 = (ranking_df[score_col] >= 75).sum() if score_col in ranking_df.columns else 0
        col1.metric("总候选数", total)
        col2.metric("重点研究 (≥85)", above85)
        col3.metric("观察池 (75-85)", above75 - above85)
        col4.metric("剔除/低分", total - above75)

        # Top5
        st.markdown("**Top 5 核心候选**")
        display_cols = ["code", "name", score_col]
        if "industry" in ranking_df.columns:
            display_cols.append("industry")
        if "rule_score" in ranking_df.columns:
            display_cols.append("rule_score")
        if "diverge_flag" in ranking_df.columns:
            display_cols.append("diverge_flag")
        if "kline_score" in ranking_df.columns:
            display_cols.append("kline_score")

        top5 = ranking_df.nlargest(5, score_col)
        display_cols = [c for c in display_cols if c in top5.columns]
        st.dataframe(top5[display_cols], use_container_width=True, hide_index=True)

    # ---- 预警面板 ----
    st.markdown("---")
    st.markdown("## 预警面板")

    warnings_list = []

    # 行业暴露
    if exposure_report and not exposure_report.get("passed", True):
        warnings_list.extend(exposure_report.get("warnings", []))

    # 因子衰减
    if decay_report is not None and not decay_report.empty:
        n_red = (decay_report["状态"].str.contains("红色")).sum()
        if n_red > 0:
            red_factors = decay_report[decay_report["状态"].str.contains("红色")]["因子"].tolist()
            warnings_list.append(f"因子衰减红灯: {', '.join(red_factors[:5])}")

    # 模型分歧
    if divergence_stats:
        ms = divergence_stats.get("severe", 0)
        if ms > 0:
            warnings_list.append(f"模型严重分歧: {ms} 只股票")

    if warnings_list:
        for w in warnings_list:
            st.warning(w)
    else:
        st.success("本周无重大预警")
