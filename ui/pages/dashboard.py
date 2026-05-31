"""
总览驾驶舱 — 首页：系统状态 + 今日结论 + 池分布 + Gate 状态
"""
import streamlit as st
import pandas as pd
import numpy as np


def render(ranking_df, index_detail, model_mode_data, gates, exposure_report, decay_report):
    st.title("总览驾驶舱")

    # ==== 系统状态提示条 ====
    mode = (model_mode_data or {}).get("mode", "UNKNOWN")
    mode_name = (model_mode_data or {}).get("name", "未知")
    mode_desc = (model_mode_data or {}).get("desc", "")

    status_colors = {
        "MULTIFACTOR_PASS": "green", "MULTIFACTOR_WARN": "orange",
        "MODEL_MECHANISM_PASS_DATA_PENDING": "blue", "MOMENTUM_ONLY": "violet",
        "FUNDAMENTAL_PASS_VALUATION_FAIL": "orange", "TRAINING_PASS_BACKTEST_PENDING": "blue",
        "LIVE_READY": "green", "DATA_FAIL": "red",
    }
    color = status_colors.get(mode, "gray")
    st.markdown(f"<div style='padding:12px;background:{color};color:white;border-radius:8px;font-size:1.1em'>"
                f"当前系统状态：{mode} — {mode_name}<br><small>{mode_desc}</small></div>",
                unsafe_allow_html=True)

    # ==== Gate 状态条 ====
    if gates:
        st.markdown("### Gate 状态")
        gate_cols = st.columns(min(6, len(gates)))
        gate_icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}
        for i, g in enumerate(gates[:6]):
            with gate_cols[i]:
                st.metric(g["name"], gate_icons.get(g["status"], "?") + " " + g["status"])

    # ==== 今日结论 ====
    st.markdown("### 今日结论")
    if ranking_df is not None and not ranking_df.empty:
        pools = ranking_df["stock_pool"].value_counts()
        core_n = int(pools.get("核心候选池", 0))
        value_n = int(pools.get("价值观察池", 0))
        model_n = int(pools.get("模型观察池", 0))
        tech_n = int(pools.get("技术强势观察池", 0))
        normal_n = int(pools.get("普通观察池", 0))
        pending_n = int(pools.get("数据待补池", 0))
        exclude_n = int(pools.get("剔除池", 0))

        if core_n > 0:
            conclusion = f"核心候选池 {core_n} 只可重点研究。价值观察池 {value_n} 只，技术强势观察池 {tech_n} 只。"
            advice = "优先研究核心候选池；价值观察池可做备选研究；技术强势池只作趋势观察。"
        else:
            conclusion = f"当前无核心候选。价值观察池 {value_n} 只，技术强势观察池 {tech_n} 只，剔除池 {exclude_n} 只。"
            advice = "优先研究价值观察池；技术强势池只作趋势观察；当前不建议把技术强势股直接视为买入候选。"

        st.info(conclusion)
        st.caption(f"建议：{advice}")

        # 池分布卡片
        st.markdown("### 池分布")
        pc = st.columns(7)
        pool_data = [
            ("核心候选池", core_n), ("价值观察池", value_n), ("模型观察池", model_n),
            ("技术强势观察池", tech_n), ("普通观察池", normal_n),
            ("数据待补池", pending_n), ("剔除池", exclude_n),
        ]
        for i, (name, cnt) in enumerate(pool_data):
            pc[i].metric(name, cnt)

        if core_n == 0:
            st.info("核心候选池为空是正常结果，说明当前没有同时满足规则、模型、技术、风险条件的标的。")

    # ==== 指数环境 ====
    if index_detail:
        st.markdown("### 指数环境")
        ic = st.columns(3)
        score = index_detail.get("total_score", "?")
        state = index_detail.get("state_cn", "?")
        ic[0].metric("指数评分", f"{score}/10")
        ic[1].metric("市场状态", state)
        ic[2].metric("操作建议", index_detail.get("suggestion", "")[:30] + "...")

    # ==== 模型审计摘要 ====
    if model_mode_data and model_mode_data.get("audit"):
        audit = model_mode_data["audit"]
        st.markdown("### 模型摘要")
        ac = st.columns(4)
        ac[0].metric("Total Gain", f"{audit.get('total_gain', 0):.0f}")
        ac[1].metric("基本面贡献", f"{audit.get('fundamental_ratio', 0):.0%}")
        ac[2].metric("动量占比", f"{audit.get('momentum_ratio', 0):.0%}")
        ac[3].metric("非零特征", f"{audit.get('nonzero_features', 0)}/{audit.get('feature_count', 0)}")
