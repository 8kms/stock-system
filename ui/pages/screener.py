"""
智能选股器 — 三层筛选
"""
import streamlit as st
import pandas as pd
import numpy as np
from ui.screener import apply_screener_filters, SCREENER_TEMPLATES

DISPLAY_COLS = ["code", "name", "stock_pool", "final_action_score_adj",
                "rule_score", "model_score", "technical_score", "risk_score",
                "model_consensus", "action_text", "downgrade_reasons", "industry"]


def render(ranking_df, industry_df=None):
    st.title("智能选股器")
    if ranking_df is None or ranking_df.empty:
        st.warning("暂无数据，请先运行 run_weekly.py")
        return

    df = ranking_df.copy()
    # Add ma columns
    try:
        import pickle; from config import DATA_CACHE
        hist = pickle.load(open(DATA_CACHE / "stock_hist.pkl", "rb"))
        c_d, m20, m60 = {}, {}, {}
        for c, h in hist.items():
            if len(h) >= 60:
                c_d[c] = h["close"].iloc[-1]
                m20[c] = h["close"].rolling(20).mean().iloc[-1]
                m60[c] = h["close"].rolling(60).mean().iloc[-1]
        df["close"] = df["code"].map(c_d)
        df["ma20"] = df["code"].map(m20)
        df["ma60"] = df["code"].map(m60)
    except: pass

    # Layer 1: Template
    st.markdown("### 策略模板")
    tmpl_names = list(SCREENER_TEMPLATES.keys())
    sel = st.selectbox("选择模板", tmpl_names + ["自定义"], index=tmpl_names.index("价值低估型"))
    if sel != "自定义":
        filters = SCREENER_TEMPLATES[sel].copy()
        st.caption(filters.pop("desc", ""))
    else:
        filters = {"stock_pools": [], "industries": [], "sort_col": "final_action_score_adj",
                   "score_min": {}, "score_max": {}, "model_consensus": [],
                   "exclude_data_fail": True, "exclude_valuation_fail": False,
                   "exclude_model_disagree": False, "above_ma20": False,
                   "above_ma60": False, "ret20_positive": False}

    # Layer 2: Core
    st.markdown("### 核心筛选")
    c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1])
    all_pools = ["核心候选池", "价值观察池", "模型观察池", "技术强势观察池", "普通观察池", "数据待补池", "剔除池"]
    dp = filters.get("stock_pools", ["价值观察池", "核心候选池"])
    with c1: pools = st.multiselect("股票池", all_pools, default=[p for p in dp if p in all_pools])
    with c2:
        all_i = ["全部"] + sorted(df["industry"].dropna().unique().tolist()) if "industry" in df.columns else ["全部"]
        di = filters.get("industries", [])
        inds = st.multiselect("行业", all_i, default=di if di else ["全部"])
    sm = {"调整后操作分": "final_action_score_adj", "规则评分": "rule_score",
          "模型评分": "model_score", "技术评分": "technical_score", "风险评分": "risk_score"}
    ds = filters.get("sort_col", "final_action_score_adj")
    si = list(sm.values()).index(ds) if ds in sm.values() else 0
    with c3: sort_col = st.selectbox("排序", list(sm.keys()), index=si)
    with c4: n_show = st.selectbox("显示", [20, 50, 100], index=0)
    filters["stock_pools"] = pools
    filters["industries"] = [] if "全部" in inds else inds
    filters["sort_col"] = sm[sort_col]

    # Layer 3: Advanced (collapsed)
    with st.expander("高级筛选", expanded=False):
        use_def = st.checkbox("使用模板默认阈值", value=(sel != "自定义"))
        smin = filters.get("score_min", {})
        if not use_def:
            a1, a2, a3, a4 = st.columns(4)
            smin["rule_score"] = a1.slider("规则≥", 0, 100, smin.get("rule_score", 70))
            smin["model_score"] = a2.slider("模型≥", 0, 100, smin.get("model_score", 50))
            smin["technical_score"] = a3.slider("技术≥", 0, 100, smin.get("technical_score", 50))
            smin["risk_score"] = a4.slider("风险≥", 0, 100, smin.get("risk_score", 60))
            filters["score_min"] = smin
        elif smin:
            st.caption("、".join(str(k) + "≥" + str(v) for k, v in smin.items()))
        mc = ["一致偏强", "一致中性", "轻度分歧", "模型分歧", "严重分歧", "一致偏弱"]
        filters["model_consensus"] = st.multiselect("模型一致性", mc, default=filters.get("model_consensus", []))
        filters["exclude_model_disagree"] = st.checkbox("排除模型分歧", value=filters.get("exclude_model_disagree", False))
        t1, t2, t3 = st.columns(3)
        filters["above_ma20"] = t1.checkbox("仅站上MA20", value=filters.get("above_ma20", False))
        filters["above_ma60"] = t2.checkbox("仅站上MA60", value=filters.get("above_ma60", False))
        filters["ret20_positive"] = t3.checkbox("仅20日收益>0", value=filters.get("ret20_positive", False))
        d1, d2 = st.columns(2)
        filters["exclude_data_fail"] = d1.checkbox("排除数据FAIL", value=filters.get("exclude_data_fail", True))
        filters["exclude_valuation_fail"] = d2.checkbox("排除估值FAIL", value=filters.get("exclude_valuation_fail", True))

    # Results
    result = apply_screener_filters(df, filters)

    # Threshold summary (template conditions)
    threshold_lines = []
    smin = filters.get("score_min", {})
    smax = filters.get("score_max", {})
    if smin: threshold_lines.append("条件: " + ", ".join(k + "≥" + str(v) for k, v in smin.items()))
    if smax: threshold_lines.append("上限: " + ", ".join(k + "≤" + str(v) + "%" for k, v in smax.items()))
    vqa = filters.get("valuation_quality_allowed", [])
    if vqa: threshold_lines.append("估值质量: " + "/".join(vqa))
    if threshold_lines:
        st.caption("模板阈值 — " + " | ".join(threshold_lines))

    # Summary
    sp = "、".join(filters.get("stock_pools", [])) or "全部"
    si = "、".join(filters.get("industries", [])) or "全部"
    ex = []
    if filters.get("exclude_data_fail"): ex.append("数据FAIL")
    if filters.get("exclude_valuation_fail"): ex.append("估值FAIL")
    if filters.get("exclude_model_disagree"): ex.append("模型分歧")
    if filters.get("above_ma60"): ex.append("仅MA60")
    if filters.get("ret20_positive"): ex.append("20日>0")
    et = "、".join(ex) if ex else "无"
    st.info(sp + " / " + si + " / 排除:" + et + " / " + sort_col + " / 结果:" + str(len(result)) + "只")

    st.markdown("### 筛选结果: " + str(len(result)) + " 只")
    dcols = [c for c in DISPLAY_COLS if c in result.columns]

    if len(result) == 0:
        st.warning("当前筛选条件下没有符合条件的股票。")
        # Funnel diagnosis
        steps = [("初始", len(df))]
        cur = df.copy()
        if filters.get("exclude_data_fail") and "data_quality_flag" in cur.columns:
            cur = cur[cur["data_quality_flag"] == "PASS"]; steps.append(("排除数据FAIL", len(cur)))
        if filters.get("exclude_valuation_fail") and "valuation_quality" in cur.columns:
            vqa = filters.get("valuation_quality_allowed", ["PASS", "WARN"])
            cur = cur[cur["valuation_quality"].isin(vqa)]; steps.append(("估值质量=" + "/".join(vqa), len(cur)))
        pools = filters.get("stock_pools", [])
        if pools: cur = cur[cur["stock_pool"].isin(pools)]; steps.append(("股票池", len(cur)))
        for k, v in (smin or {}).items():
            if k in cur.columns: cur = cur[cur[k] >= v]; steps.append((k + "≥" + str(v), len(cur)))
        for k, v in (smax or {}).items():
            if k in cur.columns: cur = cur[cur[k] <= v]; steps.append((k + "≤" + str(v), len(cur)))
        with st.expander("漏斗诊断（哪一步筛没了）", expanded=True):
            for name, cnt in steps: st.write("• " + name + ": " + str(cnt) + "只")
        # Relax
        st.markdown("**一键放宽:**")
        rb1, rb2, rb3 = st.columns(3)
        if rb1.button("加入普通观察池"): st.session_state["relax_pool"] = True; st.rerun()
        if rb2.button("规则降到65"): st.session_state["relax_rule"] = True; st.rerun()
        if rb3.button("查看最接近10只"): st.session_state["show_near"] = True; st.rerun()
        # Apply relax
        for k in ("relax_pool", "relax_rule"):
            if st.session_state.get(k):
                st.session_state[k] = False
        # Nearest
        if st.session_state.get("show_near"):
            near = df[df["data_quality_flag"] == "PASS"].copy()
            fit = pd.Series(0.0, index=near.index)
            if "rule_score" in near.columns: fit += near["rule_score"].fillna(0) * 0.35
            if "valuation_score" in near.columns: fit += near["valuation_score"].fillna(0) * 0.30
            if "risk_score" in near.columns: fit += near["risk_score"].fillna(0) * 0.15
            if "pe_hist_pct_5y" in near.columns: fit += (100 - near["pe_hist_pct_5y"].fillna(100)) * 0.10
            if "pb_hist_pct_5y" in near.columns: fit += (100 - near["pb_hist_pct_5y"].fillna(100)) * 0.10
            near["_fit"] = fit.round(1)
            near10 = near.nlargest(10, "_fit")
            st.info("最接近「" + sel + "」的 10 只:")
            nd = [c for c in (dcols + ["_fit"]) if c in near10.columns]
            st.dataframe(near10[nd], use_container_width=True, hide_index=True)
    else:
        st.dataframe(result[dcols].head(n_show), use_container_width=True, hide_index=True)
        if len(result) > 0:
            csv_data = result[dcols].head(n_show).to_csv(index=False)
            st.download_button("导出CSV", csv_data, "screener_" + sel + ".csv", "text/csv")
