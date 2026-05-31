"""
A 股多因子选股系统 — Streamlit 可视化界面
运行: streamlit run app.py
"""
import pickle
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="A 股多因子选股系统",
    page_icon=":chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_CACHE


@st.cache_resource
def load_data():
    """加载所有缓存数据"""
    data = {}
    cache_files = {
        "stock_list": "stock_list.pkl",
        "hist_data": "stock_hist.pkl",
        "index_data": "index_hist.pkl",
        "industry_data": "industry.pkl",
        "ranking_df": "ranking.pkl",
        "factor_df": "factors_v2.pkl",
        "importance_df": "importance_v2.pkl",
        "index_detail": "index_detail.pkl",
        "kline_results": "kline_results.pkl",
        "risk_df": "risk_df.pkl",
    }
    # 优先读 candidate_pool.parquet (含所有筛选字段)
    cp_path = DATA_CACHE / "candidate_pool.parquet"
    if cp_path.exists():
        try:
            data["ranking_df"] = pd.read_parquet(cp_path)
        except Exception:
            pass
    for key, fname in cache_files.items():
        fpath = DATA_CACHE / fname
        if fpath.exists():
            try:
                with open(fpath, "rb") as f:
                    data[key] = pickle.load(f)
            except Exception:
                data[key] = None
        else:
            data[key] = None

    # 优先从 valuation.pkl 加载（含真实价格和估算 PE/PB）
    val_path = DATA_CACHE / "valuation.pkl"
    if val_path.exists():
        try:
            with open(val_path, "rb") as f:
                data["valuation_data"] = pickle.load(f)
        except Exception:
            pass

    # 降级：从 stock_list 构建
    if data.get("valuation_data") is None and data.get("stock_list") is not None:
        sl = data["stock_list"]
        pe_ok = (sl.get("pe", pd.Series([0]*len(sl))) > 0).sum()
        if "price" in sl.columns and pe_ok > 10:
            data["valuation_data"] = sl[["code", "name", "price", "pe", "pb", "market_cap"]].copy()
        else:
            data["valuation_data"] = pd.DataFrame({
                "code": sl["code"], "name": sl.get("name", sl["code"]),
                "price": 10.0, "pe": 0, "pb": 0, "market_cap": 1e10,
            })
    elif data.get("valuation_data") is None and data.get("ranking_df") is not None:
        r = data["ranking_df"]
        data["valuation_data"] = pd.DataFrame({
            "code": r["code"], "name": r.get("name", r["code"]),
            "price": 10.0, "pe": 0, "pb": 0, "market_cap": 1e10,
        })

    # Load decay_report + exposure_report + model_mode
    for fname, key in [("decay_report.pkl", "decay_report"), ("exposure_report.pkl", "exposure_report"),
                        ("model_mode.pkl", "model_mode")]:
        p = DATA_CACHE / fname
        if p.exists():
            try:
                with open(p, "rb") as f:
                    data[key] = pickle.load(f)
            except Exception:
                pass
    exp_path = DATA_CACHE / "exposure_report.pkl"
    if exp_path.exists():
        try:
            with open(exp_path, "rb") as f:
                data["exposure_report"] = pickle.load(f)
        except Exception:
            pass

    # Load model_mode (Gate statuses)
    mm_path = DATA_CACHE / "model_mode.pkl"
    if mm_path.exists():
        try:
            with open(mm_path, "rb") as f:
                data["model_mode"] = pickle.load(f)
        except Exception:
            pass

    return data


# ==== 侧栏 ====
st.sidebar.title("A 股多因子选股系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航",
    ["总览驾驶舱", "智能选股", "候选池", "个股详情", "模型审计", "数据质量", "回测分析"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**数据状态**")

data = load_data()

has_hist = data.get("hist_data") is not None and len(data.get("hist_data", {})) > 0
has_ranking = data.get("ranking_df") is not None and len(data.get("ranking_df", pd.DataFrame())) > 0
has_index = data.get("index_detail") is not None

if has_hist:
    n = len(data.get("hist_data", {}))
    st.sidebar.success(f"市场数据: {n} 只股票")
else:
    st.sidebar.warning("市场数据: 未加载")

if has_ranking:
    n = len(data.get("ranking_df", pd.DataFrame()))
    st.sidebar.success(f"排名数据: {n} 条")
else:
    st.sidebar.warning("排名数据: 未加载")

if has_index:
    d = data.get("index_detail", {})
    score = d.get("total_score", "?")
    st.sidebar.success(f"指数评分: {score}/10")
else:
    st.sidebar.warning("指数评分: 未加载")

st.sidebar.markdown("---")
st.sidebar.caption("1. 运行 python run_weekly.py 更新数据")
st.sidebar.caption("2. 刷新浏览器查看最新结果")


# ==== 页面路由 ====
try:
    # Load extra modules for Gate/exposure/decay
    gates = None
    exposure_report = data.get("exposure_report")
    decay_report = data.get("decay_report")
    divergence_stats = None

    try:
        from models.gate_checker import check_all_gates
        from models.industry_exposure import check_industry_exposure
        from models.factor_decay import generate_decay_report

        ranking_df = data.get("ranking_df")
        if ranking_df is not None and not ranking_df.empty:
            if exposure_report is None:
                exposure_report = check_industry_exposure(ranking_df)
            if decay_report is None:
                decay_report = generate_decay_report(ranking_df)
            divergence_stats = {"severe": int((ranking_df.get("rank_diff", 0) >= 30).sum()),
                               "mild": int(((ranking_df.get("rank_diff", 0) >= 10) & (ranking_df.get("rank_diff", 0) < 30)).sum())}
            gates, _ = check_all_gates(ranking_df, data.get("index_detail"),
                                       decay_report, exposure_report, divergence_stats)
    except Exception:
        pass

    if page == "总览驾驶舱":
        model_mode_data = data.get("model_mode", {})
        from ui.pages.dashboard import render as dash_render
        dash_render(ranking_df, data.get("index_detail"), model_mode_data, gates, exposure_report, decay_report)

    elif page == "智能选股":
        from ui.pages.screener import render as screener_render
        screener_render(data.get("ranking_df"), data.get("industry_data"))

    elif page == "候选池":
        from ui.pages.top_stocks import render as top_render
        top_render(
            data.get("ranking_df"),
            data.get("importance_df"),
            data.get("industry_data"),
        )

    elif page == "个股详情":
        from ui.pages.stock_detail import render as detail_render
        detail_render(
            data.get("ranking_df"),
            data.get("hist_data"),
            data.get("factor_df"),
            data.get("kline_results", {}),
            data.get("valuation_data"),
        )

    elif page == "模型审计":
        st.title("模型审计")
        imp = data.get("importance_v2")
        mm = data.get("model_mode", {})
        audit = mm.get("audit", {}) if mm else {}
        if audit:
            st.markdown("### 模型健康摘要")
            ac = st.columns(4)
            ac[0].metric("Total Gain", f"{audit.get('total_gain', 0):.0f}")
            ac[1].metric("基本面贡献", f"{audit.get('fundamental_ratio', 0):.0%}")
            ac[2].metric("估值贡献", f"{audit.get('valuation_ratio', 0):.0%}")
            ac[3].metric("动量占比", f"{audit.get('momentum_ratio', 0):.0%}")
            ac2 = st.columns(3)
            ac2[0].metric("非零特征", f"{audit.get('nonzero_features', 0)}/{audit.get('feature_count', 0)}")
            ac2[1].metric("Total Split", str(audit.get('total_split', 0)))
            mr = audit.get('momentum_ratio', 0)
            if mr > 0.50: st.error(f"动量占比 {mr:.0%} > 50%，模型偏动量")
            elif mr > 0.35: st.warning(f"动量占比 {mr:.0%} 在 35-50% 之间")
        if imp is not None and not imp.empty:
            st.markdown("### 因子重要性 Top 15")
            from ui.components import factor_importance_chart
            st.plotly_chart(factor_importance_chart(imp), use_container_width=True)
        dr = data.get("decay_report")
        if dr is not None and not dr.empty:
            st.markdown("### 因子衰减状态")
            st.dataframe(dr, use_container_width=True, hide_index=True)

    elif page == "数据质量":
        st.title("数据质量 Gate")
        mm_q = data.get("model_mode", {}) if data.get("model_mode") else {}
        st.markdown("### Gate 状态")
        qc = st.columns(4)
        qc[0].metric("Gate 0A 行情", mm_q.get("price_status", "?"))
        qc[1].metric("Gate 0B 财务", mm_q.get("financial_status", "?"))
        qc[2].metric("Gate 0C 估值", mm_q.get("val_status", "?"))
        qc[3].metric("Gate 0D 特征", mm_q.get("feature_status", "?"))
        val_data = data.get("valuation_data")
        if val_data is not None and not val_data.empty and "valuation_source" in val_data.columns:
            st.markdown("### 估值来源分布")
            src_counts = val_data["valuation_source"].value_counts()
            sc = st.columns(len(src_counts))
            for i, (src, cnt) in enumerate(src_counts.items()):
                sc[i].metric(src, cnt)
        idx = data.get("index_detail")
        if idx:
            st.markdown("### 指数环境")
            details = idx.get("details", [])
            if details:
                st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

    elif page == "回测分析":
        st.title("回测分析")
        mm_bt = data.get("model_mode", {}) if data.get("model_mode") else {}
        val_status = mm_bt.get("val_status", "FAIL") if mm_bt else "FAIL"
        if val_status == "FAIL":
            st.warning("回测暂不可用：估值历史数据不足，当前不输出完整多因子回测结论。请补齐 valuation_daily 或 Tushare daily_basic 后重试。")
        else:
            from ui.pages.backtest import render as bt_render
            bt_render(data.get("ranking_df"), data.get("hist_data"))

except Exception as e:
    st.error(f"页面加载失败: {e}")
    st.info("请确保已运行 python run_weekly.py 生成数据缓存")
