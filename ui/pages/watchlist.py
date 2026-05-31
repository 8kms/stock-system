"""
观察池页
"""
import streamlit as st
import pandas as pd
import numpy as np

from config import WATCHLIST_STOCKS


def render(ranking_df, kline_results):
    st.title("观察池")

    # 自定义观察池
    st.markdown("### 管理观察池")
    custom_codes = st.text_input(
        "输入股票代码（逗号分隔）",
        value=",".join(WATCHLIST_STOCKS),
        help="输入要跟踪的股票代码，用逗号分隔",
    )
    watch_codes = [c.strip() for c in custom_codes.split(",") if c.strip()]

    if not watch_codes:
        st.info("请添加至少一个股票代码到观察池")
        return

    # 从排名数据中筛选
    if ranking_df is None or ranking_df.empty:
        st.warning("暂无排名数据")
        return

    watch_df = ranking_df[ranking_df["code"].isin(watch_codes)].copy()

    if watch_df.empty:
        st.warning("观察池中的股票均未出现在排名数据中")
        # 显示哪些没找到
        missing = set(watch_codes) - set(ranking_df["code"].tolist())
        if missing:
            st.info(f"未找到的股票: {', '.join(missing)}")
        return

    st.markdown(f"### 观察池表现 ({len(watch_df)} 只)")

    # 关键指标
    score_col = "total_score" if "total_score" in watch_df.columns else "model_score"
    avg_score = watch_df[score_col].mean()
    above_75 = (watch_df[score_col] >= 75).sum()
    above_85 = (watch_df[score_col] >= 85).sum()

    m_cols = st.columns(4)
    m_cols[0].metric("池中股票", len(watch_df))
    m_cols[1].metric("平均得分", f"{avg_score:.1f}" if not pd.isna(avg_score) else "N/A")
    m_cols[2].metric("得分 75+", above_75)
    m_cols[3].metric("得分 85+", above_85)

    # 详细表格
    display = watch_df.copy()
    display = display.sort_values(score_col, ascending=False)

    # 添加 K 线状态
    if kline_results:
        display["K线状态"] = display["code"].apply(
            lambda c: _kline_status(kline_results.get(c))
        )

    table_cols = ["code", "name", score_col, "model_score"]
    if "kline_score" in display.columns:
        table_cols.append("kline_score")
    if "K线状态" in display.columns:
        table_cols.append("K线状态")
    if "industry" in display.columns:
        table_cols.append("industry")

    table_cols = [c for c in table_cols if c in display.columns]

    def color_row(row):
        s = row.get(score_col, 50)
        if s >= 85:
            return ["background-color: #C6EFCE"] * len(row)
        elif s >= 75:
            return ["background-color: #FFEB9C"] * len(row)
        elif s < 65:
            return ["background-color: #FFC7CE"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display[table_cols].style.apply(color_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # 风险提示
    st.markdown("### 风险提示")
    if kline_results:
        warnings = []
        for code in watch_codes:
            if code in kline_results:
                kr = kline_results[code]
                if kr.get("is_overbought"):
                    name = watch_df[watch_df["code"] == code].iloc[0].get("name", code) if code in watch_df["code"].values else code
                    warnings.append(f"⚠ {code} {name}: 短期过热，注意追高风险")
                if not kr.get("is_healthy"):
                    name = watch_df[watch_df["code"] == code].iloc[0].get("name", code) if code in watch_df["code"].values else code
                    warnings.append(f"🔴 {code} {name}: K 线走坏，建议减仓观望")

        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("观察池中暂无显著 K 线风险信号")
    else:
        st.info("暂无 K 线数据")


def _kline_status(kr):
    if kr is None:
        return "无数据"
    if kr.get("is_overbought"):
        return "过热"
    if not kr.get("is_healthy"):
        return "走坏"
    if kr.get("is_stabilizing"):
        return "企稳"
    return "正常"
