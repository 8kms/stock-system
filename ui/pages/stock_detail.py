"""
个股详情页 — P5 升级版
新增: 估值历史分位图 | 行业横向对比 | 风险红黄绿清单 | 因子分项得分
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.components import kline_chart, score_gauge_chart


def render(ranking_df, hist_data, factor_df, kline_results, valuation_data):
    st.title("个股详情")

    if ranking_df is None or ranking_df.empty:
        st.warning("暂无数据")
        return

    code_options = ranking_df["code"].tolist()
    name_map = dict(zip(ranking_df["code"], ranking_df.get("name", ranking_df["code"]))) if "name" in ranking_df.columns else {}
    search_options = [f"{c} - {name_map.get(c, '')}" for c in code_options]

    selected = st.selectbox("选择股票", search_options)
    selected_code = selected.split(" - ")[0] if " - " in selected else selected

    if not selected_code:
        return

    stock_hist = hist_data.get(selected_code) if hist_data else None
    stock_kline = kline_results.get(selected_code) if kline_results else None
    info_row = ranking_df[ranking_df["code"] == selected_code]
    stock_name = info_row.iloc[0].get("name", selected_code) if not info_row.empty else selected_code

    st.markdown(f"## {selected_code} — {stock_name}")

    # ---- KPI 卡片: 四维评分 ----
    kpi_cols = st.columns(8)
    rule_s = info_row.iloc[0].get("rule_score", info_row.iloc[0].get("total_score", 50)) if not info_row.empty else 50
    model_s = info_row.iloc[0].get("model_score", 50) if not info_row.empty else 50
    tech_s = info_row.iloc[0].get("technical_score", 50) if not info_row.empty else 50
    final_s = info_row.iloc[0].get("final_action_score", 50) if not info_row.empty else 50
    pool_s = info_row.iloc[0].get("stock_pool", "普通观察池") if not info_row.empty else "未知"
    action_s = info_row.iloc[0].get("action_text", "") if not info_row.empty else ""

    kpi_cols[0].metric("规则评分", f"{rule_s:.0f}" if not pd.isna(rule_s) else "N/A",
                       help="基本面/估值/现金流/风险综合")
    kpi_cols[1].metric("模型评分", f"{model_s:.0f}" if not pd.isna(model_s) else "N/A",
                       help="LightGBM + XGBoost 排名转换分")
    kpi_cols[2].metric("技术评分", f"{tech_s:.0f}" if not pd.isna(tech_s) else "N/A",
                       help="K线/均线/动量/波动状态")
    kpi_cols[3].metric("操作分", f"{final_s:.0f}" if not pd.isna(final_s) else "N/A",
                       help="四维加权最终可操作评分")

    if valuation_data is not None:
        vrow = valuation_data[valuation_data["code"] == selected_code]
        if not vrow.empty:
            price = vrow.iloc[0].get("price", 0)
            pe = vrow.iloc[0].get("pe", 0)
            pb = vrow.iloc[0].get("pb", 0)
            mc = vrow.iloc[0].get("market_cap", 0)
            kpi_cols[4].metric("最新价", f"{price:.2f}" if price > 0 else "N/A")
            kpi_cols[5].metric("PE", f"{pe:.1f}" if pe > 0 else "N/A")
            kpi_cols[6].metric("PB", f"{pb:.2f}" if pb > 0 else "N/A")
            kpi_cols[7].metric("总市值", f"{mc/1e8:.0f}亿" if mc > 1e10 else (f"{mc/1e8:.1f}亿" if mc > 0 else "N/A"))
        else:
            for i in range(4, 8): kpi_cols[i].metric("-", "N/A")
    else:
        for i in range(4, 8): kpi_cols[i].metric("-", "N/A")

    # 池分类 + 建议
    if pool_s:
        pool_color = {"核心候选池": "green", "价值观察池": "orange", "模型观察池": "blue",
                      "技术强势观察池": "violet", "普通观察池": "gray", "数据待补池": "gray", "剔除池": "red"}
        c = pool_color.get(pool_s, "gray")
        st.markdown(f"**分类**: :{c}[{pool_s}] | **建议**: {action_s}")

    # ---- K 线图 ----
    st.markdown("### K 线图")
    if stock_hist is not None:
        st.plotly_chart(kline_chart(stock_hist, f"{selected_code} {stock_name}"), use_container_width=True)
    else:
        st.info("暂无历史行情数据")

    # ---- P5: 估值历史分位 ----
    st.markdown("### 估值历史分位")
    if stock_hist is not None and len(stock_hist) >= 250:
        _render_valuation_percentile(stock_hist, selected_code)
    else:
        st.info("历史数据不足，无法计算估值分位")

    # ---- 得分仪表盘 + 降级原因 ----
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(score_gauge_chart(final_s, "最终操作分"), use_container_width=True)
    with col2:
        # 降级原因
        reasons = info_row.iloc[0].get("downgrade_reasons", "") if not info_row.empty else ""
        consensus = info_row.iloc[0].get("model_consensus", "") if not info_row.empty else ""
        if reasons:
            st.markdown("### 降级原因")
            for r in str(reasons).split("；"):
                if r.strip(): st.warning(r.strip())
        if consensus:
            st.caption(f"模型一致性: {consensus}")
        if stock_kline:
            st.markdown("### K 线信号")
            for sig in stock_kline.get("signals", []):
                if "正常" in str(sig):
                    st.success(sig)
                elif "过热" in str(sig) or "异常" in str(sig):
                    st.warning(sig)
                elif "跌破" in str(sig) or "走坏" in str(sig):
                    st.error(sig)
                else:
                    st.info(sig)
        else:
            st.info("暂无 K 线信号")

    # ---- P5: 因子分项 + 分歧检测 ----
    if ranking_df is not None and "rank_diff" in ranking_df.columns:
        st.markdown("### 模型分歧检测")
        div_row = ranking_df[ranking_df["code"] == selected_code]
        if not div_row.empty:
            rd = div_row.iloc[0].get("rank_diff", 0)
            ag = div_row.iloc[0].get("agreement_score", 1)
            flag = div_row.iloc[0].get("diverge_flag", "")
            dc = st.columns(3)
            dc[0].metric("LightGBM排名", f"{div_row.iloc[0].get('lgb_rank', 0):.0f}" if 'lgb_rank' in div_row.columns else "N/A")
            dc[1].metric("XGBoost排名", f"{div_row.iloc[0].get('xgb_rank', 0):.0f}" if 'xgb_rank' in div_row.columns else "N/A")
            dc[2].metric("分歧度", f"{rd:.0f}%")
            if rd >= 30: st.error(f"严重分歧 — 不进核心池")
            elif rd >= 10: st.warning(f"降权 — {flag}")

    # ---- P5: 行业对比 ----
    if ranking_df is not None and "industry" in ranking_df.columns:
        st.markdown("### 行业横向对比")
        ind = info_row.iloc[0].get("industry", "") if not info_row.empty else ""
        if ind:
            peers = ranking_df[ranking_df["industry"] == ind]
            if len(peers) > 1:
                compare_col = "final_action_score" if "final_action_score" in ranking_df.columns else "total_score"
                _render_industry_comparison(info_row, peers, compare_col, selected_code)

    # ---- P5: 风险清单 ----
    if stock_kline or stock_hist is not None:
        st.markdown("### 风险信号清单")
        _render_risk_checklist(stock_hist, stock_kline, ranking_df, selected_code)

    # ---- 近期涨跌幅 ----
    if stock_hist is not None and len(stock_hist) >= 20:
        st.markdown("### 近期涨跌幅")
        close = stock_hist["close"]
        cols = st.columns(4)
        for i, (label, days) in enumerate([("5日", 5), ("20日", 20), ("60日", 60), ("120日", 120)]):
            if len(close) > days:
                ret = close.iloc[-1] / close.iloc[-days] - 1
                cols[i].metric(label, f"{ret:+.2%}")


def _render_valuation_percentile(hist_df, code):
    """估值历史分位图"""
    close = hist_df["close"]
    n = len(close)

    # 计算各周期分位
    periods = {"1年": min(242, n), "2年": min(484, n), "3年": min(726, n), "全部": n}
    current = close.iloc[-1]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("PE 代理分位 (价格位置)", "历史价格区间"))

    pct_data = []
    labels = []
    for label, p in periods.items():
        seg = close.tail(p)
        pct = (seg < current).sum() / len(seg) * 100
        pct_data.append(pct)
        labels.append(label)

    fig.add_trace(go.Bar(x=labels, y=pct_data, marker_color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
                         text=[f"{v:.0f}%" for v in pct_data], textposition="outside"),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=hist_df["date"].tail(726), y=close.tail(726), mode="lines",
                             name="价格", line=dict(color="#2F5496")), row=1, col=2)
    fig.add_hline(y=current, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_industry_comparison(info_row, peers, score_col, selected_code):
    """行业横向对比表"""
    stock_score = info_row.iloc[0].get(score_col, 50)
    ind_avg = peers[score_col].mean()
    ind_rank = (peers[score_col] > stock_score).sum() + 1
    ind_total = len(peers)

    comp_data = {
        "指标": ["综合得分", "行业内排名", "行业均值", "行业最高", "行业内股票数"],
        "数值": [
            f"{stock_score:.1f}",
            f"{int(ind_rank)}/{ind_total}",
            f"{ind_avg:.1f}",
            f"{peers[score_col].max():.1f}",
            str(ind_total),
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


def _render_risk_checklist(hist_df, kline_result, ranking_df, code):
    """风险红黄绿清单"""
    items = []

    # MA60 状态
    if hist_df is not None and len(hist_df) >= 60:
        close = hist_df["close"]
        ma60 = close.rolling(60).mean().iloc[-1]
        on_ma60 = close.iloc[-1] > ma60
        vol = close.iloc[-1] / close.iloc[-2] - 1 if len(close) >= 2 else 0

        if on_ma60 and vol > 0:
            items.append(("MA60 状态", "绿", "站上MA60"))
        elif on_ma60:
            items.append(("MA60 状态", "黄", "站上但缩量"))
        else:
            items.append(("MA60 状态", "红", "跌破MA60"))

    # 波动率变化
    if hist_df is not None and len(hist_df) >= 60:
        rets = hist_df["close"].pct_change().dropna()
        vol20 = rets.tail(20).std()
        vol60 = rets.tail(60).std()
        vol_ratio = vol20 / vol60 if vol60 > 0 else 1
        if vol_ratio < 1.3:
            items.append(("波动率", "绿", "正常"))
        elif vol_ratio < 2.0:
            items.append(("波动率", "黄", "放大"))
        else:
            items.append(("波动率", "红", "异常放大"))

    # K 线信号
    if kline_result:
        if kline_result.get("is_overbought"):
            items.append(("短期过热", "红", "是"))
        else:
            items.append(("短期过热", "绿", "否"))

        if not kline_result.get("is_healthy"):
            items.append(("K线走坏", "红", "是"))
        else:
            items.append(("K线走坏", "绿", "否"))

    # 模型分歧
    if ranking_df is not None and "rank_diff" in ranking_df.columns:
        crow = ranking_df[ranking_df["code"] == code]
        if not crow.empty:
            rd = crow.iloc[0].get("rank_diff", 0)
            if rd < 10: items.append(("模型分歧", "绿", f"{rd:.0f}%"))
            elif rd < 30: items.append(("模型分歧", "黄", f"{rd:.0f}%"))
            else: items.append(("模型分歧", "红", f"{rd:.0f}%"))

    # Display
    cols = st.columns(4)
    for i, (name, color, value) in enumerate(items):
        emoji = {"绿": "🟢", "黄": "🟡", "红": "🔴"}[color]
        cols[i % 4].markdown(f"{emoji} **{name}**: {value}")
