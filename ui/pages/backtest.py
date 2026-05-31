"""
回测页
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ui.components import cumulative_return_chart, drawdown_chart


def render(ranking_df, hist_data):
    st.title("回测分析")

    if ranking_df is None or ranking_df.empty:
        st.warning("暂无数据")
        return

    if hist_data is None or len(hist_data) == 0:
        st.warning("暂无历史行情数据")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        n_top = st.selectbox("Top N", [5, 10, 20, 30], index=1)
    with col2:
        lookback_days = st.selectbox("回看天数", [60, 120, 250], index=1)
    with col3:
        score_col = "total_score" if "total_score" in ranking_df.columns else "model_score"

    st.markdown("---")

    # 简单回测：Top N 等权组合 vs 全市场等权
    top_codes = ranking_df.sort_values(score_col, ascending=False).head(n_top)["code"].tolist()

    # 展示 Top N 股票列表
    top_names = ranking_df[ranking_df["code"].isin(top_codes)][["code", "name", score_col]].drop_duplicates("code")
    top_names = top_names.sort_values(score_col, ascending=False)
    name_str = "、".join([f"{r['name']}({r[score_col]:.0f}分)" for _, r in top_names.iterrows()])
    st.caption(f"Top {n_top} 组合: {name_str}")
    st.caption("等权买入 Top N 股票 vs 全市场等权，对比累计收益走势。红线跑赢灰线 = 模型选股有效。")

    # 收集所有日收益率，统一对齐到日期索引
    all_ret_dict = {}
    top_ret_dict = {}

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < lookback_days + 1:
            continue
        try:
            close = df.set_index("date")["close"].tail(lookback_days + 1)
            daily_ret = close.pct_change().dropna()
            if code in top_codes:
                top_ret_dict[code] = daily_ret
            all_ret_dict[code] = daily_ret
        except Exception:
            continue

    if len(top_ret_dict) == 0:
        st.warning("回测数据不足")
        return

    # 合并到一个 DataFrame 并取等权均值（inner join 对齐日期）
    top_ret_df = pd.DataFrame(top_ret_dict).dropna(how="all")
    all_ret_df = pd.DataFrame(all_ret_dict).dropna(how="all")

    top_combined = top_ret_df.mean(axis=1, skipna=True).dropna()
    all_combined = all_ret_df.mean(axis=1, skipna=True).dropna()

    top_cum = (1 + top_combined).cumprod()
    all_cum = (1 + all_combined).cumprod()

    # 对齐到共同日期（inner join）
    common_dates = top_cum.index.intersection(all_cum.index)
    top_cum = top_cum.loc[common_dates]
    all_cum = all_cum.loc[common_dates]

    # 收益对比
    st.markdown("### 组合收益对比")
    returns_df = pd.DataFrame({
        f"Top{n_top}等权": top_cum.values,
        "全市场等权": all_cum.values,
    }, index=common_dates)

    st.plotly_chart(
        cumulative_return_chart(returns_df, benchmark_col="全市场等权"),
        use_container_width=True,
    )

    # 关键指标
    st.markdown("### 关键指标")

    def calc_metrics(cum_ret, daily_ret):
        total_ret = cum_ret.iloc[-1] - 1
        annual_ret = (1 + total_ret) ** (252 / len(daily_ret)) - 1
        annual_vol = daily_ret.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min()
        win_rate = (daily_ret > 0).mean()
        return {
            "累计收益": f"{total_ret:+.2%}",
            "年化收益": f"{annual_ret:+.2%}",
            "年化波动": f"{annual_vol:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "胜率": f"{win_rate:.1%}",
        }

    top_metrics = calc_metrics(top_cum, top_combined.loc[common_dates])
    all_metrics = calc_metrics(all_cum, all_combined.loc[common_dates])

    m_cols = st.columns(6)
    for i, (key, val) in enumerate(zip(top_metrics.keys(), top_metrics.values())):
        m_cols[i].metric(
            f"Top{n_top} {key}",
            val,
            delta=None,
        )

    # 回撤图
    st.markdown("### 回撤对比")
    top_dd = (top_cum - top_cum.cummax()) / top_cum.cummax()

    st.plotly_chart(drawdown_chart(top_dd), use_container_width=True)

    # Rank IC 近似
    st.markdown("### 模型评估")
    if score_col in ranking_df.columns:
        # 计算近 60 日收益排名并比较
        future_rets = {}
        for code, df in hist_data.items():
            if df is None or df.empty or len(df) < 60:
                continue
            try:
                close = df["close"]
                fwd_ret = close.iloc[-1] / close.iloc[-min(60, len(close)-1)] - 1
                future_rets[code] = fwd_ret
            except Exception:
                continue

        future_df = pd.Series(future_rets, name="fwd_ret")
        # 去重：取每个 code 的第一条
        score_df = ranking_df.drop_duplicates(subset="code").set_index("code")[score_col]

        common = score_df.index.intersection(future_df.index)
        if len(common) >= 20:
            from scipy.stats import spearmanr
            ic, pval = spearmanr(
                score_df.loc[common],
                future_df.loc[common].rank(),
            )
            col_a, col_b = st.columns(2)
            col_a.metric("Rank IC (Spearman)", f"{ic:.4f}", delta=None)
            col_b.metric("IC P-Value", f"{pval:.4f}", delta=None)

            if abs(ic) > 0.1:
                st.success("IC 显著，模型排序有效")
            elif abs(ic) > 0.05:
                st.info("IC 一般，模型有一定区分能力")
            else:
                st.warning("IC 偏低，模型区分能力较弱")
        else:
            st.info("数据不足，无法计算 Rank IC")
