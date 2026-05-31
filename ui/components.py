"""
通用可视化组件
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def market_state_card(total_score, state, state_cn, suggestion):
    """市场状态大卡片"""
    colors = {
        "strong": "#2ecc71",
        "oscillation": "#f39c12",
        "weak": "#e74c3c",
        "risk": "#c0392b",
    }
    color = colors.get(state, "#95a5a6")

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=total_score,
        title={"text": state_cn, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 3], "color": "#ffcccc"},
                {"range": [3, 5], "color": "#ffe0cc"},
                {"range": [5, 8], "color": "#ffffcc"},
                {"range": [8, 10], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 8,
            },
        },
        number={"suffix": " / 10", "font": {"size": 28}},
    ))
    fig.update_layout(height=250)
    return fig


def index_radar_chart(details):
    """指数评分雷达图"""
    if not details:
        return go.Figure()

    names = [d.get("指数名称", d.get("指数代码", "")) for d in details]
    scores = [d.get("得分", 0) for d in details]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=names,
        fill="toself",
        name="指数评分",
        line_color="#2F5496",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 2.5], tickvals=[0, 0.5, 1, 1.5, 2])),
        height=350,
    )
    return fig


def ranking_bar_chart(ranking_df, n=15):
    """Top N 得分柱状图"""
    if ranking_df is None or ranking_df.empty:
        return go.Figure()

    top = ranking_df.head(n).copy()
    score_col = "total_score" if "total_score" in top.columns else "model_score"
    top = top.sort_values(score_col, ascending=True)

    colors = []
    for s in top[score_col]:
        if s >= 85:
            colors.append("#2ecc71")
        elif s >= 75:
            colors.append("#3498db")
        elif s >= 65:
            colors.append("#f39c12")
        else:
            colors.append("#95a5a6")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top[score_col],
        y=top["code"].str.cat(top.get("name", top["code"]), sep=" "),
        orientation="h",
        marker_color=colors,
        text=top[score_col].round(1),
        textposition="outside",
    ))
    fig.update_layout(
        height=500,
        xaxis_title="综合得分",
        xaxis_range=[0, 105],
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# 因子英文名 → 中文名 + 分类映射
FEATURE_LABELS = {
    # 质量因子
    "roe": "ROE(净资产收益率)", "roe_approx": "ROE近似(100/PE)",
    "gross_margin": "毛利率", "net_margin": "净利率",
    "stability": "收益稳定性(1/月波动)", "daily_sharpe": "日夏普比率",
    "amount_stability": "成交额稳定性",
    # 估值因子
    "pe": "PE(市盈率)", "pb": "PB(市净率)", "ps_approx": "PS近似",
    "pe_percentile": "PE历史分位", "pb_percentile": "PB历史分位",
    # 成长因子
    "ret_5d": "5日收益率", "ret_10d": "10日收益率",
    "ret_20d": "20日收益率", "ret_60d": "60日收益率",
    "ret_120d": "120日收益率", "ret_250d": "250日收益率",
    "momentum_accel": "动量加速度(60d-120d)",
    "amount_growth": "成交额增速", "profit_growth": "利润增速",
    "revenue_growth": "营收增速",
    # 分红因子
    "div_count": "分红次数", "div_stability": "分红稳定性",
    "avg_dividend": "平均分红金额", "div_yield": "股息率",
    "div_yield_approx": "股息率近似", "fcf_proxy": "自由现金流代理",
    # 技术因子
    "above_ma20": "站上MA20", "above_ma60": "站上MA60",
    "above_ma120": "站上MA120",
    "ma20_slope": "MA20斜率(方向)", "ma60_slope": "MA60斜率(方向)",
    "ma120_slope": "MA120斜率(方向)",
    "rps": "相对强弱(RPS)", "volatility_20d": "20日波动率",
    "amount_ratio": "成交额比(5日/20日)", "avg_turnover": "平均换手率",
    "max_drawdown_60d": "60日最大回撤",
    # 风险因子
    "annual_vol": "年化波动率", "downside_vol": "下行波动率",
    "max_drawdown": "最大回撤(250日)", "avg_drawdown": "平均回撤",
    "neg_day_ratio": "负收益天数比", "var_95": "95%VaR",
    "debt_ratio": "资产负债率",
}

FEATURE_CATEGORY = {
    "roe": "质量", "roe_approx": "质量", "gross_margin": "质量",
    "net_margin": "质量", "stability": "质量", "daily_sharpe": "质量",
    "amount_stability": "质量",
    "pe": "估值", "pb": "估值", "ps_approx": "估值",
    "pe_percentile": "估值", "pb_percentile": "估值",
    "ret_5d": "成长", "ret_10d": "成长", "ret_20d": "成长",
    "ret_60d": "成长", "ret_120d": "成长", "ret_250d": "成长",
    "momentum_accel": "成长", "amount_growth": "成长",
    "profit_growth": "成长", "revenue_growth": "成长",
    "div_count": "分红", "div_stability": "分红", "avg_dividend": "分红",
    "div_yield": "分红", "div_yield_approx": "分红", "fcf_proxy": "分红",
    "above_ma20": "技术", "above_ma60": "技术", "above_ma120": "技术",
    "ma20_slope": "技术", "ma60_slope": "技术", "ma120_slope": "技术",
    "rps": "技术", "volatility_20d": "技术", "amount_ratio": "技术",
    "avg_turnover": "技术", "max_drawdown_60d": "技术",
    "annual_vol": "风险", "downside_vol": "风险", "max_drawdown": "风险",
    "avg_drawdown": "风险", "neg_day_ratio": "风险", "var_95": "风险",
    "debt_ratio": "风险",
}


def factor_importance_chart(importance_df):
    """因子重要性柱状图（含中文标签 + 分类着色）"""
    if importance_df is None or importance_df.empty:
        return go.Figure()

    imp = importance_df.head(15).copy()
    imp["label"] = imp["feature"].map(FEATURE_LABELS).fillna(imp["feature"])
    imp["category"] = imp["feature"].map(FEATURE_CATEGORY).fillna("其他")
    imp = imp.sort_values("importance", ascending=True)

    # 分类着色
    cat_colors = {"质量": "#2ecc71", "估值": "#3498db", "成长": "#e74c3c",
                  "分红": "#f39c12", "技术": "#9b59b6", "风险": "#e67e22", "其他": "#95a5a6"}
    colors = [cat_colors.get(c, "#95a5a6") for c in imp["category"]]

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=imp["importance"], y=imp["label"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in imp["importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>重要性: %{x:.2f}<br>分类: %{customdata}<extra></extra>",
        customdata=imp["category"],
    ))
    fig.update_layout(
        title="因子重要性 Top 15（LightGBM Gain）",
        height=500,
        xaxis_title="重要性",
        margin=dict(l=10, r=80, t=40, b=10),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


def kline_chart(df, title="K线图"):
    """Plotly K 线图"""
    if df is None or df.empty:
        return go.Figure()

    df = df.tail(120).copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="K线",
        increasing_line_color="#e74c3c",
        decreasing_line_color="#2ecc71",
    ))

    # 添加均线
    if len(df) >= 20:
        ma20 = df["close"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df["date"], y=ma20,
            mode="lines", name="MA20",
            line=dict(color="#f39c12", width=1),
        ))
    if len(df) >= 60:
        ma60 = df["close"].rolling(60).mean()
        fig.add_trace(go.Scatter(
            x=df["date"], y=ma60,
            mode="lines", name="MA60",
            line=dict(color="#3498db", width=1),
        ))

    # 成交量
    if "volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"],
            name="成交量", yaxis="y2",
            marker_color="rgba(128,128,128,0.3)",
        ))

    fig.update_layout(
        title=title,
        height=500,
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="价格"),
        yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
    )
    return fig


def factor_radar_chart(factor_values, factor_names):
    """个股因子雷达图"""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=factor_values,
        theta=factor_names,
        fill="toself",
        name="因子得分",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        height=350,
    )
    return fig


def cumulative_return_chart(returns_df, benchmark_col=None):
    """累计收益曲线"""
    if returns_df is None or returns_df.empty:
        return go.Figure()

    fig = go.Figure()
    for col in returns_df.columns:
        if benchmark_col and col == benchmark_col:
            fig.add_trace(go.Scatter(
                x=returns_df.index, y=returns_df[col],
                mode="lines", name=col,
                line=dict(dash="dash", color="gray"),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=returns_df.index, y=returns_df[col],
                mode="lines", name=col,
            ))

    fig.update_layout(
        title="累计收益曲线",
        height=400,
        yaxis_title="累计收益率",
        template="plotly_white",
    )
    return fig


def drawdown_chart(drawdown_series):
    """回撤图"""
    if drawdown_series is None or drawdown_series.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index, y=drawdown_series,
        mode="lines", name="回撤",
        fill="tozeroy",
        line=dict(color="#e74c3c"),
    ))
    fig.update_layout(
        title="最大回撤",
        height=300,
        yaxis_title="回撤幅度",
        template="plotly_white",
    )
    return fig


def score_gauge_chart(score, title="综合得分"):
    """得分仪表盘"""
    color = "#2ecc71" if score >= 85 else "#f39c12" if score >= 65 else "#e74c3c"
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 65], "color": "#ffe0cc"},
                {"range": [65, 75], "color": "#ffffcc"},
                {"range": [75, 85], "color": "#ccffcc"},
                {"range": [85, 100], "color": "#aaffaa"},
            ],
        },
    ))
    fig.update_layout(height=250)
    return fig
