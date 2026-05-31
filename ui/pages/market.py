"""
市场环境页
"""
import streamlit as st
import pandas as pd

from ui.components import market_state_card, index_radar_chart


def render(index_detail):
    st.title("市场环境")

    if index_detail is None:
        st.warning("暂无指数数据，请先运行数据拉取。")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.plotly_chart(
            market_state_card(
                index_detail.get("total_score", 5),
                index_detail.get("state", "oscillation"),
                index_detail.get("state_cn", "震荡"),
                index_detail.get("suggestion", ""),
            ),
            use_container_width=True,
        )

    with col2:
        details = index_detail.get("details", [])
        if details:
            st.plotly_chart(index_radar_chart(details), use_container_width=True)

    # 操作建议
    st.markdown("### 操作建议")
    state = index_detail.get("state", "oscillation")
    if state == "strong":
        st.success(index_detail.get("suggestion", ""))
    elif state == "oscillation":
        st.warning(index_detail.get("suggestion", ""))
    elif state == "weak":
        st.error(index_detail.get("suggestion", ""))
    else:
        st.error(index_detail.get("suggestion", ""))

    # 指数明细表
    st.markdown("### 指数评分明细")
    if details:
        detail_df = pd.DataFrame(details)

        # 只显示关键列
        display_cols = [c for c in detail_df.columns if not c.startswith("站上") and not c.startswith("20日") and not c.startswith("成交")]
        display_cols = [c for c in display_cols if "(" not in c] if len(display_cols) > 3 else detail_df.columns
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # 市场广度指标
    if details:
        st.markdown("### 市场广度")
        up_ratio = sum(1 for d in details if d.get("20日线向上", False)) / len(details)

        cols = st.columns(4)
        cols[0].metric("均线上方比例", f"{up_ratio:.0%}")
        cols[1].metric("指数均分", f"{index_detail.get('total_score', 0):.1f}/10")
        cols[2].metric("强势指数", f"{sum(1 for d in details if d.get('得分', 0) >= 1.5)}/{len(details)}")
        cols[3].metric("弱势指数", f"{sum(1 for d in details if d.get('得分', 0) < 1.0)}/{len(details)}")
