"""
汇总报告生成
"""
from datetime import datetime

from config import SCORE_THRESHOLDS


def generate_summary(ranking_df, index_detail, kline_results):
    """生成文本汇总报告"""
    date_str = datetime.now().strftime("%Y年%m月%d日")

    lines = []
    lines.append("=" * 50)
    lines.append(f"  A 股多因子选股报告 — {date_str}")
    lines.append("=" * 50)
    lines.append("")

    # 市场环境
    if isinstance(index_detail, dict):
        lines.append("【市场环境】")
        lines.append(f"  综合评分: {index_detail.get('total_score', 'N/A')} / 10")
        lines.append(f"  市场状态: {index_detail.get('state_cn', 'N/A')}")
        lines.append(f"  操作建议: {index_detail.get('suggestion', '')}")
        lines.append("")

    # Top 推荐
    if ranking_df is not None and not ranking_df.empty:
        lines.append("【Top 10 推荐】")
        lines.append(f"  {'排名':<6}{'代码':<10}{'名称':<10}{'得分':<8}{'判定'}")
        lines.append("  " + "-" * 45)

        for i, (_, row) in enumerate(ranking_df.head(10).iterrows()):
            score = row.get("total_score", row.get("model_score", 50))
            if score >= SCORE_THRESHOLDS["focus"]:
                v = "★ 重点研究"
            elif score >= SCORE_THRESHOLDS["watchlist"]:
                v = "● 进入观察"
            else:
                v = "○ 普通"
            lines.append(
                f"  {i+1:<6}{row.get('code',''):<10}{str(row.get('name',''))[:8]:<10}"
                f"{score:<8.1f}{v}"
            )
        lines.append("")

    # K 线风险提示
    if kline_results:
        unhealthy = [c for c, r in kline_results.items() if not r.get("is_healthy")]
        overbought = [c for c, r in kline_results.items() if r.get("is_overbought")]
        if unhealthy:
            lines.append(f"⚠ K 线走坏 ({len(unhealthy)} 只): {', '.join(unhealthy[:10])}")
        if overbought:
            lines.append(f"⚠ 短期过热 ({len(overbought)} 只): {', '.join(overbought[:10])}")
        lines.append("")

    lines.append("=" * 50)
    lines.append("免责声明: 本报告仅供辅助决策参考，不构成投资建议。")
    lines.append("=" * 50)

    return "\n".join(lines)


def print_report(ranking_df, index_detail, kline_results):
    """打印报告到控制台"""
    report = generate_summary(ranking_df, index_detail, kline_results)
    print(report)
