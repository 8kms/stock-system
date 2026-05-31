"""
风险因子：负债率、商誉、应收账款、波动率、最大回撤
"""
import numpy as np
import pandas as pd

from config import FILTERS


def calc_risk_factors(hist_data, financial_data=None, valuation_data=None):
    rows = []

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < 250:
            continue

        try:
            row = {"code": code}
            close = df["close"]

            # 波动率（年化）
            daily_ret = close.pct_change().tail(250).dropna()
            if len(daily_ret) > 0:
                row["annual_vol"] = daily_ret.std() * np.sqrt(252)
                row["downside_vol"] = daily_ret[daily_ret < 0].std() * np.sqrt(252)
            else:
                row["annual_vol"] = 0
                row["downside_vol"] = 0

            # 最大回撤（250 日）
            if len(close) >= 250:
                c250 = close.tail(250)
                rolling_max = c250.cummax()
                drawdown = (c250 - rolling_max) / rolling_max
                row["max_drawdown"] = drawdown.min()
                row["avg_drawdown"] = drawdown.mean()
            else:
                c_all = close
                rolling_max = c_all.cummax()
                drawdown = (c_all - rolling_max) / rolling_max
                row["max_drawdown"] = drawdown.min()
                row["avg_drawdown"] = drawdown.mean()

            # 负收益天数占比
            row["neg_day_ratio"] = (daily_ret < 0).mean()

            # 极端负收益（VaR 近似）
            if len(daily_ret) >= 50:
                row["var_95"] = daily_ret.quantile(0.05)
            else:
                row["var_95"] = 0

            # 从财务数据补充
            if financial_data and code in financial_data:
                fin = financial_data[code]
                fin_cols = {c.lower(): c for c in fin.columns}

                debt_col = _find_col(fin_cols, ["资产负债率", "负债率", "debtratio"])
                if debt_col:
                    val = pd.to_numeric(fin[debt_col].iloc[0], errors="coerce")
                    row["debt_ratio"] = val if not pd.isna(val) else 50
                else:
                    row["debt_ratio"] = 50
            else:
                row["debt_ratio"] = 50

            rows.append(row)
        except Exception:
            continue

    result = pd.DataFrame(rows)
    return result


def _find_col(col_map, candidates):
    for c in candidates:
        for k in col_map:
            if c in k:
                return col_map[k]
    return None
