"""
成长因子：营收增速、扣非利润增速、现金流增速
"""
import numpy as np
import pandas as pd


def calc_growth_factors(hist_data, financial_data=None):
    rows = []

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < 250:
            continue

        try:
            row = {"code": code}

            recent = df.tail(250)
            close = recent["close"]

            # 价格动量（营收增速的代理）
            row["ret_60d"] = close.iloc[-1] / close.iloc[-min(60, len(close))] - 1
            row["ret_120d"] = close.iloc[-1] / close.iloc[-min(120, len(close))] - 1

            # 收益加速度 = 近 60 日 - 近 120 日
            row["momentum_accel"] = row["ret_60d"] - row["ret_120d"]

            # 成交额增速（代表市场关注度增长）
            if "amount" in recent.columns:
                amt = recent["amount"]
                amt_60 = amt.tail(60).mean()
                amt_120 = amt.head(min(120, len(amt) - 60)).mean()
                row["amount_growth"] = (amt_60 / (amt_120 + 1)) - 1 if amt_120 > 0 else 0
            else:
                row["amount_growth"] = 0

            # 利润增速代理：从财务数据
            if financial_data and code in financial_data:
                fin = financial_data[code]
                fin_cols = {c.lower(): c for c in fin.columns}
                profit_col = _find_col(fin_cols, ["净利润增长率", "归属净利润同比增长", "profitgrowth"])
                if profit_col:
                    val = pd.to_numeric(fin[profit_col].iloc[0], errors="coerce")
                    row["profit_growth"] = val if not pd.isna(val) else 0
                else:
                    row["profit_growth"] = 0

                rev_col = _find_col(fin_cols, ["营业收入增长率", "营收同比增长", "revenuegrowth"])
                if rev_col:
                    val = pd.to_numeric(fin[rev_col].iloc[0], errors="coerce")
                    row["revenue_growth"] = val if not pd.isna(val) else 0
                else:
                    row["revenue_growth"] = 0
            else:
                row["profit_growth"] = 0
                row["revenue_growth"] = 0

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
