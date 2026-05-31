"""
分红现金流因子：股息率、派息率、分红稳定性、自由现金流
"""
import numpy as np
import pandas as pd


def calc_dividend_factors(hist_data, dividend_data=None, valuation_data=None):
    rows = []

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < 250:
            continue

        try:
            row = {"code": code}

            recent = df.tail(250)
            close = recent["close"]

            # 从分红数据计算
            if dividend_data and code in dividend_data:
                div_df = dividend_data[code]

                # 分红次数
                if "年度" in div_df.columns or "报告期" in div_df.columns:
                    date_col = "年度" if "年度" in div_df.columns else "报告期"
                    recent_divs = div_df[div_df[date_col] >= "2020"]
                    row["div_count"] = len(recent_divs)
                    row["div_stability"] = min(row["div_count"] / 5, 1.0)  # 5年分红频率
                else:
                    row["div_count"] = 0
                    row["div_stability"] = 0

                # 派息金额
                amount_cols = [c for c in div_df.columns if "派息" in c or "分红" in c or "税前" in c]
                if amount_cols:
                    avg_div = pd.to_numeric(div_df[amount_cols[0]], errors="coerce").mean()
                    row["avg_dividend"] = avg_div if not pd.isna(avg_div) else 0
                else:
                    row["avg_dividend"] = 0
            else:
                row["div_count"] = 0
                row["div_stability"] = 0
                row["avg_dividend"] = 0

            # 股息率估算
            if valuation_data is not None:
                vrow = valuation_data[valuation_data["code"] == code]
                # 部分版本 AKShare 有股息率字段
            if close.iloc[-1] > 0:
                row["div_yield"] = row["avg_dividend"] / close.iloc[-1] if row["avg_dividend"] > 0 else 0
            else:
                row["div_yield"] = 0

            # 自由现金流代理：用成交额 / 市值波动近似
            if "amount" in recent.columns:
                row["fcf_proxy"] = recent["amount"].mean() / (close.std() + 1)
            else:
                row["fcf_proxy"] = 0

            rows.append(row)
        except Exception:
            continue

    result = pd.DataFrame(rows)
    return result
