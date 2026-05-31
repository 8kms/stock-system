"""
技术趋势因子：均线位置、收益率、相对强弱、成交额变化
"""
import numpy as np
import pandas as pd

from config import TECH_MA_PERIODS


def calc_technical_factors(hist_data):
    rows = []

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < 250:
            continue

        try:
            row = {"code": code}
            close = df["close"]
            amount = df["amount"] if "amount" in df.columns else None
            turnover = df["turnover"] if "turnover" in df.columns else None

            # 均线 & 均线位置
            for ma in TECH_MA_PERIODS:
                ma_val = close.rolling(ma).mean().iloc[-1]
                row[f"above_ma{ma}"] = 1 if close.iloc[-1] > ma_val else 0
                # 均线斜率（方向）
                ma_series = close.rolling(ma).mean().dropna()
                if len(ma_series) >= 10:
                    row[f"ma{ma}_slope"] = (ma_series.iloc[-1] / ma_series.iloc[-10] - 1)
                else:
                    row[f"ma{ma}_slope"] = 0

            # 各周期收益
            for days in [5, 10, 20, 60, 120]:
                if len(close) > days:
                    row[f"ret_{days}d"] = close.iloc[-1] / close.iloc[-days] - 1
                else:
                    row[f"ret_{days}d"] = 0

            # 相对强弱 = 短均 / 长均
            ma20 = close.rolling(20).mean().iloc[-1]
            ma120 = close.rolling(120).mean().iloc[-1]
            row["rps"] = ma20 / ma120 - 1 if ma120 > 0 else 0

            # 波动率（20 日年化）
            ret_20d = close.pct_change().tail(20).dropna()
            row["volatility_20d"] = ret_20d.std() * np.sqrt(252) if len(ret_20d) > 0 else 0

            # 成交额变化
            if amount is not None:
                amt_5 = amount.tail(5).mean()
                amt_20 = amount.tail(20).mean()
                row["amount_ratio"] = amt_5 / amt_20 if amt_20 > 0 else 1
            else:
                row["amount_ratio"] = 1

            # 换手率（流动性）
            if turnover is not None:
                row["avg_turnover"] = turnover.tail(20).mean()
            else:
                row["avg_turnover"] = 0

            # 最大回撤（近 60 日）
            if len(close) >= 60:
                c60 = close.tail(60)
                rolling_max = c60.cummax()
                drawdown = (c60 - rolling_max) / rolling_max
                row["max_drawdown_60d"] = drawdown.min()
            else:
                row["max_drawdown_60d"] = 0

            rows.append(row)
        except Exception:
            continue

    result = pd.DataFrame(rows)
    return result
