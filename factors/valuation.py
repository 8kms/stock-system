"""
估值因子：PE、PB、PS、股息率、PE/PB 历史分位
"""
import numpy as np
import pandas as pd


def calc_valuation_factors(hist_data, valuation_data=None):
    rows = []

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < 250:
            continue

        try:
            row = {"code": code}

            recent = df.tail(250)

            # PE/PB 从 valuation_data 获取
            if valuation_data is not None:
                vrow = valuation_data[valuation_data["code"] == code]
                if not vrow.empty:
                    row["pe"] = vrow["pe"].iloc[0] if "pe" in vrow.columns else np.nan
                    row["pb"] = vrow["pb"].iloc[0] if "pb" in vrow.columns else np.nan
                else:
                    row["pe"] = np.nan
                    row["pb"] = np.nan
            else:
                row["pe"] = np.nan
                row["pb"] = np.nan

            # PE 历史分位（从日线 close 估算：最近 PE 在历史区间的位次）
            if "close" in recent.columns:
                close = recent["close"]
                # 以价格的相对位置近似估值分位
                row["pe_percentile"] = (
                    (close.iloc[-1] - close.min())
                    / (close.max() - close.min() + 0.01)
                )
                # PB 分位同样近似
                row["pb_percentile"] = row["pe_percentile"]

                # PS 近似：市值 / 价格变化隐含
                vol_ratio = recent.tail(60)["close"].mean() / recent["close"].mean()
                row["ps_approx"] = 1 / (vol_ratio + 0.01)

            # 股息率估算（从涨跌幅反推，粗略）
            if "close" in recent.columns and len(recent) >= 250:
                row["div_yield_approx"] = max(0, recent["close"].iloc[-1] / recent["close"].iloc[-250] - 1) * 0.3

            rows.append(row)
        except Exception:
            continue

    result = pd.DataFrame(rows)
    return result
