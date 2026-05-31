"""
质量因子：ROE、ROA、毛利率、净利率、经营现金流/净利润
"""
import numpy as np
import pandas as pd


def calc_quality_factors(hist_data, financial_data=None, valuation_data=None):
    """
    从日线+财务数据计算质量因子
    返回 DataFrame: code + quality 子因子
    """
    rows = []

    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < 250:
            continue

        try:
            row = {"code": code}

            # === 从日线推导的质量指标 ===
            if "close" in df.columns and "amount" in df.columns:
                recent = df.tail(250)
                # 累积收益 = 最近 250 日收益率
                row["ret_250d"] = (
                    recent["close"].iloc[-1] / recent["close"].iloc[0] - 1
                )

                # 收益稳定性 = 月收益标准差倒数（越高越稳）
                monthly = recent.set_index("date")["close"].resample("ME").last().dropna()
                if len(monthly) >= 6:
                    monthly_ret = monthly.pct_change().dropna()
                    row["stability"] = 1 / (monthly_ret.std() + 0.01)
                else:
                    row["stability"] = 0

                # 夏普比率近似 = 日均收益 / 日波动
                daily_ret = recent["close"].pct_change().dropna()
                if len(daily_ret) > 0 and daily_ret.std() > 0:
                    row["daily_sharpe"] = daily_ret.mean() / daily_ret.std()
                else:
                    row["daily_sharpe"] = 0

                # 成交额稳定性
                row["amount_stability"] = 1 / (recent["amount"].pct_change().std() + 0.01)

            # === 从财务数据来的质量指标 ===
            if financial_data and code in financial_data:
                fin = financial_data[code]
                fin_cols = {c.lower(): c for c in fin.columns}

                # ROE
                roe_col = _find_col(fin_cols, ["净资产收益率", "加权净资产收益率", "roe"])
                if roe_col:
                    val = pd.to_numeric(fin[roe_col].iloc[0], errors="coerce")
                    row["roe"] = val if not pd.isna(val) else np.nan
                else:
                    row["roe"] = np.nan

                # 毛利率
                gp_col = _find_col(fin_cols, ["销售毛利率", "毛利率", "grossprofitmargin"])
                if gp_col:
                    val = pd.to_numeric(fin[gp_col].iloc[0], errors="coerce")
                    row["gross_margin"] = val if not pd.isna(val) else np.nan
                else:
                    row["gross_margin"] = np.nan

                # 净利率
                np_col = _find_col(fin_cols, ["销售净利率", "净利率", "netprofitmargin"])
                if np_col:
                    val = pd.to_numeric(fin[np_col].iloc[0], errors="coerce")
                    row["net_margin"] = val if not pd.isna(val) else np.nan
                else:
                    row["net_margin"] = np.nan

            # === 从估值数据来的补充 ===
            if valuation_data is not None:
                vrow = valuation_data[valuation_data["code"] == code]
                if not vrow.empty:
                    if "pe" in vrow.columns and row.get("roe") is None:
                        pe = vrow["pe"].iloc[0]
                        if pe and pe > 0:
                            row["roe_approx"] = 100 / pe  # ROE ≈ 100/PE

            rows.append(row)
        except Exception:
            continue

    result = pd.DataFrame(rows)
    return result


def _find_col(col_map, candidates):
    for c in candidates:
        for k, v in col_map.items():
            if c in k:
                return v
    return None
