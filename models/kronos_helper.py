"""
Kronos K 线辅助模块（可选）
默认用技术指标替代；安装 Kronos 后可切换为真实模型
"""
import numpy as np
import pandas as pd


def analyze_kline(hist_df, use_kronos=False):
    """
    分析单只股票的 K 线节奏
    返回:
        - is_healthy: K 线是否走坏
        - is_overbought: 是否短期过热
        - is_stabilizing: 是否回踩企稳
        - vol_abnormal: 波动是否异常放大
        - score: K 线健康分 (0-10)
    """
    if hist_df is None or hist_df.empty or len(hist_df) < 60:
        return _default_result()

    close = hist_df["close"].values
    volume = hist_df["volume"].values if "volume" in hist_df.columns else np.ones_like(close)

    result = {
        "is_healthy": True,
        "is_overbought": False,
        "is_stabilizing": False,
        "vol_abnormal": False,
        "score": 7.0,
        "signals": [],
    }

    # 1. 是否走坏：价格连续跌破短、中、长期均线
    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
    ma120 = np.mean(close[-120:]) if len(close) >= 120 else ma60

    below_all = close[-1] < ma20 and close[-1] < ma60 and close[-1] < ma120
    below_ma60 = close[-1] < ma60

    if below_all:
        result["is_healthy"] = False
        result["score"] -= 4
        result["signals"].append("跌破所有主要均线")
    elif below_ma60:
        result["is_healthy"] = False
        result["score"] -= 2
        result["signals"].append("跌破 60 日均线")

    # 2. 是否过热：短期涨幅过大
    if len(close) >= 20:
        ret_5d = close[-1] / close[-5] - 1
        ret_20d = close[-1] / close[-20] - 1
        if ret_5d > 0.15 or ret_20d > 0.35:
            result["is_overbought"] = True
            result["score"] -= 3
            result["signals"].append(f"短期过热 (5日涨{ret_5d:.1%}, 20日涨{ret_20d:.1%})")

    # 3. 是否企稳：回踩均线后反弹
    if len(close) >= 30:
        recent_low = np.min(close[-10:])
        ma20_val = ma20
        near_ma = abs(recent_low - ma20_val) / ma20_val < 0.03
        bounced = close[-1] > close[-3] and close[-2] > close[-3]
        if near_ma and bounced:
            result["is_stabilizing"] = True
            result["score"] += 2
            result["signals"].append("回踩 20 日线企稳反弹")

    # 4. 波动异常
    if len(close) >= 60:
        vol_20 = np.std(close[-20:] / np.mean(close[-20:]))
        vol_60 = np.std(close[-60:] / np.mean(close[-60:]))
        if vol_20 > vol_60 * 2:
            result["vol_abnormal"] = True
            result["score"] -= 2
            result["signals"].append("波动率异常放大")

    # 5. 量价配合
    if len(close) >= 20 and len(volume) >= 20:
        price_up = close[-1] > close[-20]
        vol_up = np.mean(volume[-5:]) > np.mean(volume[-20:])
        if price_up and vol_up:
            result["score"] += 1
            result["signals"].append("量价配合良好")
        elif not price_up and vol_up:
            result["score"] -= 1
            result["signals"].append("放量下跌，注意风险")

    result["score"] = max(0, min(10, result["score"]))
    return result


def batch_kline_analysis(hist_data, codes=None):
    """批量 K 线分析"""
    results = {}
    target = codes if codes else list(hist_data.keys())
    for code in target:
        if code in hist_data:
            results[code] = analyze_kline(hist_data[code])
    return results


def _default_result():
    return {
        "is_healthy": True,
        "is_overbought": False,
        "is_stabilizing": False,
        "vol_abnormal": False,
        "score": 5.0,
        "signals": ["数据不足"],
    }
