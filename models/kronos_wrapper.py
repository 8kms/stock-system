"""
Kronos K 线基础模型 — PyTorch 实现

基于 Kronos 论文架构的精简版 Transformer K 线编码器：
- 输入: 60 根日 K 线 (open/high/low/close/volume)
- 输出: K 线健康度评分 (0-10)、波动异常概率、过热概率、破位概率
- 用 MPS (Apple Silicon GPU) 或 CPU 推理

完整 Kronos 模型安装:
  git clone https://github.com/shiyu-coder/Kronos.git
  cd Kronos && pip install -r requirements.txt
"""
import numpy as np
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DATA_CACHE


# ============================================================
# Kronos 风格 Transformer K 线编码器
# ============================================================

class KlineEncoder(nn.Module):
    """
    Kronos 风格 Transformer Encoder for K-line sequences.

    结构:
      Input (60, 5) → Linear Projection → Positional Encoding
      → 2× TransformerEncoderLayer → Global Pooling
      → MLP Head → [health_score, overbought_prob, breakdown_prob, vol_abnormal_prob]
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(5, d_model)  # OHLCV → d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len=120, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 4),  # 4 outputs
        )

    def forward(self, x):
        # x: (batch, seq_len, 5) — OHLCV
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)  # (batch, seq, d_model)
        x = x.transpose(1, 2)   # (batch, d_model, seq)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        x = self.head(x)  # (batch, 4)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# Kronos 风格 K 线分析器
# ============================================================

class KronosAnalyzer:
    """
    Kronos K 线分析器

    - 支持加载预训练权重
    - 无预训练权重时使用启发式规则（基于 Kronos 设计原则）
    - 用 MPS/CPU 推理
    """

    def __init__(self, model_path=None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = KlineEncoder().to(self.device)
        self.model.eval()

        self._pretrained = False
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self._pretrained = True

    def analyze(self, hist_df, seq_len=60):
        """
        分析单只股票的 K 线

        参数:
            hist_df: DataFrame with columns [open, high, low, close, volume]
            seq_len: 使用的 K 线根数

        返回:
            dict with:
                - is_healthy: K线是否健康
                - is_overbought: 是否短期过热
                - is_stabilizing: 是否回踩企稳
                - vol_abnormal: 波动是否异常
                - score: 综合 K 线健康分 (0-10)
                - signals: 信号文字列表
        """
        if hist_df is None or hist_df.empty or len(hist_df) < seq_len:
            return _default_result("数据不足")

        # 提取最近 seq_len 根 K 线
        recent = hist_df.tail(seq_len)
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(c in recent.columns for c in required_cols):
            return _default_result("缺少K线字段")

        # 如果加载了预训练模型，用模型推理
        if self._pretrained:
            return self._model_inference(recent)
        else:
            return self._heuristic_analysis(recent)

    def _model_inference(self, recent_df):
        """用 Transformer 模型推理"""
        ohlcv = recent_df[["open", "high", "low", "close", "volume"]].values

        # 归一化
        mean = ohlcv.mean(axis=0, keepdims=True) + 1e-8
        std = ohlcv.std(axis=0, keepdims=True) + 1e-8
        normalized = (ohlcv - mean) / std

        # 推理
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x).squeeze(0).cpu().numpy()
            # outputs: [health_raw, overbought_prob, breakdown_prob, vol_abnormal_prob]

        health_raw = float(outputs[0])
        overbought_prob = float(torch.sigmoid(torch.tensor(outputs[1])).item())
        breakdown_prob = float(torch.sigmoid(torch.tensor(outputs[2])).item())
        vol_abnormal_prob = float(torch.sigmoid(torch.tensor(outputs[3])).item())

        # 映射到 0-10
        score = max(0, min(10, (health_raw + 2) * 2.5))

        signals = []
        if overbought_prob > 0.6:
            signals.append(f"Kronos 检测短期过热 (概率 {overbought_prob:.0%})")
        if breakdown_prob > 0.6:
            signals.append(f"Kronos 检测破位风险 (概率 {breakdown_prob:.0%})")
        if vol_abnormal_prob > 0.6:
            signals.append(f"Kronos 检测波动异常 (概率 {vol_abnormal_prob:.0%})")
        if not signals:
            signals.append("Kronos: K线结构正常")

        return {
            "is_healthy": breakdown_prob < 0.5,
            "is_overbought": overbought_prob > 0.5,
            "is_stabilizing": breakdown_prob < 0.3 and overbought_prob < 0.3,
            "vol_abnormal": vol_abnormal_prob > 0.5,
            "score": round(score, 1),
            "signals": signals,
        }

    def _heuristic_analysis(self, recent_df):
        """
        启发式 K 线分析（Kronos 设计原则的规则化实现）
        基于 Kronos 论文的三个核心维度：
        1. 趋势完整性（均线排列 + 价格位置）
        2. 波动异常度（波动率锥 + 跳空检测）
        3. 过热/过冷（RSI + 布林带极端位置）
        """
        close = recent_df["close"].values
        high = recent_df["high"].values
        low = recent_df["low"].values
        volume = recent_df["volume"].values

        signals = []
        score = 7.0
        is_healthy = True
        is_overbought = False
        is_stabilizing = False
        vol_abnormal = False

        # ---- 1. 趋势完整性 ----
        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20

        # 均线多头排列
        if ma5 > ma10 > ma20 > ma60:
            score += 1.0
        elif close[-1] < ma60:
            is_healthy = False
            score -= 2.5
            signals.append("跌破 60 日均线，趋势走坏")

        # 价格相对均线位置
        if close[-1] < ma20 and close[-3] < ma20:
            score -= 1.0
            signals.append("连续在 20 日均线下方")

        # ---- 2. 波动异常度 ----
        rets = np.diff(close) / close[:-1]
        vol_20 = np.std(rets[-20:]) if len(rets) >= 20 else np.std(rets)
        vol_60 = np.std(rets[-60:]) if len(rets) >= 60 else vol_20

        if vol_20 > vol_60 * 2.0:
            vol_abnormal = True
            score -= 2.0
            signals.append(f"波动率异常放大 (20日波动={vol_20:.2%}, 60日={vol_60:.2%})")
        elif vol_20 < vol_60 * 0.5:
            signals.append("波动率收缩，可能蓄势")

        # 跳空检测
        gap_ups = 0
        for i in range(-20, -1):
            if low[i] > high[i - 1]:
                gap_ups += 1
        if gap_ups >= 3:
            is_overbought = True
            score -= 1.5
            signals.append(f"近 20 日出现 {gap_ups} 次跳空高开，短期过热")

        # ---- 3. 过热/过冷 ----
        # RSI 近似
        gains = rets[-14:] * (rets[-14:] > 0)
        losses = -rets[-14:] * (rets[-14:] < 0)
        avg_gain = np.mean(gains) if np.any(gains > 0) else 0
        avg_loss = np.mean(losses) if np.any(losses > 0) else 1e-8
        rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100

        if rsi > 80:
            is_overbought = True
            score -= 2.0
            signals.append(f"RSI 超买 ({rsi:.0f})，短期过热")
        elif rsi < 20:
            score += 1.0
            signals.append(f"RSI 超卖 ({rsi:.0f})，可能超跌反弹")
        elif 60 <= rsi <= 75:
            signals.append(f"RSI 偏强 ({rsi:.0f})")

        # 布林带位置
        bb_mid = np.mean(close[-20:])
        bb_std = np.std(close[-20:])
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        if close[-1] > bb_upper:
            is_overbought = True
            score -= 1.5
            signals.append("突破布林带上轨，短期过热")
        elif close[-1] < bb_lower:
            score += 0.5
            signals.append("跌破布林带下轨，关注反弹")

        # ---- 4. 企稳检测 ----
        # 回踩短期均线后反弹
        near_ma20 = abs(close[-3] - ma20) / ma20 < 0.02
        bounced = close[-1] > close[-3] * 1.01
        if near_ma20 and bounced:
            is_stabilizing = True
            score += 1.5
            signals.append("回踩 20 日线后反弹企稳")

        # ---- 5. 量价配合 ----
        vol_5_avg = np.mean(volume[-5:])
        vol_20_avg = np.mean(volume[-20:])
        price_up = close[-1] > close[-5]

        if price_up and vol_5_avg > vol_20_avg * 1.1:
            score += 1.0
            signals.append("价涨量增，配合良好")
        elif not price_up and vol_5_avg > vol_20_avg * 1.3:
            score -= 1.5
            signals.append("放量下跌，警惕出货")

        # ---- 综合 ----
        if not signals:
            signals.append("K 线结构正常")

        score = max(0, min(10, score))

        return {
            "is_healthy": is_healthy and score > 3,
            "is_overbought": is_overbought,
            "is_stabilizing": is_stabilizing,
            "vol_abnormal": vol_abnormal,
            "score": round(score, 1),
            "signals": signals,
        }


def _default_result(reason="数据不足"):
    return {
        "is_healthy": True,
        "is_overbought": False,
        "is_stabilizing": False,
        "vol_abnormal": False,
        "score": 5.0,
        "signals": [reason],
    }


# ============================================================
# 批量分析接口
# ============================================================

def create_analyzer(model_path=None):
    """创建 Kronos 分析器实例"""
    return KronosAnalyzer(model_path=model_path)


def batch_kline_analysis(hist_data, codes=None, model_path=None):
    """批量 K 线分析（Kronos 模型）"""
    analyzer = KronosAnalyzer(model_path=model_path)
    results = {}
    target = codes if codes else list(hist_data.keys())
    for code in target:
        if code in hist_data:
            results[code] = analyzer.analyze(hist_data[code])
    return results
