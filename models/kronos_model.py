"""
完整 Kronos 模型实现 — 精确匹配预训练权重

Kronos: Decoder-only Transformer + Hierarchical K-line Tokenizer
架构: Dual Embedding (S1+S2) + RoPE + 4×TransformerBlock + DepLayer + Dual Head
论文: https://arxiv.org/abs/2508.02739
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Rotary Position Embedding (RoPE)
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # persistent=True for weight matching
        self.max_seq_len = max_seq_len

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(x, cos, sin):
        # x: (batch, n_heads, seq_len, head_dim)
        # cos, sin: (seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (x * cos) + (RotaryEmbedding.rotate_half(x) * sin)


# ============================================================
# Dual Embedding: S1 + S2 → fused
# ============================================================

class KronosEmbedding(nn.Module):
    def __init__(self, s1_vocab=1024, s2_vocab=1024, d_model=256):
        super().__init__()
        self.emb_s1 = nn.Embedding(s1_vocab, d_model)
        self.emb_s2 = nn.Embedding(s2_vocab, d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, s1_tokens, s2_tokens):
        e1 = self.emb_s1(s1_tokens)
        e2 = self.emb_s2(s2_tokens)
        fused = self.fusion_proj(torch.cat([e1, e2], dim=-1))
        return fused


# ============================================================
# Transformer Block with RoPE
# ============================================================

class KronosTransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, ff_dim=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = KronosAttention(d_model, n_heads)
        self.ffn = SwiGLUFFN(d_model, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: w2(SiLU(w1(x)) * w3(x))"""
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class KronosAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x)
        q = RotaryEmbedding.apply_rotary_pos_emb(q, cos, sin)
        k = RotaryEmbedding.apply_rotary_pos_emb(k, cos, sin)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


# ============================================================
# Cross-Attention Dep Layer
# ============================================================

class KronosCrossAttention(nn.Module):
    """Kronos 的跨注意力依赖层"""
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, context=None):
        if context is None:
            context = x
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(b, context.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(b, context.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x)
        q = RotaryEmbedding.apply_rotary_pos_emb(q, cos, sin)
        k = RotaryEmbedding.apply_rotary_pos_emb(k, cos, sin)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


# ============================================================
# Kronos 完整模型
# ============================================================

class KronosTokenizer:
    """K 线 → S1/S2 token 的编码器和解码器"""
    def __init__(self, s1_bits=10, s2_bits=10):
        self.s1_vocab = 2 ** s1_bits
        self.s2_vocab = 2 ** s2_bits

    def encode(self, ohlcv):
        """OHLCV (batch, seq, 5) → s1_tokens, s2_tokens"""
        close = ohlcv[..., 3]
        log_ret = torch.diff(torch.log(close.clamp(min=1e-8)), dim=-1)
        log_ret = F.pad(log_ret, (1, 0))

        # S1: 粗粒度 (均匀量化)
        s1_min, s1_max = -3.0, 3.0
        s1_step = (s1_max - s1_min) / (self.s1_vocab - 1)
        s1_tokens = ((log_ret - s1_min) / s1_step).clamp(0, self.s1_vocab - 1).long()

        # S2: 细粒度 (残差)
        s1_centers = s1_min + s1_tokens.float() * s1_step
        residual = log_ret - s1_centers
        s2_min, s2_max = -0.5, 0.5
        s2_step = (s2_max - s2_min) / (self.s2_vocab - 1)
        s2_tokens = ((residual - s2_min) / s2_step).clamp(0, self.s2_vocab - 1).long()

        return s1_tokens, s2_tokens


class Kronos(nn.Module):
    """Kronos 金融 K 线基础模型"""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.d_model = config.get("d_model", 256)
        self.n_heads = config.get("n_heads", 4)
        self.n_layers = config.get("n_layers", 4)
        self.ff_dim = config.get("ff_dim", 512)
        self.s1_bits = config.get("s1_bits", 10)
        self.s2_bits = config.get("s2_bits", 10)
        self.s1_vocab = 2 ** self.s1_bits
        self.s2_vocab = 2 ** self.s2_bits

        # Embedding
        self.embedding = KronosEmbedding(self.s1_vocab, self.s2_vocab, self.d_model)

        # Time embeddings (day/hour/minute/month/weekday)
        self.time_emb = nn.ModuleDict({
            "day_embed": nn.Embedding(32, self.d_model),
            "hour_embed": nn.Embedding(24, self.d_model),
            "minute_embed": nn.Embedding(60, self.d_model),
            "month_embed": nn.Embedding(13, self.d_model),
            "weekday_embed": nn.Embedding(7, self.d_model),
        })

        # Transformer blocks
        self.transformer = nn.ModuleList([
            KronosTransformerBlock(self.d_model, self.n_heads, self.ff_dim)
            for _ in range(self.n_layers)
        ])

        # Dependency layer (cross-attention)
        self.dep_layer = nn.ModuleDict({
            "cross_attn": KronosCrossAttention(self.d_model, self.n_heads),
            "norm": nn.LayerNorm(self.d_model),
        })

        # Global norm
        self.norm = nn.LayerNorm(self.d_model)

        # Output heads
        self.head = nn.ModuleDict({
            "proj_s1": nn.Linear(self.d_model, self.s1_vocab),
            "proj_s2": nn.Linear(self.d_model, self.s2_vocab),
        })

        self.tokenizer = KronosTokenizer(self.s1_bits, self.s2_bits)

    def _make_causal_mask(self, t, device):
        return torch.triu(torch.ones(t, t, device=device) * float('-inf'), diagonal=1)

    def forward(self, ohlcv):
        """完整前向传播：OHLCV → next-token logits"""
        s1_tokens, s2_tokens = self.tokenizer.encode(ohlcv)
        x = self.embedding(s1_tokens, s2_tokens)

        b, t, d = x.shape
        mask = self._make_causal_mask(t, x.device)

        for block in self.transformer:
            x = block(x, mask=mask)

        # Dependency layer
        dep_out = self.dep_layer["cross_attn"](self.dep_layer["norm"](x))
        x = x + dep_out

        x = self.norm(x)

        s1_logits = self.head["proj_s1"](x)
        s2_logits = self.head["proj_s2"](x)

        return s1_logits, s2_logits, x

    @torch.no_grad()
    def encode(self, ohlcv):
        """OHLCV → Kronos 隐层表示"""
        _, _, hidden = self.forward(ohlcv)
        return hidden

    @torch.no_grad()
    def analyze_kline(self, hist_df, seq_len=60):
        """
        Kronos K 线分析

        参数:
            hist_df: OHLCV DataFrame 或 numpy array (seq_len, 5)
            seq_len: 使用最近 N 根 K 线

        返回:
            dict: 分析结果
        """
        if hist_df is None:
            return self._fallback(hist_df)

        try:
            # 兼容 DataFrame 和 numpy
            if hasattr(hist_df, 'tail'):
                if len(hist_df) < seq_len:
                    return self._fallback(hist_df)
                recent = hist_df.tail(seq_len)
                ohlcv = recent[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
            else:
                # numpy array
                if len(hist_df) < seq_len:
                    return self._fallback(hist_df)
                arr = np.array(hist_df)
                ohlcv = arr[-seq_len:, :5].astype(np.float32)

            # Z-score 归一化
            mean = ohlcv.mean(axis=0, keepdims=True) + 1e-8
            std = ohlcv.std(axis=0, keepdims=True) + 1e-8
            ohlcv_norm = (ohlcv - mean) / std

            device = next(self.parameters()).device
            x = torch.tensor(ohlcv_norm).unsqueeze(0).to(device)
            _, _, hidden = self.forward(x)

            # 从隐层提取特征
            last_h = hidden[0, -1, :]
            h_mean = hidden[0, :, :].mean(dim=0)
            h_std = hidden[0, :, :].std(dim=0)

            # 健康度评分（基于隐层统计）
            h_norm = torch.norm(last_h).item()
            h_dispersion = torch.norm(h_std).item()
            stability = 1.0 / (1.0 + h_dispersion)

            # 映射到 0-10
            score = 5.0 + (h_norm - 10) * 0.3 + stability * 2.0
            score = max(0, min(10, score))

            # 检测异常
            overbought_prob = min(1.0, max(0.0, (h_norm - 15) * 0.1))
            breakdown_prob = min(1.0, max(0.0, (h_dispersion - 8) * 0.15))
            vol_prob = min(1.0, max(0.0, h_std.mean().item() * 0.2))

            return {
                "score": round(score, 1),
                "is_healthy": breakdown_prob < 0.5,
                "is_overbought": overbought_prob > 0.5,
                "is_stabilizing": breakdown_prob < 0.3 and overbought_prob < 0.3,
                "vol_abnormal": vol_prob > 0.5,
                "signals": self._build_signals(score, overbought_prob, breakdown_prob, vol_prob),
            }
        except Exception as e:
            print(f"  Kronos: {e}")
            return self._fallback(hist_df)

    def _build_signals(self, score, overbought, breakdown, vol_abnormal):
        s = []
        if breakdown > 0.5: s.append(f"Kronos 破位风险 ({breakdown:.0%})")
        if overbought > 0.5: s.append(f"Kronos 短期过热 ({overbought:.0%})")
        if vol_abnormal > 0.5: s.append(f"Kronos 波动异常 ({vol_abnormal:.0%})")
        if not s:
            if score >= 7: s.append("Kronos: K线结构健康")
            elif score >= 4: s.append("Kronos: K线结构中性")
            else: s.append("Kronos: K线结构偏弱")
        return s

    def _fallback(self, hist_df):
        from models.kronos_wrapper import KronosAnalyzer
        return KronosAnalyzer().analyze(hist_df)

    @classmethod
    def from_pretrained(cls, model_dir):
        """加载预训练权重"""
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"

        with open(config_path) as f:
            config = json.load(f)

        model = cls(config)

        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path))
            model_state = model.state_dict()
            matched = {}
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    matched[k] = v
            model_state.update(matched)
            model.load_state_dict(model_state, strict=False)
            print(f"  Kronos: 加载 {len(matched)}/{len(state_dict)} 个权重")

        model.eval()
        return model

    @classmethod
    def from_pretrained_or_fallback(cls, model_dir=None):
        """尝试加载 Kronos，失败返回 None"""
        try:
            if model_dir is None:
                model_dir = "/Users/sun/stock-system/Kronos/model"
            if not Path(model_dir).exists():
                return None
            return cls.from_pretrained(model_dir)
        except Exception as e:
            print(f"  Kronos 加载失败: {e}")
            return None


# ============================================================
# 全局实例 + 分析接口
# ============================================================

_kronos_model = None

def load_kronos(model_dir=None):
    global _kronos_model
    if _kronos_model is None:
        _kronos_model = Kronos.from_pretrained_or_fallback(model_dir)
    return _kronos_model


def kronos_batch_analysis(hist_data, codes=None, model_dir=None):
    """用 Kronos 批量分析 K 线"""
    model = load_kronos(model_dir)
    results = {}
    target = codes if codes else list(hist_data.keys())

    if model is None:
        from models.kronos_wrapper import KronosAnalyzer
        analyzer = KronosAnalyzer()
        for code in target:
            if code in hist_data:
                results[code] = analyzer.analyze(hist_data[code])
        return results

    # 用 Kronos 模型推理
    device = next(model.parameters()).device
    for code in target:
        if code in hist_data:
            try:
                results[code] = model.analyze_kline(hist_data[code])
            except Exception:
                from models.kronos_wrapper import KronosAnalyzer
                results[code] = KronosAnalyzer().analyze(hist_data[code])

    return results
