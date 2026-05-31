"""
A 股多因子选股系统 — 全局配置
"""
from pathlib import Path

# === 路径 ===
BASE_DIR = Path(__file__).parent
DATA_CACHE = BASE_DIR / "data_cache"
OUTPUT_DIR = BASE_DIR / "output_files"
DATA_CACHE.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# === 股票池 ===
INDEX_STOCKS = {
    "hs300": "000300",
    "csi500": "000905",
    "csi1000": "000852",
    "cyb": "399006",
    "dividend": "000922",
}

INDEX_NAMES = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "399006": "创业板指",
    "000922": "中证红利",
}

# 指数环境评分的 5 个观察指数
WATCH_INDICES = ["000300", "000905", "000852", "399006", "000922"]

# 自选优质公司池
WATCHLIST_STOCKS = [
    "600519", "600900", "600036", "000333", "000858",
    "601318", "600276", "000651", "002415", "300750",
]

# 剔除条件
FILTERS = {
    "min_market_cap": 5e9,        # 最小市值 50 亿
    "min_avg_volume": 5e7,        # 最小日均成交额 5000 万
    "max_debt_ratio": 80,         # 最大负债率
    "min_list_days": 250,         # 最少上市天数（剔除次新）
    "max_goodwill_ratio": 30,     # 最大商誉占净资产比
}

# === 因子权重（100 分制） ===
FACTOR_WEIGHTS = {
    "quality": 25,
    "valuation": 15,
    "growth": 15,
    "dividend": 15,
    "technical": 10,
    "industry_relative": 10,
    "risk_penalty": -20,
    "human_override": 10,
}

# === 模型参数 ===
MODEL_PARAMS = {
    "lightgbm": {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 30],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "num_iterations": 200,
        "early_stopping_rounds": 30,
        "verbose": -1,
        "seed": 42,
    },
    "xgboost": {
        "objective": "rank:ndcg",
        "eval_metric": ["ndcg@5", "ndcg@10"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "early_stopping_rounds": 30,
        "verbosity": 0,
        "random_state": 42,
    },
}

# 目标变量：未来 60 个交易日相对行业超额收益
TARGET_FORWARD_DAYS = 60

# === 指数评分参数 ===
INDEX_SCORE_RULES = {
    "above_ma20": 0.5,
    "above_ma60": 0.5,
    "ma20_up": 0.5,
    "volume_expand": 0.5,
}

# 市场状态阈值
MARKET_STATE = {
    "strong": (8, 10),
    "oscillation": (5, 7),
    "weak": (3, 4),
    "risk": (0, 2),
}

# === 评分标准 ===
SCORE_THRESHOLDS = {
    "focus": 85,
    "watchlist": 75,
    "normal": 65,
    "observe": 50,
}

# === 技术指标参数 ===
TECH_MA_PERIODS = [20, 60, 120]
TECH_LOOKBACK = 250  # 至少需要的历史天数
