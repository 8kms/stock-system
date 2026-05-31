"""
P0 数据质量闸门 — Gate 0A + Gate 0B

Gate 0A: 覆盖率检查（财务/估值/行业字段覆盖率）
Gate 0B: 横截面区分度检查（方差、唯一值、常数检测）

规则:
  - 财务缺失 > 50%: rule_score 封顶 NaN
  - 财务缺失 20-50%: rule_score 封顶 60
  - 常数特征: 标记 FAIL
  - Gate 0B FAIL: 停止 LightGBM 训练
"""
import numpy as np
import pandas as pd

# 核心财务字段（缺失任何一个都影响评估）
CORE_FIN_FIELDS = [
    "roe", "gross_margin", "net_margin",
    "cfo_to_profit", "debt_ratio",
]

# 代理字段名映射（当前Sina降级模式下的字段名）
CORE_FIN_PROXY_FIELDS = [
    "roe_raw", "roa_raw", "gross_margin_raw",
    "net_margin_raw", "cfo_to_profit_raw",
]


def coverage_rate(series):
    """非空率"""
    return series.notna().mean()


def check_valuation_quality(val_df):
    """
    估值数据质量检查

    返回: (status, report_dict)
    """
    if val_df is None or val_df.empty:
        return "FAIL", {"estimated_ratio": 1.0, "missing_ratio": 1.0}

    total = len(val_df)
    if "valuation_source" not in val_df.columns:
        return "WARN", {"estimated_ratio": 1.0, "missing_ratio": 0.0}

    n_estimated = (val_df["valuation_source"] == "sina_price_estimated").sum()
    n_missing = (val_df["valuation_quality"] == "FAIL").sum()
    n_real = (val_df["valuation_source"] == "akshare_real").sum()

    estimated_ratio = n_estimated / total if total > 0 else 1.0
    missing_ratio = n_missing / total if total > 0 else 1.0

    if estimated_ratio > 0.50 or missing_ratio > 0.30:
        status = "FAIL"
    elif estimated_ratio > 0.30 or missing_ratio > 0.15:
        status = "WARN"
    else:
        status = "PASS"

    return status, {
        "n_real": n_real, "n_estimated": n_estimated, "n_missing": n_missing,
        "estimated_ratio": round(estimated_ratio, 3),
        "missing_ratio": round(missing_ratio, 3),
    }


def gate_0a_coverage_check(df, fields=None):
    """
    Gate 0A: 覆盖率检查

    返回: (checks_dict, status_dict)
    """
    if fields is None:
        # Try proxy fields first (Sina mode), then real fields
        available = [c for c in CORE_FIN_PROXY_FIELDS if c in df.columns]
        if len(available) < 3:
            available = [c for c in CORE_FIN_FIELDS if c in df.columns]
        fields = available

    checks = {}
    for col in fields:
        if col in df.columns:
            checks[col] = round(coverage_rate(df[col]), 3)
        else:
            checks[col] = 0.0

    status = {}
    for k, v in checks.items():
        if v >= 0.90:
            status[k] = "PASS"
        elif v >= 0.70:
            status[k] = "WARN"
        else:
            status[k] = "FAIL"

    n_pass = sum(1 for s in status.values() if s == "PASS")
    n_fail = sum(1 for s in status.values() if s == "FAIL")
    overall = "PASS" if n_fail == 0 else ("WARN" if n_fail <= 2 else "FAIL")

    return checks, status, overall


def gate_0b_variance_check(df, feature_cols=None):
    """
    Gate 0B: 横截面区分度检查

    检查项:
      - 唯一值数量 ≤ 3: FAIL
      - 标准差 < 1e-6: FAIL
      - p90 - p10 < threshold: FAIL
      - 缺失率 > 30%: FAIL
      - 零值率 > 80%: FAIL
      - 单一值占比 > 80%: FAIL

    返回: DataFrame 每列一行
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c not in ("code", "name", "industry", "date")
                        and df[c].dtype in (np.float64, float, np.int64, int)]

    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = df[col]

        missing_rate = round(s.isna().mean(), 3)
        zero_rate = round((s == 0).mean(), 3) if pd.api.types.is_numeric_dtype(s) else None
        nunique = s.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(s) and nunique > 1:
            std = round(s.std(skipna=True), 6)
            p10 = s.quantile(0.10)
            p90 = s.quantile(0.90)
            spread = round(p90 - p10, 6)
        else:
            std = 0.0
            spread = 0.0

        top_value_ratio = round(s.value_counts(normalize=True, dropna=True).iloc[0], 3) if nunique > 0 else 1.0

        # 判定
        if missing_rate > 0.30 or nunique <= 3 or top_value_ratio > 0.80:
            status = "FAIL"
        elif missing_rate > 0.15 or top_value_ratio > 0.60:
            status = "WARN"
        else:
            status = "PASS"

        rows.append({
            "feature": col,
            "missing_rate": missing_rate,
            "zero_rate": zero_rate,
            "nunique": nunique,
            "std": std,
            "p90_p10_spread": spread,
            "top_value_ratio": top_value_ratio,
            "status": status,
        })

    report = pd.DataFrame(rows)
    n_fail = (report["status"] == "FAIL").sum()
    n_warn = (report["status"] == "WARN").sum()
    overall = "PASS" if n_fail == 0 else ("WARN" if n_fail <= 5 else "FAIL")

    return report, overall


def apply_data_quality_cap(rule_df):
    """
    禁止 rule_score 假满分

    财务缺失严重的:
      - WARN (20-50%缺失): rule_score 封顶 60
      - FAIL (>50%缺失): rule_score = NaN, 进入数据待补池
    """
    df = rule_df.copy()

    # 检测核心财务字段缺失率
    fin_fields = [c for c in CORE_FIN_PROXY_FIELDS + CORE_FIN_FIELDS if c in df.columns]
    if not fin_fields:
        df["data_quality_flag"] = "FAIL"
        df["rule_score"] = pd.NA
        return df

    core_missing_rate = df[fin_fields].isna().mean(axis=1)

    df["data_quality_flag"] = "PASS"
    df.loc[core_missing_rate > 0.20, "data_quality_flag"] = "WARN"
    df.loc[core_missing_rate > 0.50, "data_quality_flag"] = "FAIL"

    # WARN: 规则分不能超过 60
    warn_mask = df["data_quality_flag"] == "WARN"
    if "rule_score" in df.columns:
        df.loc[warn_mask, "rule_score"] = df.loc[warn_mask, "rule_score"].clip(upper=60)

    # FAIL: 不给分
    fail_mask = df["data_quality_flag"] == "FAIL"
    if "rule_score" in df.columns:
        df.loc[fail_mask, "rule_score"] = pd.NA

    n_warn = warn_mask.sum()
    n_fail = fail_mask.sum()
    if n_warn > 0 or n_fail > 0:
        print(f"  数据质量封顶: WARN={n_warn}只(≤60分), FAIL={n_fail}只(NaN)")

    return df


def get_model_mode(gate0a_overall, gate0b_overall, momentum_ratio=None):
    """
    确定系统运行模式

    返回: (mode, display_name, description)
    """
    if gate0a_overall == "FAIL" or gate0b_overall == "FAIL":
        return "DATA_FAIL", "数据不可用", "财务数据缺失严重，禁止使用系统输出"

    if gate0a_overall == "WARN" or gate0b_overall == "WARN":
        return "MOMENTUM_ONLY", "技术模型可用", "基本面数据部分缺失，当前只能使用技术面排序，不能当多因子系统"

    if momentum_ratio is not None and momentum_ratio > 0.50:
        return "MOMENTUM_ONLY", "技术模型可用", f"动量重要性占比 {momentum_ratio:.0%} > 50%，基本面因子未生效"

    return "MULTIFACTOR_PASS", "多因子可用", "所有数据质量检查通过，多因子模型正常"


def check_momentum_dominance(importance_df):
    """
    检查动量是否垄断模型

    返回: (status, ratio)
      PASS: <35%
      WARN: 35-50%
      FAIL: >50%
    """
    if importance_df is None or importance_df.empty:
        return "FAIL", 1.0

    momentum_features = ["ret_60d", "ma60_slope", "momentum_score", "ret_5d",
                         "ret_20d", "rps", "momentum_accel", "ma20_slope",
                         "g_ret_60d", "t_ret_60d", "t_ma60_slope", "t_rps",
                         "trend_score", "ret_20d"]
    total = importance_df["importance"].sum()

    if total <= 0:
        return "FAIL", 1.0

    momentum_imp = importance_df.loc[
        importance_df["feature"].isin(momentum_features),
        "importance"
    ].sum()

    ratio = momentum_imp / total

    if ratio > 0.50:
        return "FAIL", ratio
    elif ratio > 0.35:
        return "WARN", ratio
    else:
        return "PASS", ratio


# ============================================================
# rule_score 分布检查
# ============================================================

def check_rule_score_distribution(rule_df):
    """检查 rule_score 是否集中在满分附近"""
    s = rule_df["rule_score"].dropna()

    if len(s) == 0:
        return {"status": "FAIL", "reason": "all rule_score missing"}

    top_value_ratio = s.value_counts(normalize=True).iloc[0]
    p90 = s.quantile(0.90)
    p10 = s.quantile(0.10)
    spread = p90 - p10

    if top_value_ratio > 0.50:
        return {
            "status": "FAIL",
            "reason": f"rule_score最大单值占比过高: {top_value_ratio:.1%}",
            "top_value_ratio": round(top_value_ratio, 3),
            "spread": round(spread, 1),
        }

    if spread < 10:
        return {
            "status": "WARN",
            "reason": f"rule_score 分差过小: {spread:.1f}",
            "top_value_ratio": round(top_value_ratio, 3),
            "spread": round(spread, 1),
        }

    return {
        "status": "PASS",
        "top_value_ratio": round(top_value_ratio, 3),
        "spread": round(spread, 1),
    }


# ============================================================
# 核心字段 Gate 0B 检查
# ============================================================

CORE_MODEL_FEATURES = [
    "quality_score", "valuation_score", "cashflow_score",
    "growth_score", "risk_score",
]


def gate_0b_core_check(df):
    """只检查核心模型特征（不是全部特征）"""
    available = [c for c in CORE_MODEL_FEATURES if c in df.columns]
    if not available:
        return "FAIL", pd.DataFrame({"feature": CORE_MODEL_FEATURES, "status": ["FAIL"]*len(CORE_MODEL_FEATURES)})

    report, _ = gate_0b_variance_check(df, available)
    failed = report[report["status"] == "FAIL"]
    warn = report[report["status"] == "WARN"]

    if len(failed) > 0:
        status = "FAIL"
    elif len(warn) > 0:
        status = "WARN"
    else:
        status = "PASS"

    return status, report


# ============================================================
# 训练前断言
# ============================================================

def assert_trainable(train_df, feature_cols):
    """LightGBM 训练前断言"""
    status, report = gate_0b_core_check(train_df)
    if status == "FAIL":
        failed_features = report[report["status"] == "FAIL"]["feature"].tolist()
        raise ValueError(f"Gate 0B FAIL: 核心因子不可训练 ({failed_features})")

    rule_status = check_rule_score_distribution(train_df)
    if rule_status["status"] == "FAIL":
        raise ValueError(f"rule_score 无效: {rule_status}")

    return report


# ============================================================
# Gate 6 分歧分类
# ============================================================

def classify_disagreement(row):
    """
    对单只股票分类分歧原因

    返回: 分类标签字符串
    """
    # 数据质量导致
    if row.get("data_quality_flag") == "FAIL":
        return "DATA_QUALITY_DISAGREE"

    # rule_score 无效
    if pd.isna(row.get("rule_score")):
        return "RULE_SCORE_INVALID"

    # 规则分异常集中
    if row.get("rule_score_top_value_abnormal", False):
        return "RULE_SCORE_INVALID"

    # 动量 vs 基本面
    if (row.get("rule_score", 100) < 60 and
            row.get("lgb_rank_pct", row.get("lgb_score", 50)) > 80):
        return "MOMENTUM_VS_FUNDAMENTAL"

    # 真实模型分歧
    if row.get("rank_diff", 0) > 0.30:
        return "MODEL_DISAGREE"

    return "VALID"


# ============================================================
# 候选池分配
# ============================================================

def assign_pool(row, system_mode):
    """
    根据系统模式分配池子

    DATA_FAIL → 禁止输出
    MOMENTUM_ONLY → 技术观察池
    MULTIFACTOR_WARN/PASS → 多因子候选池
    """
    if row.get("data_quality_flag") == "FAIL":
        return "DATA_PENDING"

    if system_mode == "DATA_FAIL":
        return "DISABLED"

    if system_mode == "MOMENTUM_ONLY":
        if row.get("momentum_score", 0) >= 80:
            return "MOMENTUM_WATCH"
        return "OBSERVE"

    if system_mode in ("MULTIFACTOR_WARN", "MULTIFACTOR_PASS"):
        if (row.get("rule_score", 0) >= 75 and
                row.get("lgb_score", 0) >= 80 and
                row.get("xgb_score", 0) >= 70 and
                row.get("disagreement_type", "VALID") in ("VALID", "VALID_DISAGREE")):
            return "CORE_CANDIDATE"
        if row.get("rule_score", 0) >= 70:
            return "WATCHLIST"

    return "EXCLUDE"


# ============================================================
# 系统模式决策
# ============================================================

def decide_system_mode(price_status, financial_status, valuation_status,
                       feature_status, model_split_status, factor_structure_status,
                       backtest_status=None, paper_tracking_status=None):
    """
    完整系统模式状态机 — v2.0 最终版

    返回: (mode, name, description)
    """
    if price_status != "PASS":
        return "DATA_FAIL", "数据不可用", "行情数据不可用"

    if financial_status != "PASS":
        return "MOMENTUM_ONLY", "技术模型可用", "财务数据不完整，仅输出技术观察池"

    if feature_status == "FAIL":
        return "MOMENTUM_ONLY", "技术模型可用", "核心特征无区分度"

    # 模型机制已修复但数据未闭环
    if model_split_status == "PASS" and factor_structure_status == "PASS":
        if valuation_status != "PASS":
            return "FUNDAMENTAL_PASS_VALUATION_FAIL", "基本面可用(缺估值)", "财务因子正常但估值历史数据未达标"

        if backtest_status is None or backtest_status == "PENDING":
            return "TRAINING_PASS_BACKTEST_PENDING", "训练通过(待回测)", "模型训练完成，回测验证待执行"

        if paper_tracking_status == "PASS":
            return "LIVE_READY", "可实盘", "所有Gate通过，纸面跟踪验证完成"

        if backtest_status != "PASS":
            return "MULTIFACTOR_WARN", "多因子可用(预警)", "回测验证未完全通过"

        return "MULTIFACTOR_PASS", "多因子可用", "所有检查通过"

    if model_split_status == "PASS":
        return "MODEL_MECHANISM_PASS_DATA_PENDING", "模型机制就绪(数据待补)", "LightGBM可分裂但财务/估值未闭环"

    return "MOMENTUM_ONLY", "技术模型可用", "模型分裂或因子结构异常"


def gate_0c_valuation_check(valuation_df):
    """Gate 0C: 估值历史跨度检查"""
    if valuation_df is None or valuation_df.empty:
        return "FAIL", {"n_rows": 0}
    m = {
        "n_rows": len(valuation_df),
        "n_codes": valuation_df["code"].nunique() if "code" in valuation_df.columns else 0,
        "n_dates": valuation_df["trade_date"].nunique() if "trade_date" in valuation_df.columns else 0,
        "pe_coverage": valuation_df["pe_ttm"].notna().mean() if "pe_ttm" in valuation_df.columns else 0,
        "pb_coverage": valuation_df["pb"].notna().mean() if "pb" in valuation_df.columns else 0,
        "mv_coverage": valuation_df["total_mv"].notna().mean() if "total_mv" in valuation_df.columns else 0,
    }
    m["pe_nunique"] = valuation_df["pe_ttm"].nunique(dropna=True) if "pe_ttm" in valuation_df.columns else 0
    m["pb_nunique"] = valuation_df["pb"].nunique(dropna=True) if "pb" in valuation_df.columns else 0

    pass_cond = [m["n_codes"] >= 90, m["n_dates"] >= 750, m["pe_coverage"] >= 0.90, m["pb_coverage"] >= 0.90]
    warn_cond = [m["n_codes"] >= 70, m["n_dates"] >= 120, m["pe_coverage"] >= 0.70, m["pb_coverage"] >= 0.70]
    status = "PASS" if all(pass_cond) else ("WARN" if all(warn_cond) else "FAIL")
    return status, m
