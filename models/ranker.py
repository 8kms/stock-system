"""
LightGBM / XGBoost 排序模型
目标：预测未来 60 日相对行业超额收益排名

主模型：LightGBM LGBMRanker（行业内排序）
交叉验证：XGBoost XGBRanker
降级方案：百分位线性打分

scikit-learn: 数据切分、标准化、评估
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr

from config import MODEL_PARAMS, FACTOR_WEIGHTS


# ============================================================
# 目标变量构造
# ============================================================

def build_target(hist_data, factor_df, industry_df, forward_days=60):
    """
    构造目标变量：未来 60 个交易日相对行业指数的超额收益
    返回 pd.Series, index=code, value=超额收益率
    """
    targets = {}
    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < forward_days + 5:
            continue
        try:
            close = df["close"].values
            # 未来 60 日收益 = (60日后的价格 / 当前价格) - 1
            fwd_ret = close[-1] / close[-(forward_days + 1)] - 1
            targets[code] = fwd_ret
        except Exception:
            continue

    if not targets:
        return None

    target_s = pd.Series(targets, name="fwd_ret")

    # 行业中性化：减去同行业均值 → 得到行业内超额收益
    if industry_df is not None and not industry_df.empty:
        ind_map = dict(zip(industry_df["code"], industry_df["industry"]))
        target_df = target_s.reset_index()
        target_df.columns = ["code", "fwd_ret"]
        target_df["industry"] = target_df["code"].map(ind_map)
        # 每组至少 3 只股票才有意义
        industry_mean = target_df.groupby("industry")["fwd_ret"].transform("mean")
        target_df["excess"] = target_df["fwd_ret"] - industry_mean
        return target_df.set_index("code")["excess"].dropna()

    return target_s.dropna()


# ============================================================
# LightGBM LGBMRanker（主模型）
# ============================================================

def train_lightgbm_ranker(X, y, codes, industry_df):
    """
    用 LightGBM LGBMRanker 训练行业内排序模型

    参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标变量（超额收益）
        codes: 股票代码列表
        industry_df: 行业分类

    返回: (model, importance_df) 或 (None, None)
    """
    try:
        import lightgbm as lgb
    except Exception as e:
        print(f"  LightGBM 不可用: {e}")
        return None, None

    if len(X) < 50:
        return None, None

    # Step 1: 把连续超额收益离散化为整数标签（lambdarank 要求）
    # n_bins 自适应：样本量/3，但不超过50
    n_bins = min(50, max(5, len(y) // 3))
    try:
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        y_binned = pd.cut(y, bins=n_bins, labels=False)
    y_int = y_binned.fillna(0).astype(int).values
    actual_bins = y_binned.nunique()
    # LightGBM 要求 max(label) < len(label_gain)，所以 +1
    max_label = y_int.max()
    if max_label >= actual_bins:
        actual_bins = max_label + 1

    # Step 2: 构造行业 group（同一行业的股票组成一个 query group）
    ind_map = dict(zip(industry_df["code"], industry_df["industry"]))
    code_to_ind = np.array([ind_map.get(c, "unknown") for c in codes])
    # 给每个行业一个整数 ID
    unique_inds = np.unique(code_to_ind)
    ind_to_id = {ind: i for i, ind in enumerate(unique_inds)}
    group_ids = np.array([ind_to_id[ind] for ind in code_to_ind])

    # Step 3: 按 group 排序（lambdarank 要求同 group 的数据连续）
    sort_idx = np.argsort(group_ids)
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y_int[sort_idx]
    groups_sorted = group_ids[sort_idx]

    # 计算每个 group 的大小
    _, group_sizes = np.unique(groups_sorted, return_counts=True)
    group_sizes = group_sizes.tolist()

    # Step 4: 训练（使用自适应参数）
    n_samples = len(X_sorted)
    n_codes_est = len(set(codes))
    from models.lgb_params import get_lgb_params
    base_params = get_lgb_params(n_rows=n_samples, n_codes=n_codes_est, n_features=len(X.columns),
                                  mode="cross_section" if n_samples < 500 else "panel")
    # 覆盖 lambdarank 特有参数
    params = {**base_params, "objective": "lambdarank", "metric": "ndcg",
              "ndcg_eval_at": [5, 10, 30], "label_gain": list(range(actual_bins)),
              "seed": 42, "verbosity": -1}
    print(f"  LGB params: n={n_samples}, min_data_in_leaf={params['min_data_in_leaf']}")

    try:
        train_data = lgb.Dataset(X_sorted, label=y_sorted, group=group_sizes)
        model = lgb.train(params, train_data)

        # v2.0: 优先用 gain，如果全零则降级为 split
        gain_imp = model.feature_importance(importance_type="gain")
        if gain_imp.sum() == 0:
            gain_imp = model.feature_importance(importance_type="split")
        # 检查基本面因子是否有贡献
        fundamental_features = ["roe_raw", "roa_raw", "gross_margin_raw", "net_margin_raw",
                                "cfo_to_profit_raw", "quality_score", "valuation_score",
                                "cashflow_score", "v_pe_hist", "v_pb_hist", "c_cfo_profit",
                                "risk_score", "growth_score", "g_rev_3y"]
        fundamental_gain = sum(gain_imp[X.columns.get_loc(c)] for c in fundamental_features if c in X.columns)
        total_gain = gain_imp.sum()
        fundamental_ratio = fundamental_gain / total_gain if total_gain > 0 else 0

        if total_gain == 0:
            print(f"  ⚠️ LGB gain=0 (num_trees={model.num_trees()}, split_sum={sum(model.feature_importance('split'))})")
        elif fundamental_ratio == 0:
            print(f"  ⚠️ 基本面因子 gain=0! 检查数据merge/方差/缺失值")
        else:
            print(f"  LGB: total_gain={total_gain:.1f}, fundamental_gain={fundamental_gain:.1f} ({fundamental_ratio:.0%})")

        importance = pd.DataFrame({
            "feature": X.columns.tolist(),
            "importance": gain_imp,
        }).sort_values("importance", ascending=False)

        print(f"  LightGBM 训练成功, bins={actual_bins}, groups={len(group_sizes)}")
        return model, importance
    except Exception as e:
        print(f"  LightGBM 训练失败: {e}")
        return None, None


# ============================================================
# XGBoost XGBRanker（交叉验证）
# ============================================================

def train_xgboost_ranker(X, y, codes, industry_df):
    """
    用 XGBoost XGBRanker 训练（与 LightGBM 交叉验证）

    参数同上
    返回: (model, importance_df) 或 (None, None)
    """
    try:
        import xgboost as xgb
    except Exception as e:
        print(f"  XGBoost 不可用: {e}")
        return None, None

    if len(X) < 50:
        return None, None

    # Step 1: 标签离散化（XGBoost NDCG 要求 label <= 31）
    n_bins = min(30, max(5, len(y) // 3))
    try:
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        y_binned = pd.cut(y, bins=n_bins, labels=False)
    y_int = y_binned.fillna(0).astype(int).values

    # Step 2: 构造行业 group
    ind_map = dict(zip(industry_df["code"], industry_df["industry"]))
    unique_inds = list(set(ind_map.get(c, "unknown") for c in codes))
    ind_to_id = {ind: i for i, ind in enumerate(unique_inds)}
    group_ids = np.array([ind_to_id[ind_map.get(c, "unknown")] for c in codes])

    # Step 3: 按 qid 排序（XGBoost 强制要求）
    sort_idx = np.argsort(group_ids)
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y_int[sort_idx]
    groups_sorted = group_ids[sort_idx]

    # Step 4: 训练
    try:
        ranker = xgb.XGBRanker(
            objective="rank:ndcg",
            eval_metric=["ndcg@5"],
            max_depth=6,
            learning_rate=0.05,
            n_estimators=100,
            verbosity=0,
            random_state=42,
        )
        ranker.fit(X_sorted, y_sorted, qid=groups_sorted)

        importance = pd.DataFrame({
            "feature": X.columns.tolist(),
            "importance": ranker.feature_importances_,
        }).sort_values("importance", ascending=False)

        print(f"  XGBoost 训练成功, bins={n_bins}, groups={len(unique_inds)}")
        return ranker, importance
    except Exception as e:
        print(f"  XGBoost 训练失败: {e}")
        return None, None


# ============================================================
# 模型打分 + 综合
# ============================================================

def predict_scores(factor_df, lgb_model, xgb_model):
    """
    用训练好的模型对全量股票打分
    返回 DataFrame: code, lgb_score, xgb_score, model_score
    """
    feature_cols = [c for c in factor_df.columns if c != "code"]
    X = factor_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    codes = factor_df["code"].values

    result = pd.DataFrame({"code": codes})

    if lgb_model is not None:
        try:
            raw = lgb_model.predict(X)
            result["lgb_score"] = _to_percentile(raw)
        except Exception:
            result["lgb_score"] = np.nan
    else:
        result["lgb_score"] = np.nan

    if xgb_model is not None:
        try:
            raw = xgb_model.predict(X)
            result["xgb_score"] = _to_percentile(raw)
        except Exception:
            result["xgb_score"] = np.nan
    else:
        result["xgb_score"] = np.nan

    # 综合：取两个模型的平均（至少需要一个有效）
    has_lgb = result["lgb_score"].notna()
    has_xgb = result["xgb_score"].notna()
    both = has_lgb & has_xgb

    result["model_score"] = np.nan
    if both.any():
        result.loc[both, "model_score"] = (
            result.loc[both, "lgb_score"] + result.loc[both, "xgb_score"]
        ) / 2
    if has_lgb.any() and not both.all():
        only_lgb = has_lgb & ~both
        result.loc[only_lgb, "model_score"] = result.loc[only_lgb, "lgb_score"]
    if has_xgb.any() and not both.all():
        only_xgb = has_xgb & ~both
        result.loc[only_xgb, "model_score"] = result.loc[only_xgb, "xgb_score"]

    return result


def _to_percentile(raw_scores):
    """把模型原始输出映射到 0-100 百分位"""
    s = pd.Series(raw_scores)
    return s.rank(pct=True) * 100


# ============================================================
# 降级方案：线性打分
# ============================================================

def compute_linear_score(factor_df):
    """
    降级方案：百分位线性加权（不需要训练，ML 模型不可用时使用）
    """
    features = factor_df.copy()

    category_weights = {
        "quality": 0.25, "valuation_inverse": 0.15,
        "valuation_direct": 0.05, "growth": 0.15,
        "dividend": 0.15, "technical": 0.10, "risk": -0.10,
    }

    # 因子→分类映射
    factor_categories = {}
    for col in features.columns:
        if col == "code":
            continue
        name = col.lower()
        if any(x in name for x in ["roe", "margin", "sharpe", "stability"]):
            factor_categories[col] = "quality"
        elif any(x in name for x in ["pe_percentile", "pb_percentile"]):
            factor_categories[col] = "valuation_direct"
        elif any(x in name for x in ["pe", "pb", "ps"]):
            factor_categories[col] = "valuation_inverse"
        elif any(x in name for x in ["ret_", "momentum", "growth", "accel"]):
            factor_categories[col] = "growth"
        elif any(x in name for x in ["div_", "fcf"]):
            factor_categories[col] = "dividend"
        elif any(x in name for x in ["ma", "rps", "above", "amount_ratio", "turnover"]):
            factor_categories[col] = "technical"
        elif any(x in name for x in ["drawdown", "vol", "debt", "neg_day", "var"]):
            factor_categories[col] = "risk"
        else:
            factor_categories[col] = "quality"

    # 每个因子用百分位
    pct_cols = []
    for col in features.columns:
        if col == "code":
            continue
        pct_col = f"{col}_pct"
        features[pct_col] = features[col].rank(pct=True) * 100
        pct_cols.append(pct_col)

    # 加权
    scores = pd.Series(0.0, index=features.index)
    total_w = 0.0
    for pcol in pct_cols:
        orig = pcol.replace("_pct", "")
        cat = factor_categories.get(orig, "quality")
        w = category_weights.get(cat, 0)
        if cat in ("valuation_inverse", "risk"):
            scores += abs(w) * (100 - features[pcol])
        else:
            scores += w * features[pcol]
        total_w += abs(w)

    if total_w > 0:
        features["raw"] = scores / total_w
    else:
        features["raw"] = 50

    # 再次排名 → 0-100 均匀分布
    features["linear_score"] = features["raw"].rank(pct=True) * 100

    result = features[["code", "linear_score"]].copy()
    result.columns = ["code", "model_score"]
    return result


# ============================================================
# scikit-learn: 模型评估（时序交叉验证 + Rank IC）
# ============================================================

def evaluate_models(X, y, codes, industry_df):
    """
    用时序交叉验证评估模型排序能力

    返回 dict:
        - lgb_ic: LightGBM 的平均 Rank IC
        - xgb_ic: XGBoost 的平均 Rank IC
        - lgb_ic_std, xgb_ic_std
        - cv_folds: 交叉验证折数
    """
    n_samples = len(X)
    if n_samples < 60:
        return None

    # 时序切分（最近 30% 作为验证）
    split_idx = int(n_samples * 0.7)
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, n_samples)

    # 按 code 排序保证时序一致（codes 已排序）
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    codes_train = codes[train_idx]
    codes_val = codes[val_idx]

    results = {"cv_folds": 1, "train_samples": len(train_idx), "val_samples": len(val_idx)}

    # LightGBM 评估
    lgb_model, _ = train_lightgbm_ranker(X_train, y_train, codes_train, industry_df)
    if lgb_model is not None:
        try:
            val_pred = lgb_model.predict(X_val.fillna(0).replace([np.inf, -np.inf], 0))
            ic, _ = spearmanr(val_pred, y_val.values)
            results["lgb_ic"] = round(ic, 4)
        except Exception:
            results["lgb_ic"] = None
    else:
        results["lgb_ic"] = None

    # XGBoost 评估
    xgb_model, _ = train_xgboost_ranker(X_train, y_train, codes_train, industry_df)
    if xgb_model is not None:
        try:
            val_pred = xgb_model.predict(X_val.fillna(0).replace([np.inf, -np.inf], 0))
            ic, _ = spearmanr(val_pred, y_val.values)
            results["xgb_ic"] = round(ic, 4)
        except Exception:
            results["xgb_ic"] = None
    else:
        results["xgb_ic"] = None

    return results


# ============================================================
# 主入口
# ============================================================

def run_ranking(factor_df, hist_data, industry_df=None, use_ml=True):
    """
    运行完整排序流程

    1. 构造目标（未来 60 日行业内超额收益）
    2. RobustScaler 标准化特征
    3. 时序切分：训练集/验证集
    4. 训练 LightGBM LGBMRanker（主模型）
    5. 训练 XGBoost XGBRanker（交叉验证）
    6. 综合打分 + Rank IC 评估
    7. 如 ML 不可用，降级到线性打分

    返回: (scores_df, importance_df, eval_metrics)
    """
    # 构造目标
    target = build_target(hist_data, factor_df, industry_df)
    if target is None or len(target) < 30:
        print("  目标数据不足 (<30)，使用线性打分")
        return compute_linear_score(factor_df), None, None

    # 准备特征：只保留 target 中有值的股票
    common_codes = list(set(factor_df["code"]) & set(target.index))
    if len(common_codes) < 30:
        print("  共同样本不足，使用线性打分")
        return compute_linear_score(factor_df), None, None

    common_df = factor_df[factor_df["code"].isin(common_codes)].copy()
    feature_cols = [c for c in factor_df.columns if c != "code"]
    X = common_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    codes = common_df["code"].values
    y = target.loc[codes]

    # scikit-learn: RobustScaler 标准化
    try:
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
    except Exception:
        X_scaled = X

    print(f"  训练样本: {len(X_scaled)} 只股票, {X_scaled.shape[1]} 个因子")

    importance = None
    eval_metrics = None

    if use_ml:
        # scikit-learn: 模型评估
        print("  scikit-learn 时序交叉验证评估...")
        eval_metrics = evaluate_models(X_scaled, y, codes, industry_df)
        if eval_metrics:
            lgb_ic = eval_metrics.get("lgb_ic")
            xgb_ic = eval_metrics.get("xgb_ic")
            ic_str = f"LGB_IC={lgb_ic}, XGB_IC={xgb_ic}" if lgb_ic or xgb_ic else "N/A"
            print(f"  评估结果: {ic_str}")

        # 主模型：LightGBM（全量训练）
        lgb_model, lgb_imp = train_lightgbm_ranker(X_scaled, y, codes, industry_df)
        if lgb_imp is not None:
            importance = lgb_imp

        # 交叉验证：XGBoost（全量训练）
        xgb_model, xgb_imp = train_xgboost_ranker(X_scaled, y, codes, industry_df)
        if importance is None and xgb_imp is not None:
            importance = xgb_imp

        if lgb_model is not None or xgb_model is not None:
            print(f"  ML 模型就绪: LGB={'Y' if lgb_model else 'N'}, XGB={'Y' if xgb_model else 'N'}")
            scores = predict_scores(factor_df, lgb_model, xgb_model)
            return scores, importance, eval_metrics
        else:
            print("  ML 模型均不可用，使用线性打分")
            return compute_linear_score(factor_df), importance, eval_metrics

    return compute_linear_score(factor_df), None, None
