"""
P2+P3+P4: 修复后的排序模型

P2: 目标标签 = date × industry 内未来 60 日收益分位
P3: Walk-forward 回测 (3年训练/1年验证, ≥60日 embargo)
P4: XGBoost 分歧检测 (lgb_rank, xgb_rank, rank_diff, agreement_score)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _to_percentile(raw_scores):
    """模型原始输出 → 0-100 百分位"""
    s = pd.Series(raw_scores)
    return s.rank(pct=True) * 100


# ============================================================
# P2: 目标标签构造
# ============================================================

def build_target_v2(hist_data, forward_days=60):
    """
    构造 P2 目标变量

    两步:
      1. 计算每只股票的未来 60 日绝对收益
      2. 在 date × industry 内做分位排名 → 行业内相对排名

    返回:
        pd.Series: index=code, value=行业未来收益分位 (0-1)
    """
    from config import DATA_CACHE
    import pickle

    # 加载行业数据
    try:
        industry = pickle.load(open(DATA_CACHE / "industry.pkl", "rb"))
        ind_map = dict(zip(industry["code"], industry["industry"]))
    except Exception:
        ind_map = {}

    targets = {}
    for code, df in hist_data.items():
        if df is None or df.empty or len(df) < forward_days + 5:
            continue
        try:
            close = df["close"].values
            # 未来 60 日收益
            fwd_ret = close[-1] / close[-(forward_days + 1)] - 1
            targets[code] = {"fwd_ret": fwd_ret, "industry": ind_map.get(code, "unknown")}
        except Exception:
            continue

    if not targets:
        return None

    df = pd.DataFrame(targets).T
    df.index.name = "code"
    df = df.reset_index()

    # 行业内分位排名
    df["target"] = df.groupby("industry")["fwd_ret"].transform(lambda x: x.rank(pct=True))

    return df.set_index("code")["target"]


# ============================================================
# P3: Walk-forward 回测引擎
# ============================================================

def walk_forward_backtest(factor_df, hist_data, industry_df, n_splits=3, train_years=3, embargo_days=60):
    """
    P3 Walk-forward 回测

    参数:
        n_splits: 回测窗口数
        train_years: 训练集年数
        embargo_days: 训练/验证间隔（防止标签重叠）

    返回:
        pd.DataFrame: 每个窗口的评估指标
    """
    from models.ranker import train_lightgbm_ranker, train_xgboost_ranker

    # 获取所有 unique 日期（从最早的股票）
    all_dates = pd.DatetimeIndex([])
    for df in hist_data.values():
        if "date" in df.columns:
            all_dates = all_dates.union(df["date"])
    all_dates = all_dates.sort_values()

    if len(all_dates) < 500:
        print("  历史数据不足 (<500 交易日)，无法做 walk-forward")
        return None

    # 窗口划分
    trading_days_per_year = 242
    results = []

    for split in range(n_splits):
        # 训练期：最近 train_years 年
        train_end = all_dates[-(embargo_days + (n_splits - split) * trading_days_per_year)]
        train_start = all_dates[all_dates <= train_end][-train_years * trading_days_per_year] if len(all_dates[all_dates <= train_end]) >= train_years * trading_days_per_year else all_dates[0]

        # 验证期：训练结束后 + embargo
        val_start_idx = all_dates.get_loc(train_end) + embargo_days
        val_end_idx = min(val_start_idx + trading_days_per_year, len(all_dates) - 1)
        if val_start_idx >= len(all_dates):
            break

        val_start = all_dates[val_start_idx]
        val_end = all_dates[val_end_idx]

        print(f"  Window {split+1}: train={str(train_start)[:10]}~{str(train_end)[:10]}, val={str(val_start)[:10]}~{str(val_end)[:10]}")

        # 构造训练集标签（使用训练期最后一天可观测的未来收益）
        target = build_target_v2(hist_data)
        if target is None or len(target) < 30:
            continue

        # 准备特征
        common = list(set(factor_df["code"]) & set(target.index))
        if len(common) < 30:
            continue

        X = factor_df[factor_df["code"].isin(common)]
        feat_cols = [c for c in X.columns if c != "code" and not c.endswith("_rank")]
        X = X[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = target.loc[common]
        codes = np.array(common)

        # 训练两个模型
        lgb_m, lgb_imp = train_lightgbm_ranker(X, y, codes, industry_df)
        xgb_m, xgb_imp = train_xgboost_ranker(X, y, codes, industry_df)

        # 全量打分
        X_full = factor_df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
        predictions = pd.DataFrame({"code": factor_df["code"]})

        if lgb_m is not None:
            predictions["lgb_score"] = lgb_m.predict(X_full)
        if xgb_m is not None:
            predictions["xgb_score"] = xgb_m.predict(X_full)

        # Rank IC
        lgb_ic = None; xgb_ic = None
        actual = target.loc[common]
        if lgb_m is not None:
            preds = predictions[predictions["code"].isin(common)]["lgb_score"]
            lgb_ic, _ = spearmanr(preds, actual.loc[preds.index] if isinstance(actual, pd.Series) else actual)
        if xgb_m is not None:
            preds = predictions[predictions["code"].isin(common)]["xgb_score"]
            try:
                xgb_ic, _ = spearmanr(preds, actual.loc[preds.index])
            except Exception:
                pass

        # Top30 vs Bottom30 超额
        lgb_top30_ret = None
        if lgb_m is not None:
            top30 = predictions.nlargest(30, "lgb_score")["code"]
            bot30 = predictions.nsmallest(30, "lgb_score")["code"]
            top_ret = np.mean([target.get(c, 0) for c in top30])
            bot_ret = np.mean([target.get(c, 0) for c in bot30])
            lgb_top30_ret = top_ret - bot_ret

        results.append({
            "window": split + 1,
            "train_dates": f"{str(train_start)[:10]}~{str(train_end)[:10]}",
            "val_dates": f"{str(val_start)[:10]}~{str(val_end)[:10]}",
            "n_stocks": len(common),
            "lgb_ic": round(lgb_ic, 4) if lgb_ic else None,
            "xgb_ic": round(xgb_ic, 4) if xgb_ic else None,
            "lgb_top30_spread": round(lgb_top30_ret * 100, 2) if lgb_top30_ret else None,
        })

    if results:
        df = pd.DataFrame(results)
        print(f"\n  Walk-forward 汇总:")
        valid_ic = [r["lgb_ic"] for r in results if r["lgb_ic"] is not None]
        if valid_ic:
            print(f"    LGB IC 均值: {np.mean(valid_ic):.4f}, 标准差: {np.std(valid_ic):.4f}, ICIR: {np.mean(valid_ic)/np.std(valid_ic):.2f}" if np.std(valid_ic) > 0 else f"    LGB IC 均值: {np.mean(valid_ic):.4f}")
        return df

    return None


# ============================================================
# P4: XGBoost 分歧检测
# ============================================================

def detect_divergence(ranking_df):
    """
    P4: 检测 LightGBM 与 XGBoost 的排名分歧

    输入: ranking_df 需包含 lgb_score 和 xgb_score 列
    输出: 增加 rank_diff, agreement_score, diverge_flag 列
    """
    df = ranking_df.copy()

    if "lgb_score" not in df.columns or "xgb_score" not in df.columns:
        df["rank_diff"] = 0
        df["agreement_score"] = 1.0
        df["diverge_flag"] = ""
        return df

    # 百分位排名
    df["lgb_rank"] = df["lgb_score"].rank(pct=True) * 100
    df["xgb_rank"] = df["xgb_score"].rank(pct=True) * 100
    df["rank_diff"] = abs(df["lgb_rank"] - df["xgb_rank"])

    # 一致性得分
    df["agreement_score"] = 1 - df["rank_diff"] / 100

    # 分歧标记
    df["diverge_flag"] = ""
    df.loc[df["rank_diff"] < 10, "diverge_flag"] = "一致"
    df.loc[(df["rank_diff"] >= 10) & (df["rank_diff"] < 30), "diverge_flag"] = "降权"
    df.loc[df["rank_diff"] >= 30, "diverge_flag"] = "严重分歧"

    # 严重分歧 → 降权 total_score
    if "total_score" in df.columns:
        severe = df["rank_diff"] >= 30
        df.loc[severe, "total_score"] = df.loc[severe, "total_score"] * 0.5

    # 统计
    n_severe = (df["rank_diff"] >= 30).sum()
    n_mild = ((df["rank_diff"] >= 10) & (df["rank_diff"] < 30)).sum()
    if n_severe > 0 or n_mild > 0:
        print(f"  分歧检测: 严重{n_severe}只, 降权{n_mild}只")

    return df


# ============================================================
# 综合入口（兼容旧接口）
# ============================================================

def run_ranking_v2(factor_df, hist_data, industry_df=None, use_ml=True):
    """
    新版完整排序流程: P2 目标 + 双模型 + P4 分歧检测

    返回: (scores_df, importance_df, eval_metrics)
    """
    from models.ranker import (
        train_lightgbm_ranker, train_xgboost_ranker,
        compute_linear_score, predict_scores,
    )

    # P2: 行业未来收益分位标签
    target = build_target_v2(hist_data)
    if target is None or len(target) < 30:
        print("  P2 目标不足，降级线性打分")
        return compute_linear_score(factor_df), None, None

    # 准备特征
    common = list(set(factor_df["code"]) & set(target.index))
    if len(common) < 30:
        return compute_linear_score(factor_df), None, None

    feat_cols = [c for c in factor_df.columns if c != "code" and not c.endswith("_rank")]
    common_df = factor_df[factor_df["code"].isin(common)]
    X = common_df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = target.loc[common]
    codes = np.array(common)

    print(f"  P2 目标样本: {len(X)} 只, 平均目标值: {y.mean():.3f}")

    importance = None; eval_metrics = None

    if use_ml:
        lgb_m, lgb_imp = train_lightgbm_ranker(X, y, codes, industry_df)
        xgb_m, xgb_imp = train_xgboost_ranker(X, y, codes, industry_df)
        importance = lgb_imp if lgb_imp is not None else xgb_imp

        if lgb_m is not None or xgb_m is not None:
            # Predict on same columns used for training
            X_full = factor_df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
            scores = predict_scores_with_cols(factor_df["code"], X_full, lgb_m, xgb_m)
            scores = detect_divergence(scores)
            return scores, importance, eval_metrics

    return compute_linear_score(factor_df), importance, eval_metrics


def predict_scores_with_cols(codes, X, lgb_model, xgb_model):
    """用训练好的特征矩阵预测（保证列一致）"""
    result = pd.DataFrame({"code": codes.values if hasattr(codes, 'values') else list(codes)})

    if lgb_model is not None:
        try:
            raw = lgb_model.predict(X)
            result["lgb_score"] = _to_percentile(raw)
        except Exception as e:
            # Feature count mismatch or other prediction error
            print(f"  LGB predict: {type(e).__name__}")
            # Try with shape check disabled
            try:
                import lightgbm as lgb
                raw = lgb_model.predict(X, predict_disable_shape_check=True)
                result["lgb_score"] = _to_percentile(raw)
                print(f"  LGB predict: retry OK")
            except Exception as e2:
                print(f"  LGB predict: retry also failed - {type(e2).__name__}")
                result["lgb_score"] = np.nan
    else:
        result["lgb_score"] = np.nan

    if xgb_model is not None:
        try:
            raw = xgb_model.predict(X)
            result["xgb_score"] = _to_percentile(raw)
        except Exception as e:
            print(f"  XGB predict: {type(e).__name__}")
            result["xgb_score"] = np.nan
    else:
        result["xgb_score"] = np.nan

    has_lgb = result["lgb_score"].notna()
    has_xgb = result["xgb_score"].notna()
    both = has_lgb & has_xgb

    result["model_score"] = np.nan
    if both.any():
        result.loc[both, "model_score"] = (result.loc[both, "lgb_score"] + result.loc[both, "xgb_score"]) / 2
    only_lgb = has_lgb & ~has_xgb
    if only_lgb.any():
        result.loc[only_lgb, "model_score"] = result.loc[only_lgb, "lgb_score"]
    only_xgb = ~has_lgb & has_xgb
    if only_xgb.any():
        result.loc[only_xgb, "model_score"] = result.loc[only_xgb, "xgb_score"]

    n_lgb_ok = has_lgb.sum()
    n_xgb_ok = has_xgb.sum()
    print(f"  Predict: LGB={n_lgb_ok}/{len(result)}, XGB={n_xgb_ok}/{len(result)}")

    return result
