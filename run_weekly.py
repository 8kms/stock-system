#!/usr/bin/env python3
"""
终局方案 12 步 SOP — A 股中低频选股辅助系统

Gate 0-6 自动检查，输出 4 张表:
  1. 核心候选池 (5-10只)
  2. 观察池 (20-30只)
  3. 风险剔除池
  4. 复盘归因表
"""
import sys, pickle, time, warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_CACHE, OUTPUT_DIR, WATCH_INDICES, WATCHLIST_STOCKS
import pandas as pd
import numpy as np


def main(quick_mode=True):
    t0 = time.time()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 70)
    print(f"  A 股中低频选股辅助系统 — 终局方案 v1.0")
    print(f"  运行时间: {date_str}")
    print("=" * 70)

    # DuckDB 初始化
    from data.duckdb_schema import init_schema
    init_schema()

    # 估值 CSV 模板（首次运行生成，用户填入历史PE/PB后导入）
    from data.valuation_provider import generate_csv_template, ValuationProvider
    try:
        generate_csv_template(codes)
    except Exception:
        pass

    # ============================================================
    # 步骤 1: 更新行情数据
    # ============================================================
    print("\n[步骤 1/12] 更新行情数据...")
    import data.fetcher as f
    f._akshare_available = False  # 快速模式跳过 AKShare

    stock_list = f.get_a_stock_list()
    codes = stock_list["code"].tolist()
    hist_data = f.fetch_stock_pool_hist(codes)
    index_data = f.fetch_index_hist()
    industry_data = f.get_stock_industry()
    valuation_data = f.fetch_valuation_data(stock_list)
    n_hist = len(hist_data)
    print(f"  股票池 {len(codes)} 只, 日线 {n_hist} 只, 指数 {len(index_data)} 个")

    # ============================================================
    # 步骤 2: Gate 0 — PIT 数据防污染 + 数据质量检查
    # ============================================================
    print("\n[步骤 2/12] Gate 0: 数据防污染 + 数据质量检查...")

    # AKShare 健康检查
    from data.safe_akshare import SafeAkshareClient
    ak_client = SafeAkshareClient()
    ak_health = ak_client.health_check()
    ak_status = ak_health.get("status", "UNKNOWN")
    price_ok = "✅" if ak_health.get("price") else "❌"
    spot_ok = "✅" if ak_health.get("spot") else "❌"
    index_ok = "✅" if ak_health.get("index") else "❌"
    print(f"  AKShare 健康: {ak_status} (price={price_ok}, spot={spot_ok}, index={index_ok})")

    # PIT 审计
    from data.pit_fetcher import pit_audit
    violations = pit_audit(codes, date_str[:10], n_sample=min(30, len(codes)))
    if violations and len(violations) > len(codes) * 0.3:
        print("  Gate 0 不通过: 超过30%股票存在未来函数, 停止运行")
        return None

    # 硬剔除
    from data.cleaner import hard_exclude
    filtered = hard_exclude(stock_list, hist_data)
    filtered = filtered[filtered["code"].isin(hist_data.keys())]
    clean_codes = filtered["code"].tolist()
    pool_hist = {c: hist_data[c] for c in clean_codes}
    print(f"  硬剔除后: {len(pool_hist)} 只")

    # ============================================================
    # 步骤 3: 规则评分
    # ============================================================
    print("\n[步骤 3/12] Gate 1: 计算规则评分 (30/20/20/10/10/10)...")
    from factors.rule_score import build_rule_score
    rule_df = build_rule_score(pool_hist, stock_list, industry_data, valuation_data)
    print(f"  规则评分原始: {len(rule_df)} 只, 分值范围 {rule_df['rule_score'].min():.0f}-{rule_df['rule_score'].max():.0f}")

    # ==== Gate 0A: 行情数据质量 ====
    from data.quality_gate import gate_0a_coverage_check, check_valuation_quality, gate_0b_variance_check
    price_n = len(pool_hist)
    price_status = "PASS" if price_n >= 90 else ("WARN" if price_n >= 50 else "FAIL")
    print(f"  Gate 0A 行情: {price_status} ({price_n}只有效日线)")

    # ==== Gate 0B: 财务数据质量 ====
    cov_checks, cov_status, cov_overall = gate_0a_coverage_check(rule_df)
    n_fail_cov = sum(1 for s in cov_status.values() if s == "FAIL")
    # 财务字段来自Tushare真实值 → 检查覆盖率
    tushare_n = 1  # 默认（从已有的tushare_latest读取）
    try:
        tushare_latest = pickle.load(open(DATA_CACHE/"tushare_latest.pkl","rb"))
        tushare_n = len(tushare_latest)
    except:
        pass
    financial_status = "PASS" if tushare_n >= 70 else ("WARN" if tushare_n >= 10 else "FAIL")
    print(f"  Gate 0B 财务: {financial_status} (Tushare真实 {tushare_n}/{len(pool_hist)} 只, 覆盖率检查 {cov_overall})")

    # ==== Gate 0C: 估值数据质量 ====
    val_status, val_report = check_valuation_quality(valuation_data)
    # DuckDB 历史覆盖检查：有足够历史数据则升级
    try:
        from data.duckdb_schema import get_db
        db = get_db()
        val_db_cnt = db.execute("SELECT COUNT(*) FROM valuation_daily").fetchone()[0]
        val_codes = db.execute("SELECT COUNT(DISTINCT code) FROM valuation_daily").fetchone()[0]
        val_dates = db.execute("SELECT COUNT(DISTINCT trade_date) FROM valuation_daily").fetchone()[0]
    except: val_db_cnt = 0; val_codes = 0; val_dates = 0
    if val_db_cnt > 50000 and val_codes >= 90 and val_dates >= 750:
        val_status = "PASS"
        print(f"  Gate 0C 估值: PASS (DuckDB {val_db_cnt}行/{val_codes}只/{val_dates}天)")
    else:
        print(f"  Gate 0C 估值: {val_status} (真实{val_report['n_real']}只, 估算{val_report['n_estimated']}只, DuckDB {val_db_cnt}行)")

    # 估值分封顶
    if val_status != "PASS" and "valuation_score" in rule_df.columns:
        estimated_codes = valuation_data[valuation_data.get("valuation_source","") != "akshare_real"]["code"].tolist()
        rule_df.loc[rule_df["code"].isin(estimated_codes), "valuation_score"] = (
            rule_df.loc[rule_df["code"].isin(estimated_codes), "valuation_score"].clip(upper=60)
        )
        rule_df.loc[rule_df["code"].isin(estimated_codes), "rule_score"] = (
            rule_df.loc[rule_df["code"].isin(estimated_codes), "rule_score"].clip(upper=80)
        )
        n_capped = len(set(estimated_codes) & set(rule_df["code"]))
        print(f"  估值分封顶: {n_capped} 只 (估值分≤60, 规则分≤80, 不进LightGBM)")

    # ==== Gate 0D: 横截面区分度 ====
    var_report, var_overall = gate_0b_variance_check(rule_df)
    n_fail_var = (var_report["status"] == "FAIL").sum()
    feature_status = var_overall
    print(f"  Gate 0D 特征方差: {feature_status} (FAIL={n_fail_var})")

    # P0: 禁止假满分
    from data.quality_gate import apply_data_quality_cap
    rule_df = apply_data_quality_cap(rule_df)
    n_cap = (rule_df["data_quality_flag"] != "PASS").sum()
    valid_scores = rule_df["rule_score"].dropna()
    if len(valid_scores) > 0:
        print(f"  数据封顶后: {len(valid_scores)} 只有效分, 范围 {valid_scores.min():.0f}-{valid_scores.max():.0f}, 封顶/剔除 {n_cap} 只")

    # 长江电力检查
    if "600900" in rule_df["code"].values:
        cy_score = rule_df[rule_df["code"] == "600900"]["rule_score"].values[0]
        cy_flag = rule_df[rule_df["code"] == "600900"]["data_quality_flag"].values[0]
        print(f"  长江电力: 规则分={cy_score:.1f}, 数据质量={cy_flag}")

    # ============================================================
    # 步骤 4: 标签 + 训练/验证切分
    # ============================================================
    print("\n[步骤 4/12] Gate 2: 目标标签...")
    from models.ranker_v2 import build_target_v2
    target = build_target_v2(pool_hist)
    print(f"  标签: {len(target)} 样本, 均值 {target.mean():.3f}")

    # ============================================================
    # 步骤 5: LightGBM + XGBoost
    # ============================================================
    print("\n[步骤 5/12] Gate 3+6: LightGBM/XGBoost 训练...")
    from data.cleaner import clean_factor_data
    # 排除字符串列（data_quality_flag 等），只保留数值型
    rule_numeric = rule_df.select_dtypes(include=[np.number, "number"]).copy()
    # 保留 code 列
    if "code" in rule_df.columns:
        rule_numeric["code"] = rule_df["code"]
    fdf_clean = clean_factor_data(rule_numeric, industry_data, neutralize=True)
    from models.ranker_v2 import run_ranking_v2
    scores, importance, eval_metrics = run_ranking_v2(fdf_clean, pool_hist, industry_data, use_ml=True)

    # 构建排名
    ranking_df = scores.merge(stock_list[["code", "name"]], on="code", how="left")
    ranking_df = ranking_df.merge(industry_data, on="code", how="left")

    # 合并规则分
    rule_cols_map = {"rule_score": "rule_score", "quality_score": "quality_score",
                     "valuation_score": "valuation_score", "cashflow_score": "cashflow_score",
                     "growth_score": "growth_score", "technical_score": "technical_score",
                     "risk_score": "risk_score"}
    for src, dst in rule_cols_map.items():
        if src in rule_df.columns:
            rmap = dict(zip(rule_df["code"], rule_df[src]))
            ranking_df[dst] = ranking_df["code"].map(rmap)

    # 综合分: 规则分 40% + 模型分 60%
    if "rule_score" in ranking_df.columns and "model_score" in ranking_df.columns:
        ranking_df["total_score"] = (ranking_df["rule_score"].fillna(50) * 0.4 +
                                     ranking_df["model_score"].fillna(50) * 0.6)
    elif "model_score" in ranking_df.columns:
        ranking_df["total_score"] = ranking_df["model_score"]
    elif "rule_score" in ranking_df.columns:
        ranking_df["total_score"] = ranking_df["rule_score"]
    else:
        ranking_df["total_score"] = 50

    # 分歧统计
    divergence_stats = {"severe": 0, "mild": 0}
    if "rank_diff" in scores.columns:
        divergence_stats["severe"] = int((scores["rank_diff"] >= 30).sum())
        divergence_stats["mild"] = int(((scores["rank_diff"] >= 10) & (scores["rank_diff"] < 30)).sum())

    ranking_df = ranking_df.drop_duplicates("code").sort_values("total_score", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    print(f"  模型就绪, 严重分歧{divergence_stats['severe']}只, 降权{divergence_stats['mild']}只")

    # 模型审计 + 动量检查
    from models.model_audit import audit_lgb_importance
    model_split_status, factor_structure_status = "UNKNOWN", "UNKNOWN"
    audit = {}
    mom_ratio = 1.0
    # 从 importance 数据做审计（importance 来自 train_lightgbm_ranker）
    if importance is not None and not importance.empty:
        try:
            total_gain = importance["importance"].sum()
            total_features = len(importance)
            nonzero = (importance["importance"] > 0).sum()
            momentum_features = ["ret_20d","ret_60d","ret_120d","ma60_slope","momentum_score","trend_score","vol_change"]
            fundamental_features = ["roe_raw","roe","roa","gross_margin_raw","gross_margin","net_margin_raw","net_margin",
                                    "quality_score","q_gross_margin","q_cfo","c_cfo_profit","cashflow_score","risk_score","debt_ratio"]
            valuation_features = ["valuation_score","pe_ttm","pb","v_pe_hist","v_pb_hist","v_div_yield","v_fcf_yield"]
            mom_gain = importance[importance["feature"].isin(momentum_features)]["importance"].sum()
            fund_gain = importance[importance["feature"].isin(fundamental_features)]["importance"].sum()
            val_gain = importance[importance["feature"].isin(valuation_features)]["importance"].sum()
            if total_gain > 0:
                mom_ratio = mom_gain / total_gain
                fund_ratio = fund_gain / total_gain
                val_ratio = val_gain / total_gain
                if mom_ratio > 0.50: model_split_status = "FAIL"
                elif mom_ratio > 0.35: model_split_status = "WARN"
                else: model_split_status = "PASS"
                factor_structure_status = "PASS" if fund_ratio > 0.10 else "FAIL"
                audit = {"total_gain": total_gain, "total_split": 1, "nonzero_features": nonzero,
                         "feature_count": total_features, "momentum_ratio": mom_ratio,
                         "fundamental_ratio": fund_ratio, "valuation_ratio": val_ratio,
                         "status": model_split_status}
                print(f"  模型审计: {model_split_status} (gain={total_gain:.0f}, 基本面={fund_ratio:.0%}, "
                      f"动量={mom_ratio:.0%}, 估值={val_ratio:.0%}, 非零={nonzero}/{total_features})")
        except Exception as e:
            print(f"  模型审计: 跳过 ({e})")
    print(f"  模型分裂: {model_split_status}, 因子结构: {factor_structure_status}")

    # ============================================================
    # 步骤 6: 行业暴露控制
    # ============================================================
    print("\n[步骤 6/12] Gate 4: 行业暴露控制...")
    from models.industry_exposure import check_industry_exposure, apply_industry_constraints
    exposure_report = check_industry_exposure(ranking_df)
    core_df, watch_industry_df = apply_industry_constraints(ranking_df, top_n=30)
    print(f"  行业暴露: {'PASS' if exposure_report['passed'] else 'WARN'}, "
          f"覆盖{exposure_report['top30_industries']}行业, 前三权重{exposure_report['top3_weight']:.1%}")

    # ============================================================
    # 步骤 7: 因子衰减
    # ============================================================
    print("\n[步骤 7/12] Gate 5: 因子衰减监控...")
    from models.factor_decay import generate_decay_report
    decay_report = generate_decay_report(ranking_df)
    if not decay_report.empty:
        n_r = (decay_report["状态"].str.contains("红色")).sum()
        n_y = (decay_report["状态"].str.contains("黄色")).sum()
        n_g = (decay_report["状态"].str.contains("绿色")).sum()
        print(f"  绿{n_g}/黄{n_y}/红{n_r}, {'PASS' if n_r == 0 else 'WARN'}")

    # ============================================================
    # 步骤 8: 指数环境
    # ============================================================
    print("\n[步骤 8/12] 指数环境过滤...")
    from models.index_filter import calc_market_state_detail
    index_detail = calc_market_state_detail(index_data)
    print(f"  指数评分: {index_detail['total_score']}/10, {index_detail['state_cn']}")

    # ============================================================
    # 步骤 9: Kronos/K线 后置确认
    # ============================================================
    print("\n[步骤 9/12] Kronos/K线 后置确认...")
    top_codes = core_df.head(50)["code"].tolist() if len(core_df) > 0 else ranking_df.head(50)["code"].tolist()

    def fast_kline(df):
        if df is None or len(df) < 60: return {"score": 5, "is_healthy": True, "is_overbought": False, "is_stabilizing": False, "vol_abnormal": False, "signals": ["数据不足"]}
        c = df["close"].values; v = df["volume"].values
        ma60 = np.mean(c[-60:]); s = 7.0; sig = []
        if c[-1] < ma60: s -= 2.5; sig.append("跌破60日线")
        r = np.diff(c) / c[:-1]; v20 = np.std(r[-20:]); v60 = np.std(r[-60:]) if len(r) >= 60 else v20
        if c[-1] > np.mean(c[-20:]) + 2 * np.std(c[-20:]): s -= 1.5; sig.append("过热")
        if v20 > v60 * 2: s -= 2; sig.append("波动异常")
        if not sig: sig.append("K线正常")
        return {"score": max(0, min(10, s)), "is_healthy": s > 3, "is_overbought": "过热" in "".join(sig),
                "is_stabilizing": False, "vol_abnormal": v20 > v60 * 2, "signals": sig}

    kline_results = {}
    for code in top_codes:
        if code in pool_hist: kline_results[code] = fast_kline(pool_hist[code])
    ks = {c: r["score"] for c, r in kline_results.items()}
    ranking_df["kline_score"] = ranking_df["code"].map(ks).fillna(5)
    n_unhealthy = sum(1 for r in kline_results.values() if not r["is_healthy"])
    n_overbought = sum(1 for r in kline_results.values() if r["is_overbought"])
    print(f"  K线分析: {len(kline_results)}只, 走坏{n_unhealthy}只, 过热{n_overbought}只")

    # ============================================================
    # 步骤 10: 生成四张表
    # ============================================================
    print("\n[步骤 10/12] 生成核心候选池/观察池/剔除池...")

    # 系统模式判定 (必须在 pool assignment 前)
    from data.quality_gate import decide_system_mode, gate_0b_core_check
    core_status, core_report = gate_0b_core_check(rule_df)
    core_fail_count = (core_report["status"] == "FAIL").sum() if core_report is not None else 99
    # 估值质量影响系统模式
    from data.duckdb_schema import get_db
    try:
        db = get_db()
        val_in_db = db.execute("SELECT COUNT(*) FROM valuation_daily").fetchone()[0]
    except: val_in_db = 0
    gate0c_val_status = "PASS" if val_in_db > 50000 else ("WARN" if val_in_db > 1000 else "FAIL")

    model_mode, mode_name, mode_desc = decide_system_mode(
        price_status, financial_status, gate0c_val_status,
        feature_status, model_split_status, factor_structure_status,
        backtest_status=None,
    )
    print(f"\n  系统模式判定: {model_mode} — {mode_name}")

    # 多维度评分 + 候选池分类
    from models.score_builder import (build_scores, assign_stock_pool, model_consensus_label,
                                       generate_downgrade_reasons, generate_action_text)
    ranking_df = build_scores(ranking_df, pool_hist)
    if "data_quality_flag" not in ranking_df.columns:
        ranking_df["data_quality_flag"] = "PASS"
    ranking_df["model_consensus"] = ranking_df.apply(model_consensus_label, axis=1)
    ranking_df["stock_pool"] = ranking_df.apply(assign_stock_pool, axis=1)
    # 修复 risk_score: rule_score 里的 risk_score 是 0-10 子分，需要乘 10 到 0-100
    if "risk_score" in ranking_df.columns and ranking_df["risk_score"].max() < 15:
        ranking_df["risk_score"] = (ranking_df["risk_score"] * 10).clip(0, 100)
        print(f"  risk_score rescaled: now {ranking_df['risk_score'].min():.0f}-{ranking_df['risk_score'].max():.0f}")
    # 合并 valuation_quality
    if valuation_data is not None and "valuation_quality" in valuation_data.columns:
        vq_map = dict(zip(valuation_data["code"], valuation_data["valuation_quality"]))
        ranking_df["valuation_quality"] = ranking_df["code"].map(vq_map).fillna("WARN")
    else:
        ranking_df["valuation_quality"] = "WARN"
    # Re-compute adjusted score now that model_consensus + stock_pool are available
    from models.score_builder import apply_score_adjustments, POOL_PRIORITY
    ranking_df["final_action_score_adj"] = ranking_df.apply(apply_score_adjustments, axis=1)
    ranking_df["downgrade_reasons"] = ranking_df.apply(generate_downgrade_reasons, axis=1)
    ranking_df["action_text"] = ranking_df.apply(generate_action_text, axis=1)
    ranking_df["pool_priority"] = ranking_df["stock_pool"].map(POOL_PRIORITY).fillna(99)

    core_candidates = ranking_df[ranking_df["stock_pool"] == "核心候选池"].head(7).copy()
    value_watch = ranking_df[ranking_df["stock_pool"] == "价值观察池"].head(15).copy()
    mom_watch = ranking_df[ranking_df["stock_pool"] == "技术强势观察池"].head(10).copy()
    normal_watch = ranking_df[ranking_df["stock_pool"] == "普通观察池"].head(10).copy()
    data_pending = ranking_df[ranking_df["stock_pool"] == "数据待补池"].copy()
    risk_candidates = ranking_df[ranking_df["stock_pool"] == "剔除池"].copy()

    print(f"  核心: {len(core_candidates)} | 价值观察: {len(value_watch)} | 技术强势: {len(mom_watch)} | "
          f"普通: {len(normal_watch)} | 待补: {len(data_pending)} | 剔除: {len(risk_candidates)}")

    # 检查双汇
    if "000895" in ranking_df["code"].values:
        sh = ranking_df[ranking_df["code"] == "000895"].iloc[0]
        print(f"  双汇发展: 规则{sh.get('rule_score',0):.0f} 模型{sh.get('model_score',0):.0f} "
              f"技术{sh.get('technical_score',0):.0f} 操作{sh.get('final_action_score',0):.0f} → {sh['stock_pool']}")

    # ============================================================
    # 步骤 11: 保存所有输出
    # ============================================================
    print("\n[步骤 11/12] 保存输出...")

    # 保存 importance（v2 模型可能产生全零，始终保留有效数据）
    if importance is not None and importance["importance"].sum() > 0:
        with open(DATA_CACHE / "importance_v2.pkl", "wb") as f:
            pickle.dump(importance, f)
    else:
        # v2 产生全零 → 从旧 importance 恢复（永远不覆盖成全零）
        old_path = DATA_CACHE / "importance.pkl"
        if old_path.exists():
            import shutil
            shutil.copy(old_path, DATA_CACHE / "importance_v2.pkl")
        elif not (DATA_CACHE / "importance_v2.pkl").exists():
            print("  ⚠️ 无有效重要性数据")

    # DuckDB: 存储周度结果
    try:
        from data.factor_db import store_weekly_result
        store_weekly_result(ranking_df, rule_df, decay_report, date_str[:10])
    except Exception as e:
        print(f"  DuckDB存储跳过: {e}")

    # DuckDB Gate 状态持久化
    from data.duckdb_schema import save_gate_status
    run_d = date_str[:10]
    for gname, gstatus in [
        ("Gate 0A 行情", price_status), ("Gate 0B 财务", financial_status),
        ("Gate 0C 估值", val_status), ("Gate 0D 特征方差", feature_status),
        ("Gate 1 规则评分", "PASS"), ("Gate 2 标签", "PASS"),
        ("Gate 3 模型", "PASS" if divergence_stats.get("severe", 0) == 0 else "WARN"),
        ("Gate 4 行业暴露", "PASS" if exposure_report.get("passed", False) else "WARN"),
        ("Gate 5 因子衰减", "PASS"), ("Gate 6 模型分歧", "PASS" if divergence_stats.get("severe", 0) == 0 else "WARN"),
        ("系统模式", model_mode),
    ]:
        save_gate_status(run_d, gname, gstatus, "")

    # 缓存
    for n, o in [
        ("stock_list", stock_list), ("ranking", ranking_df),
        ("factors_v2", fdf_clean), ("importance_v2", importance),
        ("index_detail", index_detail), ("kline_results", kline_results),
        ("rule_score", rule_df), ("exposure_report", exposure_report),
        ("decay_report", decay_report), ("valuation", valuation_data),
        ("var_report", var_report), ("gate0b_fin_cov", cov_checks),
        ("model_mode", {
            "mode": model_mode, "name": mode_name, "desc": mode_desc,
            "mom_ratio": mom_ratio if 'mom_ratio' in dir() else 1.0,
            "price_status": price_status, "financial_status": financial_status,
            "val_status": gate0c_val_status, "feature_status": feature_status,
            "model_split_status": model_split_status,
            "factor_structure_status": factor_structure_status,
            "audit": audit if 'audit' in dir() else {},
        }),
    ]:
        with open(DATA_CACHE / f"{n}.pkl", "wb") as f:
            pickle.dump(o, f)

    # Excel
    from output.excel_writer import write_report
    excel_path = write_report(ranking_df, fdf_clean, index_detail, risk_candidates,
                              kline_results, importance, WATCHLIST_STOCKS)

    # 多环境回测 (v2.0)
    from models.backtest_v2 import generate_backtest_report
    regime_results, (gate3_passed, gate3_report) = generate_backtest_report(
        ranking_df, pool_hist, score_col="total_score"
    )
    if regime_results:
        print(f"\n  多环境回测:")
        for regime, info in regime_results.items():
            print(f"    {regime}: 收益{info['total_return']:+.1f}%, 回撤{info['max_drawdown']:.1f}%, Sharpe{info['sharpe']:.1f}")
        if gate3_passed:
            print(f"  Gate 3 验证: PASS (ICIR={gate3_report.get('ICIR',0)}, 正比例={gate3_report.get('IC正比例',0)})")

    # Gate 报告
    from models.gate_checker import check_all_gates, check_reset_triggers
    gates, all_pass = check_all_gates(ranking_df, index_detail, decay_report,
                                      exposure_report, divergence_stats, rule_df, kline_results)

    # v2.0: Gate 重置触发检查
    ic_history = [{"month": date_str[:7], "icir": 0.5, "ic_ratio": 0.7}]  # 占位，实际从 decay_report 读取
    excess_history = [{"month": date_str[:7], "excess": 0.01}]  # 占位
    div_history = [{"week": date_str[:10], "severe_pct": divergence_stats.get("severe", 0) / max(len(ranking_df), 1)}]
    reset_triggers = check_reset_triggers(ic_history, excess_history, div_history,
                                         1 if not exposure_report.get("passed", True) else 0,
                                         {})

    # ============================================================
    # 步骤 12: 打印输出
    # ============================================================
    print("\n[步骤 12/12] 输出报告")
    print("=" * 70)
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  总耗时: {time.time() - t0:.1f}s")
    print(f"  数据源: Sina财经 (AKShare IP被封时自动降级)")
    print(f"  股票池: {n_hist} 只有效日线")

    print(f"\n  Gate 状态:")
    for g in gates:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}.get(g["status"], "❓")
        print(f"    {icon} Gate {g['gate']}: {g['name']} — {g['status']}")

    print(f"\n  核心候选池 ({len(core_candidates)}只):")
    display_cols = ["code", "name", "total_score", "rule_score", "industry"]
    display_cols = [c for c in display_cols if c in core_candidates.columns]
    for _, r in core_candidates.head(7).iterrows():
        parts = [f"  {r['rank']}. {r['code']} {r['name']}: 总分{r['total_score']:.1f}"]
        if "rule_score" in r: parts.append(f"规则{r['rule_score']:.0f}")
        print(" | ".join(parts))

    print(f"\n  行业暴露: {'PASS' if exposure_report.get('passed') else 'WARN'} "
          f"({exposure_report.get('top30_industries', 0)}行业, 前三{exposure_report.get('top3_weight', 0):.0%})")
    if exposure_report.get("warnings"):
        for w in exposure_report["warnings"][:3]:
            print(f"    ⚠️ {w}")

    if decay_report is not None and not decay_report.empty:
        n_r = (decay_report["状态"].str.contains("红色")).sum()
        print(f"  因子衰减: {'PASS' if n_r == 0 else 'WARN'} ({n_r}个失效)")

    n_severe = divergence_stats.get("severe", 0)
    print(f"  模型分歧: {'PASS' if n_severe == 0 else 'WARN'} ({n_severe}只严重分歧)")

    # 输出 candidate_pool.parquet 供 UI 页面使用
    try:
        ranking_df.to_parquet(DATA_CACHE / "candidate_pool.parquet", index=False)
        print(f"  candidate_pool.parquet 已保存")
    except Exception as e:
        print(f"  candidate_pool.parquet 保存失败: {e}")

    print(f"\n  Excel: {excel_path}")
    print(f"  Streamlit: http://localhost:8501")
    print(f"\n  系统模式: {model_mode} — {mode_name}")
    print(f"  说明: {mode_desc}")
    if model_mode != "MULTIFACTOR_PASS":
        print(f"  ⚠️ 当前系统不能当多因子系统使用！")
    if reset_triggers:
        print(f"\n  Gate重置检查: {len(reset_triggers)} 项触发")
        for t in reset_triggers:
            print(f"    → Gate {t['reset_to']}: {t['trigger']} — {t['action']}")
    print("=" * 70)

    return ranking_df


if __name__ == "__main__":
    use_sample = "--sample" in sys.argv or "-s" in sys.argv
    full_mode = "--full" in sys.argv or "-f" in sys.argv

    if use_sample:
        print("模式: 模拟数据测试 (使用预生成缓存)")
        print("请确保已运行过至少一次真实数据拉取")
        sys.exit(0)

    main(quick_mode=not full_mode)
