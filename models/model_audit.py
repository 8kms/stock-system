"""
模型训练后审计 — 自动拦截 gain=0、动量垄断、基本面缺失
"""
import pandas as pd

def audit_lgb_importance(model, feature_cols):
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    imp = pd.DataFrame({"feature": feature_cols, "gain": gain, "split": split})
    total_gain = imp["gain"].sum()
    total_split = imp["split"].sum()

    momentum_cols = ["ret_20d","ret_60d","ret_120d","ma60_slope","momentum_score","trend_score","vol_change"]
    fundamental_cols = ["roe_raw","roe","roa","gross_margin_raw","gross_margin","net_margin_raw","net_margin",
                        "quality_score","q_gross_margin","q_cfo","c_cfo_profit","cashflow_score","risk_score","debt_ratio"]
    valuation_cols = ["valuation_score","pe_ttm","pb","v_pe_hist","v_pb_hist","v_div_yield","v_fcf_yield","fcf_yield"]

    momentum_gain = imp.loc[imp["feature"].isin(momentum_cols),"gain"].sum()
    fundamental_gain = imp.loc[imp["feature"].isin(fundamental_cols),"gain"].sum()
    valuation_gain = imp.loc[imp["feature"].isin(valuation_cols),"gain"].sum()

    if total_split == 0 or total_gain == 0:
        status, reason = "FAIL", "LightGBM has no effective split"
    else:
        m_ratio = momentum_gain / total_gain
        f_ratio = fundamental_gain / total_gain
        if m_ratio > 0.50: status, reason = "FAIL", f"Momentum dominance {m_ratio:.0%} > 50%"
        elif m_ratio > 0.35: status, reason = "WARN", f"Momentum {m_ratio:.0%} between 35-50%"
        elif f_ratio < 0.10: status, reason = "WARN", f"Fundamental contribution too low ({f_ratio:.0%})"
        else: status, reason = "PASS", "Model factor structure healthy"

    return {
        "status": status, "reason": reason,
        "total_gain": float(total_gain), "total_split": int(total_split),
        "nonzero_features": int((imp["gain"] > 0).sum()), "feature_count": len(feature_cols),
        "momentum_gain": float(momentum_gain), "fundamental_gain": float(fundamental_gain),
        "valuation_gain": float(valuation_gain),
        "momentum_ratio": float(m_ratio) if total_gain > 0 else 1.0,
        "fundamental_ratio": float(f_ratio) if total_gain > 0 else 0.0,
        "valuation_ratio": float(valuation_gain / total_gain) if total_gain > 0 else 0.0,
        "importance": imp.sort_values("gain", ascending=False),
    }
