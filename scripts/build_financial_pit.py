#!/usr/bin/env python3
"""合并所有 parquet → DuckDB financial_pit"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from pathlib import Path
from data.duckdb_schema import get_db

SRC_DIR = Path("data_cache/tushare/fina_indicator")

def build():
    files = list(SRC_DIR.glob("*.parquet"))
    if not files: raise RuntimeError("No fina_indicator parquet files")
    dfs = [pd.read_parquet(p) for p in files]
    raw = pd.concat(dfs, ignore_index=True)

    raw["code"] = raw["code"].astype(str).str[:6].str.zfill(6)
    raw["ann_date"] = pd.to_datetime(raw["ann_date"], errors="coerce")
    raw["end_date"] = pd.to_datetime(raw["end_date"], errors="coerce")
    raw["effective_date"] = raw["ann_date"]
    raw["report_period"] = raw["end_date"]

    out = pd.DataFrame()
    out["code"] = raw["code"]
    out["report_period"] = raw["report_period"]
    out["ann_date"] = raw["ann_date"]
    out["effective_date"] = raw["effective_date"]
    out["roe"] = pd.to_numeric(raw.get("roe"), errors="coerce")
    out["gross_margin"] = pd.to_numeric(raw.get("grossprofit_margin"), errors="coerce")
    out["net_margin"] = pd.to_numeric(raw.get("netprofit_margin"), errors="coerce")
    out["roa"] = pd.to_numeric(raw.get("roa"), errors="coerce")
    out["debt_ratio"] = pd.to_numeric(raw.get("debt_to_assets"), errors="coerce") if "debt_to_assets" in raw.columns else pd.NA
    out["cfo_to_profit"] = pd.to_numeric(raw.get("ocf_to_profit"), errors="coerce") if "ocf_to_profit" in raw.columns else pd.NA
    out["revenue_yoy"] = pd.to_numeric(raw.get("or_yoy"), errors="coerce")
    out["profit_yoy"] = pd.to_numeric(raw.get("netprofit_yoy"), errors="coerce")
    out["source"] = "tushare_fina_indicator"
    out["data_quality"] = "PASS"
    out["updated_at"] = pd.Timestamp.now()

    core = ["roe","gross_margin","net_margin"]
    miss = out[core].isna().mean(axis=1)
    out.loc[miss > 0.34, "data_quality"] = "WARN"
    out.loc[miss > 0.67, "data_quality"] = "FAIL"

    db = get_db()
    db.execute("DELETE FROM financial_pit WHERE source = 'tushare_fina_indicator'")
    db.register("out", out)
    db.execute("INSERT INTO financial_pit SELECT * FROM out")
    db.close()
    print(f"financial_pit: {len(out)} rows, {out['code'].nunique()} codes")
    print(out["data_quality"].value_counts().to_string())

if __name__ == "__main__": build()
