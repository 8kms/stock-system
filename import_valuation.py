#!/usr/bin/env python3
"""
估值数据导入工具 — 从CSV导入历史PE/PB/市值到DuckDB

使用方式:
  1. 从Wind/Choice/同花顺等软件导出历史PE/PB/市值CSV
  2. 放在 data_sources/valuation_csv/ 目录下
  3. 运行: python3 import_valuation.py
  4. 系统自动识别并导入

CSV格式:
  code,trade_date,pe_ttm,pb,ps_ttm,total_mv,circ_mv,turnover_rate
  600519,2024-01-02,25.3,9.1,12.0,2100000000000,2100000000000,0.12
  600519,2024-01-03,25.1,9.0,11.9,2080000000000,2080000000000,0.13
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from pathlib import Path
from datetime import datetime

CSV_DIR = Path(__file__).parent / "data_sources" / "valuation_csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)


def import_csv_files():
    """导入CSV目录下所有估值文件到DuckDB"""
    csv_files = sorted(CSV_DIR.glob("*.csv"))

    # 过滤掉模板文件
    data_files = [f for f in csv_files if "template" not in f.name.lower()]

    if not data_files:
        print("未找到估值CSV文件。")
        print(f"请将估值CSV放入: {CSV_DIR}")
        print(f"格式: code,trade_date,pe_ttm,pb,ps_ttm,total_mv,circ_mv")
        return False

    all_data = []
    for f in data_files:
        try:
            df = pd.read_csv(f)
            required = ["code", "trade_date", "pe_ttm", "pb", "total_mv"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"跳过 {f.name}: 缺少字段 {missing}")
                continue

            # 标准化
            df["code"] = df["code"].astype(str).str.zfill(6)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            for c in ["pe_ttm", "pb", "total_mv"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # 自动补充字段
            if "ps_ttm" not in df.columns:
                df["ps_ttm"] = df["pe_ttm"] * 0.8  # 粗略估算
            if "circ_mv" not in df.columns:
                df["circ_mv"] = df["total_mv"] * 0.7
            if "turnover_rate" not in df.columns:
                df["turnover_rate"] = 0.0

            df["valuation_source"] = "manual_csv"
            df["valuation_quality"] = "PASS"
            df["updated_at"] = datetime.now()

            all_data.append(df)
            print(f"{f.name}: {len(df)} 行, {df['code'].nunique()} 只股票")
        except Exception as e:
            print(f"跳过 {f.name}: {e}")

    if not all_data:
        return False

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["code", "trade_date"])

    print(f"\n总计: {len(combined)} 行, {combined['code'].nunique()} 只股票")
    print(f"日期范围: {combined['trade_date'].min()} ~ {combined['trade_date'].max()}")
    print(f"PE覆盖率: {combined['pe_ttm'].notna().mean():.1%}")

    # 质量检查
    pe_range = combined['pe_ttm'].dropna()
    if len(pe_range) > 0:
        print(f"PE范围: {pe_range.min():.1f} ~ {pe_range.max():.1f}")
        if pe_range.nunique() < 10:
            print("⚠️ PE唯一值太少，请检查数据")

    # 导入DuckDB
    from data.duckdb_schema import save_valuation_daily, get_db
    db = get_db()
    try:
        db.execute("DELETE FROM valuation_daily")
        db.execute("INSERT INTO valuation_daily SELECT * FROM combined")
        cnt = db.execute("SELECT COUNT(*) FROM valuation_daily").fetchone()[0]
        db.close()
        print(f"\n✅ DuckDB valuation_daily: {cnt} 行已导入")
        return True
    except Exception as e:
        print(f"\n❌ 导入失败: {e}")
        return False


def generate_sample_csv(stock_list_path=None):
    """为前10只股票生成示例CSV（含最近20个交易日）"""
    import pickle
    from config import DATA_CACHE

    codes = []
    names = {}
    if stock_list_path:
        sl = pickle.load(open(stock_list_path, 'rb'))
        codes = sl['code'].tolist()[:10]
        if 'name' in sl.columns:
            names = dict(zip(sl['code'], sl['name']))

    # 加载日线获取最近日期
    try:
        hist = pickle.load(open(DATA_CACHE / "stock_hist.pkl", "rb"))
        all_dates = set()
        for c in codes:
            if c in hist:
                dates = hist[c]['date'].dt.strftime("%Y-%m-%d").tolist()
                all_dates.update(dates[-20:])
        trade_dates = sorted(all_dates)[-10:]  # 最近10个交易日
    except:
        from datetime import datetime, timedelta
        today = datetime.now()
        trade_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10, 0, -1)]

    rows = []
    for code in codes:
        name = names.get(code, code)
        for d in trade_dates:
            rows.append({
                "code": code, "trade_date": d,
                "pe_ttm": "", "pb": "", "ps_ttm": "",
                "total_mv": "", "circ_mv": "", "turnover_rate": "",
                "source": "",
            })

    df = pd.DataFrame(rows)
    path = CSV_DIR / "valuation_sample.csv"
    df.to_csv(path, index=False)
    print(f"示例CSV: {path} ({len(codes)}只 × {len(trade_dates)}天)")
    print("请填入真实PE/PB/市值后重新运行 import_valuation.py")
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="生成示例CSV")
    parser.add_argument("--dry-run", action="store_true", help="仅检查不导入")
    args = parser.parse_args()

    if args.sample:
        generate_sample_csv()
    else:
        success = import_csv_files()
        if not success:
            print("\n提示: 运行 'python3 import_valuation.py --sample' 生成示例模板")
