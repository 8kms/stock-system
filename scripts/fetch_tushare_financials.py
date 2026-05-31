#!/usr/bin/env python3
"""Tushare 财务数据拉取 — 单票 parquet 落盘 + 断点续跑"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["TUSHARE_TOKEN"] = "977d313022aa6c0ece3fcd34b10a5d54e13b921de5de681514712082"

import pandas as pd
from pathlib import Path
from data.tushare_provider import TushareProvider

CACHE_DIR = Path("data_cache/tushare/fina_indicator")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def normalize_code(code):
    code = str(code)
    if code.endswith((".SH", ".SZ")): return code
    return code + ".SH" if code.startswith("6") else code + ".SZ"

def fetch_one(ts, code, force=False):
    ts_code = normalize_code(code)
    out_path = CACHE_DIR / f"{ts_code[:6]}.parquet"
    if out_path.exists() and not force: return "SKIP"
    try:
        df = ts.fina_indicator(ts_code=ts_code, force=force)
        if df is None or df.empty: raise ValueError("empty")
        df["code"] = ts_code[:6]
        df.to_parquet(out_path, index=False)
        return "OK"
    except Exception as e: return f"FAIL:{e}"

def main():
    from data.fetcher import get_a_stock_list
    stock_list = get_a_stock_list()
    codes = stock_list["code"].tolist()
    ts = TushareProvider(sleep_sec=1.0)
    failed = []
    for i, code in enumerate(codes, 1):
        status = fetch_one(ts, code)
        print(f"[{i}/{len(codes)}] {code}: {status}")
        if status.startswith("FAIL"): failed.append(str(code))
        if i < len(codes):
            wait = 90 + random.randint(0, 15)
            print(f"  sleep {wait}s")
            time.sleep(wait)
    if failed:
        pd.DataFrame({"code": failed}).to_csv("data_cache/tushare/failed_fina.csv", index=False)
    print(f"Done: {len(codes)-len(failed)}/{len(codes)}")

if __name__ == "__main__": main()
