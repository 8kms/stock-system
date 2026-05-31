#!/usr/bin/env python3
"""后台批量拉取 Tushare 财务数据（每90秒一只，约2.5小时完成）"""
import os, sys, pickle, time
os.environ["TUSHARE_TOKEN"] = "977d313022aa6c0ece3fcd34b10a5d54e13b921de5de681514712082"
sys.path.insert(0, '/Users/sun/stock-system')
from config import DATA_CACHE
from data.tushare_provider import TushareProvider
import pandas as pd

ts = TushareProvider(sleep_sec=1.0)
stock_list = pickle.load(open(DATA_CACHE/'stock_list.pkl','rb'))
codes = stock_list['code'].tolist()

# Load already-fetched
done_codes = set()
all_records = []
try:
    existing = pickle.load(open(DATA_CACHE/'tushare_financials.pkl','rb'))
    done_codes = set(existing['code'].unique())
    all_records.append(existing)
except:
    pass

ok = len(done_codes)
total = len(codes)
WAIT = 90  # 90秒安全间隔
print(f'Starting: {ok}/{total} done, {total-ok} remaining ({WAIT}s interval)', flush=True)

for i, code in enumerate(codes):
    if code in done_codes:
        continue
    ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
    remaining = total - ok
    eta_min = remaining * WAIT / 60
    print(f'[{ok+1}/{total}] {code} {ts_code}: ETA {eta_min:.0f}min...', flush=True)
    
    for retry in range(3):
        try:
            time.sleep(WAIT)
            df = ts.fina_indicator(ts_code, force=True)
            if df is not None and len(df) > 0:
                df['code'] = code
                all_records.append(df)
                ok += 1
                roe = float(df['roe'].dropna().iloc[-1]) if len(df['roe'].dropna()) > 0 else '?'
                print(f'  OK: ROE={roe}  ({ok}/{total})', flush=True)
                break
        except Exception as e:
            err = str(e)[:80]
            if '频率' in err or '超限' in err:
                extra = 30 * (retry + 1)
                print(f'  限频, 额外等待{extra}s...', flush=True)
                time.sleep(extra)
            else:
                print(f'  SKIP: {err}', flush=True)
                break

    # Save every 3 new stocks
    if ok % 3 == 0 and len(all_records) > 0:
        combined = pd.concat(all_records, ignore_index=True)
        latest = combined.sort_values('ann_date').groupby('code').tail(1)
        with open(DATA_CACHE/'tushare_financials.pkl','wb') as f: pickle.dump(combined, f)
        with open(DATA_CACHE/'tushare_latest.pkl','wb') as f: pickle.dump(latest, f)
        print(f'  [SAVED] {ok}/{total}', flush=True)

# Final save
if len(all_records) > 0:
    combined = pd.concat(all_records, ignore_index=True)
    latest = combined.sort_values('ann_date').groupby('code').tail(1)
    with open(DATA_CACHE/'tushare_financials.pkl','wb') as f: pickle.dump(combined, f)
    with open(DATA_CACHE/'tushare_latest.pkl','wb') as f: pickle.dump(latest, f)
    print(f'\nDONE: {ok}/{total} stocks saved!', flush=True)
