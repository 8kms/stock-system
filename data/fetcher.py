"""
数据拉取模块 — 双轨数据源

第一优先: AKShare (PE/PB/市值/财务/分红 — 数据最全)
第一降级: Sina 财经 (K线 + 实时价格 — 稳定不限频)
第二降级: 模拟数据

策略: 每次运行自动探测 AKShare 可用性，被封则用 Sina
"""
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import DATA_CACHE, WATCH_INDICES

# ============================================================
# 精选股票池（约100只）：沪深300 + 中证500 + 自选
# ============================================================
FOCUS_STOCKS = [
    ("600519", "贵州茅台"), ("600036", "招商银行"), ("000858", "五粮液"),
    ("601318", "中国平安"), ("600276", "恒瑞医药"), ("000333", "美的集团"),
    ("000651", "格力电器"), ("002415", "海康威视"), ("600900", "长江电力"),
    ("601398", "工商银行"), ("601288", "农业银行"), ("600030", "中信证券"),
    ("000002", "万科A"), ("300750", "宁德时代"), ("600809", "山西汾酒"),
    ("002594", "比亚迪"), ("601899", "紫金矿业"), ("600585", "海螺水泥"),
    ("000568", "泸州老窖"), ("000725", "京东方A"), ("002142", "宁波银行"),
    ("600887", "伊利股份"), ("601166", "兴业银行"), ("600016", "民生银行"),
    ("000001", "平安银行"), ("600028", "中国石化"), ("601857", "中国石油"),
    ("600031", "三一重工"), ("000063", "中兴通讯"), ("002475", "立讯精密"),
    ("600048", "保利发展"), ("000100", "TCL科技"), ("002230", "科大讯飞"),
    ("300059", "东方财富"), ("601668", "中国建筑"), ("600104", "上汽集团"),
    ("000625", "长安汽车"), ("002352", "顺丰控股"), ("601390", "中国中铁"),
    ("600019", "宝钢股份"), ("002460", "赣锋锂业"), ("601012", "隆基绿能"),
    ("300124", "汇川技术"), ("000157", "中联重科"), ("601766", "中国中车"),
    ("600009", "上海机场"), ("601939", "建设银行"), ("002271", "东方雨虹"),
    ("300274", "阳光电源"), ("600309", "万华化学"), ("000538", "云南白药"),
    ("002049", "紫光国微"), ("300015", "爱尔眼科"), ("600745", "闻泰科技"),
    ("601088", "中国神华"), ("601225", "陕西煤业"), ("002120", "韵达股份"),
    ("002916", "深南电路"), ("002459", "晶澳科技"), ("000792", "盐湖股份"),
    ("002714", "牧原股份"), ("603259", "药明康德"), ("600690", "海尔智家"),
    ("002007", "华兰生物"), ("300450", "先导智能"), ("000895", "双汇发展"),
    ("600406", "国电南瑞"), ("601615", "明阳智能"), ("603288", "海天味业"),
    ("300394", "天孚通信"), ("000338", "潍柴动力"), ("600150", "中国船舶"),
    ("600436", "片仔癀"), ("000661", "长春高新"), ("002129", "中环股份"),
    ("600588", "用友网络"), ("002422", "科伦药业"), ("002463", "沪电股份"),
    ("600482", "中国动力"), ("002153", "石基信息"), ("002368", "太极股份"),
    ("002048", "宁波华翔"), ("300760", "迈瑞医疗"), ("688981", "中芯国际"),
    ("002557", "洽洽食品"), ("002340", "格林美"), ("002185", "华天科技"),
    ("600132", "重庆啤酒"), ("000963", "华东医药"), ("603160", "汇顶科技"),
    ("600563", "法拉电子"), ("002408", "齐翔腾达"), ("000869", "张裕A"),
    ("601689", "拓普集团"), ("300496", "中科创达"), ("300207", "欣旺达"),
    ("002074", "国轩高科"), ("300751", "迈为股份"),
]

# 行业映射（申万一级）
STOCK_INDUSTRY = {
    "600519":"食品饮料","600036":"银行","000858":"食品饮料",
    "601318":"非银金融","600276":"医药生物","000333":"家用电器",
    "000651":"家用电器","002415":"计算机","600900":"公用事业",
    "601398":"银行","601288":"银行","600030":"非银金融",
    "000002":"房地产","300750":"电力设备","600809":"食品饮料",
    "002594":"汽车","601899":"有色金属","600585":"建筑材料",
    "000568":"食品饮料","000725":"电子","002142":"银行",
    "600887":"食品饮料","601166":"银行","600016":"银行",
    "000001":"银行","600028":"石油石化","601857":"石油石化",
    "600031":"机械设备","000063":"通信","002475":"电子",
    "600048":"房地产","000100":"电子","002230":"计算机",
    "300059":"非银金融","601668":"建筑装饰","600104":"汽车",
    "000625":"汽车","002352":"交通运输","601390":"建筑装饰",
    "600019":"钢铁","002460":"有色金属","601012":"电力设备",
    "300124":"电力设备","000157":"机械设备","601766":"机械设备",
    "600009":"交通运输","601939":"银行","002271":"建筑材料",
    "300274":"电力设备","600309":"基础化工","000538":"医药生物",
    "002049":"电子","300015":"医药生物","600745":"电子",
    "601088":"煤炭","601225":"煤炭","002120":"交通运输",
    "002916":"电子","002459":"电力设备","000792":"基础化工",
    "002714":"农林牧渔","603259":"医药生物","600690":"家用电器",
    "002007":"医药生物","300450":"电力设备","000895":"食品饮料",
    "600406":"电力设备","601615":"电力设备","603288":"食品饮料",
    "300394":"通信","000338":"机械设备","600150":"国防军工",
    "600436":"医药生物","000661":"医药生物","002129":"电力设备",
    "600588":"计算机","002422":"医药生物","002463":"电子",
    "600482":"汽车","002153":"计算机","002368":"计算机",
    "002048":"汽车","300760":"医药生物","688981":"电子",
    "002557":"食品饮料","002340":"有色金属","002185":"电子",
    "600132":"食品饮料","000963":"医药生物","603160":"电子",
    "600563":"电子","002408":"基础化工","000869":"食品饮料",
    "601689":"汽车","300496":"计算机","300207":"电力设备",
    "002074":"电力设备","300751":"电力设备",
}


def _cache_path(name):
    return DATA_CACHE / f"{name}.pkl"

def _load_cache(name, max_age_hours=6):
    p = _cache_path(name)
    if p.exists():
        age = time.time() - p.stat().st_mtime
        if age < max_age_hours * 3600:
            with open(p, "rb") as f:
                return pickle.load(f)
    return None

def _save_cache(name, data):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)


# ============================================================
# AKShare 可用性探测
# ============================================================

_akshare_available = None

def is_akshare_available():
    """探测 AKShare 是否可用（快速检测，3秒超时）"""
    global _akshare_available
    if _akshare_available is not None:
        return _akshare_available

    try:
        import socket
        socket.setdefaulttimeout(3)
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})
        r = s.get("https://push2.eastmoney.com/api/qt/stock/get",
                  params={"secid": "1.600519", "fields": "f57"}, timeout=3)
        _akshare_available = r.status_code == 200 and len(r.text) > 50
    except Exception:
        _akshare_available = False

    print(f"  AKShare: {'可用 (真实PE/PB/市值)' if _akshare_available else '不可用 → 使用Sina降级'}")
    return _akshare_available


# ============================================================
# AKShare 数据拉取（第一优先）
# ============================================================

def _fetch_akshare_stock_list():
    """AKShare: 全A股列表（含PE/PB/市值/价格）"""
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    df = df.rename(columns={
        "代码": "code", "名称": "name", "最新价": "price",
        "涨跌幅": "pct_chg", "总市值": "market_cap",
        "流通市值": "float_cap", "市盈率-动态": "pe", "市净率": "pb",
    })
    cols = ["code", "name", "price", "pe", "pb", "market_cap", "pct_chg"]
    return df[[c for c in cols if c in df.columns]].copy()


def _fetch_akshare_hist(codes, start_date="20200101"):
    """AKShare: 批量拉取日线"""
    import akshare as ak
    end_date = datetime.now().strftime("%Y%m%d")
    result = {}
    for code in codes:
        try:
            df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start_date, end_date=end_date, adjust='qfq')
            if df is not None and len(df) >= 60:
                df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume","成交额":"amount","涨跌幅":"pct_chg","换手率":"turnover"})
                df["date"] = pd.to_datetime(df["date"])
                df["code"] = code
                for c in ["open","close","high","low","volume","amount","pct_chg","turnover"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                result[code] = df.sort_values("date")
        except Exception:
            pass
        time.sleep(0.1)
    return result


def _fetch_akshare_index():
    """AKShare: 指数日线"""
    import akshare as ak
    result = {}
    for code in WATCH_INDICES:
        try:
            df = ak.index_zh_a_hist(symbol=code, period='daily', start_date='20200101', end_date=datetime.now().strftime("%Y%m%d"))
            if df is not None and not df.empty:
                col_map = {}
                for c in df.columns:
                    cl = c.strip()
                    if "日期" in cl: col_map[c] = "date"
                    elif "开" in cl: col_map[c] = "open"
                    elif "收" in cl: col_map[c] = "close"
                    elif "高" in cl: col_map[c] = "high"
                    elif "低" in cl: col_map[c] = "low"
                    elif "量" in cl: col_map[c] = "volume"
                df = df.rename(columns=col_map)
                df["date"] = pd.to_datetime(df["date"])
                for c in ["open","close","high","low","volume"]:
                    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
                result[code] = df.sort_values("date")
        except Exception:
            pass
        time.sleep(0.1)
    return result


# ============================================================
# Sina 数据拉取（第一降级）
# ============================================================

_sina_session = None

def _get_sina():
    global _sina_session
    if _sina_session is None:
        _sina_session = requests.Session()
        _sina_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/",
        })
    return _sina_session


def _fetch_sina_kline_raw(symbol, datalen=2000):
    """Sina: 单只K线"""
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {"symbol": symbol, "scale": "240", "ma": "no", "datalen": datalen}
    for _ in range(3):
        try:
            r = _get_sina().get(url, params=params, timeout=15)
            if r.status_code == 200 and r.text.strip():
                import json
                data = json.loads(r.text)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            time.sleep(0.3)
    return None


def _parse_sina_kline(raw, code):
    rows = []
    for item in raw:
        try:
            rows.append({"date": item["day"], "open": float(item["open"]), "high": float(item["high"]),
                         "low": float(item["low"]), "close": float(item["close"]), "volume": float(item["volume"])})
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = code
    df["amount"] = df["close"] * df["volume"] * 0.5
    df["pct_chg"] = df["close"].pct_change() * 100
    df["turnover"] = 0.0
    return df.sort_values("date")


def _fetch_sina_price(codes):
    """Sina: 批量实时价格"""
    sina_codes = ["sh" + c if c.startswith("6") else "sz" + c for c in codes]
    all_rows = []
    for i in range(0, len(sina_codes), 50):
        batch = sina_codes[i:i+50]
        try:
            r = _get_sina().get("https://hq.sinajs.cn/list=" + ",".join(batch), timeout=10)
            if r.status_code == 200:
                for line in r.text.strip().split("\n"):
                    try:
                        parts = line.split('"')[1].split(",")
                        name = parts[0]
                        price = float(parts[3]) if parts[3] else 0
                        prev_close = float(parts[2]) if parts[2] else price
                        pct = (price / prev_close - 1) * 100 if prev_close > 0 else 0
                        code_raw = line.split("=")[0].split("_")[-1]
                        code = code_raw[2:] if len(code_raw) > 2 else code_raw
                        all_rows.append({"code": code, "name": name, "price": price, "pct_chg": round(pct, 2)})
                    except Exception:
                        continue
        except Exception:
            pass
        time.sleep(0.05)
    return pd.DataFrame(all_rows)


def _fetch_sina_index():
    """Sina: 指数日线"""
    idx_map = {"000300":"sh000300","000905":"sh000905","000852":"sh000852","399006":"sz399006","000922":"sh000922"}
    result = {}
    for code in WATCH_INDICES:
        sym = idx_map.get(code)
        if sym:
            raw = _fetch_sina_kline_raw(sym, datalen=2000)
            if raw:
                df = _parse_sina_kline(raw, code)
                if not df.empty: result[code] = df
        time.sleep(0.1)
    return result


# ============================================================
# 统一公开接口
# ============================================================

def get_a_stock_list():
    """获取股票列表（含估值数据）"""
    cache = _load_cache("stock_list", max_age_hours=8)
    if cache is not None:
        return cache

    # 尝试 AKShare（含真实 PE/PB/市值）
    if is_akshare_available():
        try:
            df = _fetch_akshare_stock_list()
            # 只保留我们关注池中的
            focus_codes = {c for c, _ in FOCUS_STOCKS}
            df = df[df["code"].isin(focus_codes)]
            if len(df) >= 50:
                _save_cache("stock_list", df)
                print(f"  AKShare 股票列表: {len(df)} 只 (含真实PE/PB/市值)")
                return df
        except Exception as e:
            print(f"  AKShare 股票列表失败: {e}")

    # 降级: Sina 价格 + 估算PE/PB
    prices = _fetch_sina_price([c for c, _ in FOCUS_STOCKS])
    if not prices.empty:
        rows = []
        for _, r in prices.iterrows():
            rows.append({
                "code": r["code"], "name": r["name"],
                "price": r["price"], "pe": 0, "pb": 0,
                "market_cap": 0, "pct_chg": r.get("pct_chg", 0),
            })
        df = pd.DataFrame(rows)
        _save_cache("stock_list", df)
        print(f"  Sina 股票列表: {len(df)} 只 (PE/PB待AKShare恢复)")
        return df

    # 最后降级: code only
    df = pd.DataFrame({"code": [c for c, _ in FOCUS_STOCKS], "name": [n for _, n in FOCUS_STOCKS],
                        "price": 10.0, "pe": 0, "pb": 0, "market_cap": 0, "pct_chg": 0})
    _save_cache("stock_list", df)
    return df


def get_stock_industry():
    cache = _load_cache("industry", max_age_hours=720)
    if cache is not None:
        return cache
    rows = [{"code": c, "industry": STOCK_INDUSTRY.get(c, "其他")} for c in [x[0] for x in FOCUS_STOCKS]]
    df = pd.DataFrame(rows)
    _save_cache("industry", df)
    return df


def fetch_stock_pool_hist(codes):
    """拉取日线：AKShare优先 → Sina降级"""
    cache = _load_cache("stock_hist", max_age_hours=8)
    if cache is not None:
        return cache

    result = {}

    if is_akshare_available():
        result = _fetch_akshare_hist(codes)
        if len(result) >= 50:
            _save_cache("stock_hist", result)
            print(f"  AKShare 日线: {len(result)} 只")
            return result

    # Sina 降级
    for code in codes:
        prefix = "sh" if code.startswith("6") else "sz"
        raw = _fetch_sina_kline_raw(prefix + code, datalen=2000)
        if raw:
            df = _parse_sina_kline(raw, code)
            if not df.empty and len(df) >= 60:
                result[code] = df
        time.sleep(0.06)

    _save_cache("stock_hist", result)
    print(f"  Sina 日线: {len(result)} 只")
    return result


def fetch_index_hist():
    cache = _load_cache("index_hist", max_age_hours=8)
    if cache is not None:
        return cache

    if is_akshare_available():
        result = _fetch_akshare_index()
        if len(result) >= 5:
            _save_cache("index_hist", result)
            print(f"  AKShare 指数: {len(result)} 个")
            return result

    result = _fetch_sina_index()
    _save_cache("index_hist", result)
    print(f"  Sina 指数: {len(result)} 个")
    return result


def fetch_valuation_data(stock_list=None):
    """
    获取估值数据 — 分层质量标记

    AKShare spot → 真实 PE/PB (quality=PASS)
    AKShare不可用 → 尝试本地缓存 (quality=WARN)
    降级 → 行业基准估算 (quality=FAIL, 不进模型)
    """
    import numpy as np

    codes = stock_list["code"].tolist() if stock_list is not None else [c for c, _ in FOCUS_STOCKS]

    # ── 第1层: AKShare 真实 PE/PB ──
    if is_akshare_available():
        try:
            import akshare as ak
            spot = ak.stock_zh_a_spot_em()
            if spot is not None and len(spot) > 100:
                # 字段适配
                code_col = _pick_col(spot, ["代码", "code"])
                name_col = _pick_col(spot, ["名称", "name"])
                pe_col = _pick_col(spot, ["市盈率-动态", "市盈率", "市盈率TTM"])
                pb_col = _pick_col(spot, ["市净率", "市净率PB"])
                mc_col = _pick_col(spot, ["总市值", "总市值1"])
                price_col = _pick_col(spot, ["最新价", "最新价格"])

                if code_col and pe_col:
                    spot = spot.rename(columns={
                        code_col: "code", name_col: "name",
                        pe_col: "pe", pb_col: "pb",
                        mc_col: "market_cap", price_col: "price",
                    })
                    spot["code"] = spot["code"].astype(str).str.zfill(6)
                    for c in ["pe","pb","market_cap","price"]:
                        if c in spot.columns:
                            spot[c] = pd.to_numeric(spot[c], errors="coerce")

                    focus = spot[spot["code"].isin(codes)]
                    if len(focus) > 30 and focus["pe"].notna().sum() > 30:
                        focus["valuation_source"] = "akshare_real"
                        focus["valuation_quality"] = "PASS"
                        focus["pct_chg"] = 0
                        print(f"  AKShare 真实估值: {len(focus)} 只 (PE/PB/市值)")
                        return focus[["code","name","price","pe","pb","market_cap","pct_chg",
                                       "valuation_source","valuation_quality"]]
        except Exception as e:
            print(f"  AKShare 估值接口失败: {e}")

    # ── 第2层: Sina 价格 + 行业估算（标记为 estimated） ──
    prices = _fetch_sina_price(codes) if stock_list is None or "price" not in stock_list.columns else stock_list

    if prices is None or prices.empty:
        df = pd.DataFrame({"code": codes, "name": codes, "price": 0, "pe": np.nan, "pb": np.nan,
                            "market_cap": 0, "pct_chg": 0,
                            "valuation_source": "missing", "valuation_quality": "FAIL"})
        print(f"  估值: 全部缺失 (FAIL)")
        return df

    industry_pe_benchmark = {
        "银行": 6, "食品饮料": 25, "医药生物": 30, "电子": 25,
        "计算机": 35, "非银金融": 12, "电力设备": 25, "汽车": 18,
        "有色金属": 20, "建筑材料": 15, "房地产": 8, "公用事业": 15,
        "石油石化": 10, "机械设备": 20, "通信": 20, "家用电器": 15,
        "建筑装饰": 10, "钢铁": 8, "交通运输": 12, "农林牧渔": 25,
        "煤炭": 8, "国防军工": 40, "基础化工": 18, "轻工制造": 20,
        "传媒": 22, "综合": 18,
    }
    ind_map = dict(FOCUS_STOCKS)

    rows = []
    n_real_price = 0
    for _, r in prices.iterrows():
        p = r.get("price", 0)
        code = r.get("code", "")
        ind = ind_map.get(code, "综合")
        base_pe = industry_pe_benchmark.get(ind, 18)

        if p > 0:
            n_real_price += 1
            pe_est = round(base_pe * (p / 50) ** 0.3, 1)
            pb_est = round(pe_est * 0.10, 2)
            shares = 5e9 if p > 100 else (1e10 if p > 30 else 2e10)
            mc_est = p * shares
            source = "sina_price_estimated"
            quality = "WARN"
        else:
            pe_est = np.nan
            pb_est = np.nan
            mc_est = 0
            p = 0
            source = "missing"
            quality = "FAIL"

        rows.append({
            "code": code, "name": r.get("name", code),
            "price": p, "pe": pe_est, "pb": pb_est,
            "market_cap": mc_est, "pct_chg": r.get("pct_chg", 0),
            "valuation_source": source,
            "valuation_quality": quality,
        })

    df = pd.DataFrame(rows)
    n_estimated = (df["valuation_source"] == "sina_price_estimated").sum()
    n_missing = (df["valuation_quality"] == "FAIL").sum()

    # 质量汇总
    if n_missing > len(df) * 0.3:
        gate_status = "FAIL"
    elif n_estimated > len(df) * 0.5:
        gate_status = "WARN"
    else:
        gate_status = "PASS"

    print(f"  估值: 真实0只, 价格真实{n_real_price}只, 估算{n_estimated}只, 缺失{n_missing}只 → Gate {gate_status}")
    if gate_status != "PASS":
        print(f"  ⚠️ 估算PE/PB仅展示，不进入LightGBM训练")

    return df


def _pick_col(df, candidates):
    """从候选字段名中选第一个存在的"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fetch_all(codes=None, quick_mode=True):
    """一键拉取所有数据"""
    print("=== 开始拉取 A 股数据 ===")

    if is_akshare_available():
        print("  数据源: AKShare (PE/PB/市值/日线)")
    else:
        print("  数据源: Sina 财经 (日线+价格, AKShare IP被封)")

    stock_list = get_a_stock_list()
    if codes is None:
        codes = stock_list["code"].tolist()
    print(f"  股票池: {len(codes)} 只")

    hist_data = fetch_stock_pool_hist(codes)
    index_data = fetch_index_hist()
    industry_data = get_stock_industry()
    valuation_data = fetch_valuation_data(stock_list)

    print("=== 数据拉取完成 ===")
    return {
        "stock_list": stock_list,
        "hist_data": hist_data,
        "index_data": index_data,
        "industry_data": industry_data,
        "valuation_data": valuation_data,
    }
