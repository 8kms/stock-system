"""
DuckDB 本地数据湖 — 完整表结构

表:
  price_daily, index_daily, financial_pit, valuation_daily,
  factor_panel, gate_status
"""
from pathlib import Path
import duckdb

DB_PATH = Path(__file__).parent.parent / "data" / "stock_system.duckdb"


def get_db():
    return duckdb.connect(str(DB_PATH))


def init_schema():
    """创建所有表"""
    db = get_db()

    db.execute("""
    CREATE TABLE IF NOT EXISTS price_daily (
        code VARCHAR,
        trade_date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        amount DOUBLE,
        source VARCHAR,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (code, trade_date)
    )""")

    db.execute("""
    CREATE TABLE IF NOT EXISTS index_daily (
        index_code VARCHAR,
        trade_date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        amount DOUBLE,
        source VARCHAR,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (index_code, trade_date)
    )""")

    db.execute("""
    CREATE TABLE IF NOT EXISTS financial_pit (
        code VARCHAR,
        report_period DATE,
        ann_date DATE,
        effective_date DATE,
        roe DOUBLE,
        gross_margin DOUBLE,
        net_margin DOUBLE,
        roa DOUBLE,
        debt_ratio DOUBLE,
        goodwill_to_equity DOUBLE,
        cfo_to_profit DOUBLE,
        revenue_yoy DOUBLE,
        profit_yoy DOUBLE,
        source VARCHAR,
        data_quality VARCHAR DEFAULT 'UNKNOWN',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (code, report_period)
    )""")

    db.execute("""
    CREATE TABLE IF NOT EXISTS valuation_daily (
        code VARCHAR,
        trade_date DATE,
        pe_ttm DOUBLE,
        pb DOUBLE,
        ps_ttm DOUBLE,
        total_mv DOUBLE,
        circ_mv DOUBLE,
        turnover_rate DOUBLE,
        valuation_source VARCHAR,
        valuation_quality VARCHAR DEFAULT 'UNKNOWN',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (code, trade_date)
    )""")

    db.execute("""
    CREATE TABLE IF NOT EXISTS factor_panel (
        date DATE,
        code VARCHAR,
        industry VARCHAR,
        quality_score DOUBLE,
        valuation_score DOUBLE,
        cashflow_score DOUBLE,
        growth_score DOUBLE,
        risk_score DOUBLE,
        momentum_score DOUBLE,
        rule_score DOUBLE,
        data_quality_flag VARCHAR DEFAULT 'PASS',
        valuation_quality VARCHAR DEFAULT 'UNKNOWN',
        model_mode VARCHAR DEFAULT 'UNKNOWN',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (date, code)
    )""")

    db.execute("""
    CREATE TABLE IF NOT EXISTS gate_status (
        run_date DATE,
        gate_name VARCHAR,
        status VARCHAR,
        message VARCHAR,
        metrics_json VARCHAR,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_date, gate_name)
    )""")

    db.close()
    print("DuckDB 数据湖已初始化")


def save_price_daily(df):
    """保存日线行情"""
    db = get_db()
    db.execute("DELETE FROM price_daily WHERE (code, trade_date) IN (SELECT code, trade_date FROM df)")
    db.execute("INSERT INTO price_daily SELECT * FROM df")
    db.close()


def save_valuation_daily(df):
    """保存估值日数据"""
    db = get_db()
    try:
        db.execute("DELETE FROM valuation_daily WHERE (code, trade_date) IN (SELECT code, trade_date FROM df)")
        db.execute("INSERT INTO valuation_daily SELECT * FROM df")
    except Exception:
        pass
    db.close()


def load_valuation_daily(codes, start_date, end_date):
    """加载估值数据"""
    db = get_db()
    code_list = "','".join(codes)
    df = db.execute(f"""
        SELECT * FROM valuation_daily
        WHERE code IN ('{code_list}')
        AND trade_date BETWEEN '{start_date}' AND '{end_date}'
    """).df()
    db.close()
    return df


def save_financial_pit(df):
    """保存PIT财务数据"""
    db = get_db()
    try:
        db.execute("DELETE FROM financial_pit WHERE (code, report_period) IN (SELECT code, report_period FROM df)")
        db.execute("INSERT INTO financial_pit SELECT * FROM df")
    except Exception:
        pass
    db.close()


def save_gate_status(run_date, gate_name, status, message, metrics_json=""):
    """保存Gate检查结果"""
    db = get_db()
    db.execute("""
        INSERT OR REPLACE INTO gate_status (run_date, gate_name, status, message, metrics_json)
        VALUES (?, ?, ?, ?, ?)
    """, [run_date, gate_name, status, message, metrics_json])
    db.close()
