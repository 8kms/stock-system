"""
DuckDB 本地因子库 (v2.0 终局方案)

替代 pickle 缓存，提供:
  - SQL 查询因子数据
  - 按日期/行业/代码快速筛选
  - PIT 快照存储
"""
import duckdb
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "factors" / "factor_library.db"


def get_connection():
    """获取 DuckDB 连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def store_factors(df, table_name, replace=True):
    """存储因子 DataFrame 到 DuckDB"""
    con = get_connection()
    if replace:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
    con.close()
    print(f"  DuckDB: {table_name} 表已存储 ({len(df)} 行)")


def load_factors(table_name, filters=None):
    """从 DuckDB 加载因子数据"""
    con = get_connection()
    try:
        query = f"SELECT * FROM {table_name}"
        if filters:
            conditions = " AND ".join(filters)
            query += f" WHERE {conditions}"
        df = con.execute(query).df()
        con.close()
        return df
    except Exception:
        con.close()
        return pd.DataFrame()


def store_pit_snapshot(df, as_of_date):
    """存储 PIT 快照"""
    table_name = f"pit_{as_of_date.replace('-', '_')}"
    store_factors(df, table_name, replace=True)


def store_weekly_result(ranking_df, rule_df, decay_report, week_date):
    """存储每周结果"""
    con = get_connection()
    date_tag = week_date.replace("-", "_")

    for name, df in [("ranking", ranking_df), ("rule", rule_df), ("decay", decay_report)]:
        if df is not None and not df.empty:
            table = f"{name}_{date_tag}"
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")

    con.close()
    print(f"  DuckDB: 周度结果 {week_date} 已存储")


def list_stored_weeks():
    """列出所有已存储的周"""
    con = get_connection()
    tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
    con.close()
    return [t[0] for t in tables if t[0].startswith("ranking_")]
