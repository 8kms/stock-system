# A 股多因子选股系统

> 部署日期：2025-05-31 | 版本：v1.0 | 60 文件 | 本地辅助决策系统

AKShare/Sina/Tushare 数据 → 6 类因子 → LightGBM/XGBoost/Kronos 排序 → Gate 0-8 质量检查 → Streamlit 7 页工作台。

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/8kms/stock-system.git
cd stock-system

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 配置 Tushare Token（获取真实财务数据）
export TUSHARE_TOKEN=你的token

# 4. 运行
python3 run_weekly.py          # 拉数据 + 因子 + 模型 + 输出
streamlit run app.py           # 启动可视化 → http://localhost:8501
```

首次运行使用 Sina 财经作为行情源（免费、不限频），PE/PB 值基于行业基准估算。配置 Tushare Token 后自动切换真实财务数据。

## 系统架构

```
Sina/AKShare 行情 + Tushare 财务
        ↓
   PIT 数据清洗 + 硬剔除 (Gate 0)
        ↓
  DuckDB 本地数据湖 (6 张表)
        ↓
  规则评分 (30/20/20/10/10/10)
        ↓
  LightGBM LGBMRanker + XGBoost XGBRanker
        ↓
  行业暴露控制 + 因子衰减监控 + 模型分歧检测
        ↓
  Streamlit 决策工作台 (7 页)
```

## 模型

| 模型 | 定位 | 说明 |
|------|------|------|
| LightGBM LGBMRanker | 主排序 | 行业内 lambdarank，预测未来60日收益分位 |
| XGBoost XGBRanker | 交叉验证 | 分歧检测，防止单模型误判 |
| Kronos-mini | K线后置 | 80/80 权重，仅用于波动/过热/破位确认 |

## 7 页决策工作台

| 页面 | 功能 |
|------|------|
| 总览驾驶舱 | Gate 状态 + 池分布 + 今日结论 |
| 智能选股 | 6 策略模板 + 自定义筛选 + 漏斗诊断 |
| 候选池 | 池优先级排序 + 四维评分 + 降级原因 |
| 个股详情 | 规则/模型/技术/风险 四维拆解 + 估值分位 |
| 模型审计 | Gain/Split/贡献占比 + 因子重要性 |
| 数据质量 | Gate 0A-0D 四层数据质量 |
| 回测分析 | 5 市场环境回测 + 交易成本 |

## Gate 体系

| Gate | 检查项 | 标准 |
|------|--------|------|
| 0A | 行情覆盖 | >98% |
| 0B | 财务覆盖 | >90% |
| 0C | 估值覆盖 | >90% / 历史 >750 天 |
| 0D | 特征方差 | nunique>3 / std>0 / 单值<80% |
| 1 | 规则评分 | 6 分项 30/20/20/10/10/10 |
| 2 | 标签 | 行业内未来 60 日收益分位 + embargo |
| 3 | 模型 | ICIR>0.3 + IC正比例>55% |
| 4 | 行业暴露 | 单行业≤5 只/≤20% |
| 5 | 因子衰减 | 绿/黄/红/灰 四级 |
| 6 | 模型分歧 | LGB/XGB rank_diff + agreement_score |

## 目录结构

```
stock-system/
├── app.py                      # Streamlit 入口
├── run_weekly.py               # 一键运行 12 步 SOP
├── config.py                   # 全局配置
├── data/                       # 数据层
│   ├── fetcher.py              #   Sina/AKShare 行情拉取
│   ├── tushare_provider.py     #   Tushare 财务数据
│   ├── valuation_provider.py   #   估值三源 (Tushare→CSV→AKShare)
│   ├── duckdb_schema.py        #   DuckDB 6 表
│   ├── quality_gate.py         #   Gate 0A-0D + 状态机
│   ├── safe_akshare.py         #   AKShare 限速+缓存客户端
│   └── pit_fetcher.py          #   PIT 公告日对齐
├── factors/
│   └── rule_score.py           # 规则评分 30/20/20/10/10/10
├── models/
│   ├── ranker.py               # LightGBM/XGBoost 训练
│   ├── ranker_v2.py            # P2 标签 + walk-forward
│   ├── score_builder.py        # 四维评分 + 候选池分类
│   ├── model_audit.py          # 训练后审计
│   ├── backtest_v2.py          # 5 市场环境回测
│   ├── industry_exposure.py    # 行业暴露 Gate 4
│   ├── factor_decay.py         # 因子衰减 Gate 5
│   ├── gate_checker.py         # Gate 重置触发器
│   └── kronos_model.py         # Kronos-mini K线
├── ui/
│   ├── pages/
│   │   ├── dashboard.py        # 总览驾驶舱
│   │   ├── screener.py         # 智能选股器
│   │   ├── top_stocks.py       # 候选池
│   │   ├── stock_detail.py     # 个股详情
│   │   └── backtest.py         # 回测分析
│   ├── screener.py             # 筛选器逻辑
│   └── components.py           # 通用图表
└── scripts/
    ├── fetch_tushare_financials.py
    └── build_financial_pit.py
```
