# Household Power Forecast

本仓库只保留家庭电力消耗多步预测实验的可运行代码。代码基于 UCI Individual
household electric power consumption 数据集，将分钟级记录汇总为日级序列，并使用过去
90 天预测未来 90 天和 365 天的 `global_active_power`。

GitHub 链接：<https://github.com/couragec/household-power-forecast>

## Files

- `src/train.py`：完整训练、评测和预测绘图代码。
- `experiments/model_search.py`：改进模型候选搜索脚本。
- `scripts/make_report_figures.py`：报告辅助图生成脚本。
- `requirements.txt`：Python 依赖。

## Run

```bash
pip install -r requirements.txt
python src/train.py --epochs 30 --runs 5
```

快速检查：

```bash
python src/train.py --epochs 2 --runs 1
```

生成报告图表：

```bash
python scripts/make_report_figures.py
```

## Models

1. `LSTM`：使用单层 LSTM 编码过去 90 天多变量序列，并由 MLP 直接输出多步预测。
2. `Transformer`：使用线性嵌入、位置编码和 Transformer Encoder 建模长期依赖。
3. `Calibrated Multi-scale Conv-Transformer`：使用多尺度一维卷积提取局部用电形态，
   再结合 Transformer Encoder 建模全局上下文；长期预测中使用训练集内部验证窗口做线性校准。

## Data Split

若目录中存在 `train.csv` 和 `test.csv`，脚本会优先读取本地 CSV；否则使用
UCI 原始数据自动构造日级数据集。默认采用时间顺序划分：最后 365 天作为测试集，
其余日期作为训练集。运行后生成的数据缓存、指标表和图片会写入本地 `data/` 与
`outputs/` 目录，这些结果文件不纳入 Git 跟踪。
