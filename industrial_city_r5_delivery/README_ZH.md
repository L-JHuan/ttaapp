# 城市级 R@5 相对提升分析

本目录用于比较同一 TAP 仓库产生的 TAP-SID 与 Residual SID 正式测试
结果。脚本不加载模型、不重新评估，只读取两套已经落盘的预测、指标、
码本以及共享预处理数据。

## 1. 放置位置

将整个目录复制到 TAP 仓库根目录，例如：

```text
TAP/
├── configs/
│   └── local.env
├── industrial_city_r5_delivery/
│   ├── analyze_city_r5.py
│   ├── plot_city_r5.py
│   ├── run_city_r5.sh
│   └── boundaries/
└── ...
```

目录名称可以修改；运行脚本会根据自身位置定位分析代码和边界文件。

## 2. 标准输入

脚本按照 TAP 仓库的正式输出结构读取文件。

TAP-SID：

```text
TAP_RUN_ROOT/
├── codebook/tap_sid.csv
├── data/llm_test.json
└── eval/
    ├── test_predictions.json
    └── test_metrics.json
```

Residual SID：

```text
RESIDUAL_RUN_ROOT/
├── codebook/gnpr_sid.csv
├── data/llm_test.json
└── eval/
    ├── test_predictions.json
    └── test_metrics.json
```

共享预处理目录：

```text
PROCESSED_ROOT/
├── poi_info.csv
└── v1_sequence/
    └── test_poi_sequence.csv
```

对方如果完全使用本仓库的数据准备、训练和评估代码，无需改写上述文件。

## 3. local.env 参数

在现有 `configs/local.env` 末尾增加：

```bash
TAP_RUN_ROOT=./outputs/your_tap_run
RESIDUAL_RUN_ROOT=./outputs/your_residual_run
CITY_ANALYSIS_ROOT=./outputs/city_r5_analysis

CITY_TOP_N=8
CITY_BOOTSTRAP=2000
CITY_SEED=42
CITY_FIGURE_DPI=300
```

已有的 `PROCESSED_ROOT` 继续使用，不要重复定义。

默认读取本交付目录内的行政区边界。只有边界文件被移动时，才需要额外
设置：

```bash
PROVINCE_GEOJSON=/path/to/province_full.json
CITY_BOUNDARY_DIR=/path/to/city_boundary_directory
```

如需指定 Python，可设置：

```bash
PYTHON_BIN=/path/to/python
```

## 4. 依赖

原 TAP 环境已经包含 NumPy。绘图还需要 Matplotlib：

```bash
pip install -r industrial_city_r5_delivery/requirements_city.txt
```

不需要 GPU。

## 5. 运行

在 TAP 仓库根目录执行：

```bash
bash industrial_city_r5_delivery/run_city_r5.sh
```

也可以显式指定配置文件：

```bash
bash industrial_city_r5_delivery/run_city_r5.sh configs/local.env
```

## 6. 统计口径

每条测试样本根据目标 POI 的经纬度匹配到地级行政区。城市级 R@5 定义为：

```text
该城市中 Gold 位于前 5 的测试样本数 / 该城市测试样本总数
```

TAP-SID 相对 Residual SID 的提升为：

```text
(TAP R@5 - Residual R@5) / Residual R@5 × 100%
```

城市先按测试样本数从高到低排序，再选择前 8 个城市。图中不显示样本数，
只显示各城市的 R@5 相对提升率。

## 7. 自动校验

统计脚本会检查：

1. 两套测试数据和预测文件的样本数一致；
2. 两套测试查询的用户和目标时间一致；
3. 两种 SID 反向映射后的目标 PID 与测试序列一致；
4. 预测文件中的 Gold 与各自测试输出一致；
5. 每条记录至少包含 5 个预测候选；
6. 每个测试目标 POI 都有经纬度；
7. 重新计算的 R@1、R@5、R@10 和 NDCG@10 与各自
   `test_metrics.json` 一致；
8. 未匹配城市的样本数和比例被写入验证报告。

任一关键对齐检查失败时，脚本会停止，不生成正式图。

## 8. 输出

`CITY_ANALYSIS_ROOT` 中将生成：

```text
city_r5_metrics_all.csv
city_r5_metrics_all.json
validation_report.json
top8_city_r5_relative_improvement.png
top8_city_r5_relative_improvement.pdf
top8_city_r5_relative_improvement.svg
```

条形图采用与论文当前版本一致的横向布局和 `#8FBC8F` 配色。

