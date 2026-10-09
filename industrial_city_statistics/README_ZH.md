# 已完成工业实验的城市级基础统计

仅读取已完成的 TAP-SID 和 GNPR/Residual SID 测试产物，用 CPU 统计。
不加载模型、不重训、不重新推理、不运行 Spark、不扫描原始行为日志；不划分城市规模组。
独立于旧 `industrial_city_r5_delivery`，不修改训练、评估或旧分析入口。

## 配置与运行

在工业仓库根目录，先复制本文件夹的 `statistics.example.env` 为
`statistics.local.env`，填写三个**真实且属于同一批实验**的目录：

- `TAP_RUN_ROOT`：TAP-SID 的输出目录。
- `GNPR_RUN_ROOT`：GNPR 的输出目录。
- `PROCESSED_ROOT`：两种方法共用的预处理目录。

填写 `POI_CITY_JSONL` 为收到的 `poi_info_字段清理.jsonl` 的真实路径。
这会通过已有编号映射关联城市名称，**绝不假定原始 POI 编号等于内部 pid**。
不填写时使用预处理坐标与仓库既有离线地级行政区边界；坐标需与边界坐标系一致。
直辖市按整市统计；边界未匹配的 POI/样本归入“未匹配”，保留在总量和校验报告中。

在已有 TAP/GNPR Python 环境中运行。若需脱离终端，可先单独执行：

```bash
tmux new -s city_statistics
```

进入工业仓库根目录后执行：

```bash
bash industrial_city_statistics/run_statistics.sh
```

也可以传入另一个已填写的配置文件：

```bash
bash industrial_city_statistics/run_statistics.sh industrial_city_statistics/statistics.local.env
```

不需要训练 env 中的卡数、模型等配置。运行器显式清空 `CUDA_VISIBLE_DEVICES`。
相对路径均相对于工业仓库根目录。终端只提示成功/失败与文件位置，不打印结果表。
CSV/JSON 输入仅依赖 Python 标准库；已有输入为 Parquet 时需要 `pyarrow`，
如环境中尚无该依赖可执行 `pip install pyarrow`。不需要启动 Spark 或安装绘图依赖。

## 已有输入结构

两种方法各读取：

```text
TAP_RUN_ROOT/codebook/tap_sid.csv
GNPR_RUN_ROOT/codebook/gnpr_sid.csv
各自 RUN_ROOT/data/llm_test.json 或 llm_test.jsonl（只能有一份）
各自 RUN_ROOT/eval/test_predictions.json
各自 RUN_ROOT/eval/test_metrics.json
```

测试数据支持 JSON 对象列表、JSONL 文件、Spark 分片 JSON 目录。
预测必须保留原评估代码输出的 `input`、`gold`、`predictions`；
多卡合并预测若包含 `sample_index`，必须完整、唯一且覆盖所有测试样本。
无需导出或转换已经完成的预测。

城市映射有两种方式：

1. 填写 `POI_CITY_JSONL`：读取 JSONL 中的 `poi编号`、`区县编码`、`区县中文`，
   并使用已有 `PROCESSED_ROOT/id_mappings.json` 的 `poi_id_to_internal`，
   或 Spark 已有 `PROCESSED_ROOT/mappings/pois` Parquet（`_poi`、`pid`）。
   本文件里的城市字段可能包含自治州、盟等行政地区，统计保持原始行政口径。
2. 留空：读取已有 `PROCESSED_ROOT/poi_info.csv`，
   或 `PROCESSED_ROOT/metadata/catalog` Parquet（`pid`、`longitude`、`latitude`），
   并使用仓库已有 `industrial_city_r5_delivery/boundaries`。

每类输入若同时存在新旧两种版本，脚本拒绝自动选择，以防混入过期文件。
不需要原始事件数据，也不需要重跑预处理。

## 统计与校验口径

- 城市归属依据**测试目标 POI**，不表示用户常住地。
- `catalog_pois`：实际共同候选目录中的 POI 数，不是整个元数据文件的 POI 数。
  例如元数据有 95,700 个、正式候选目录有 79,155 个时，只统计实际目录。
- `test_target_pois`：该城市测试目标的去重 POI 数。
- `test_users`：该城市测试样本涉及的去重用户数。
  跨城用户在各城市各计一次；总体用户数跨城去重，不能把城市用户数直接相加。
- `test_samples` / `test_sample_share_pct`：测试样本数与占总体比例。
- `tap_*` / `gnpr_*`：R@1、R@5、R@10、NDCG@10，按样本计算。
- `delta_*`：TAP 减 GNPR；`delta_*_pp`：百分点差；
  `relative_*_pct`：相对 GNPR 的提升百分比，基线为 0 时为空而不是无穷大。
- 有目录 POI 但无测试样本的城市仍输出，指标为空而不是 0。

先分别校验预测与各自测试集，再把历史 SID 映射回内部 POI，
按完整历史、用户、目标时间和目标 POI 对齐两种方法。
不能假定两个 Spark 输出的行顺序相同。
检查相同候选目录、样本覆盖、预测合法性、正式指标复算、城市映射及总量守恒。
任一关键校验失败时返回非零状态并写 `validation_report.json` 的 `FAILED`；
**只使用状态为 PASS 的结果**。未匹配边界的数量另行报告，不静默删样本。

## 输出文件

每次在 `CITY_STATS_ROOT`（默认仓库 `outputs/city_statistics`）下新建时间戳目录：

```text
时间戳_进程号/
├── run.log                       # 运行器日志
└── results/
    ├── city_statistics.csv        # 全城市计数、两方法指标及提升；Excel可直接打开
    ├── city_statistics.json       # 同表完整精度
    ├── city_statistics.md         # 可复制的全部城市结果表，指标保留0.前缀
    ├── summary.json               # 总体去重人数、样本量、城市数、指标与定义
    ├── validation_report.json     # PASS/FAILED、对齐校验、输入文件指纹
    └── analysis.log               # 阶段进度、城市匹配进度、错误详情
```

按测试样本数降序输出全部城市，不选前八、不自动分组、不作显著性声明。
这些统计支持按城市规模和覆盖讨论既有结果，但不提供线上全流量、随机 A/B
分配比例或未记录的部署信息。输出不含用户标识、POI 标识或精确坐标。
原始数据和实际输出不要提交到代码仓库。

## 测试

在仓库根目录运行：

```bash
python -m unittest discover -s industrial_city_statistics/tests -v
```
