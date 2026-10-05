# GeoGR full 工业基线

本目录提供 GeoGR full 的工业数据一键运行版本，用于与同一仓库中的 TAP-SID 和 Residual SID 进行统一协议比较。它只面向工业数据，不包含 NYC/TKY 专属入口或参数。

## 复用的公共数据

GeoGR full 直接读取 TAP-SID 已完成的共享预处理产物，不重新扫描原始行为日志，也不重新执行用户筛选、POI 过滤、时间切分、Last-5 训练样本或滚动测试样本构造。支持两种既有布局：

- Spark：`metadata/catalog`、`mappings/` 和 `sequence_parquet/`；
- CSV：`poi_info.csv`、`role_priors.csv` 和 `v1_sequence/`。

GeoGR 自身仍会执行方法必需的地理约束共访、P2P 表示学习、三级 RQ、两轮 EM SID 更新以及 CPT 和 SFT；这些不是公共数据预处理。

## 完整流程

```text
共享 TAP 预处理
  -> 地理约束共访对与 P2P 表示学习
  -> 三层初始 RQ SID
  -> 两轮 EM-style SID 更新
  -> GeoGR 多模板 CPT
  -> 下一 POI SFT
  -> 目录约束 beam-10 全量滚动测试
```

工业版本使用与 TAP-SID、Residual SID 相同的 `Meta-Llama-3-8B-Instruct`、训练/测试划分、Last-5 训练样本和测试协议，不单独使用验证集。`EM_BEAMS=20` 只用于 GeoGR 内部 SID 更新；正式推荐评估使用 `TEST_BEAMS=10`。

## 配置

仓库提供两个内容完整的工业配置：

- `configs/industrial.example.env`：公开模板；
- `configs/local.env`：可直接编辑的本地配置。

两者可以独立使用。请将其中的 `PROCESSED_ROOT`、`RUN_ROOT` 和 `BASE_MODEL` 替换为真实路径；GPU 列表可按运行环境调整。

默认使用 `local.env`：

```bash
cd baselines/geogr_full_pipeline
bash scripts/run_industrial_pipeline.sh
```

如果只使用模板文件，则显式传入：

```bash
cd baselines/geogr_full_pipeline
bash scripts/run_industrial_pipeline.sh "$PWD/configs/industrial.example.env"
```

当未传配置参数且 `local.env` 不存在时，脚本会自动回退到 `industrial.example.env`。

## 输出

正式 GeoGR full 结果位于：

```text
RUN_ROOT/cpt_sft/eval/test_predictions.json
RUN_ROOT/cpt_sft/eval/test_metrics.json
```

训练、SID 构建、EM 更新和评估日志分别写入 `RUN_ROOT/logs/`。脚本会复用已经完整落盘的中间阶段；检测到不完整的正式输出目录时会停止，避免覆盖或继续使用损坏产物。
