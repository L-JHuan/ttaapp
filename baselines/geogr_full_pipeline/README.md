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

## P2P 训练完成后的恢复导出

P2P 详细日志在 `RUN_ROOT/logs/p2p_train.log`，不直接显示在终端。P2P 训练或恢复导出失败时，一键脚本会在终端提示日志位置并显示该阶段最后 40 行。

训练完成后所有 rank 先同步并关闭训练通信组，再由 rank 0 保存 adapter、训练完成报告和导出向量；其他 rank 不再等待长时间目录编码，因此不会因为导出超过 10 分钟而触发原来的 NCCL 超时。

如果已有 `RUN_ROOT/p2p/encoder_adapter`，但向量导出失败，一键脚本会优先恢复导出，不重新训练。新版训练完成报告记录输入 SHA256、基础模型路径、精度和长度，恢复时必须一致；权重还会与加载后的全部 LoRA 参数严格核对键、形状和数值。

一键脚本默认开启旧版 adapter 恢复（`P2P_RECOVER_LEGACY=1`），无需修改 env 或额外添加开关。旧版没有保存完成报告，因此运行前仍应核对原 `p2p_train.log` 中三轮训练均达到 100%，原基础模型、目录、类别及训练序列未变，且没有训练期异常。在原 GeoGR 文件夹、原运行环境沿用原命令：

```bash
bash scripts/run_industrial_pipeline.sh
```

该命令保留原 adapter 和训练日志，单进程导出全部目录向量后继续 RQ、EM、CPT、SFT 和测试。恢复日志使用 `p2p_export_recovery_时间戳.log`。旧训练的损失、步数及输入来源无法从权重独立还原，报告会明确标记为缺少原训练完成记录，并将无法追溯的训练统计留空；不会伪造训练完成或人工确认信息。无需改动 env 的训练参数。如需禁止恢复没有完成报告的旧权重，可显式设置 `P2P_RECOVER_LEGACY=0`。

如果 adapter 文件缺失、损坏、参数不匹配，或已存在不完整的向量/最终报告，恢复会报错而非覆盖产物。此时先检查日志，不要直接删除旧权重或重跑训练。

## 初始 RQ 的线程限制与续跑

初始 RQ 构建仅在自己的子进程中设置 `OPENBLAS_NUM_THREADS=1`、`OPENBLAS_DEFAULT_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，减少 OpenBLAS 与 K-means 并行线程叠加造成的问题；同时启用 `PYTHONFAULTHANDLER=1` 记录底层错误栈。这些设置不会修改父进程环境或后续 EM、CPT、SFT 和评估的多卡配置，也不改变三层 K-means、码本大小、`n_init=20` 或随机种子。

初始 RQ 日志使用 `RUN_ROOT/logs/initial_rq_时间戳.log`，终端会提示实际路径；失败时显示最后40行并停止，不继续训练。旧的 `initial_rq.log` 保留，不覆盖。

如 P2P 向量和完成报告已成功保存，而初始 RQ 尚未完成，保留原 env、`RUN_ROOT` 和 P2P 产物，更新代码后沿用一键入口即可从 RQ 继续，不重新训练或导出 P2P。若仍然段错误，请提供最新 RQ 日志及错误栈；线程限制的本机测试不等同于已验证工业机器问题消失。
