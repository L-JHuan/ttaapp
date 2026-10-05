# GeoGR paper-guided matched-protocol adaptation

本目录用于回答“在 TAP-SID 的统一实验协议下，GeoGR 的地理协同量化流程是否更优”，不是复现 GeoGR 原论文的绝对数值。

## 保留的 GeoGR 专属模块

1. 训练集 Swing 共访对与 3 km 地理过滤；
2. POI-to-POI 对比微调；
3. 三层 RQ-Kmeans（NYC `K=32`，TKY `K=64`）；
4. LLM 驱动的 EM-style SID 迭代更新，内部使用 beam-20 候选；
5. 四类模板的 CPT；
6. 下一 POI 监督微调。

## 与 TAP-SID 对齐的公共协议

- 下游主干和 P2P 编码主干均使用 `Meta-Llama-3-8B-Instruct`；
- 使用同一 NYC/TKY 数据划分、目录、历史输入、last-5 训练样本和测试样本；
- CPT、EM 和推荐训练均采用 LoRA，不使用原论文的全参数 Qwen-4B 设置；
- 不引入公开数据中不存在的 POI 名称、品牌、地址、消费水平、天气、查询和用户画像；
- 不新增原子 SID token，沿用原始 Llama tokenizer；
- 正式推荐 SFT 为 3 epochs、学习率 `1e-5`、LoRA `r=16/alpha=32`；
- 正式测试使用与 TAP-SID 相同的完整目录 Trie、beam=10、top-10；
- 指标使用当前主实验的 Recall@1/5/10 与 NDCG@10。

EM 内部的 beam=20 只用于短 POI 描述到三层 SID 的候选搜索，不是正式推荐测试 beam。候选生成按 GPU 分片并行，先运行单样本显存 smoke；若 smoke 失败，流水线直接停止，不会静默降低 beam。

## 报告的两个版本

- `GeoGR-EM + SFT`：最终 EM SID 从基础 Llama 直接做统一 SFT，用于隔离 SID 构造贡献；
- `GeoGR-EM + CPT + SFT`：在相同最终 SID 上增加 GeoGR 多模板 CPT，再做相同 SFT，用于展示完整流程。

运行前复制 `configs/matched_example.env`，填写真实绝对路径，然后执行：

```bash
bash baselines/geogr_full_pipeline/scripts/run_matched_pipeline.sh /absolute/path/to/geogr.env
```

脚本支持从已经验收的阶段继续，但不会覆盖已有的不完整正式输出。各训练、候选生成和评估阶段使用独立日志。

## 工业数据一键运行

工业版本只复用 TAP-SID 已完成的共享预处理，不重新扫描原始行为日志，也不重新构造用户、POI、时间切分、last-5 训练样本或滚动测试样本。它同时支持：

- Spark 产物：`metadata/catalog`、`mappings/` 与 `sequence_parquet/`；
- CSV 产物：`poi_info.csv`、`role_priors.csv` 与 `v1_sequence/`。

复制配置模板并填写真实路径：

```bash
cd baselines/geogr_full_pipeline
cp configs/industrial.example.env configs/industrial.local.env
# 修改 PROCESSED_ROOT、RUN_ROOT、BASE_MODEL 和 GPU 列表
bash scripts/run_industrial_pipeline.sh "$PWD/configs/industrial.local.env"
```

工业入口默认使用与 TAP-SID 相同的四卡 Llama-3-8B、last-5 训练、无验证集和全量滚动测试协议。`EM_BEAMS=20` 仅用于 GeoGR 内部 SID 更新；正式推荐评估固定使用 `TEST_BEAMS=10`。

完整流水线依次执行：

```text
共享 TAP 预处理
  -> 地理约束共访对与 P2P 表示学习
  -> 三层初始 RQ SID
  -> EM-style SID 更新
  -> GeoGR 多模板 CPT
  -> SFT-only 与 CPT->SFT
  -> 目录约束 beam-10 测试
```

最终结果分别位于：

```text
RUN_ROOT/sft_only/eval/test_predictions.json
RUN_ROOT/sft_only/eval/test_metrics.json
RUN_ROOT/cpt_sft/eval/test_predictions.json
RUN_ROOT/cpt_sft/eval/test_metrics.json
```
