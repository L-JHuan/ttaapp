# GNPR 工业数据复现实验

## 1. 公平比较原则

GNPR 与 TAP 必须使用同一份预处理结果。以下内容不能为 GNPR 单独重做：

- 用户、POI 和类别内部 ID；
- 连续相同 POI 合并；
- 全局时间切分；
- 训练期已见用户和闭集 POI 过滤；
- 每条样本的滚动历史与 50 事件上限；
- 训练集每用户 last-5 目标；
- 验证集和测试集目标集合。

因此，已经跑完 TAP 时，应将 `PROCESSED_ROOT` 指向 TAP 的处理目录，并设置：

```bash
REUSE_PREPROCESSED=1
```

这里的“复用预处理”不等于复用 TAP-SID。GNPR 仍会从训练期统计自己的
79 维连续 POI 表征，重新训练三层 RQ-VAE，并生成独立 residual SID。

## 2. 输入

原始工业 JSONL 每行至少包含：

```text
user_id, poiid, new_key_type, longitude, latitude, log_time
```

若已经存在 TAP 处理产物，GNPR 不再读取原始 JSONL 生成时间序列：

- 小规模流程读取 `PROCESSED_ROOT/v1_sequence` 和 `id_mappings.json`；
- Spark 流程读取 `sequence_parquet`、`metadata/catalog`、`mappings/pois`
  和 `_spark_stages/catalog_filtered_states`。

## 3. GNPR 标识构造

每个训练目录 POI 的连续表示由三部分拼接：

| 部分 | 维度 | 数据来源 |
|---|---:|---|
| L2 类别文本 embedding + PCA | 64 | POI 类别元数据 |
| 球面经纬度坐标 | 3 | POI 坐标 |
| 访问小时 Fourier 统计 | 12 | 仅训练期事件 |
| 合计 | 79 | - |

RQ-VAE 使用三层 64 大小码本、64 维量化空间、`[512,256,128]` 编码器、
MSE 重构损失、0.5 量化损失权重、3000 epoch。对残差路径完全相同的
POI，按内部 `pid` 升序添加确定性 collision suffix，保证完整 SID 唯一。

## 4. 执行

```bash
cp configs/industrial.example.env configs/local.env
# 修改路径、时间边界、GPU 和模型目录
set -a
source configs/local.env
set +a
export PYTHONPATH=$PWD
```

没有共享预处理时：

```bash
bash scripts/prepare_data.sh
```

数亿事件使用 Spark：

```bash
bash scripts/prepare_data_spark.sh
```

已有 TAP 预处理时不需要再次扫描原始事件，直接执行：

```bash
bash scripts/build_codebook.sh
bash scripts/train.sh
bash scripts/evaluate.sh
```

## 5. 产物

```text
RUN_ROOT/
  embeddings/
    poi_Emb_dict.pkl
    poi_info_for_sid.csv
    embedding_report.json
  codebook/
    rqvae/best_loss_model.pth
    rqvae/training_report.json
    gnpr_sid.csv
    codebook_report.json
  data/
    llm_train.json or llm_train.jsonl/
    llm_val.json or llm_val.jsonl/
    llm_test.json or llm_test.jsonl/
  checkpoint/
    final_sft/
    training_summary.json
  eval/
    test_predictions.json
    test_metrics.json
```

`evaluate.sh` 将 `EVAL_GPUS` 中的 GPU 各自作为独立分片，并在全部分片
成功后合并。最终必须确认两个 JSON 可解析，不能只依据进度条结束判断成功。

学校服务器验证环境为 Python 3.10.20、PyTorch 2.11.0+cu130 和 CUDA
13.0。`requirements.txt` 固定 Python 直接依赖；Spark 集群应使用集群
已有版本，客户端 PySpark 的主、次版本需与集群一致。
