# 工业 JSONL 运行说明

本文档说明如何从工业 POI 活动 JSONL 完成数据处理、TAP-SID 构造、LLM 微调和测试。

## 1. 输入文件

准备一个 JSONL 文件，每行包含一个完整 JSON 对象：

```json
{"user_id":"u_hash","poiid":"p_hash","new_key_type":"生活服务/住宅区/住宅小区","longitude":116.5,"latitude":35.7,"log_time":"2026-07-01 08:00:00"}
```

必需字段：

| 字段 | 含义 |
|---|---|
| `user_id` | 稳定匿名用户 ID |
| `poiid` | 稳定匿名 POI ID |
| `new_key_type` | POI 层级类别路径 |
| `longitude` | 经度 |
| `latitude` | 纬度 |
| `log_time` | 行为发生时间 |

用户和 POI 可以使用不可逆哈希，但同一实体必须始终使用相同值。原始文件、生成数据、
checkpoint 和预测结果均被 `.gitignore` 排除，不应提交到 Git。

## 2. 处理协议

程序依次执行：

```text
解析 JSONL
  -> 生成标准 events.csv 与 pois.csv
  -> 用户内按时间排序
  -> 删除完全重复事件
  -> 合并连续相同 POI 报告
  -> 按全局时间划分 train/test
  -> 仅保留训练期可见用户和 POI
  -> 滚动构造下一不同 POI 状态预测样本
  -> 每用户保留最后 5 个训练目标
  -> 构造 TAP-SID
  -> 生成 LLM 训练与测试 JSON
```

连续状态合并示例：

```text
A, A, A, B, B, A
-> A, B, A
```

稍后再次返回 A 时仍保留该状态。测试采用滚动历史：测试窗口中较早发生的真实状态可以
进入更晚目标的历史，但任何未来事件都不会进入当前样本。

## 3. 安装

```bash
git clone https://github.com/L-JHuan/ttaapp.git
cd ttaapp
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

基础模型应准备为 Transformers 可以读取的本地 Hugging Face 模型目录。
PyTorch 需要根据业务服务器的 CUDA 版本安装。本项目已在 Python 3.10、
Transformers 4.46.3 和 PEFT 0.19.1 环境验证。

## 4. 配置

```bash
cp configs/industrial.example.env configs/local.env
```

至少修改以下字段：

```bash
INDUSTRIAL_JSONL=/path/to/output.jsonl
PROCESSED_ROOT=/path/to/processed_data
RUN_ROOT=/path/to/experiment_outputs

TRAIN_END=2026-07-17T15:59:59Z
NO_VALIDATION=1

BASE_MODEL=/path/to/Meta-Llama-3-8B-Instruct
NPROC_PER_NODE=2
EVAL_GPUS=0,1
```

`TRAIN_END` 是包含端点的 UTC 时间。若原始时间按北京时间解释，北京时间
`2026-07-17 23:59:59` 对应 `2026-07-17T15:59:59Z`。应根据实际数据范围调整。

加载配置：

```bash
set -a
source configs/local.env
set +a
```

## 5. 数据准备

```bash
bash scripts/prepare_data.sh > prepare.log 2>&1
```

主要输出：

```text
$PROCESSED_ROOT/
  converted_raw/events.csv
  converted_raw/pois.csv
  converted_raw/conversion_report.json
  poi_info.csv
  role_priors.csv
  v1_sequence/train_poi_sequence.csv
  v1_sequence/test_poi_sequence.csv
  protocol_report.json

$RUN_ROOT/
  codebook/tap_sid.csv
  codebook/tap_sid_report.json
  data/llm_train.json
  data/llm_test.json
```

开始训练前检查：

1. `conversion_report.json` 可解析且没有异常大量丢弃记录；
2. `protocol_report.json` 中 train/test 样本数均大于 0；
3. `catalog_pois`、`categories_l1` 和 `categories_l2` 符合预期；
4. `tap_sid_report.json` 中完整 SID 无碰撞；
5. `llm_train.json` 与 `llm_test.json` 可解析。

## 6. 训练

双卡训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
bash scripts/train.sh > train.log 2>&1
```

训练成功后应存在：

```text
$RUN_ROOT/checkpoint/final_sft/adapter_config.json
$RUN_ROOT/checkpoint/final_sft/adapter_model.bin
$RUN_ROOT/checkpoint/training_summary.json
```

无验证集模式使用固定 3 个 epoch，并保存最终 checkpoint，不根据测试指标选择模型。

## 7. 多卡测试

```bash
export EVAL_GPUS=0,1
bash scripts/evaluate.sh > evaluate.log 2>&1
```

`EVAL_GPUS` 接受任意数量的逗号分隔 GPU，例如 `0,1,2,3`。评估采用数据并行：
每张 GPU 独立加载模型并处理一个测试分片，全部分片成功后按原始样本索引合并，并基于
完整测试集重新计算指标。该方式不要求特定 GPU 型号，但每张 GPU 都必须能够独立加载
基础模型和 LoRA adapter。

各分片的预测、指标和日志保存在：

```text
$RUN_ROOT/eval/shards/<run_id>/
```

若任意分片失败，脚本不会写入新的正式合并结果。可通过 `EVAL_RUN_ID` 为重跑指定独立名称。

输出：

```text
$RUN_ROOT/eval/test_predictions.json
$RUN_ROOT/eval/test_metrics.json
```

`test_metrics.json` 包含 Recall@1、Recall@5、Recall@10 和 NDCG@10。
评估脚本还会检查全部样本恰好出现一次，并在结束前验证两个 JSON 文件均可解析。

## 8. 串联运行

确认配置和 GPU 后，可串联执行：

```bash
bash scripts/prepare_data.sh > prepare.log 2>&1 &&
bash scripts/train.sh > train.log 2>&1 &&
bash scripts/evaluate.sh > evaluate.log 2>&1
```

`&&` 保证前一阶段失败后不会继续运行后续阶段。正式运行建议保留三个独立日志。

## 9. 结果解释

工业 JSONL 的行为语义由数据提供方决定。若数据表示设备报活或围栏状态，实验应描述为
下一 POI 状态预测；只有在业务方确认事件代表真实到店或访问时，才可描述为下一到店预测。
当前目录限定为训练期可见 POI，不属于训练期未见 POI 冷启动评估。
