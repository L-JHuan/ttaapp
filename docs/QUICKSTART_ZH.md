# TAP-SID 快速运行指南

本文档说明如何从工业 JSONL、TSMC2014 格式签到文件或两张通用 CSV 表，完成数据准备、训练和测试。

## 1. 获取代码与安装环境

```bash
git clone https://github.com/L-JHuan/ttaapp.git
cd ttaapp
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

准备一个可被 Hugging Face Transformers 读取的基础模型目录。模型不需要放入本仓库。

## 2. 准备输入数据

### 2.1 工业 JSONL

工业报活数据可直接提供每行一个 JSON 对象的文件，字段为：

```text
user_id, poiid, new_key_type, longitude, latitude, log_time
```

在配置中设置 `INDUSTRIAL_JSONL` 后，程序会生成标准事件表和 POI 表，并自动合并每位用户
连续上报的相同 POI。合并后仍采用逐位置滚动预测，但目标变为下一个不同 POI。

工业 JSONL 每行必须是一个完整 JSON 对象，例如：

```json
{"user_id":"u_hash","poiid":"p_hash","new_key_type":"生活服务/住宅区/住宅小区","longitude":116.5,"latitude":35.7,"log_time":"2026-07-01 08:00:00"}
```

用户和 POI 可以使用稳定哈希值，但同一实体在所有记录中必须保持同一个值。

### 2.2 单个签到文件（推荐）

将八列制表符分隔文件放入：

```text
data/raw/checkins.txt
```

八列依次为：用户 ID、POI ID、细类别 ID、细类别名称、纬度、经度、时区分钟偏移和 UTC 签到时间。程序会自动生成 `events.csv` 与 `pois.csv`，并通过固定 Foursquare taxonomy 从细类别构造粗类别。

### 2.3 两张 CSV（兼容模式）

将数据放入以下位置：

```text
data/raw/events.csv
data/raw/pois.csv
```

`events.csv` 每行表示一次真实 POI 访问，至少包含：

```csv
user_id,poi_id,timestamp
u_1,p_12,2024-10-01T08:30:00Z
```

`pois.csv` 每行表示一个 POI，至少包含：

```csv
poi_id,latitude,longitude,category_l1,category_l2
p_12,31.2304,121.4737,Food,Coffee
```

两张表中的 `poi_id` 必须一致。类别数量不需要预先写死，程序会根据数据自动编号。完整要求见 [DATA_REQUIREMENTS_ZH.md](DATA_REQUIREMENTS_ZH.md)。

## 3. 设置时间切分与运行目录

复制配置模板：

```bash
cp configs/example.env configs/local.env
```

工业 JSONL 建议直接复制专用模板：

```bash
cp configs/industrial.example.env configs/local.env
```

编辑 `configs/local.env`。使用单文件输入时配置为：

```bash
TSMC_FILE=./data/raw/checkins.txt
TSMC_ENCODING=latin-1
TSMC_CATEGORY_L1_MAP=./configs/foursquare_category_l1_map.json

PROCESSED_ROOT=./data/processed/example
RUN_ROOT=./outputs/example

TRAIN_END=2024-10-31T23:59:59Z
VALIDATION_END=2024-11-30T23:59:59Z
DEFAULT_TIMEZONE_OFFSET_MINUTES=480

BASE_MODEL=/path/to/Meta-Llama-3-8B-Instruct

N_COARSE_REGIONS=64
N_FINE_REGIONS=256

NPROC_PER_NODE=1
DEVICE=cuda:0
```

使用双表输入时令 `TSMC_FILE` 为空，并配置：

```bash
TSMC_FILE=
EVENTS=./data/raw/events.csv
POIS=./data/raw/pois.csv

PROCESSED_ROOT=./data/processed/example
RUN_ROOT=./outputs/example

TRAIN_END=2024-10-31T23:59:59Z
VALIDATION_END=2024-11-30T23:59:59Z
DEFAULT_TIMEZONE_OFFSET_MINUTES=480

BASE_MODEL=/path/to/Meta-Llama-3-8B-Instruct

N_COARSE_REGIONS=64
N_FINE_REGIONS=256

NPROC_PER_NODE=1
DEVICE=cuda:0
```

两个时间点按目标访问发生时间切分样本：

```text
target_time <= TRAIN_END                         -> train
TRAIN_END < target_time <= VALIDATION_END       -> validation
target_time > VALIDATION_END                    -> test
```

例如，上述配置使用 2024 年 10 月及以前的数据训练，使用 2024 年 11 月验证，使用 2024 年 12 月及以后数据测试。时间点应根据实际数据覆盖范围填写，确保三个区间都有事件。

### 3.1 工业数据的 train/test 配置

若工业数据只划分训练集和测试集，可使用：

```bash
INDUSTRIAL_JSONL=./data/raw/output.jsonl
INDUSTRIAL_TIMEZONE_OFFSET_MINUTES=480

TSMC_FILE=
EVENTS=
POIS=

PROCESSED_ROOT=./data/processed/industrial_run
RUN_ROOT=./outputs/industrial_run

# 时间边界必须带时区。该示例对应北京时间 2026-07-17 23:59:59。
TRAIN_END=2026-07-17T15:59:59Z
VALIDATION_END=
NO_VALIDATION=1
DEFAULT_TIMEZONE_OFFSET_MINUTES=480

BASE_MODEL=/path/to/Meta-Llama-3-8B-Instruct

N_COARSE_REGIONS=64
N_FINE_REGIONS=256

NPROC_PER_NODE=2
DEVICE=cuda:0
EVAL_GPUS=0,1
```

此时：

```text
target_time <= TRAIN_END   -> train
target_time > TRAIN_END    -> test
```

程序不会生成 `llm_val.json`。训练脚本使用固定轮数并保存最终 checkpoint，不会读取测试指标选择模型。

`DEFAULT_TIMEZONE_OFFSET_MINUTES=480` 表示当地时间为 UTC+8。若 `timestamp` 已带有 `Z` 或明确时区，程序先统一为 UTC，再使用该偏移生成提示中的当地时间。

加载配置：

```bash
set -a
source configs/local.env
set +a
```

## 4. 数据准备

```bash
bash scripts/prepare_data.sh
```

主要输出：

```text
data/processed/example/
  converted_raw/                 # 仅单文件输入产生
    events.csv
    pois.csv
    conversion_report.json
  poi_info.csv
  role_priors.csv
  v1_sequence/
  protocol_report.json

outputs/example/
  codebook/tap_sid.csv
  codebook/tap_sid_report.json
  data/llm_train.json
  data/llm_val.json
  data/llm_test.json
```

首先检查 `protocol_report.json`，确认启用的 split 均包含有效样本，再开始训练。三段切分应检查 train、validation 和 test；`NO_VALIDATION=1` 时只检查 train 和 test。若使用原始单文件输入，还应检查 `converted_raw/conversion_report.json`。

## 5. 训练

单卡训练：

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
bash scripts/train.sh
```

双卡训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
bash scripts/train.sh
```

正式 adapter 输出到：

```text
outputs/example/checkpoint/final_sft/
```

当 `VALIDATION_END` 非空且 `NO_VALIDATION=0` 时，训练脚本会在每个 epoch 后保存
`checkpoint/checkpoints/epoch_XXX/`，并按验证集 teacher-forcing `lm_loss` 选择最优
epoch。所选 adapter 会复制到兼容现有评估入口的 `checkpoint/final_sft/`；测试指标不参与
checkpoint 选择。`KEEP_LAST_K_TRAIN=5` 只限制训练目标数，不限制验证集和测试集。

## 6. 多卡测试

```bash
export EVAL_GPUS=0,1
bash scripts/evaluate.sh
```

`EVAL_GPUS` 可以配置为任意数量的 GPU，例如 `0,1,2,3`。每张 GPU 处理一个独立
测试分片，全部完成后按照原始样本索引合并并重新计算全量指标。单卡运行时设置为
`EVAL_GPUS=0`。

输出文件：

```text
outputs/example/eval/test_predictions.json
outputs/example/eval/test_metrics.json
```

评估脚本会检查分片是否完整覆盖测试集，以及两个正式 JSON 文件是否能够正常解析。
指标文件包含 Recall@1、Recall@5、Recall@10 和 NDCG@10。

## 7. 完整工业运行顺序

配置加载后，按顺序执行：

```bash
bash scripts/prepare_data.sh &&
bash scripts/train.sh &&
bash scripts/evaluate.sh
```

使用 `&&` 可保证前一阶段失败后不会继续执行。正式长任务建议分别保存日志：

```bash
bash scripts/prepare_data.sh > prepare.log 2>&1 &&
bash scripts/train.sh > train.log 2>&1 &&
bash scripts/evaluate.sh > evaluate.log 2>&1
```

## 8. 常见检查

- 数据准备后某个 split 为空：调整时间边界，使其落在实际数据时间范围内。
- 找不到 POI：确认 `events.csv` 与 `pois.csv` 使用完全一致的 `poi_id`。
- 类别校验失败：同一个 `category_l2` 只能属于一个 `category_l1`。
- 模型加载失败：确认 `BASE_MODEL` 指向完整的 Hugging Face 模型目录。
- 多卡训练未启用：确认可见 GPU 数量与 `NPROC_PER_NODE` 相同。
- 多卡测试分片失败：检查 `$RUN_ROOT/eval/shards/<run_id>/logs/` 下对应 GPU 的日志。
- 更换数据集重跑：为 `PROCESSED_ROOT` 和 `RUN_ROOT` 使用新的目录，避免覆盖已有产物。
