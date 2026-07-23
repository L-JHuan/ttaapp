# 数据处理协议

## 1. 时间划分

使用两个时间点将目标行为按时间顺序划分：

```text
target_time <= train_end                       -> train
train_end < target_time <= validation_end     -> validation
target_time > validation_end                  -> test
```

设置 `NO_VALIDATION=1` 时只使用一个训练截止点：

```text
target_time <= train_end   -> train
target_time > train_end    -> test
```

默认只保留训练期出现过的用户和 POI。每个用户内部按时间排序，当前访问作为预测目标，
更早的访问作为历史。每条样本最多包含 49 个历史事件和 1 个目标事件。训练数据在每个
用户内部保留最后 5 条样本，validation 和 test 保留全部有效样本。

该协议不是 leave-one-out。一个用户可以在 validation 或 test 中贡献多条样本；每条样本
只使用目标发生前已经观测到的真实轨迹。训练集的“最后 5 条”仅用于控制微调样本规模，
不会把每个用户的最后一条和倒数第二条分别指定为 test 与 validation。

对于 `INDUSTRIAL_JSONL` 报活数据，程序会先将每位用户连续上报的相同 POI 合并为一个
位置状态，并保留该状态的首次上报时间，然后逐位置预测下一个不同 POI。该处理避免模型
主要学习“重复上一条报活 POI”。TSMC 单文件和通用双表输入默认不执行该合并。

测试采用滚动历史：测试窗口内较早发生的真实状态可以进入更晚测试目标的历史，但目标
之后的事件不会进入当前样本。目录默认仅包含训练期出现的用户和 POI，因此该设置不评估
训练期未见 POI 冷启动。

## 2. 处理流程

```text
工业 JSONL 或 TSMC 格式签到文件（可选）
  -> convert_industrial_jsonl.py / convert_tsmc2014.py
  -> events.csv + pois.csv
  -> prepare_realworld_data.py
  -> 时间切分、历史序列、匿名 ID 映射
  -> build_tap_sid.py
  -> TAP-SID 码本
  -> build_llm_data.py
  -> llm_train.json / llm_val.json / llm_test.json
  -> train_tap_sid.py
  -> LoRA checkpoint
  -> evaluate_tap_sid.py
  -> Recall@1/5/10 与 NDCG@10
```

TAP-SID 只使用 POI 坐标和类别构造，不使用测试期交互统计。评估阶段使用目录 Trie，
保证完整输出对应真实 POI。
