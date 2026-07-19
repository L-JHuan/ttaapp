# 数据处理协议

## 1. 时间划分

使用两个时间点将目标行为按时间顺序划分：

```text
target_time <= train_end                       -> train
train_end < target_time <= validation_end     -> validation
target_time > validation_end                  -> test
```

默认只保留训练期出现过的用户和 POI。每个用户内部按时间排序，当前访问作为预测目标，
更早的访问作为历史。每条样本最多包含 49 个历史事件和 1 个目标事件。训练数据在每个
用户内部保留最后 5 条样本，validation 和 test 保留全部有效样本。

## 2. 处理流程

```text
events.csv + pois.csv
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

