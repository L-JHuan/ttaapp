# 为什么 GNPR 不应重复预处理

TAP-SID 与 GNPR-SID 的正式比较只允许标识构造不同。如果 GNPR 重新进行
用户过滤、POI 过滤、时间切分或样本生成，两者可能面对不同的目录、历史或
测试目标，结果将不再是 matched comparison。

推荐流程：

```text
同一原始行为日志
        |
        v
一次统一预处理
        |
        +--> TAP-SID codebook --> TAP SFT --> TAP evaluation
        |
        +--> GNPR 79D embedding --> RQ-VAE SID --> GNPR SFT --> GNPR evaluation
```

唯一需要分别执行的是 codebook 之后的路径。GNPR 的访问小时统计只使用
统一训练边界之前的事件，不能读取验证期或测试期行为。
