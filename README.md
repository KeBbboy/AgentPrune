## 🚀 Quick Start
 
 ### 📦 Install packages
 
 ```bash 
 uv venv --python 3.10 
 source .venv/bin/activate 
 uv pip install -r requirements.txt 
 ``` 
 
 ### 📊 Datasets

Please download the `MMLU`, `HumanEval`, and `GSM8K` datasets and place them in the `dataset` folder. The file structure should be organized as follows:

```
dataset
└── mmlu
    └── data
└── mmlu_pro
    ├── validation.parquet|json|jsonl
    └── test.parquet|json|jsonl
└── humaneval
    └── humaneval-py.jsonl
└── gsm8k
    └── gsm8k.jsonl
...
```

### 🧩 消息压缩框架 (Message Compression Framework)

`feat/message-compression` 分支在 `AgentDropout/graph/message_compression/` 下提供 3 个 **ours 方法**开关
(默认全部关闭, 与基线行为一致), **每个开关对应一个独立模块**, 复用到现有 6 个数据集与 autogen runner:

| CLI 开关                         | 触发模块                          | 含义                                              |
|----------------------------------|-----------------------------------|---------------------------------------------------|
| `--role_aware_compression`       | `role_aware_compression.py`       | 按 self_role 差异化截断前驱消息                   |
| `--node_aware_aggregation`       | `node_aware_aggregation.py`       | 按 self_role 重新聚合前驱输出 (同 role 合并)      |
| `--temporal_api_summary`         | `temporal_api_summary.py`         | 每轮按 answer 一致性聚合多智能体输出 (纯字符)     |

模块 3 按"答案"为聚合键分流: 答案一致时合并 reasoning 取首句作共识理由;
答案分裂时每个答案单独一行, 理由取 sample reasoning 首句. **不调 LLM**.

快速使用示例:

```bash
python experiments/run_mmlu.py \
    --mode FullConnected --agent_names AnalyzeAgent --agent_nums 5 \
    --domain mmlu --limit_questions 8 \
    --role_aware_compression true \
    --node_aware_aggregation true \
    --temporal_api_summary true
```

完整说明见 [`AgentDropout/graph/message_compression/`](AgentDropout/graph/message_compression)。
