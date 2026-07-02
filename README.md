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
