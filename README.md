# LLM Sampling Parameters Benchmark

测试LLM采样参数对摘要任务性能影响的基准测试工具。

## 功能特性

- **支持数据集**: LCSTS (中文)、XSum (英文)、TruthfulQA
- **批量推理**: 基于 OpenAI Batch API 实现高效批量推理
- **评估指标**: ROUGE-1, ROUGE-2, ROUGE-L (支持中文)
- **参数扫描**: 支持对 temperature、top_p 等参数进行扫描测试

## 安装

```bash
cd /home/ai/code/python/lm-eval-harness
uv sync
```

## 快速开始

### 单次测试

```bash
# 测试 LCSTS 数据集
python main.py --dataset lcsts --max-samples 100

# 测试 XSum 数据集
python main.py --dataset xsum --max-samples 100

# 测试 TruthfulQA
python main.py --dataset truthfulqa --max-samples 50
```

### 参数扫描

```bash
# 对 temperature 和 top_p 进行扫描测试
python main.py --dataset lcsts --sweep --max-samples 100
```

### 自定义采样参数

```bash
python main.py \
    --dataset lcsts \
    --temperature 0.5 \
    --top-p 0.95 \
    --max-tokens 128 \
    --max-samples 200
```

### 指定 API 服务器

```bash
python main.py \
    --dataset xsum \
    --base-url http://your-server:8000/v1 \
    --model-name your-model-name
```

## 配置

### 环境变量

创建 `.env` 文件:

```env
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_API_KEY=sk-dummy
OPENAI_MODEL_NAME=local-model
```

### 代码中使用

```python
import asyncio
from sampling_benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    ClientConfig,
    SamplingConfig,
)

async def run_benchmark():
    # 配置客户端
    client_config = ClientConfig(
        base_url="http://127.0.0.1:1234/v1",
        api_key="sk-dummy",
        model_name="local-model",
    )

    # 配置基准测试
    config = BenchmarkConfig(
        dataset_name="lcsts",
        split="test",
        max_samples=100,
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )

    # 运行测试
    runner = BenchmarkRunner(client_config=client_config)
    result = await runner.run(config)

    # 查看结果
    print(f"ROUGE-1: {result.avg_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {result.avg_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {result.avg_scores['rougeL']:.4f}")

asyncio.run(run_benchmark())
```

## 输出结果

结果保存在 `results/` 目录:

- `benchmark_summary.csv` - 所有测试的汇总结果
- `{dataset}_{timestamp}.csv` - 单次测试的详细结果
- `benchmark.log` - 运行日志

### 结果格式

详细结果 CSV 包含以下列:

| 列名        | 说明            |
| ----------- | --------------- |
| sample_id   | 样本ID          |
| text        | 原始文本        |
| reference   | 参考摘要        |
| prediction  | 模型生成的摘要  |
| rouge1      | ROUGE-1 F1 分数 |
| rouge2      | ROUGE-2 F1 分数 |
| rougeL      | ROUGE-L F1 分数 |
| temperature | 采样温度        |
| top_p       | Top-p 采样参数  |
| timestamp   | 时间戳          |

## 数据集说明

### LCSTS (中文短文本摘要)

- 路径: `/home/ai/code/datasets/LCSTS/`
- 格式: JSONL
- 语言: 中文
- 使用 `rouge-chinese` 进行评估

### XSum (英文新闻摘要)

- 路径: `/home/ai/code/datasets/xsum/`
- 格式: Parquet (通过 HuggingFace datasets 加载)
- 语言: 英文
- 使用 `rouge-score` 进行评估

### TruthfulQA (真实性问答)

- 路径: `/home/ai/code/datasets/TruthfulQA/`
- 格式: CSV
- 语言: 英文
- 生成模式评估

## 采样参数

| 参数              | 默认值 | 范围     | 说明         |
| ----------------- | ------ | -------- | ------------ |
| temperature       | 0.7    | 0.0-2.0  | 控制随机性   |
| top_p             | 0.9    | 0.0-1.0  | 核采样       |
| top_k             | 50     | 0+       | Top-k 采样   |
| max_tokens        | 256    | >0       | 最大生成长度 |
| frequency_penalty | 0.0    | -2.0-2.0 | 频率惩罚     |
| presence_penalty  | 0.0    | -2.0-2.0 | 存在惩罚     |

## API 兼容性

支持任何 OpenAI 兼容的 API:

- LM Studio
- vLLM
- Ollama
- Text Generation WebUI
- 其他兼容服务器
