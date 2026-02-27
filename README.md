---
license: apache-2.0
language:
- en
tags:
- zen
- zen-lm
- embeddings
- retrieval
- multilingual
- mteb
- matryoshka
library_name: transformers
pipeline_tag: feature-extraction
---

<p align="center">
  <img src="https://zenlm.org/logo.png" width="300"/>
</p>

<h1 align="center">Zen Embedding</h1>

<p align="center">
  <strong>Multilingual text embedding models by Zen LM — #1 MTEB multilingual (score 70.58)</strong>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/zenlm/zen-embedding-8b">HuggingFace</a> &nbsp;|&nbsp;
  📖 <a href="https://zenlm.org">Docs</a> &nbsp;|&nbsp;
  💻 <a href="https://github.com/zenlm">GitHub</a>
</p>

---

## Introduction

**Zen Embedding** is a family of multilingual text embedding models from Zen LM by Hanzo AI. Available in three sizes (0.6B, 4B, 8B), the 8B model ranks **#1 on the MTEB multilingual leaderboard** (score 70.58, as of June 2025).

These models support over 100 languages, 32K context, Matryoshka Representation Learning (MRL) for flexible vector dimensions, and instruction-aware embeddings for task-specific tuning.

## Model Family

| Model | Parameters | Embedding Dim | Context | MRL |
|-------|------------|---------------|---------|-----|
| [zen-embedding-0.6b](https://huggingface.co/zenlm/zen-embedding-0.6b) | 0.6B | 1024 | 32K | Yes |
| [zen-embedding-4b](https://huggingface.co/zenlm/zen-embedding-4b) | 4B | 2560 | 32K | Yes |
| [zen-embedding-8b](https://huggingface.co/zenlm/zen-embedding-8b) | 8B | 4096 | 32K | Yes |

## Key Features

- **#1 MTEB Multilingual**: 8B model achieves 70.58 mean score (June 2025)
- **100+ languages**: Robust multilingual, cross-lingual, and code retrieval
- **32K context**: Long document and passage embeddings
- **MRL support**: Flexible vector dimensions — truncate at any size without quality loss
- **Instruction-aware**: Custom task instructions improve retrieval by 1-5%
- **Dual use**: Both embedding and reranking tasks

## Quick Start

### Install

```bash
pip install transformers torch
```

> Important: Use `AutoModel`, not `AutoModelForCausalLM`, for embedding inference.

### Transformers Usage

```python
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


# Task instruction (applied to queries only, not documents)
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is the capital of France?'),
    get_detailed_instruct(task, 'Explain gradient descent'),
]
documents = [
    "Paris is the capital of France.",
    "Gradient descent is an optimization algorithm that minimizes a function by iteratively moving in the direction of steepest descent.",
]

input_texts = queries + documents

model_name = 'zenlm/zen-embedding-8b'
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModel.from_pretrained(model_name)

# Enable flash attention for better performance (optional)
# model = AutoModel.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()

max_length = 8192
batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# Normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = embeddings[:2] @ embeddings[2:].T
print(scores.tolist())
```

### vLLM Usage

```python
# Requires vllm>=0.8.5
import torch
from vllm import LLM


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is the capital of France?'),
    get_detailed_instruct(task, 'Explain gradient descent'),
]
documents = [
    "Paris is the capital of France.",
    "Gradient descent minimizes a loss function iteratively.",
]

input_texts = queries + documents

model = LLM(model="zenlm/zen-embedding-8b", task="embed")
outputs = model.embed(input_texts)

embeddings = torch.tensor([o.outputs.embedding for o in outputs])
scores = embeddings[:2] @ embeddings[2:].T
print(scores.tolist())
```

### Sentence Transformers Usage

```python
# Requires sentence-transformers>=2.7.0
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("zenlm/zen-embedding-8b")

queries = ["What is the capital of France?", "Explain gradient descent"]
documents = [
    "Paris is the capital of France.",
    "Gradient descent minimizes a loss function by following the negative gradient.",
]

with torch.no_grad():
    query_embeddings = model.encode(queries, prompt_name="query")
    document_embeddings = model.encode(documents)

similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
```

## Instruction Guidance

For best results, write task-specific instructions in English even when working with other languages. Examples:

```python
# Retrieval
task = 'Given a web search query, retrieve relevant passages that answer the query'

# Classification
task = 'Classify the sentiment of the following text'

# Clustering
task = 'Identify the topic of the following document'

# Code retrieval
task = 'Given a code comment, retrieve the code implementation that matches'
```

Instructions improve performance by 1-5% on most retrieval tasks. Do not apply instructions to retrieval documents — only to queries.

## Matryoshka Representation Learning (MRL)

All Zen Embedding models support MRL: you can truncate embeddings to smaller dimensions without retraining.

```python
import torch.nn.functional as F

# Full dimension (8B model: 4096)
full_embeddings = F.normalize(embeddings, p=2, dim=1)

# Truncate to 1024 dimensions for storage efficiency
truncated = F.normalize(embeddings[:, :1024], p=2, dim=1)
```

## Multilingual Support

100+ languages including all major European, Asian, Middle Eastern, and South Asian language families. See the full list of supported languages below.

<details>
<summary>Supported Languages</summary>

| Family | Languages |
|--------|-----------|
| Indo-European | English, French, German, Spanish, Portuguese, Italian, Russian, Polish, Dutch, Czech, Romanian, Greek, Hindi, Bengali, Urdu, and 40+ more |
| Sino-Tibetan | Chinese (Simplified/Traditional/Cantonese), Burmese |
| Afro-Asiatic | Arabic (Standard + 7 dialects), Hebrew, Maltese |
| Austronesian | Indonesian, Malay, Tagalog, Javanese, and others |
| Dravidian | Tamil, Telugu, Kannada, Malayalam |
| Turkic | Turkish, Uzbek, Kazakh, Azerbaijani, Tatar |
| Tai-Kadai | Thai, Lao |
| Uralic | Finnish, Estonian, Hungarian |
| Austroasiatic | Vietnamese, Khmer |
| Other | Japanese, Korean, Georgian, Basque, Swahili |

</details>

## MTEB Benchmark Results

### Multilingual (MTEB)

| Model | Size | Mean (Task) | Retrieval | STS |
|-------|------|-------------|-----------|-----|
| multilingual-e5-large-instruct | 0.6B | 63.22 | 57.12 | 76.81 |
| gemini-embedding-exp-03-07 | - | 68.37 | 67.71 | 79.40 |
| **zen-embedding-0.6b** | 0.6B | 64.33 | 64.64 | 76.17 |
| **zen-embedding-4b** | 4B | 69.45 | 69.60 | 80.86 |
| **zen-embedding-8b** | 8B | **70.58** | **70.88** | **81.08** |

### English (MTEB v2)

| Model | Size | Mean (Task) | Retrieval | STS |
|-------|------|-------------|-----------|-----|
| NV-Embed-v2 | 7.8B | 69.81 | 62.84 | 83.82 |
| gemini-embedding-exp-03-07 | - | 73.3 | 64.35 | 85.29 |
| **zen-embedding-0.6b** | 0.6B | 70.70 | 61.83 | 86.57 |
| **zen-embedding-4b** | 4B | 74.60 | 68.46 | **88.72** |
| **zen-embedding-8b** | 8B | **75.22** | **69.44** | 88.58 |

## Hardware Requirements

| Model | VRAM (FP16) | Notes |
|-------|-------------|-------|
| zen-embedding-0.6b | 2GB | CPU-compatible |
| zen-embedding-4b | 8GB | Single GPU |
| zen-embedding-8b | 16GB | Single GPU (A100 recommended) |

## License

Apache 2.0

## Citation

```bibtex
@misc{zenlm2025zen-embedding,
    title={Zen Embedding: Multilingual Text Embedding Models by Zen LM},
    author={Hanzo AI and Zoo Labs Foundation},
    year={2025},
    publisher={HuggingFace},
    howpublished={\url{https://huggingface.co/zenlm/zen-embedding-8b}}
}
```

---

<p align="center">
  <strong>Zen LM by Hanzo AI</strong> - Clarity Through Intelligence<br>
  <a href="https://zenlm.org">zenlm.org</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/zenlm">HuggingFace</a> &nbsp;|&nbsp;
  <a href="https://github.com/zenlm">GitHub</a>
</p>
