# Makefile for Zen Embedding (0.6B)

MODEL_NAME = zen-embedding-0.6b
BASE_MODEL = Qwen/Qwen3-Embedding-0.6B
HF_REPO = zenlm/${MODEL_NAME}

.PHONY: all
all: train quantize upload

.PHONY: train
train:
	@echo "🎯 Training zen-embedding..."
	@python train_zen_embedding.py

.PHONY: quantize
quantize:
	@echo "🗜️ Creating GGUF quantizations..."
	@make gguf-q4 gguf-q5 gguf-q8

.PHONY: gguf-q4
gguf-q4:
	@../llama.cpp/build/bin/llama-quantize \
		gguf/${MODEL_NAME}-f16.gguf \
		gguf/${MODEL_NAME}-Q4_K_M.gguf Q4_K_M

.PHONY: mlx
mlx:
	@echo "🍎 Converting to MLX..."
	@mlx_lm.convert --hf-path finetuned --mlx-path mlx --quantize

.PHONY: test
test:
	@echo "🧪 Testing zen-embedding..."
	@python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
		model = AutoModelForCausalLM.from_pretrained('finetuned'); \
		tokenizer = AutoTokenizer.from_pretrained('finetuned'); \
		print('Model loaded successfully')"

.PHONY: upload
upload:
	@echo "📤 Uploading to HuggingFace..."
	@huggingface-cli upload ${HF_REPO} . --repo-type model

.PHONY: clean
clean:
	@rm -rf finetuned/ gguf/ mlx/
