# nanogpt

A from-scratch GPT-style language model project for learning modern LLM internals.

The code started from basic masked self-attention and has been extended toward a more modern decoder-only architecture.

## Implemented

- GPT-style decoder-only Transformer
- Causal masked attention
- Multi-head attention
- Grouped-query attention, including MQA when `n_kv_heads=1`
- RoPE positional encoding
- Learned and sinusoidal positional encoding options
- RMSNorm
- SwiGLU feedforward block
- Weight tying between token embeddings and LM head
- GPT-2 BPE tokenization through `tiktoken`
- Top-k and top-p sampling
- Train/validation split
- Warmup + cosine learning-rate schedule
- bf16 mixed precision on CUDA
- Gradient clipping
- Checkpoints with model config, train config, model weights, and optimizer state
- Inference that reconstructs model architecture from checkpoint config

## Files

- `model.py`: GPT wrapper, embeddings, final norm, LM head, generation
- `transformer.py`: Transformer blocks, attention variants, RoPE, RMSNorm, SwiGLU
- `dataloader.py`: tokenizes text and creates next-token training chunks
- `tokenizer.py`: GPT-2 `tiktoken` encode helper
- `schedulers.py`: warmup + cosine LR schedule
- `train.py`: training loop, validation loss, checkpointing
- `inference.py`: interactive generation from a checkpoint
- `small_dataset.py`: TinyStories/GSM8K/TriviaQA mix
- `medium_dataset.py`: larger streamed mixture of FineWeb-Edu, Cosmopedia, OpenWebMath, TinyStories, and QA/math data

## Build A Dataset

Small dataset:

```bash
python small_dataset.py
```

This writes:

```text
mixed_train.txt
```

Medium dataset:

```bash
python medium_dataset.py --output medium_train.txt --num-examples 200000
```

The medium mixture is approximately:

- 60% FineWeb-Edu
- 20% Cosmopedia
- 10% OpenWebMath
- 5% TinyStories
- 5% QA/math sources: GSM8K, TriviaQA, OpenBookQA

Generated dataset files and checkpoints are ignored by git.

## Train

Small smoke test:

```bash
python train.py \
  --data-path mixed_train.txt \
  --block-size 128 \
  --batch-size 8 \
  --d-model 128 \
  --n-heads 4 \
  --n-kv-heads 2 \
  --n-layers 2 \
  --attention-mode gqa \
  --encoding-mode rope \
  --max-steps 100
```

Larger H200-style run:

```bash
python train.py \
  --data-path medium_train.txt \
  --block-size 1024 \
  --batch-size 16 \
  --d-model 768 \
  --n-heads 12 \
  --n-kv-heads 4 \
  --n-layers 12 \
  --attention-mode gqa \
  --encoding-mode rope \
  --base-lr 3e-4 \
  --min-lr 3e-5 \
  --warmup-steps 500 \
  --max-steps 50000
```

Checkpoints are saved under names like:

```text
checkpoints/gqa_q12_kv4_rope/checkpoint_1000.pt
```

## Inference

Run interactive inference from a checkpoint:

```bash
python inference.py \
  --checkpoint checkpoints/gqa_q12_kv4_rope/checkpoint_1000.pt \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9
```

Type a prompt at:

```text
Prompt>
```

Exit with:

```text
q
```

or:

```text
quit
```

## Next Steps

- Add checkpoint resume using saved optimizer state and step
- Add tokenized dataset caching so startup does not re-tokenize large text files
- Add PyTorch SDPA / FlashAttention-style attention backend
- Add KV cache for faster autoregressive generation
- Add sliding-window attention
- Add Mixture of Experts feedforward blocks
- Explore toy PagedAttention for KV-cache memory management
- Explore Multi-head Latent Attention
