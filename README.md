# nanogpt

A from-scratch GPT-style language model project for learning modern LLM internals.

This started as a small masked self-attention implementation and grew into a compact
LLaMA-style decoder-only model with pretraining, annealing, SFT, inference, checkpoint
resume, and multi-GPU training support.

The point of this repo is not to produce a production chatbot. It is a learning
artifact: build the stack, train a real small model, observe where it works, and
observe where scale, data quality, and inference systems become the real bottlenecks.

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
- EOT stopping during generation
- Interactive single-line and multiline inference
- Train/validation split
- Tokenized dataset caching
- Warmup + cosine learning-rate schedule
- Relative LR schedule for continuation/annealing runs
- bf16 mixed precision on CUDA
- Gradient clipping
- Checkpoint resume with optimizer state and step
- Checkpoints with model config, train config, model weights, and optimizer state
- Inference that reconstructs model architecture from checkpoint config
- Multi-GPU DDP pretraining with `torchrun`
- Multi-GPU DDP SFT with response-only loss masking
- Dataset builders for small pretraining, medium pretraining, annealing, and SFT

## Files

- `model.py`: GPT wrapper, embeddings, final norm, LM head, generation
- `transformer.py`: Transformer blocks, attention variants, RoPE, RMSNorm, SwiGLU
- `dataloader.py`: tokenizes text, caches token tensors, creates next-token chunks
- `tokenizer.py`: GPT-2 `tiktoken` encode helper
- `schedulers.py`: warmup + cosine LR schedule
- `train.py`: pretraining loop, validation, DDP, checkpointing, resume
- `finetune.py`: SFT loop, response-only masking, DDP, checkpointing
- `inference.py`: interactive generation from a checkpoint
- `small_dataset.py`: TinyStories/GSM8K/TriviaQA mix
- `medium_dataset.py`: larger streamed pretraining mixture
- `anneal_dataset.py`: high-quality raw-document annealing mixture
- `sft_dataset.py`: JSONL SFT builders, including a simple pristine SFT mode

Generated datasets, token caches, checkpoints, and logs are ignored by git.

## Pretraining Data

Small dataset:

```bash
python small_dataset.py
```

Medium dataset:

```bash
python medium_dataset.py --output medium_train.txt --num-examples 1000000
```

The medium mixture is roughly:

- 60% FineWeb-Edu
- 20% Cosmopedia
- 10% OpenWebMath
- 5% TinyStories
- 5% QA/math sources: GSM8K, TriviaQA, OpenBookQA

## Pretraining

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

Main 115M-ish model:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
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
  --max-steps 300000
```

With `--batch-size 16` and `--nproc_per_node=2`, the global batch is 32 sequences,
or about 32,768 tokens per optimizer step at block size 1024.

Checkpoints are saved under names like:

```text
checkpoints/gqa_q12_kv4_rope/checkpoint_300000.pt
```

## Annealing

The annealing phase switches to a higher-quality raw-document mixture and decays LR
over a shorter continuation window.

Build annealing data:

```bash
python anneal_dataset.py \
  --output anneal_train.txt \
  --num-examples 1000000
```

Continue from a base checkpoint:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data-path anneal_train.txt \
  --block-size 1024 \
  --batch-size 16 \
  --d-model 768 \
  --n-heads 12 \
  --n-kv-heads 4 \
  --n-layers 12 \
  --attention-mode gqa \
  --encoding-mode rope \
  --base-lr 1e-5 \
  --min-lr 5e-7 \
  --warmup-steps 0 \
  --max-steps 330000 \
  --relative-lr \
  --resume-from checkpoints/gqa_q12_kv4_rope/checkpoint_300000.pt
```

In the main run, annealing from step 300k to 330k improved validation loss on the
annealing distribution from roughly 3.49 to roughly 3.38. This was the cleanest
measured improvement in the project.

## SFT

Build a simple, consistent SFT dataset:

```bash
python sft_dataset.py \
  --mode pristine_small \
  --format jsonl \
  --output sft_pristine_small_100k.jsonl \
  --num-examples 100000 \
  --progress-interval 1000
```

This mode intentionally uses a boring template:

```text
### Instruction:
...

### Input:
...

### Response:
...
```

Run SFT from the annealed checkpoint:

```bash
torchrun --standalone --nproc_per_node=2 finetune.py \
  --data-path sft_pristine_small_100k.jsonl \
  --data-format jsonl \
  --checkpoint checkpoints/gqa_q12_kv4_rope/checkpoint_330000.pt \
  --output-dir checkpoints/sft_pristine_small_100k_from_330k \
  --batch-size 16 \
  --base-lr 2e-5 \
  --min-lr 1e-6 \
  --warmup-steps 300 \
  --max-steps 10000 \
  --eval-interval 500 \
  --save-interval 1000 \
  --debug-batch
```

`finetune.py` masks prompt tokens and trains only on the response span. The debug
batch should show the prompt under `prompt/context tokens` and only the answer plus
`<|endoftext|>` under `supervised target tokens`.

The SFT loss is not directly comparable to pretraining loss because it is computed
only on response tokens and on a narrower instruction distribution.

## Inference

Run interactive inference from a checkpoint:

```bash
python inference.py \
  --checkpoint checkpoints/gqa_q12_kv4_rope/checkpoint_330000.pt \
  --max-new-tokens 200 \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 0.9 \
  --multiline
```

For SFT checkpoints, use the training template:

```text
### Instruction:
Explain machine learning in one sentence.

### Response:
```

or:

```text
### Instruction:
Answer the question in one short sentence.

### Input:
What is the capital of France?

### Response:
```

## Lessons Learned

### Architecture

Modern decoder blocks are not mysterious once built piece by piece. RoPE, GQA,
RMSNorm, and SwiGLU are all small local changes, but together they move a GPT-2-like
implementation toward a LLaMA-like architecture.

- RoPE adds relative-position structure without learned position embeddings.
- GQA saves KV memory and inference bandwidth compared with full MHA.
- RMSNorm is a simpler normalization that skips mean-centering.
- SwiGLU adds a gated feedforward path and improves expressiveness.
- Weight tying removes a large duplicate vocab projection.

### Training From Scratch Is Hard

A 115M model can learn fluent local text, but it does not become reliably factual or
deeply coherent just because it has seen billions of tokens. The model learned style,
syntax, and topic-shaped continuation before it learned stable factual recall.

This made the Chinchilla-style lesson concrete: after enough tokens, the bottleneck
for a small model shifts from "more tokens" toward model capacity and data quality.
Continuing to train the same 115M model produced diminishing returns.

### Data Quality Matters More Than It Feels Like It Should

The medium pretraining data was broad enough to teach fluency, but noisy enough that
the model learned many generic continuations and hallucination patterns. The annealing
run on cleaner data produced a visible and measurable improvement.

Formatting also mattered. Mixed prompt templates confused the SFT objective. A small,
consistent SFT dataset was more useful than a larger, messier one.

### Annealing Worked

The high-quality annealing phase was the most clearly successful training intervention.
It used:

- a higher-quality raw-document mix
- a lower LR
- a short continuation window
- the same pretraining objective

It improved validation loss by about 0.11 nats on the annealing distribution and made
samples more topic-appropriate.

### SFT Did What SFT Can Do

SFT improved:

- instruction formatting
- concise assistant-style responses
- stopping at `<|endoftext|>`

SFT did not fix:

- unreliable factual recall
- long-form repetition
- weak reasoning

This is the main practical lesson: SFT changes behavior and presentation, but it does
not magically add knowledge that the base model does not robustly encode. The model
could answer easy, high-frequency prompts cleanly, but still failed simple factual QA
in unstable ways.

### Evaluation Must Match The Training Distribution

After SFT, raw continuation prompts like:

```text
Machine learning is
```

are out of distribution. The model was trained to respond to:

```text
### Instruction:
...

### Response:
```

Comparing raw continuation before SFT against templated instruction-following after
SFT is an apples-to-oranges test. The SFT model should be evaluated in-template.

### Debugging Lesson

When a data builder or training run hangs, observe first. Use small isolated tests,
source-by-source loading, verbose logs, and process inspection before patching. Guessing
creates code scar tissue. A concrete example from this project: an SFT builder appeared
to hang, and the useful diagnostic was isolating which dataset source froze rather than
making speculative changes.

### Systems Lessons

The project made several systems bottlenecks concrete:

- Disk quota matters because datasets, token caches, logs, and checkpoints are large.
- Token caches are expensive to rebuild but save repeated startup time.
- DDP duplicates the full model on each GPU and shards batches, so global batch size is
  `per_gpu_batch_size * world_size`.
- Checkpointing should be rank-0-only in DDP and should save unwrapped model weights.
- GPU compute is only one part of training; data loading, memory capacity, memory
  bandwidth, checkpoint I/O, and scheduler limits all matter.

## Results Summary

The main experimental arc:

1. Build a GPT-style decoder from scratch.
2. Add LLaMA-like architecture upgrades: RoPE, GQA, RMSNorm, SwiGLU.
3. Train a 115M-ish model on a medium mixed dataset.
4. Continue pretraining to roughly 300k steps.
5. Anneal from 300k to 330k on higher-quality data.
6. SFT from the annealed checkpoint on a consistent instruction dataset.

Observed behavior:

- Pretraining produced fluent but rambling continuation.
- Annealing improved validation loss and topic adherence.
- SFT improved formatting and stopping.
- The final model still lacked reliable factuality and long-form coherence.

That is the expected outcome for a small from-scratch model and is the core lesson of
the project.

## Next Directions

The pretraining arc is complete. The most useful next work is inference and systems:

- Add PyTorch SDPA / FlashAttention-style attention backend
- Add KV cache for faster autoregressive generation
- Implement vanilla speculative decoding
- Train or distill a smaller draft model for speculative decoding
- Measure acceptance rate vs. temperature, prompt type, and draft length
- Try KV-cache quantization and measure quality/perplexity impact
- Explore sliding-window attention and KV eviction ideas
- Explore toy PagedAttention for KV-cache memory management
- Study FlashAttention, speculative decoding, KV compression, MoE, and modern inference serving

