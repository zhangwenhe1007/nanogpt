import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TextDataset
from model import GPT
from schedulers import WarmupCosineScheduler

import tiktoken
from tqdm import tqdm
import os
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="mixed_train.txt")
parser.add_argument("--block-size", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--d-model", type=int, default=512)
parser.add_argument("--n-heads", type=int, default=8)
parser.add_argument("--n-kv-heads", type=int, default=2)
parser.add_argument("--n-layers", type=int, default=8)
parser.add_argument("--attention-mode", type=str, default="gqa", choices=["mha", "gqa"])
parser.add_argument("--encoding-mode", type=str, default="learned", choices=["learned", "sinusoidal", "rope"])
parser.add_argument("--base-lr", type=float, default=3e-4)
parser.add_argument("--min-lr", type=float, default=3e-5)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--max-steps", type=int, default=10000)
args = parser.parse_args()

block_size = args.block_size
data_path = args.data_path
batch_size = args.batch_size
d_model = args.d_model
n_heads = args.n_heads
n_kv_heads = args.n_kv_heads
n_layers = args.n_layers
attention_mode = args.attention_mode
encoding_mode = args.encoding_mode
base_lr = args.base_lr
min_lr = args.min_lr
warmup_steps = args.warmup_steps
max_steps = args.max_steps

checkpoint_root = "checkpoints"
if attention_mode == "gqa":
    run_name = f"gqa_q{n_heads}_kv{n_kv_heads}_{encoding_mode}"
else:
    run_name = f"{attention_mode}_{encoding_mode}"

checkpoint_dir = os.path.join(checkpoint_root, run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

model_config = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "d_model": d_model,
    "n_heads": n_heads,
    "n_layers": n_layers,
    "n_kv_heads": n_kv_heads,
    "mode": attention_mode,
    "pos_encoding": encoding_mode,
}

train_config = {
    "data_path": data_path,
    "batch_size": batch_size,
    "base_lr": base_lr,
    "min_lr": min_lr,
    "warmup_steps": warmup_steps,
    "max_steps": max_steps,
}

train_dataset = TextDataset(data_path, block_size, split="train")
val_dataset = TextDataset(data_path, block_size, split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = GPT(
    vocab_size=vocab_size,
    block_size=block_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    n_kv_heads=n_kv_heads,
    mode=attention_mode,
    pos_encoding=encoding_mode
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
scheduler = WarmupCosineScheduler(optimizer, base_lr, min_lr, warmup_steps, max_steps)

def save_checkpoint(path, step):
    torch.save({
        "step": step,
        "model_config": model_config,
        "train_config": train_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)

@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=50):
    model.eval()
    losses = []

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        B, T, C = logits.shape

        loss = F.cross_entropy(
            logits.view(B * T, C),
            y.view(B * T)
        )

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

pbar = tqdm(train_loader)

step = 0

for step, (x, y) in enumerate(pbar):
    if step >= max_steps:
        break

    lr = scheduler.step(step)

    if step > 0 and step % 1000 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
        save_checkpoint(checkpoint_path, step)
    
    if step > 0 and step % 500 == 0:
        model.eval()

        prompt = "Question: What is 2 + 2?\nAnswer:"
        ids = enc.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        out = model.generate(idx, max_new_tokens=50)
        print(enc.decode(out[0].tolist()))

        model.train()
        val_loss = estimate_loss(model, val_loader, device)
        print(f"\nstep {step}: val loss {val_loss:.4f}")

    x = x.to(device)
    y = y.to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits = model(x)  # (B, T, vocab_size)

        B, T, C = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, C),
            y.view(B * T)
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    #gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    pbar.set_description(f"loss {loss.item():.4f}")

#save last checkpoint
checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
save_checkpoint(checkpoint_path, step)
