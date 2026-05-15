import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TextDataset
from model import GPT
import tiktoken
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 512
batch_size = 32  # try 64 if memory is fine

d_model = 512
n_heads = 8
n_kv_heads = 2  # GQA
n_layers = 8

attention_mode = "gqa"
encoding_mode = "learned"

checkpoint_root = "checkpoints"
if attention_mode == "gqa":
    run_name = f"gqa_q{n_heads}_kv{n_kv_heads}"
else:
    run_name = attention_mode

checkpoint_dir = os.path.join(checkpoint_root, run_name)
os.makedirs(checkpoint_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

train_dataset = TextDataset("mixed_train.txt", block_size, split="train")
val_dataset = TextDataset("mixed_train.txt", block_size, split="val")

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

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

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
    if step > 0 and step % 1000 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    
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

    logits = model(x)  # (B, T, vocab_size)

    B, T, C = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, C),
        y.view(B * T)
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    pbar.set_description(f"loss {loss.item():.4f}")

checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
torch.save(model.state_dict(), checkpoint_path)