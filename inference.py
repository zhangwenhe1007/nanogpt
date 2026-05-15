import torch
import tiktoken
import argparse
from model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max-new-tokens", type=int, default=100)
parser.add_argument("--temperature", type=float, default=0.8)

args = parser.parse_args()

checkpoint_path = args.checkpoint

block_size = 512
d_model = 512
n_heads = 8
n_kv_heads = 2
n_layers = 8
mode = "gqa"

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

model = GPT(
    vocab_size=vocab_size,
    block_size=block_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    n_kv_heads=n_kv_heads,
    mode=mode,
)

state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state)

model = model.to(device)
model.eval()

while True:
    prompt = input("\nPrompt> ")

    if prompt.lower() in {"exit", "quit", "q"}:
        break

    ids = enc.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    text = enc.decode(out[0].tolist())
    print("\n" + text)