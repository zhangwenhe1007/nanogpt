import torch
import tiktoken
import argparse
from model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max-new-tokens", type=int, default=100)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--top-p", type=float, default=None)

args = parser.parse_args()

checkpoint_path = args.checkpoint

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

checkpoint = torch.load(checkpoint_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model_config = checkpoint["model_config"]
    state = checkpoint["model_state_dict"]
else:
    model_config = {
        "vocab_size": vocab_size,
        "block_size": 512,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "n_kv_heads": 2,
        "mode": "gqa",
        "pos_encoding": "rope",
    }
    state = checkpoint

model = GPT(
    vocab_size=model_config["vocab_size"],
    block_size=model_config["block_size"],
    d_model=model_config["d_model"],
    n_heads=model_config["n_heads"],
    n_layers=model_config["n_layers"],
    n_kv_heads=model_config["n_kv_heads"],
    mode=model_config["mode"],
    pos_encoding=model_config["pos_encoding"],
)

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
        top_k=args.top_k,
        top_p=args.top_p,
    )

    text = enc.decode(out[0].tolist())
    print("\n" + text)
