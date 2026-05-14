import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import encode


class TextDataset(Dataset):
    def __init__(self, path, block_size, split="train", train_frac=0.9):
        self.block_size = block_size

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        ids = encode(text)
        data = torch.tensor(ids, dtype=torch.long)

        n = int(train_frac * len(data))

        if split == "train":
            self.data = data[:n]
        elif split == "val":
            self.data = data[n:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
