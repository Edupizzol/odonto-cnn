import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

CLASSES = ["caries", "deep_caries", "periapical"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet


def get_transforms(train):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if not train:
        return T.Compose(norm)
    return T.Compose([
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        *norm,
    ])


class TeethDataset(Dataset):
    def __init__(self, manifest, split, train=False):
        df = pd.read_csv(manifest)
        self.df = df[df.split == split].reset_index(drop=True)
        self.paths = self.df.path.tolist()
        self.labels = torch.tensor(self.df[CLASSES].values, dtype=torch.float32)
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")  # cinza replicado em 3 canais
        return self.tf(img), self.labels[i]


def pos_weight(manifest):
    """neg/pos por classe, calculado só no treino (para BCEWithLogitsLoss)."""
    df = pd.read_csv(manifest)
    tr = df[df.split == "train"]
    pos = tr[CLASSES].sum().values
    neg = len(tr) - pos
    return torch.tensor(neg / pos, dtype=torch.float32)


def get_loaders(manifest, batch_size=32, num_workers=2, seed=42):
    g = torch.Generator().manual_seed(seed)

    def make(split, train):
        return DataLoader(
            TeethDataset(manifest, split, train),
            batch_size=batch_size, shuffle=train, num_workers=num_workers,
            pin_memory=True, generator=g if train else None)

    return make("train", True), make("val", False), make("test", False)