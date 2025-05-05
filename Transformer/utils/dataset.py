import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import glob

class DataImporter(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform

        cheater_dir = Path(data_dir) / "cheater"
        not_cheater_dir = Path(data_dir) / "not_cheater"

        # cheater data with label 1
        for file_path in glob.glob(str(cheater_dir / "*.parquet")):
            self.samples.append((file_path, 1))

        # non-cheater data with label 0
        for file_path in glob.glob(str(not_cheater_dir / "*.parquet")):
            self.samples.append((file_path, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        df = pd.read_parquet(file_path)

        # DataFrame to tensor
        data = torch.tensor(df.values, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)

        return data, label
