import random
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from training.hyperparameters import train_size, test_size, val_size
import pandas as pd

class DataImporter(Dataset):
    def __init__(self, split='train', transform=None, seed=41):
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        self.samples = []
        self.transform = transform
        random.seed(seed)

        data_dir = Path(__file__).resolve().parent
        
        cheater_dir = data_dir / "cheater"
        not_cheater_dir = data_dir / "not_cheater"

        cheater_files = self._group_files_by_file_int(cheater_dir)
        non_cheater_files = self._group_files_by_file_int(not_cheater_dir)

        all_keys = list(cheater_files.keys() | non_cheater_files.keys())
        random.shuffle(all_keys)

        total = len(all_keys)
        train_end = int(train_size * total)
        val_end = train_end + int(val_size * total)

        if split == 'train':
            selected_keys = all_keys[:train_end]
        elif split == 'val':
            selected_keys = all_keys[train_end:val_end]
        else:
            selected_keys = all_keys[val_end:]

        for key in selected_keys:
            for label, file_dict in [(1, cheater_files), (0, non_cheater_files)]:
                if key in file_dict:
                    for file_path in file_dict[key]:
                        if split == 'test' and "_aug" in file_path.name:
                            continue
                        self.samples.append((file_path, label))

    def _group_files_by_file_int(self, directory):
        """
        Returns a dictionary where key = file_int (e.g., 0 from file_0), 
        value = list of Path objects for that file_int
        """
        grouped = {}
        for file in os.listdir(directory):
            if not file.endswith('.parquet'):
                continue
            parts = file.split('-')
            file_int = parts[2].replace("file_", "")
            key = f"{parts[0]}-{parts[1]}-file_{file_int}"
            grouped.setdefault(key, []).append(directory / file)
        return grouped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        df = pd.read_parquet(file_path)

        data = torch.tensor(df.values, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)

        return data, label
