import pandas as pd
import os
import numpy as np

folder_dir = r"C:\Users\Gert\Desktop\context_windows\not_cheater"

files = os.listdir(folder_dir)

files_count = len(files)

axes = ["X", "Y", "Z"]
aug_amount = 1
noise_std = 0.01

for idx, file in enumerate(files):
    print(f"\rProcessing file {idx + 1}/{len(files)}: {file}", end="", flush=True)
    df = pd.read_parquet(folder_dir + "\\" + file)

    for aug_idx in range(aug_amount):
        df_aug = df.copy()

        for axis in axes:

            attacker_col = f"attacker_{axis}"
            victim_col = f"victim_{axis}" 

            if attacker_col in df.columns and victim_col in df.columns:
                noise = np.random.normal(loc=0.0, scale=noise_std, size=len(df))

                df_aug[attacker_col] = np.clip(df_aug[attacker_col] + noise, 0.0, 1.0)
                df_aug[victim_col] = np.clip(df_aug[victim_col] + noise, 0.0, 1.0)
            else:
                raise Exception("not all column values found")
            
        new_filename = file.replace(".parquet", f"_aug{aug_idx}.parquet")

        df_aug = df_aug.astype(np.float32)
        df_aug.to_parquet(os.path.join(folder_dir, new_filename), index=False)

files = os.listdir(folder_dir)
files_count_after = len(files)
new_files_count = files_count_after - files_count
print(f"\n {new_files_count} were created, from {files_count} to {files_count_after}")