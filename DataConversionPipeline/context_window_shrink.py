import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

souce_dir = r"C:\Users\Gert\Desktop\context_windows\not_cheater"
target_dir = r"C:\Users\Gert\Desktop\context_windows_512\not_cheater"

files = os.listdir(souce_dir)

for file in files:
    df = pd.read_parquet(souce_dir + "\\" + file)
    df_shrink = df.iloc[448:960].reset_index(drop=True)
    df_shrink = df_shrink.astype(np.float32)
    df_shrink.to_parquet(target_dir + "\\" + file, index=False)