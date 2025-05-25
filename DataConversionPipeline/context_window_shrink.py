import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

souce_dir = r"C:\Users\gluzk\Desktop\ch"
target_dir = r"C:\Users\gluzk\Desktop\ch"

files = os.listdir(souce_dir)

for file in files:
    df = pd.read_parquet(souce_dir + "\\" + file)
    df_shrink = df.iloc[224:480].reset_index(drop=True)
    df_shrink = df_shrink.astype(np.float32)
    df_shrink.to_parquet(os.path.join(target_dir, file), index=False)