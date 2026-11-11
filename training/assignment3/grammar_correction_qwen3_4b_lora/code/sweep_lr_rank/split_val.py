import os
import sys

import pandas as pd

SRC = os.getenv("VAL_FILE", "../data_a3/val.csv")
OUT_DIR = os.getenv("SWEEP_DATA_DIR", "data")

if not os.path.exists(SRC):
    sys.exit(f"{SRC} not found. Run prepare_data_a3.sh in the parent folder first.")
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(SRC, keep_default_na=False)
for source, name in (("indictalk", "val_indictalk.csv"), ("a2", "val_a2.csv")):
    part = df[df["source"] == source]
    path = os.path.join(OUT_DIR, name)
    tmp = f"{path}.{os.getpid()}.tmp"
    part.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"{path}: {len(part):,} rows")
