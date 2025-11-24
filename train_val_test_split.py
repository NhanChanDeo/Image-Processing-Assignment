import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_DIR = 'labeled/full/img'
CSV_PATH = 'labeled/full/label.csv'
OUTPUT_ROOT = 'labeled'

TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.7, 0.15, 0.15

for split in ['train', 'test', 'valid']:
    os.makedirs(os.path.join(OUTPUT_ROOT, split, "img"), exist_ok=True)

df = pd.read_csv(CSV_PATH)

train_df, temp_df = train_test_split(df, test_size=(1 - TRAIN_RATIO), random_state=42, shuffle=True)
valid_df, test_df = train_test_split(temp_df, test_size=TEST_RATIO / (TEST_RATIO + VALID_RATIO), random_state=42)

splits = {
    "train": train_df,
    "valid": valid_df,
    "test": test_df
}

for split, split_df in splits.items():
    split_dir = os.path.join(OUTPUT_ROOT, split, "img")
    csv_out = os.path.join(OUTPUT_ROOT, split, "label.csv")

    records = []
    for _, row in tqdm(split_df.iterrows(), desc=split):
        src = os.path.join(BASE_DIR, row["name"])
        dest = os.path.join(split_dir, row["name"])

        shutil.copy2(src, dest)
        records.append(row.tolist())

    pd.DataFrame(records, columns=["name", "age", "gender"]).to_csv(csv_out, index=False)
