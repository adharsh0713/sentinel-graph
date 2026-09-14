from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/CIC-IDS2018")


for file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(file, nrows=5)

    print("=" * 60)
    print(file.name)
    print("Columns:", len(df.columns))
    print(df.columns.tolist())