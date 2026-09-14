from pathlib import Path
import pandas as pd


DIR = Path("data/states")


for file in DIR.glob("*.parquet"):

    df = pd.read_parquet(file)

    print("="*50)
    print(file.name)

    print(df.shape)

    print(df.head())