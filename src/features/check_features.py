from pathlib import Path
import pandas as pd


DIR = Path("data/features")


for file in DIR.glob("*.parquet"):

    df = pd.read_parquet(file)

    print("="*50)
    print(file.name)

    print("Shape:", df.shape)

    print(
        df.columns.tolist()
    )

    print("\nMissing:")
    print(
        df.isna().sum().sum()
    )