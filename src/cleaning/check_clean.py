from pathlib import Path
import pandas as pd


DIR = Path("data/interim")


for file in DIR.glob("*.parquet"):

    df = pd.read_parquet(file)

    print("="*60)
    print(file.name)

    print(df.shape)

    print(df.dtypes)

    print("\nMissing:")
    print(
        df.isna().sum().sum()
    )

    print("\nLabels:")
    print(
        df["label"].value_counts()
    )