from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/CIC-IDS2018")


for file in DATA_DIR.glob("*.csv"):

    print("\n" + "="*80)
    print(file.name)

    df = pd.read_csv(file)

    print("\nShape:")
    print(df.shape)

    print("\nLabels:")
    print(df["Label"].value_counts())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nData types:")
    print(df.dtypes.value_counts())

    print("\nTimestamp example:")
    print(df["Timestamp"].head())