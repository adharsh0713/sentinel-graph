from pathlib import Path

import pandas as pd


INPUT_DIR = Path("data/interim")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_features(df):

    df = df.copy()

    df["total_packets"] = (
        df["fwd_packets"] +
        df["bwd_packets"]
    )

    df["total_bytes"] = (
        df["fwd_bytes"] +
        df["bwd_bytes"]
    )

    df["avg_packet_size"] = (
        df["total_bytes"] /
        df["total_packets"].replace(0, pd.NA)
    )

    df["fwd_ratio"] = (
        df["fwd_packets"] /
        df["total_packets"].replace(0, pd.NA)
    )

    df["bwd_ratio"] = (
        df["bwd_packets"] /
        df["total_packets"].replace(0, pd.NA)
    )

    df["syn_rate"] = (
        df["syn_count"] /
        df["total_packets"].replace(0, pd.NA)
    )

    df["ack_rate"] = (
        df["ack_count"] /
        df["total_packets"].replace(0, pd.NA)
    )

    df["rst_rate"] = (
        df["rst_count"] /
        df["total_packets"].replace(0, pd.NA)
    )

    return df


for file in INPUT_DIR.glob("*.parquet"):

    print("Processing:", file.name)

    df = pd.read_parquet(file)

    df = create_features(df)

    output = OUTPUT_DIR / file.name

    df.to_parquet(
        output,
        index=False
    )

    print(
        "Saved:",
        output,
        "Rows:",
        len(df),
        "Columns:",
        len(df.columns)
    )
