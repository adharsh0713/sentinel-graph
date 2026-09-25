from pathlib import Path

import pandas as pd


INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/states")
OUTPUT_DIR.mkdir(exist_ok=True)


WINDOW = "30s"


def create_states(df):

    df = df.sort_values("timestamp").copy()

    df["is_attack"] = (
        df["label"] != "Benign"
    ).astype(int)

    states = (
        df
        .set_index("timestamp")
        .resample(WINDOW)
        .agg(
            {
                "total_packets": [
                    "mean",
                    "sum",
                    "max"
                ],

                "total_bytes": [
                    "mean",
                    "sum",
                    "max"
                ],

                "avg_packet_size": [
                    "mean",
                    "max"
                ],

                "fwd_ratio": [
                    "mean",
                    "max"
                ],

                "bwd_ratio": [
                    "mean",
                    "max"
                ],

                "byte_rate": [
                    "mean",
                    "max"
                ],

                "packet_rate": [
                    "mean",
                    "max"
                ],

                "syn_rate": [
                    "mean",
                    "max"
                ],

                "ack_rate": [
                    "mean",
                    "max"
                ],

                "rst_rate": [
                    "mean",
                    "max"
                ],

                "flow_duration": [
                    "mean",
                    "max"
                ],

                "is_attack": "mean"
            }
        )
    )

    states.columns = [
        "_".join(col).strip("_")
        for col in states.columns
    ]

    states = states.reset_index()

    states = states.dropna()

    return states


for file in INPUT_DIR.glob("*.parquet"):

    print("Processing:", file.name)

    df = pd.read_parquet(file)

    states = create_states(df)

    output = OUTPUT_DIR / file.name

    states.to_parquet(
        output,
        index=False
    )

    print(
        "States:",
        len(states)
    )
