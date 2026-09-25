from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("data/states")
OUTPUT_DIR = Path("data/sequences")
OUTPUT_DIR.mkdir(exist_ok=True)

SEQ_LEN = 10

FEATURES = [
    "total_packets_mean",
    "total_packets_sum",
    "total_packets_max",
    "total_bytes_mean",
    "total_bytes_sum",
    "total_bytes_max",
    "avg_packet_size_mean",
    "avg_packet_size_max",
    "fwd_ratio_mean",
    "fwd_ratio_max",
    "bwd_ratio_mean",
    "bwd_ratio_max",
    "byte_rate_mean",
    "byte_rate_max",
    "packet_rate_mean",
    "packet_rate_max",
    "syn_rate_mean",
    "syn_rate_max",
    "ack_rate_mean",
    "ack_rate_max",
    "rst_rate_mean",
    "rst_rate_max",
    "flow_duration_mean",
    "flow_duration_max",
]


def build_sequences(df):
    X = []
    y_attack = []
    y_intensity = []
    last_attack = []

    values = df[FEATURES].values
    attack_ratio = df["is_attack_mean"].values

    for i in range(len(df) - SEQ_LEN):
        X.append(values[i:i + SEQ_LEN])

        future_attack = attack_ratio[i + SEQ_LEN]
        current_attack = attack_ratio[i + SEQ_LEN - 1]

        y_attack.append(int(future_attack > 0))
        y_intensity.append(future_attack)
        last_attack.append(int(current_attack > 0))

    return (
        np.array(X),
        np.array(y_attack),
        np.array(y_intensity),
        np.array(last_attack),
    )


for file in INPUT_DIR.glob("*.parquet"):
    print("Processing:", file.name)

    df = pd.read_parquet(file)
    df = df.sort_values("timestamp")

    X, y_attack, y_intensity, last_attack = build_sequences(df)

    output = OUTPUT_DIR / (file.stem + ".npz")

    np.savez(
        output,
        X=X,
        y_attack=y_attack,
        y_intensity=y_intensity,
        last_attack=last_attack,
    )

    print(
        "X:", X.shape,
        "y_attack:", y_attack.shape,
        "y_intensity:", y_intensity.shape,
        "last_attack:", last_attack.shape,
    )
