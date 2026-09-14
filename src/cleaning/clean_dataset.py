from pathlib import Path
import pandas as pd

from schema import CANONICAL_COLUMNS


RAW_DIR = Path("data/raw/CIC-IDS2018")
OUTPUT_DIR = Path("data/interim")

OUTPUT_DIR.mkdir(exist_ok=True)


NUMERIC_COLUMNS = [
    "dst_port",
    "flow_duration",

    "fwd_packets",
    "bwd_packets",

    "fwd_bytes",
    "bwd_bytes",

    "byte_rate",
    "packet_rate",

    "flow_iat_mean",
    "flow_iat_std",

    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",

    "packet_length_mean",
    "packet_length_std",
]


def clean_file(file):

    print(f"\nProcessing {file.name}")

    df = pd.read_csv(
        file,
        low_memory=False
    )


    # remove accidental header rows
    df = df[df["Label"] != "Label"]


    # rename columns
    df = df.rename(
        columns=CANONICAL_COLUMNS
    )


    # keep only available columns
    available = [
        col for col in CANONICAL_COLUMNS.values()
        if col in df.columns
    ]

    df = df[available]

    # normalize protocol datatype
    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].astype(str)

    # timestamp conversion
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        dayfirst=True
    )


    # remove impossible timestamps
    df = df[
        (df["timestamp"].dt.year >= 2018)
        &
        (df["timestamp"].dt.year <= 2019)
    ]


    # numeric conversion
    for col in NUMERIC_COLUMNS:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


    # missing values
    for col in NUMERIC_COLUMNS:

        if col in df.columns:

            median = df[col].median()

            df[col] = df[col].fillna(
                median
            )


    # remove invalid timestamps
    df = df.dropna(
        subset=["timestamp"]
    )


    # chronological ordering
    df = df.sort_values(
        "timestamp"
    )


    output = OUTPUT_DIR / (
        file.stem + ".parquet"
    )


    df.to_parquet(
        output,
        index=False
    )


    print("Saved:", output)
    print("Rows:", len(df))


for file in RAW_DIR.glob("*.csv"):
    clean_file(file)