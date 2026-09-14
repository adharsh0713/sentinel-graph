# Sentinel Graph

## Temporal Network Attack Forecasting System

Sentinel Graph is a cybersecurity intelligence system that analyzes network traffic behavior and predicts future attack progression.

The system transforms raw network traffic data into temporal network states by extracting security-focused features from network flows, aggregating traffic behavior over time windows, and generating sequential datasets for machine learning models.

The long-term goal is to move from traditional intrusion detection:

```
"What attack is happening now?"
```

towards predictive security:

```
"What attack is likely to happen next?"
```

The project focuses on:

- Network traffic analysis
- Attack behavior modeling
- Temporal sequence learning
- Future attack prediction
- Graph-based threat intelligence


---

# Pipeline

The complete data pipeline:

```
CIC-IDS2018 Dataset
        |
        v
Raw Network Flow Data (CSV)
        |
        v
Data Cleaning
        |
        v
Clean Flow Dataset (Parquet)
        |
        v
Feature Engineering
        |
        v
Cybersecurity Feature Dataset
        |
        v
Temporal Windowing
        |
        v
Network State Representation
        |
        v
Sequence Generation
        |
        v
ML/DL Forecasting Dataset
```

## Pipeline Stages

### 1. Data Cleaning

Input:

```
Raw CIC-IDS2018 CSV files
```

Operations:

- Schema normalization
- Datatype correction
- Timestamp conversion
- Missing value handling
- Corrupted row removal

Output:

```
data/interim/
```

---

### 2. Feature Engineering

Creates security-focused features:

Traffic behavior:

- Total packets
- Total bytes
- Average packet size

Direction behavior:

- Forward packet ratio
- Backward packet ratio

TCP behavior:

- SYN rate
- ACK rate
- RST rate

Timing behavior:

- Packet frequency


Output:

```
data/features/
```

---

### 3. Temporal Network State Generation

Converts individual flows into time-based network states.

Current window:

```
30 seconds
```

Example:

```
Flow 1
Flow 2
Flow 3
      |
      v
Network State S(t)
```

Output:

```
data/states/
```

---

### 4. Sequence Generation

Creates temporal sequences for forecasting models.

Example:

Input:

```
S1 → S2 → S3 → S4 → S5 → S6 → S7 → S8 → S9 → S10
```

Target:

```
S11
```

Output:

```
data/sequences/
```


---

# Project Structure

```
sentinel-graph/

│
├── data/
│   ├── raw/
│   │   └── CIC-IDS2018/
│   │
│   ├── interim/
│   │
│   ├── features/
│   │
│   ├── states/
│   │
│   └── sequences/
│
│
├── src/
│   │
│   ├── ingestion/
│   │
│   ├── cleaning/
│   │   ├── clean_dataset.py
│   │   └── check_clean.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── check_features.py
│   │
│   ├── windowing/
│   │   ├── create_states.py
│   │   └── check_states.py
│   │
│   └── dataset/
│       └── build_sequences.py
│
│
├── notebooks/
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset Information

## CIC-IDS2018

Dataset:

Canadian Institute for Cybersecurity Intrusion Detection System Dataset 2018

Source:

https://www.unb.ca/cic/datasets/ids-2018.html


The dataset contains:

- Network flow records
- Benign traffic
- Multiple attack scenarios


Attack categories include:

- Bot
- DoS
- DDoS
- Brute Force
- Web Attacks
- Infiltration


The dataset is **not included in this repository**.

Download separately and place it inside:

```
data/raw/CIC-IDS2018/
```

Expected structure:

```
data/

└── raw/

    └── CIC-IDS2018/

        ├── Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
        ├── Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
        └── ...
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>

cd sentinel-graph
```

---

# 2. Create Virtual Environment


## Windows PowerShell

```powershell
python -m venv .venv
```


## macOS / Linux

```bash
python3 -m venv .venv
```

---

# 3. Activate Virtual Environment


## Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```


## macOS / Linux

```bash
source .venv/bin/activate
```

---

# 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

## Windows PowerShell

Create dataset folder:

```powershell
mkdir data/raw/CIC-IDS2018
```

Download CIC-IDS2018 dataset manually and extract CSV files into:

```
data/raw/CIC-IDS2018/
```


Verify:

```powershell
dir data/raw/CIC-IDS2018
```


---

## macOS / Linux

Create dataset folder:

```bash
mkdir -p data/raw/CIC-IDS2018
```

Download CIC-IDS2018 dataset manually and extract CSV files into:

```
data/raw/CIC-IDS2018/
```


Verify:

```bash
ls data/raw/CIC-IDS2018
```

---

# Running the Pipeline

Run each stage sequentially.

---

## Step 1 — Data Cleaning

Convert raw CSV files into cleaned parquet files.

```bash
python src/cleaning/clean_dataset.py
```

Output:

```
data/interim/
```


Verify:

```bash
python src/cleaning/check_clean.py
```

Expected:

- Same schema across files
- No missing values
- Correct timestamps
- Normalized datatypes


---

## Step 2 — Feature Engineering

Generate cybersecurity features.

```bash
python src/features/build_features.py
```

Output:

```
data/features/
```


Verify:

```bash
python src/features/check_features.py
```

Expected:

```
19 original features

+

derived security features
```

---

## Step 3 — Generate Temporal Network States

Aggregate flows into 30-second windows.

```bash
python src/windowing/create_states.py
```

Output:

```
data/states/
```


Verify:

```bash
python src/windowing/check_states.py
```

Expected:

```
1 row = 1 network state
```

---

## Step 4 — Generate Forecasting Sequences

Create temporal sequences for ML models.

```bash
python src/dataset/build_sequences.py
```

Output:

```
data/sequences/
```

Generated format:

```
X:

(sequence_length, features)


y:

future attack state
```

---

# Current Status

Completed:

- [x] CIC-IDS2018 dataset integration
- [x] Data cleaning pipeline
- [x] Schema normalization
- [x] Feature engineering pipeline
- [x] Temporal network state generation
- [x] Sequence dataset generation


Next:

- Baseline ML models
- LSTM forecasting model
- Attack category prediction
- MITRE ATT&CK stage mapping
- Temporal Graph Neural Network


---

# Development Workflow

## Branch Naming

Use:

```
feature/<name>

bugfix/<name>

experiment/<name>
```


Examples:

```
feature/lstm-model

feature/mitre-mapping

experiment/gnn-baseline
```


## Commit Style

Use conventional commits:

```
feat:
fix:
docs:
refactor:
test:
chore:
```


Examples:

```
feat: add CIC-IDS2018 cleaning pipeline

feat: implement temporal state generation

feat: generate forecasting sequences
```