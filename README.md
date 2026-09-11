# Sentinel Graph

## Prerequisites

Install:

- Python 3.11+
- Git

Check Python version:

```
python --version
```

---

## Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd <project-folder>
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/Mac**

```bash
source venv/bin/activate
```

---

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Lock Dependencies

After installing new packages:

```bash
pip freeze > requirements-lock.txt
```

Install locked versions:

```bash
pip install -r requirements-lock.txt
```

