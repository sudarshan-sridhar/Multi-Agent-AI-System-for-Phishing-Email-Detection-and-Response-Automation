import os
import json
import uuid
import pandas as pd
from pandas.api.types import is_numeric_dtype

# ---------------------------------------------------------------------------
# Paths and I/O configuration
# ---------------------------------------------------------------------------
# All paths are defined relative to the `training/` folder so this script can
# be run from a consistent working directory without hard-coding absolute
# locations. The output is a single JSONL file that downstream components
# (training / evaluation) can consume.
#
#   RAW_DIR  : location of raw CSV datasets
#   OUT_DIR  : directory for processed artifacts
#   OUT_FILE : consolidated JSONL dataset (one email per line)
RAW_DIR = os.path.join("..", "data", "raw")
OUT_DIR = os.path.join("..", "data", "processed")
OUT_FILE = os.path.join(OUT_DIR, "combined.jsonl")


# ---------------------------------------------------------------------------
# Utility: robust CSV loader
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file with tolerant encoding handling.

    Tries UTF-8 first and falls back to latin1 when necessary to avoid
    hard failures on legacy datasets. Column names are normalized to
    lower-case, trimmed strings so subsequent column matching is easier.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def pick_column(df: pd.DataFrame, candidates) -> str | None:
    """
    Return the first column whose name contains any of the candidate substrings.

    This allows us to be resilient to slightly different column naming
    conventions across datasets (e.g., "Subject", "email_subject", etc.).
    If no match is found, None is returned and the caller decides how to handle
    the missing field.
    """
    cols = list(df.columns)
    for cand in candidates:
        for c in cols:
            if cand in c:
                return c
    return None


# ---------------------------------------------------------------------------
# Dataset-specific normalizers
# ---------------------------------------------------------------------------

def normalize_enron(path: str) -> pd.DataFrame:
    """
    Normalize Enron.csv to a common schema with columns:
        subject, body, label

    All Enron samples are treated as benign messages here.
    """
    df = load_csv(path)

    subject_col = pick_column(df, ["subject"])
    body_col = pick_column(df, ["body", "text", "message"])

    if subject_col is None or body_col is None:
        raise ValueError(f"Enron.csv missing subject/body columns. Columns found: {df.columns.tolist()}")

    df = df[[subject_col, body_col]].copy()
    df.rename(columns={subject_col: "subject", body_col: "body"}, inplace=True)
    df["label"] = "benign"
    return df


def normalize_ceas(path: str) -> pd.DataFrame:
    """
    Normalize CEAS_08.csv to the schema:
        subject, body, label

    The CEAS dataset can contain either numeric or string labels. This helper
    maps them into the canonical classes {"phishing", "benign"} and drops
    rows where the label cannot be confidently mapped.
    """
    df = load_csv(path)

    subject_col = pick_column(df, ["subject"])
    body_col = pick_column(df, ["body", "text", "message"])
    label_col = pick_column(df, ["label", "tag", "class", "category", "spam"])

    if subject_col is None or body_col is None or label_col is None:
        raise ValueError(
            f"CEAS_08.csv missing required columns. "
            f"Columns found: {df.columns.tolist()}"
        )

    df = df[[subject_col, body_col, label_col]].copy()
    df.rename(columns={
        subject_col: "subject",
        body_col: "body",
        label_col: "label_raw"
    }, inplace=True)

    labels = df["label_raw"]

    # Case 1: numeric labels (e.g., 1 = spam/phishing, 0 = ham/benign)
    if is_numeric_dtype(labels):
        def map_numeric(v):
            """
            Map numeric spam labels to canonical textual labels.

            Any positive value is treated as phishing/spam; zero is benign.
            Unparseable values become None and are dropped later.
            """
            try:
                if v is None:
                    return None
                # Treat >0 as phishing/spam, 0 as benign
                return "phishing" if float(v) > 0 else "benign"
            except Exception:
                return None

        df["label"] = labels.apply(map_numeric)

    else:
        # Case 2: string labels – normalize and map to canonical classes.
        s = labels.astype(str).str.lower().str.strip()

        mapping = {
            "spam": "phishing",
            "phishing": "phishing",
            "phish": "phishing",
            "malicious": "phishing",
            "fraud": "phishing",
            "ham": "benign",
            "legit": "benign",
            "legitimate": "benign",
            "benign": "benign"
        }

        def map_string(v: str):
            """
            Map raw string labels to canonical labels with some fuzzy handling.

            Handles both exact matches (e.g., 'spam') and descriptive labels
            like 'spam email', 'non-spam', etc.
            """
            if v in mapping:
                return mapping[v]
            # Handle things like "spam email", "non-spam", etc.
            if "spam" in v or "phish" in v or "fraud" in v:
                return "phishing"
            if "ham" in v or "legit" in v or "benign" in v:
                return "benign"
            return None

        df["label"] = s.apply(map_string)

    # Keep only rows with recognized labels
    df = df[df["label"].isin(["phishing", "benign"])].copy()

    df.drop(columns=["label_raw"], inplace=True)
    return df


def normalize_nazario(path: str) -> pd.DataFrame:
    """
    Normalize Nazario.csv to the schema:
        subject, body, label

    The Nazario dataset is assumed to contain only phishing examples, so all
    rows are labeled as 'phishing'.
    """
    df = load_csv(path)

    subject_col = pick_column(df, ["subject"])
    body_col = pick_column(df, ["body", "text", "message", "content"])

    if subject_col is None or body_col is None:
        raise ValueError(
            f"Nazario.csv missing subject/body columns. Columns found: {df.columns.tolist()}"
        )

    df = df[[subject_col, body_col]].copy()
    df.rename(columns={subject_col: "subject", body_col: "body"}, inplace=True)
    df["label"] = "phishing"
    return df


# ---------------------------------------------------------------------------
# Cleaning + write-out utilities
# ---------------------------------------------------------------------------

def clean_text(txt: object) -> str:
    """
    Normalize a text field into a compact, single-line string.

    - Non-string values are converted to an empty string.
    - Line breaks and excessive whitespace are collapsed.
    - Content is hard-capped at 5000 characters to avoid pathological records.
    """
    if not isinstance(txt, str):
        return ""
    txt = txt.replace("\r", " ").replace("\n", " ")
    txt = " ".join(txt.split())
    # Hard cap to avoid insanely long fields
    return txt[:5000]


def main():
    """
    Entry point for dataset preparation.

    Workflow:
      1. Load and normalize each raw dataset (Enron, CEAS_08, Nazario).
      2. Concatenate into a single DataFrame with a unified schema.
      3. Apply text cleaning and filter out empty / invalid records.
      4. Persist the combined dataset as a JSONL file (`combined.jsonl`),
         one email per line, including a synthetic UUID for each row.
    """
    print("Loading datasets...")

    enron_path = os.path.join(RAW_DIR, "Enron.csv")
    ceas_path = os.path.join(RAW_DIR, "CEAS_08.csv")
    nazario_path = os.path.join(RAW_DIR, "Nazario.csv")

    enron = normalize_enron(enron_path)
    print(f"Enron normalized: {len(enron)} rows")

    ceas = normalize_ceas(ceas_path)
    print(f"CEAS normalized: {len(ceas)} rows")

    nazario = normalize_nazario(nazario_path)
    print(f"Nazario normalized: {len(nazario)} rows")

    # Combine all sources into a single training corpus.
    df = pd.concat([enron, ceas, nazario], ignore_index=True)
    print(f"Combined dataset size (before cleaning): {len(df)} rows")

    # Basic text normalization on subject/body.
    df["subject"] = df["subject"].apply(clean_text)
    df["body"] = df["body"].apply(clean_text)

    # Remove rows with missing core fields or unknown labels.
    df.dropna(subset=["subject", "body", "label"], inplace=True)
    df = df[(df["subject"] != "") | (df["body"] != "")]
    df = df[df["label"].isin(["phishing", "benign"])]

    print("Label distribution after cleaning:")
    print(df["label"].value_counts())

    # Ensure output directory exists and stream records as JSONL.
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "id": str(uuid.uuid4()),
                "subject": row["subject"],
                "body": row["body"],
                "label": row["label"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved cleaned dataset to: {OUT_FILE}")


if __name__ == "__main__":
    main()
