import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Paths / locations
# ---------------------------------------------------------------------------
# All paths are defined relative to the `training/` folder.
# DATA_FILE : consolidated JSONL dataset created by the prep script
# MODEL_DIR : backend model directory that the FastAPI service will read from
# MODEL_PATH: final joblib artefact (vectorizer + classifier + labels)
DATA_FILE = os.path.join("..", "data", "processed", "combined.jsonl")
MODEL_DIR = os.path.join("..", "backend", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "email_classifier.joblib")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> pd.DataFrame:
    """
    Load the JSONL dataset into a DataFrame with basic validation.

    - Each line is parsed as a JSON object.
    - Malformed lines are skipped instead of killing the run.
    - Ensures required columns {subject, body, label} are present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines and continue training with remaining data.
                continue

    if not records:
        raise ValueError("No valid records found in combined.jsonl")

    df = pd.DataFrame.from_records(records)
    required_cols = {"subject", "body", "label"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"combined.jsonl must contain columns {required_cols}, "
            f"found {df.columns.tolist()}"
        )

    return df


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine `subject` + `body` into a single `text` column used for training.

    This keeps the model pipeline simple: the classifier only sees a single
    free-text field, while downstream components can still access subject/body
    separately from the raw dataset.
    """
    def combine(row):
        subj = row.get("subject") or ""
        body = row.get("body") or ""
        # Extra spacing + line breaks so subject and body remain visually distinct.
        return (str(subj) + " \n\n " + str(body)).strip()

    df = df.copy()
    df["text"] = df.apply(combine, axis=1)
    return df


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    """
    Train a TF-IDF + Logistic Regression email classifier.

    Pipeline:
      1. Load combined JSONL dataset.
      2. Normalize labels to {'phishing', 'benign'}.
      3. Build the text feature column (subject + body).
      4. Split into train/test with stratification.
      5. Fit TF-IDF vectorizer on training data.
      6. Train a balanced Logistic Regression classifier.
      7. Evaluate on held-out test set (classification report + confusion matrix).
      8. Persist artefact (vectorizer + classifier + labels) via joblib.
    """
    print("Loading processed dataset...")
    df = load_jsonl(DATA_FILE)
    print(f"Total records loaded: {len(df)}")

    # Normalize labels to canonical {phishing, benign} set.
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["phishing", "benign"])].copy()

    print("Label distribution:")
    print(df["label"].value_counts())

    # Build combined text feature from subject + body.
    df = build_text_column(df)

    X = df["text"].values
    y = df["label"].values

    # Stratified train/test split so class balance is preserved.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # TF-IDF vectorizer:
    #  - max_features caps vocabulary size for memory/speed.
    #  - ngram_range=(1,2) includes unigrams + bigrams.
    #  - unicode stripping and lowercase normalize text.
    vectorizer = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )

    print("Fitting TF-IDF on training data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression classifier with:
    #  - class_weight="balanced" to compensate class imbalance.
    #  - max_iter increased to ensure convergence with high-dimensional TF-IDF.
    #  - lbfgs solver works well for multinomial / large feature spaces.
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )

    print("Training classifier (this may take a few minutes)...")
    clf.fit(X_train_vec, y_train)

    # ---- Evaluation on held-out test set ----
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test_vec)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, y_pred, labels=["benign", "phishing"]))

    # ---- Persist model artefact ----
    # Save vectorizer + classifier + label set together so the inference
    # pipeline can reconstruct everything from a single joblib file.
    os.makedirs(MODEL_DIR, exist_ok=True)
    artefact = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "labels": np.unique(y),
    }

    joblib.dump(artefact, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
