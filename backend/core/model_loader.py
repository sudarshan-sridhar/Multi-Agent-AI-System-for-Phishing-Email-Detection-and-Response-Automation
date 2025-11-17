"""
Thread-safe model loading and prediction utilities for the phishing classifier.

This module encapsulates:
- Loading the serialized ML artefact (vectorizer + classifier + optional labels)
- Caching it in-memory to avoid repeated disk I/O
- Providing a clean helper (`predict_proba`) for computing
  (p_benign, p_phishing) from raw text

Because API endpoints and internal agents may call into the classifier
concurrently, all loading operations are guarded with a lock.
"""

import threading
from typing import Dict, Any, Tuple

import joblib
import numpy as np

from .config import classifier_config


# --------------------------------------------------------------------------
# Model Cache and Synchronization
# --------------------------------------------------------------------------
# `_model_cache` ensures the expensive joblib load happens only once.
# A lock guards the initialization path so multiple threads do not 
# simultaneously read the model from disk.
_model_lock = threading.Lock()
_model_cache: Dict[str, Any] = {}


def load_model() -> Dict[str, Any]:
    """
    Load and return the serialized ML artefact from disk.

    Performs:
        • One-time disk read of the joblib file located at classifier_config.model_path
        • Thread-safe caching to prevent duplicate loads
        • Returns artefact containing:
              - vectorizer (TF-IDF)
              - classifier (sklearn estimator)
              - labels     (optional, for consistency across deployments)

    Returns:
        Dict[str, Any]: full artefact as originally serialized.
    """
    with _model_lock:
        # Fast path: if model already loaded, return cached version immediately.
        if "artefact" in _model_cache:
            return _model_cache["artefact"]

        # Slow path: load from disk.
        artefact = joblib.load(classifier_config.model_path)

        # Cache for future calls.
        _model_cache["artefact"] = artefact
        return artefact


def predict_proba(text: str) -> Tuple[float, float]:
    """
    Predict (p_benign, p_phishing) for a single email.

    Workflow:
        1. Load the cached model artefact.
        2. Vectorize the input using the stored TF-IDF vectorizer.
        3. Use classifier.predict_proba to obtain class probabilities.
        4. Map probabilities to canonical labels ("benign", "phishing").
        5. Normalize to ensure the two probabilities sum to 1.

    Notes:
        - The classifier itself determines the class ordering via clf.classes_.
        - The normalization step prevents unexpected drift if an artefact 
          contains more than two labels or returns non-standard distributions.

    Returns:
        Tuple[float, float]:
            (p_benign, p_phishing) — each guaranteed to be in [0, 1].
    """
    artefact = load_model()
    vectorizer = artefact["vectorizer"]
    clf = artefact["classifier"]

    # Use stored label order if present; fallback ensures backward compatibility.
    labels = artefact.get(
        "labels",
        np.array([classifier_config.benign_label, classifier_config.phishing_label]),
    )

    # Transform raw text → vector → probability distribution.
    X_vec = vectorizer.transform([text])
    proba = clf.predict_proba(X_vec)[0]  # returns array of shape (n_classes,)

    # Map class labels to probabilities based on classifier-defined ordering.
    label_order = getattr(clf, "classes_", labels)
    label_to_proba = {str(label): float(p) for label, p in zip(label_order, proba)}

    # Safely retrieve probabilities with fallback defaults.
    p_benign = label_to_proba.get(classifier_config.benign_label, 0.0)
    p_phishing = label_to_proba.get(classifier_config.phishing_label, 0.0)

    # Normalize for guaranteed consistency (sum = 1).
    total = p_benign + p_phishing
    if total > 0:
        p_benign /= total
        p_phishing /= total

    return p_benign, p_phishing
