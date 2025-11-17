from backend.core.model_loader import predict_proba
from backend.core.email_utils import compute_features, heuristic_score
from backend.core.llm_manager import LLMManager

# ----------------------------------------------------------------------
# Manual smoke-test script for the ML pipeline + heuristic engine + LLM
# ----------------------------------------------------------------------
# This file is typically used as a quick functional check to ensure:
#   1. The trained sklearn model loads correctly (TF-IDF + Logistic Regression).
#   2. Feature extraction and heuristic scoring are wired correctly.
#   3. The LLM manager (Qwen or Gemini) responds and tool invocation works.
#
# This script is NOT part of the production pipeline — it is purely for
# developer validation during local debugging and integration testing.
# ----------------------------------------------------------------------

# Example input email that mimics a phishing-style pattern.
subject = "Urgent: Verify your account now"
body = (
    "Dear user, your account will be suspended. "
    "Click https://malicious.xyz/login immediately."
)

# Combine subject + body exactly as the ML model expects.
combined = subject + "\n\n" + body

# ----------------------------------------------------------------------
# 1. ML model inference (probabilities from the trained classifier)
# ----------------------------------------------------------------------
p_benign, p_phish = predict_proba(combined)
print("ML probs (benign, phishing):", p_benign, p_phish)

# ----------------------------------------------------------------------
# 2. Heuristic feature computation (URL count, suspicious phrases, etc.)
# ----------------------------------------------------------------------
feats = compute_features(subject, body)
print("Features:", feats)

# Stand-alone heuristic score (0–1 range)
print("Heuristic score:", heuristic_score(feats))

# ----------------------------------------------------------------------
# 3. LLM reasoning test (ensures model routing + prompt formatting work)
# ----------------------------------------------------------------------
# LLMManager abstracts model selection (e.g., Qwen2.5:3B or Gemini Flash)
# and provides high-level utilities such as:
#    - explain_detection(summary)
#    - draft_safe_reply(summary)
# These tests confirm connectivity, correct model name handling, and prompt flow.
llm = LLMManager("qwen2.5:3b")

# Simple diagnostic call to verify the LLM returns coherent output.
text = llm.explain_detection("Test explanation reasons here.")
print("LLM explanation snippet:", text[:200])
