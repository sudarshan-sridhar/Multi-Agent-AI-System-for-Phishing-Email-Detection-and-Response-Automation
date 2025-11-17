"""
Robustness and cross-model comparison utilities for the phishing detector.

This module provides:
  • Adversarial-style mutation generation for emails:
      - random typos
      - URL obfuscation
      - injected noise sentences
    and end-to-end analysis of each variant via `run_email_analysis`.
  • Cross-model comparison across multiple LLM backends for the *same* email:
      - helps understand how different LLMs influence final decisions and scores.

All functions return JSON-serializable dictionaries that can be consumed by
APIs or UI layers without additional transformation.
"""

from __future__ import annotations

import random
import string
from typing import Dict, Any, List

from backend.core.email_utils import extract_urls, compute_features, heuristic_score
from backend.core.model_loader import predict_proba
from backend.graph.graph_builder import run_email_analysis


def _random_typo(text: str) -> str:
    """Insert or swap a character at a random position.

    Used to simulate simple human typing errors or small perturbations
    for robustness testing of the detection pipeline.
    """
    if not text:
        return text

    t_list = list(text)
    pos = random.randint(0, len(t_list) - 1)
    op = random.choice(["delete", "swap", "insert"])

    if op == "delete" and len(t_list) > 1:
        del t_list[pos]
    elif op == "swap" and pos < len(t_list) - 1:
        t_list[pos], t_list[pos + 1] = t_list[pos + 1], t_list[pos]
    else:
        t_list.insert(pos, random.choice(string.ascii_letters))

    return "".join(t_list)


def _obfuscate_urls(text: str) -> str:
    """
    Obfuscate URLs slightly: replace dots with '[.]', add spaces, etc.

    This function is intentionally simple and only used to test whether
    the system remains robust to minor URL-format changes.
    """
    urls = extract_urls(text)
    new_text = text
    for u in urls:
        obf = u.replace(".", "[.]")
        new_text = new_text.replace(u, obf)
    return new_text


def _mutate_text(subject: str, body: str, strength: str) -> Dict[str, str]:
    """
    Create a mutated (subject, body) pair based on strength.

    strength ∈ {"light", "medium", "strong"}

    Behavior:
      - Introduces a configurable number of random typos in subject/body.
      - Optionally obfuscates URLs for medium/strong levels.
      - Adds harmless noise sentences for strong-level mutations.
    """
    strength = strength.lower()
    if strength not in {"light", "medium", "strong"}:
        strength = "medium"

    s = subject
    b = body

    # Number of typo edits per variant depends on strength.
    if strength == "light":
        n_typos = 1
    elif strength == "medium":
        n_typos = 3
    else:
        n_typos = 6

    for _ in range(n_typos):
        target = random.choice(["subject", "body"])
        if target == "subject":
            s = _random_typo(s)
        else:
            b = _random_typo(b)

    # URL obfuscation for medium and strong settings.
    if strength in {"medium", "strong"}:
        b = _obfuscate_urls(b)

    # Add random noise sentences for the strongest mutation level.
    if strength == "strong":
        noise_sentences = [
            "Please ignore this if you already responded.",
            "This is an automated message.",
            "Thank you for your prompt attention.",
        ]
        b = b + "\n\n" + random.choice(noise_sentences)

    return {"subject": s, "body": b}


# --------------------------------------------------------------------
# Adversarial / robustness-style mutations
# --------------------------------------------------------------------


def generate_adversarial_mutations(
    subject: str,
    body: str,
    num_variants: int = 8,
    strength: str = "medium",
) -> Dict[str, Any]:
    """
    Generate mutated variants of an email and run the full analysis pipeline
    (LangGraph-based) on each variant.

    Purpose:
      - Assess robustness of the overall phishing detection pipeline against
        small input perturbations (typos, obfuscation, added boilerplate).
      - Detect whether minor changes can flip classification decisions.

    Returns:
        {
          "summary": {
              "base_decision": str,
              "max_risk_score": float,
              "min_risk_score": float,
              "num_flipped_decisions": int,
              "decision_counts": {decision: count, ...},
          },
          "variants": [
              {
                  "variant_id": int,
                  "subject": str,
                  "body": str,
                  "decision": str,
                  "risk_level": str,
                  "risk_score": float,
                  "p_benign": float,
                  "p_phishing": float,
                  "heuristic_score": float,
                  "num_urls": int,
                  "num_ti_hits": int,
                  "error": Optional[str],
              },
              ...
          ]
        }
    """
    # Bound number of variants to avoid excessive compute.
    num_variants = max(1, min(int(num_variants), 50))

    variants: List[Dict[str, Any]] = []

    # Variant 0 = original email (no mutation applied).
    base_inputs = [{"subject": subject, "body": body}]

    # Additional mutated variants.
    for _ in range(num_variants - 1):
        mut = _mutate_text(subject, body, strength=strength)
        base_inputs.append(mut)

    # Run the core analysis pipeline on each version.
    for idx, inp in enumerate(base_inputs):
        try:
            result = run_email_analysis(
                subject=inp["subject"],
                body=inp["body"],
                llm_model_name="qwen2.5:3b",
            )
        except Exception as e:
            # If the graph/LLM pipeline fails, still return a structured record.
            variants.append(
                {
                    "variant_id": idx,
                    "subject": inp["subject"],
                    "body": inp["body"],
                    "decision": "error",
                    "risk_level": "low",
                    "risk_score": 0.0,
                    "p_benign": 0.0,
                    "p_phishing": 0.0,
                    "heuristic_score": 0.0,
                    "num_urls": 0,
                    "num_ti_hits": 0,
                    "error": str(e),
                }
            )
            continue

        decision = str(result.get("decision", "benign"))
        risk_level = str(result.get("risk_level", "low"))
        risk_score = float(result.get("risk_score", 0.0))
        p_benign = float(result.get("p_benign", 0.0))
        p_phishing = float(result.get("p_phishing", 0.0))
        h_score = float(result.get("heuristic_score", 0.0))

        urls = result.get("urls", []) or []
        ti_results = result.get("ti_results", []) or []
        num_urls = len(urls)

        # Count threat intelligence hits for each variant.
        num_ti_hits = 0
        for r in ti_results:
            if r.get("in_openphish") or r.get("in_urlhaus"):
                num_ti_hits += 1

        variants.append(
            {
                "variant_id": idx,
                "subject": inp["subject"],
                "body": inp["body"],
                "decision": decision,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "p_benign": p_benign,
                "p_phishing": p_phishing,
                "heuristic_score": h_score,
                "num_urls": num_urls,
                "num_ti_hits": num_ti_hits,
            }
        )

    # ----------------------------------------------------------------
    # Build aggregate summary across all variants
    # ----------------------------------------------------------------
    if variants:
        base_decision = variants[0]["decision"]
        risk_scores = [float(v["risk_score"]) for v in variants]
        max_risk = max(risk_scores)
        min_risk = min(risk_scores)

        # Count number of mutated variants whose decision differs from the original.
        num_flipped = sum(
            1 for v in variants[1:] if v.get("decision") != base_decision
        )

        decision_counts: Dict[str, int] = {}
        for v in variants:
            d = str(v.get("decision", "unknown"))
            decision_counts[d] = decision_counts.get(d, 0) + 1
    else:
        base_decision = "benign"
        max_risk = min_risk = 0.0
        num_flipped = 0
        decision_counts = {}

    summary = {
        "base_decision": base_decision,
        "max_risk_score": float(max_risk),
        "min_risk_score": float(min_risk),
        "num_flipped_decisions": int(num_flipped),
        "decision_counts": decision_counts,
    }

    return {
        "summary": summary,
        "variants": variants,
    }


# --------------------------------------------------------------------
# Cross-model LLM comparison
# --------------------------------------------------------------------


def cross_model_compare(
    subject: str,
    body: str,
    model_names: List[str],
) -> Dict[str, Any]:
    """
    Run the analysis pipeline for the same email across multiple LLM models.

    Purpose:
      - Compare how different LLM backends impact the final decision.
      - Debug LLM-specific behavior in risk scoring and classification.

    Returns:
        {
          "results": [
            {
              "llm_model_name": str,
              "decision": str,
              "risk_level": str,
              "risk_score": float,
              "p_benign": float,
              "p_phishing": float,
              "error": Optional[str],
            },
            ...
          ]
        }
    """
    results: List[Dict[str, Any]] = []

    for name in model_names:
        try:
            state = run_email_analysis(
                subject=subject,
                body=body,
                llm_model_name=name,
            )
            results.append(
                {
                    "llm_model_name": name,
                    "decision": str(state.get("decision", "benign")),
                    "risk_level": str(state.get("risk_level", "low")),
                    "risk_score": float(state.get("risk_score", 0.0)),
                    "p_benign": float(state.get("p_benign", 0.0)),
                    "p_phishing": float(state.get("p_phishing", 0.0)),
                    "error": None,
                }
            )
        except Exception as e:
            # Ensure each requested model has a corresponding result entry,
            # even if the pipeline fails for that particular model.
            results.append(
                {
                    "llm_model_name": name,
                    "decision": "error",
                    "risk_level": "low",
                    "risk_score": 0.0,
                    "p_benign": 0.0,
                    "p_phishing": 0.0,
                    "error": str(e),
                }
            )

    return {"results": results}
