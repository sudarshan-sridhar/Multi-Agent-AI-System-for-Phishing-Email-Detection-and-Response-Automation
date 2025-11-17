"""
LangGraph node implementations for the multi-stage phishing email analysis pipeline.

This module defines all stateful nodes that operate over `EmailAnalysisState`:
    1. ingest_node        – normalize / enrich raw inputs
    2. filter_node        – ML classifier + heuristic scoring
    3. threat_intel_node  – URL-based threat-intel enrichment and score update
    4. explainability_node– LLM-backed human explanation of decisions
    5. response_node      – LLM-backed safe user-facing reply + guidance
    6. soc_node           – SOC / admin recommendations
    7. forensics_node     – compact forensic summary for logging / dashboards

Each node is intentionally side-effect-free and returns a *new* state dict
based on the input `EmailAnalysisState`, which makes the graph easy to reason
about and test in isolation.
"""

from typing import Dict, Any, List

from .state import EmailAnalysisState, Decision
from backend.core.config import heuristic_config
from backend.core.model_loader import predict_proba
from backend.core.email_utils import compute_features, heuristic_score
from backend.core.llm_manager import LLMManager
from backend.core.ti_manager import ThreatIntelManager


# Single TI manager reused across runs
# Reusing one instance avoids repeatedly reloading TI feeds and
# keeps URL lookups fast across multiple graph invocations.
_ti_manager = ThreatIntelManager()


# ---------- Node 1: Ingest / initial processing ----------

def ingest_node(state: EmailAnalysisState) -> EmailAnalysisState:
    """
    Normalize raw inputs and set up basic derived fields.

    Responsibilities:
      - Ensure subject/body are non-None strings.
      - Build a combined_text blob used by the classifier.
      - Ensure an LLM model name is present (with a sane default).

    This node is the entry point of the pipeline and prepares the
    minimal fields required by downstream nodes.
    """
    subject = state.get("subject") or ""
    body = state.get("body") or ""
    combined_text = (subject + "\n\n" + body).strip()

    # Default LLM model if not explicitly provided by the caller.
    llm_model_name = state.get("llm_model_name") or "qwen2.5:3b"

    return {
        **state,
        "subject": subject,
        "body": body,
        "combined_text": combined_text,
        "llm_model_name": llm_model_name,
    }


# ---------- Node 2: ML classifier + heuristics ----------

def filter_node(state: EmailAnalysisState) -> EmailAnalysisState:
    """
    Run ML classification + heuristic scoring to produce:

      - p_benign, p_phishing from the trained classifier
      - heuristic_score from structural / lexical features
      - combined risk_score and categorical risk_level
      - initial decision (benign / suspicious / phishing)

    TI influence is *not* yet included here; that will be added by
    `threat_intel_node`, which re-computes risk_score.
    """
    combined_text = state["combined_text"]
    subject = state["subject"]
    body = state["body"]

    # 1) ML classifier – probability for benign vs phishing.
    p_benign, p_phishing = predict_proba(combined_text)

    # 2) Heuristic features extracted from subject + body.
    feats = compute_features(subject, body)
    h_score = heuristic_score(feats)

    # 3) Combine scores using configured weights.
    ml_w = heuristic_config.ml_weight
    h_w = heuristic_config.heuristic_weight
    # TI will be added later; for now assume 0
    ti_w = heuristic_config.ti_weight

    # For now ti_factor=0; threat_intel_node will update risk_score again
    ti_factor = 0.0

    base_score = (
        p_phishing * ml_w
        + h_score * h_w
        + ti_factor * ti_w
    )

    # Safety clamp – ensure risk_score is within [0, 1].
    risk_score = max(0.0, min(1.0, base_score))

    # Map numeric score to categorical risk level.
    if risk_score >= heuristic_config.high_risk_threshold:
        risk_level = "high"
    elif risk_score >= heuristic_config.phishing_threshold:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Initial decision (subject to refinement after TI).
    if risk_level == "high":
        decision: Decision = "phishing"
    elif risk_level == "medium":
        decision = "suspicious"
    else:
        decision = "benign"

    return {
        **state,
        "p_benign": p_benign,
        "p_phishing": p_phishing,
        "heuristic_score": h_score,
        "risk_score": risk_score,
        "risk_level": risk_level,  # type: ignore
        "decision": decision,
        "urls": feats.urls,       # expose extracted URLs to later stages
    }


# ---------- Node 3: Threat intel lookup ----------

def threat_intel_node(state: EmailAnalysisState) -> EmailAnalysisState:
    """
    Enrich URLs with threat-intel data and recompute risk_score.

    Steps:
      - For each URL (capped by max_urls_considered):
          • check against OpenPhish / URLHaus via ThreatIntelManager
          • record which sources flagged it
      - Convert TI hits into a normalized TI factor in [0, 1]
      - Recompute risk_score including TI weight
      - Update risk_level and final decision accordingly
    """
    urls: List[str] = state.get("urls", [])
    ti_results: List[Dict[str, Any]] = []

    if urls:
        # Ensure TI feeds are ready before performing lookups.
        _ti_manager.ensure_loaded()

    ti_hits = 0
    for url in urls[: heuristic_config.max_urls_considered]:
        in_openphish, in_urlhaus = _ti_manager.check_url(url)
        if in_openphish or in_urlhaus:
            ti_hits += 1
        ti_results.append(
            {
                "url": url,
                "in_openphish": in_openphish,
                "in_urlhaus": in_urlhaus,
            }
        )

    # Convert TI hits into a factor in [0, 1] relative to URL count.
    if urls:
        ti_factor = min(1.0, ti_hits / len(urls))
    else:
        ti_factor = 0.0

    # Recompute risk_score including TI contribution.
    p_phishing = state["p_phishing"]
    h_score = state["heuristic_score"]

    ml_w = heuristic_config.ml_weight
    h_w = heuristic_config.heuristic_weight
    ti_w = heuristic_config.ti_weight

    risk_score = (
        p_phishing * ml_w
        + h_score * h_w
        + ti_factor * ti_w
    )
    risk_score = max(0.0, min(1.0, risk_score))

    # Map new score to risk level.
    if risk_score >= heuristic_config.high_risk_threshold:
        risk_level = "high"
    elif risk_score >= heuristic_config.phishing_threshold:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Decision derived from updated risk level.
    if risk_level == "high":
        decision: Decision = "phishing"
    elif risk_level == "medium":
        decision = "suspicious"
    else:
        decision = "benign"

    return {
        **state,
        "ti_results": ti_results,
        "risk_score": risk_score,
        "risk_level": risk_level,  # type: ignore
        "decision": decision,
    }


# ---------- Helper for LLM-based nodes ----------

def _build_analysis_summary(state: EmailAnalysisState) -> str:
    """
    Construct a textual summary of the current analysis for LLM consumption.

    This summary is passed to LLMManager for:
        - end-user explanation (explainability_node)
        - safe reply drafting (response_node)

    Format is intentionally human-readable so it can double as a
    debugging artifact when needed.
    """
    lines = []
    lines.append(f"ML phishing probability: {state.get('p_phishing', 0.0):.3f}")
    lines.append(f"Heuristic score: {state.get('heuristic_score', 0.0):.3f}")
    lines.append(f"Risk score: {state.get('risk_score', 0.0):.3f}")
    lines.append(f"Risk level: {state.get('risk_level', 'low')}")
    lines.append(f"Decision: {state.get('decision', 'benign')}")
    urls = state.get("urls") or []
    if urls:
        lines.append(f"URLs found ({len(urls)}):")
        for r in state.get("ti_results", []):
            url = r["url"]
            flags = []
            if r.get("in_openphish"):
                flags.append("OpenPhish")
            if r.get("in_urlhaus"):
                flags.append("URLHaus")
            if flags:
                lines.append(f"  - {url} [TI hits: {', '.join(flags)}]")
            else:
                lines.append(f"  - {url} [no TI hit]")
    else:
        lines.append("No URLs found in the email.")

    lines.append("Subject:")
    lines.append(state.get("subject", ""))
    lines.append("Body:")
    lines.append(state.get("body", ""))

    return "\n".join(lines)


# ---------- Node 4: Explainability agent ----------

def explainability_node(state: EmailAnalysisState) -> EmailAnalysisState:
    """
    Use the configured LLM to generate a user-friendly explanation
    of why the email was classified as phishing/benign/suspicious.

    The explanation is meant for end-users (non-SOC audiences) and
    is stored under the `explanation` field in the state.
    """
    model_name = state.get("llm_model_name", "qwen2.5:3b")
    manager = LLMManager(model_name)

    summary = _build_analysis_summary(state)
    explanation = manager.explain_detection(summary)
    return {
        **state,
        "explanation": explanation,
    }


# ---------- Node 5: Response generation agent ----------

def response_node(state: EmailAnalysisState) -> EmailAnalysisState:
    """
    Use the LLM to draft a safe reply and attach human-readable guidance.

    Outputs:
      - suggested_reply: canned email response the user can send.
      - user_guidance:   bullet-style instructions tailored to decision
                         (phishing / suspicious / benign).
    """
    model_name = state.get("llm_model_name", "qwen2.5:3b")
    manager = LLMManager(model_name)

    summary = _build_analysis_summary(state)
    reply = manager.draft_safe_reply(summary)

    guidance_lines = []
    decision = state.get("decision", "benign")
    if decision == "phishing":
        guidance_lines.append(
            "Do not click any links or open attachments. "
            "Contact the organization through official channels (typing the URL manually)."
        )
        guidance_lines.append(
            "Report this email to your security / IT team and then delete it from your inbox and trash."
        )
    elif decision == "suspicious":
        guidance_lines.append(
            "Treat this email with caution. Verify the sender and content through a known channel "
            "(phone number / website you look up yourself)."
        )
        guidance_lines.append(
            "If in doubt, contact your security / IT team before interacting with the email."
        )
    else:
        guidance_lines.append(
            "This email appears benign based on automated analysis, but remain cautious with links and attachments."
        )

    return {
        **state,
        "suggested_reply": reply,
        "user_guidance": "\n".join(guidance_lines),
    }


# ---------- Node 6: SOC / admin agent ----------

def soc_node(state: EmailAnalysisState) -> EmailAnalysisState:
    """
    Generate SOC/operator-facing recommendations based on the decision.

    These recommendations are intentionally action-oriented, giving
    playbook-style steps for handling high-risk or suspicious emails.
    """
    recs: List[str] = []

    decision = state.get("decision", "benign")
    risk_level = state.get("risk_level", "low")

    if decision == "phishing" or risk_level == "high":
        recs.append("Block sender domain and originating IP at the email gateway, if possible.")
        recs.append("Search for similar emails in other mailboxes and remove them.")
        recs.append("Reset passwords / enforce MFA for potentially impacted accounts.")
        recs.append("Create an incident ticket and track containment / eradication steps.")
    elif decision == "suspicious":
        recs.append("Queue this email for manual SOC review.")
        recs.append("Consider temporarily quarantining similar emails based on subject / sender / URLs.")
    else:
        recs.append("No immediate SOC action required. Monitor as part of normal logging.")

    return {
        **state,
        "soc_recommendations": recs,
    }


# ---------- Node 7: Forensics agent (summary) ----------

def forensics_node(state: EmailAnalysisState) -> EmailAnalysisState:
    """
    Build a concise forensic summary string for logging or dashboards.

    Captures:
      - decision and scores
      - URL list + TI flags
      - subject and truncated body (to avoid unbounded payload size)

    The resulting `forensic_notes` field is primarily SOC-facing.
    """
    lines = []
    lines.append("==== Forensic Summary ====")
    lines.append(f"Decision: {state.get('decision', 'benign')}")
    lines.append(f"Risk level: {state.get('risk_level', 'low')}")
    lines.append(f"Risk score: {state.get('risk_score', 0.0):.3f}")
    lines.append(f"ML phishing probability: {state.get('p_phishing', 0.0):.3f}")
    lines.append(f"Heuristic score: {state.get('heuristic_score', 0.0):.3f}")
    urls = state.get("urls") or []
    if urls:
        lines.append("URLs:")
        for r in state.get("ti_results", []):
            url = r["url"]
            flags = []
            if r.get("in_openphish"):
                flags.append("OpenPhish")
            if r.get("in_urlhaus"):
                flags.append("URLHaus")
            flag_str = ", ".join(flags) if flags else "none"
            lines.append(f"  - {url} [TI hits: {flag_str}]")
    else:
        lines.append("URLs: none")

    lines.append("Subject:")
    lines.append(state.get("subject", ""))
    lines.append("Body (truncated to 800 chars):")
    body = state.get("body", "")
    lines.append(body[:800])

    return {
        **state,
        "forensic_notes": "\n".join(lines),
    }
