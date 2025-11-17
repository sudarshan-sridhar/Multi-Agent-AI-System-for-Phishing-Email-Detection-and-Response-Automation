from backend.graph.graph_builder import run_email_analysis

# ----------------------------------------------------------------------
# Example end-to-end pipeline test for the LangGraph-based email analyzer
# ----------------------------------------------------------------------
# This script triggers the **full** multi-stage analysis pipeline:
#   1. Ingest node → normalizes input & sets LLM model
#   2. Filter node → ML classifier + heuristics (risk score)
#   3. Threat intel node → OpenPhish + URLHaus lookups
#   4. Explainability node → LLM explanation generation
#   5. Response node → Safe reply + user guidance
#   6. SOC node → Analyst-level recommended actions
#   7. Forensics node → Structured, condensed investigation notes
#
# Running this file is a quick way to confirm:
#   • All nodes execute without exceptions
#   • The compiled StateGraph works end-to-end
#   • The LLM backend responds correctly (Qwen/Gemini)
#   • URL extraction and TI hits propagate into downstream logic
# ----------------------------------------------------------------------

subject = "Important: Confirm your bank account now"
body = """
Dear customer,

We detected unusual activity. Please log in immediately at https://secure-bank-support.xyz/login
to verify your information.

Regards,
Fake Bank
"""

# Execute full LangGraph pipeline using the specified LLM model.
# `run_email_analysis` returns the final state produced by the forensics node.
result = run_email_analysis(subject, body, llm_model_name="qwen2.5:3b")

# ----------------------------------------------------------------------
# Display the major outputs from the final state
# ----------------------------------------------------------------------
# Decision       → benign / suspicious / phishing
# Risk level     → low / medium / high
# Risk score     → normalized float in [0, 1]
# User guidance  → human-facing safety recommendations
# Suggested reply→ LLM-generated safe response
# Forensic notes → analyst-level condensed investigation log
# ----------------------------------------------------------------------

print("Decision:", result.get("decision"))
print("Risk level:", result.get("risk_level"))
print("Risk score:", result.get("risk_score"))

# Guidance to the end user (e.g., “don’t click links”, “verify externally”)
print("User guidance:\n", result.get("user_guidance"))

# Truncated fields for readability when testing
print("\nSuggested reply (first 400 chars):\n", result.get("suggested_reply", "")[:400])
print("\nForensic notes (first 400 chars):\n", result.get("forensic_notes", "")[:400])
