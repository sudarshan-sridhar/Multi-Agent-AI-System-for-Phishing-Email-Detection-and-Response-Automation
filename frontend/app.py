"""
Streamlit front-end for the Phish Multi-Agent Analyzer.

This UI:
  • Talks to a FastAPI backend exposing the multi-agent phishing analysis graph.
  • Lets users analyze a single email via `/analyze_email`.
  • Provides an evaluation dashboard over the processed dataset:
      - Offline metrics (/eval_summary, /threshold_sweep, /roc_curve, /pr_curve, /confusion_at_threshold)
      - Dataset insights (/dataset_summary, /dataset_features)
      - Robustness testing (/adversarial_mutations, /cross_model_compare)

The goal is to keep this module UI-only: no ML or security logic lives here.
All detection, scoring, and TI are delegated to the backend.
"""

import json
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st


# Default backend URL (FastAPI)
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


# --------- Helper functions ---------


def get_backend_url() -> str:
    """
    Resolve the active backend URL from Streamlit session state.

    This allows the user to override the backend base URL via the sidebar
    without hardcoding it in the code. If nothing is set, the default is used.
    """
    # Allow dynamic override via sidebar text input
    return st.session_state.get("backend_url", DEFAULT_BACKEND_URL)


def fetch_models(backend_url: str) -> Dict[str, Any]:
    """
    Query the backend `/models` endpoint for available LLM model names.

    On failure (e.g., backend down or endpoint missing), fall back to a
    static list that matches the default backend configuration so the
    UI remains usable for local demos.
    """
    try:
        resp = requests.get(f"{backend_url}/models", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Could not fetch models from backend: {e}")
        # Fallback to static list matching backend config
        return {
            "available_models": ["qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"],
            "default_model": "qwen2.5:3b",
        }


def analyze_email_request(
    backend_url: str,
    subject: str,
    body: str,
    llm_model_name: str,
) -> Dict[str, Any]:
    """
    Call the backend `/analyze_email` endpoint for a single email.

    Returns:
        JSON dict containing classification decision, scores,
        LLM explanation, guidance, SOC recommendations, etc.
    """
    payload = {
        "subject": subject,
        "body": body,
        "llm_model_name": llm_model_name,
    }

    resp = requests.post(
        f"{backend_url}/analyze_email",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def eval_summary_request(
    backend_url: str,
    max_samples: int,
) -> Dict[str, Any]:
    """
    Call `/eval_summary` to compute aggregate metrics on the processed dataset.

    Used for the main evaluation block in the 'Evaluation & Performance' tab.
    """
    payload = {
        "max_samples": max_samples,
    }
    resp = requests.post(
        f"{backend_url}/eval_summary",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def threshold_sweep_request(
    backend_url: str,
    max_samples: int,
    num_thresholds: int,
) -> Dict[str, Any]:
    """
    Call `/threshold_sweep` to evaluate performance across multiple
    phishing probability thresholds.
    """
    payload = {
        "max_samples": max_samples,
        "num_thresholds": num_thresholds,
    }
    resp = requests.post(
        f"{backend_url}/threshold_sweep",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def roc_curve_request(
    backend_url: str,
    max_samples: int,
) -> Dict[str, Any]:
    """
    Call `/roc_curve` to retrieve ROC data (FPR, TPR, thresholds, AUC)
    for phishing as the positive class.
    """
    payload = {
        "max_samples": max_samples,
    }
    resp = requests.post(
        f"{backend_url}/roc_curve",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def pr_curve_request(
    backend_url: str,
    max_samples: int,
) -> Dict[str, Any]:
    """
    Call `/pr_curve` to retrieve precision–recall data and AUC(PR)."""
    payload = {
        "max_samples": max_samples,
    }
    resp = requests.post(
        f"{backend_url}/pr_curve",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def confusion_at_threshold_request(
    backend_url: str,
    max_samples: int,
    threshold: float,
) -> Dict[str, Any]:
    """
    Call `/confusion_at_threshold` to get confusion matrix and metrics
    at a specific phishing probability threshold.
    """
    payload = {
        "max_samples": max_samples,
        "threshold": threshold,
    }
    resp = requests.post(
        f"{backend_url}/confusion_at_threshold",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def adversarial_mutations_request(
    backend_url: str,
    subject: str,
    body: str,
    num_variants: int,
    strength: str,
) -> Dict[str, Any]:
    """
    Call `/adversarial_mutations` to run robustness analysis via mutated
    variants of a base email.
    """
    payload = {
        "subject": subject,
        "body": body,
        "num_variants": num_variants,
        "strength": strength,
    }
    resp = requests.post(
        f"{backend_url}/adversarial_mutations",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def cross_model_compare_request(
    backend_url: str,
    subject: str,
    body: str,
    model_names: List[str],
) -> Dict[str, Any]:
    """
    Call `/cross_model_compare` to see how different LLM backends
    impact the final decision and risk scores for the same email.
    """
    payload = {
        "subject": subject,
        "body": body,
        "model_names": model_names,
    }
    resp = requests.post(
        f"{backend_url}/cross_model_compare",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def dataset_summary_request(
    backend_url: str,
    max_samples: int,
) -> Dict[str, Any]:
    """
    Call `/dataset_summary` to retrieve aggregate statistics about
    the processed dataset (label distribution, lengths, imbalance, etc.).
    """
    payload = {"max_samples": max_samples}
    resp = requests.post(
        f"{backend_url}/dataset_summary",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def dataset_features_request(
    backend_url: str,
    max_samples: int,
    top_k: int,
) -> Dict[str, Any]:
    """
    Call `/dataset_features` to retrieve feature-level statistics,
    including per-email word/URL counts and the most informative tokens
    per class based on TF-IDF.
    """
    payload = {"max_samples": max_samples, "top_k": top_k}
    resp = requests.post(
        f"{backend_url}/dataset_features",
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


# --------- Streamlit UI ---------


# Global Streamlit page configuration (title, favicon, layout).
st.set_page_config(
    page_title="Phish Multi-Agent Analyzer",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Phish Multi-Agent Analyzer")
st.write(
    "Analyze emails using a multi-agent pipeline with ML, heuristics, threat intel, "
    "and local LLMs (Ollama). Includes an evaluation dashboard over the training dataset."
)

# Sidebar configuration for backend connectivity and LLM selection.
st.sidebar.header("Backend & Model Settings")

backend_url_input = st.sidebar.text_input(
    "Backend URL",
    value=DEFAULT_BACKEND_URL,
    help="FastAPI backend base URL.",
)
# Persist chosen backend URL in session so it's reused in helper calls.
st.session_state["backend_url"] = backend_url_input.strip() or DEFAULT_BACKEND_URL

backend_url = get_backend_url()
models_info = fetch_models(backend_url)

available_models: List[str] = models_info.get("available_models", [])
default_model_name: str = models_info.get("default_model", "")

# If backend did not return any models, fall back to a known-good set.
if not available_models:
    available_models = ["qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"]
    default_model_name = "qwen2.5:3b"

# Allow user to choose which Ollama LLM variant is used in analysis.
selected_model = st.sidebar.selectbox(
    "LLM Model (Ollama)",
    options=available_models,
    index=available_models.index(default_model_name)
    if default_model_name in available_models
    else 0,
)

st.sidebar.markdown("---")
st.sidebar.write("**Status:**")
# Lightweight backend health check, with defensive error handling.
try:
    health_resp = requests.get(f"{backend_url}/health", timeout=3)
    if health_resp.ok:
        st.sidebar.success("Backend is online ✅")
    else:
        st.sidebar.warning("Backend health check failed.")
except Exception as e:
    st.sidebar.error(f"Backend unreachable: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("Use the tabs on the main page to switch between analysis and evaluation.")

# --------- Tabs: Analyzer / Evaluation ---------


tab_analyzer, tab_eval = st.tabs(["🔍 Email Analyzer", "📊 Evaluation & Performance"])

# ===================== TAB 1: ANALYZER =====================
with tab_analyzer:
    # Main input area for single-email analysis.
    with st.container():
        col1, col2 = st.columns([1, 2])

        with col1:
            subject = st.text_input("Email Subject", value="Urgent: Verify your account now")

        with col2:
            body = st.text_area(
                "Email Body (plain text)",
                height=220,
                value=(
                    "Dear user,\n\n"
                    "We detected unusual activity on your account. Please log in immediately at "
                    "https://secure-bank-support.xyz/login to verify your information.\n\n"
                    "Regards,\nFake Bank Security Team"
                ),
            )

    # Primary action button to trigger analysis.
    analyze_clicked = st.button("Analyze Email", type="primary")

    if analyze_clicked:
        if not subject.strip() and not body.strip():
            st.error("Please enter at least a subject or a body.")
        else:
            with st.spinner("Running multi-agent analysis..."):
                try:
                    result = analyze_email_request(
                        backend_url=backend_url,
                        subject=subject,
                        body=body,
                        llm_model_name=selected_model,
                    )
                except Exception as e:
                    st.error(f"Error calling backend: {e}")
                    result = None

            if result is not None:
                # Top summary – high-level decision and key scores.
                st.subheader("Summary")

                decision = result.get("decision", "unknown")
                risk_level = result.get("risk_level", "low")
                risk_score = result.get("risk_score", 0.0)
                p_benign = result.get("p_benign", 0.0)
                p_phishing = result.get("p_phishing", 0.0)
                heuristic_score = result.get("heuristic_score", 0.0)
                urls = result.get("urls", [])
                model_used = result.get("llm_model_name", selected_model)

                cols = st.columns(4)
                with cols[0]:
                    st.metric("Decision", decision)
                with cols[1]:
                    st.metric("Risk Level", risk_level)
                with cols[2]:
                    st.metric("Risk Score", f"{risk_score:.3f}")
                with cols[3]:
                    st.metric("Model", model_used)

                st.markdown("---")

                # More granular numeric details.
                st.subheader("Detection Details")

                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric("ML: P(benign)", f"{p_benign:.3f}")
                with mcol2:
                    st.metric("ML: P(phishing)", f"{p_phishing:.3f}")
                with mcol3:
                    st.metric("Heuristic Score", f"{heuristic_score:.3f}")

                # URLs section (surface URLs directly; TI details are in forensic notes).
                st.subheader("URLs & Threat Intelligence")
                if urls:
                    for u in urls:
                        st.write(f"- `{u}`")
                    st.caption(
                        "Threat intel hits (OpenPhish/URLHaus) are reflected in the risk score "
                        "and in the forensic notes."
                    )
                else:
                    st.write("No URLs detected in this email.")

                # Explanation + guidance + safe reply from LLM.
                st.markdown("---")
                st.subheader("Explanation & User Guidance")

                explanation = result.get("explanation", "")
                user_guidance = result.get("user_guidance", "")
                suggested_reply = result.get("suggested_reply", "")

                with st.expander("Why this decision? (LLM explanation)", expanded=True):
                    st.write(explanation or "_No explanation available._")

                with st.expander("What should the user do?", expanded=True):
                    st.write(user_guidance or "_No guidance available._")

                with st.expander("Suggested safe reply (LLM-generated)"):
                    st.write(suggested_reply or "_No reply generated._")

                # SOC / Forensics (operator-facing).
                st.markdown("---")
                st.subheader("SOC & Forensics View")

                soc_recs = result.get("soc_recommendations", [])
                forensic_notes = result.get("forensic_notes", "")

                with st.expander("SOC Recommendations", expanded=True):
                    if soc_recs:
                        for rec in soc_recs:
                            st.write(f"- {rec}")
                    else:
                        st.write("_No SOC recommendations._")

                with st.expander("Forensic Notes (text summary)"):
                    st.code(forensic_notes or "_No forensic notes._", language="text")

                # Raw JSON block – useful when debugging backend behaviour.
                with st.expander("Raw Response (JSON)"):
                    st.code(json.dumps(result, indent=2), language="json")
    else:
        st.info("Enter an email subject and body, then click **Analyze Email** to run the multi-agent pipeline.")

# ===================== TAB 2: EVALUATION & PERFORMANCE =====================
with tab_eval:
    st.subheader("Model Evaluation on Processed Dataset")

    st.write(
        "This runs the saved classifier on the processed combined dataset "
        "(`data/processed/combined.jsonl`) and reports aggregate metrics."
    )

    # Global max_samples slider used across evaluation helpers.
    max_samples = st.slider(
        "Maximum samples to evaluate (for speed)",
        min_value=1000,
        max_value=70000,
        value=5000,
        step=1000,
        help="If the dataset is larger than this, a random subset is used.",
    )

    eval_cols = st.columns(2)
    with eval_cols[0]:
        run_eval_clicked = st.button("Run Evaluation")
    with eval_cols[1]:
        st.write("")

    # ----- Offline evaluation summary -----
    if run_eval_clicked:
        with st.spinner("Running offline evaluation on processed dataset..."):
            try:
                eval_result = eval_summary_request(
                    backend_url=backend_url,
                    max_samples=max_samples,
                )
            except Exception as e:
                st.error(f"Error calling /eval_summary: {e}")
                eval_result = None

        if eval_result is not None:
            base_stats = eval_result.get("base_stats", {})
            report = eval_result.get("report", {})
            conf = eval_result.get("confusion_matrix", {})

            # Base stats section (size + class counts).
            st.markdown("### Dataset Summary")
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                st.metric("Samples evaluated", base_stats.get("n_samples", 0))
            with bcol2:
                label_counts = base_stats.get("label_counts", {})
                benign_count = label_counts.get("benign", 0)
                st.metric("Benign count", benign_count)
            with bcol3:
                label_counts = base_stats.get("label_counts", {})
                phishing_count = label_counts.get("phishing", 0)
                st.metric("Phishing count", phishing_count)

            st.caption(
                f"Mean predicted phishing probability over all samples: "
                f"{base_stats.get('p_phishing_mean', 0.0):.3f}"
            )

            # Classification report (per-class precision/recall/F1).
            st.markdown("### Classification Report")

            # Extract class-level metrics
            benign_metrics = report.get("benign", {})
            phishing_metrics = report.get("phishing", {})
            accuracy = report.get("accuracy", 0.0)
            macro_avg = report.get("macro avg", {})
            weighted_avg = report.get("weighted avg", {})

            # Show per-class metrics in three columns.
            st.write("**Per-class metrics**")
            rep_cols = st.columns(3)
            with rep_cols[0]:
                st.markdown("**Benign**")
                st.write(
                    f"- Precision: {benign_metrics.get('precision', 0.0):.3f}\n"
                    f"- Recall: {benign_metrics.get('recall', 0.0):.3f}\n"
                    f"- F1-score: {benign_metrics.get('f1-score', 0.0):.3f}\n"
                    f"- Support: {benign_metrics.get('support', 0)}"
                )
            with rep_cols[1]:
                st.markdown("**Phishing**")
                st.write(
                    f"- Precision: {phishing_metrics.get('precision', 0.0):.3f}\n"
                    f"- Recall: {phishing_metrics.get('recall', 0.0):.3f}\n"
                    f"- F1-score: {phishing_metrics.get('f1-score', 0.0):.3f}\n"
                    f"- Support: {phishing_metrics.get('support', 0)}"
                )
            with rep_cols[2]:
                st.markdown("**Overall**")
                st.write(
                    f"- Accuracy: {accuracy:.3f}\n"
                    f"- Macro F1: {macro_avg.get('f1-score', 0.0):.3f}\n"
                    f"- Weighted F1: {weighted_avg.get('f1-score', 0.0):.3f}"
                )

            # Confusion matrix (default 0.5 threshold from backend).
            st.markdown("### Confusion Matrix (default threshold 0.5)")
            labels = conf.get("labels", [])
            matrix = conf.get("matrix", [])

            if labels and matrix and len(matrix) == 2 and len(matrix[0]) == 2:
                cm_table = [
                    ["", f"Pred {labels[0]}", f"Pred {labels[1]}"],
                    [f"True {labels[0]}", matrix[0][0], matrix[0][1]],
                    [f"True {labels[1]}", matrix[1][0], matrix[1][1]],
                ]
                st.table(cm_table)
            else:
                st.write("Confusion matrix unavailable or malformed.")

            # Raw JSON for deeper debugging / reporting.
            with st.expander("Raw Evaluation JSON"):
                st.code(json.dumps(eval_result, indent=2), language="json")
    else:
        st.info(
            "Set the maximum number of samples and click **Run Evaluation** "
            "to see offline metrics for the classifier."
        )

    # --------- Dataset Insights --------------
    st.markdown("---")
    st.subheader("Dataset Insights")

    di_col1, di_col2 = st.columns(2)
    with di_col1:
        top_k_tokens = st.slider(
            "Top tokens per class (for TF-IDF importance)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
        )
    with di_col2:
        run_dataset_insights = st.button("Compute Dataset Insights")

    if run_dataset_insights:
        summary = None
        feats = None
        with st.spinner("Computing dataset summary & feature statistics..."):
            try:
                summary = dataset_summary_request(
                    backend_url=backend_url,
                    max_samples=max_samples,
                )
                feats = dataset_features_request(
                    backend_url=backend_url,
                    max_samples=max_samples,
                    top_k=top_k_tokens,
                )
            except Exception as e:
                st.error(f"Error calling dataset insight endpoints: {e}")

        if summary is not None and feats is not None:
            # High-level metrics (imbalance, ratios, etc.)
            sm_col1, sm_col2, sm_col3, sm_col4 = st.columns(4)
            with sm_col1:
                st.metric("Total samples", summary.get("n_samples", 0))
            with sm_col2:
                st.metric(
                    "Phishing ratio",
                    f"{summary.get('phishing_ratio', 0.0):.3f}",
                )
            with sm_col3:
                st.metric(
                    "Imbalance ratio (maj/min)",
                    f"{summary.get('imbalance_ratio', 0.0):.3f}",
                )
            with sm_col4:
                if summary.get("imbalance_infinite", False):
                    st.metric("Imbalance infinite?", "Yes")
                else:
                    st.metric("Imbalance infinite?", "No")

            # Label distribution bar chart for quick visual skew inspection.
            label_counts = summary.get("label_counts", {})
            if label_counts:
                st.markdown("#### Label Distribution")
                labels = list(label_counts.keys())
                counts = [label_counts[k] for k in labels]
                fig_l, ax_l = plt.subplots()
                x = np.arange(len(labels))
                ax_l.bar(x, counts)
                ax_l.set_xticks(x)
                ax_l.set_xticklabels(labels)
                ax_l.set_ylabel("Count")
                ax_l.set_title("Labels in processed dataset")
                st.pyplot(fig_l)

            # Source distribution (if dataset was combined from multiple sources).
            source_counts = summary.get("source_counts", {})
            if source_counts:
                st.markdown("#### Source Distribution")
                src_labels = list(source_counts.keys())
                src_counts = [source_counts[k] for k in src_labels]
                fig_s, ax_s = plt.subplots()
                x = np.arange(len(src_labels))
                ax_s.bar(x, src_counts)
                ax_s.set_xticks(x)
                ax_s.set_xticklabels(src_labels, rotation=20, ha="right")
                ax_s.set_ylabel("Count")
                ax_s.set_title("Samples per source dataset")
                st.pyplot(fig_s)

            # Basic token-length statistics for emails.
            st.markdown("#### Length Statistics (subject + body tokens)")
            st.write(
                f"- Avg length: {summary.get('avg_length', 0.0):.1f} tokens\n"
                f"- Median length: {summary.get('median_length', 0.0):.1f} tokens\n"
                f"- 95th percentile: {summary.get('p95_length', 0.0):.1f} tokens\n"
                f"- Min length: {summary.get('min_length', 0.0):.0f} tokens\n"
                f"- Max length: {summary.get('max_length', 0.0):.0f} tokens\n"
            )

            # Word & URL count histograms.
            st.markdown("#### Word & URL Count Distributions")
            word_counts = np.array(feats.get("word_counts", []))
            url_counts = np.array(feats.get("url_counts", []))

            if word_counts.size > 0:
                fig_wc, ax_wc = plt.subplots()
                ax_wc.hist(word_counts, bins=30)
                ax_wc.set_xlabel("Words per email")
                ax_wc.set_ylabel("Frequency")
                ax_wc.set_title("Distribution of word counts")
                st.pyplot(fig_wc)

            if url_counts.size > 0:
                fig_uc, ax_uc = plt.subplots()
                ax_uc.hist(url_counts, bins=20)
                ax_uc.set_xlabel("URLs per email")
                ax_uc.set_ylabel("Frequency")
                ax_uc.set_title("Distribution of URL counts")
                st.pyplot(fig_uc)

            # Most informative tokens for phishing vs benign class.
            st.markdown("#### Top TF-IDF Tokens")
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.write("**Tokens associated with phishing**")
                phish_tokens = feats.get("top_phishing_tokens", [])
                st.table(phish_tokens)
            with tcol2:
                st.write("**Tokens associated with benign emails**")
                benign_tokens = feats.get("top_benign_tokens", [])
                st.table(benign_tokens)

            # Raw JSON payloads for further offline analysis.
            with st.expander("Raw dataset summary JSON"):
                st.code(json.dumps(summary, indent=2), language="json")
            with st.expander("Raw dataset feature stats JSON"):
                st.code(json.dumps(feats, indent=2), language="json")
    else:
        st.info(
            "Use **Dataset Insights** to understand label balance, sources, "
            "lengths, and key tokens in the processed dataset."
        )

    # --------- Threshold Analysis (p_phishing) ---------
    st.markdown("---")
    st.subheader("Threshold Analysis (Phishing Probability)")

    st.write(
        "This analyzes different decision thresholds on the phishing probability (p_phishing) "
        "to see how precision, recall, F1, and error rates change for phishing as the positive class."
    )

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        num_thresholds = st.slider(
            "Number of thresholds",
            min_value=5,
            max_value=101,
            value=21,
            step=4,
            help="Number of evenly spaced thresholds between 0.0 and 1.0.",
        )
    with tcol2:
        run_thresh_clicked = st.button("Run Threshold Analysis")

    thresh_result = None
    if run_thresh_clicked:
        with st.spinner("Sweeping thresholds over p_phishing..."):
            try:
                thresh_result = threshold_sweep_request(
                    backend_url=backend_url,
                    max_samples=max_samples,
                    num_thresholds=num_thresholds,
                )
            except Exception as e:
                st.error(f"Error calling /threshold_sweep: {e}")
                thresh_result = None

        if thresh_result is not None:
            best = thresh_result.get("best_by_f1", {})
            thresholds = thresh_result.get("thresholds", [])
            f1_vals = thresh_result.get("f1", [])
            precision_vals = thresh_result.get("precision", [])
            recall_vals = thresh_result.get("recall", [])

            st.markdown("### Best Threshold (by F1 for 'phishing')")

            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                st.metric("Best threshold", f"{best.get('threshold', 0.5):.3f}")
            with bcol2:
                st.metric("F1 (phishing)", f"{best.get('f1', 0.0):.3f}")
            with bcol3:
                st.metric("Accuracy", f"{best.get('accuracy', 0.0):.3f}")

            bcol4, bcol5 = st.columns(2)
            with bcol4:
                st.metric("Precision (phishing)", f"{best.get('precision', 0.0):.3f}")
            with bcol5:
                st.metric("Recall (phishing)", f"{best.get('recall', 0.0):.3f}")

            st.caption(
                "Threshold tuning is based on phishing as the positive class. "
                "You can use this to justify your chosen operating point in the report."
            )

            # Simple table of thresholds and F1/precision/recall (top 10 by F1).
            st.markdown("### Threshold Sweep Table (Top 10 by F1)")
            rows = []
            for t, p, r, f in zip(thresholds, precision_vals, recall_vals, f1_vals):
                rows.append(
                    {
                        "threshold": round(t, 3),
                        "precision_phishing": round(p, 3),
                        "recall_phishing": round(r, 3),
                        "f1_phishing": round(f, 3),
                    }
                )

            # Sort rows by F1 descending and show top 10.
            rows_sorted = sorted(rows, key=lambda x: x["f1_phishing"], reverse=True)[:10]
            st.table(rows_sorted)

            with st.expander("Raw Threshold Sweep JSON"):
                st.code(json.dumps(thresh_result, indent=2), language="json")
    else:
        st.info(
            "Adjust the number of thresholds and click **Run Threshold Analysis** "
            "to explore the trade-off between precision and recall for phishing."
        )

    # --------- Performance Curves (ROC & PR) ---------
    st.markdown("---")
    st.subheader("Performance Curves (ROC & Precision–Recall)")

    st.write(
        "These curves are computed on the processed dataset with phishing as the positive class. "
        "Use them in your report to show classifier quality."
    )

    ccol1, ccol2 = st.columns(2)
    with ccol1:
        run_curves_clicked = st.button("Generate ROC & PR Curves")
    with ccol2:
        threshold_for_cm = st.slider(
            "Threshold for detailed confusion matrix",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )

    roc_result = None
    pr_result = None
    cm_thresh_result = None

    if run_curves_clicked:
        with st.spinner("Computing ROC & PR curves and confusion matrix..."):
            try:
                roc_result = roc_curve_request(
                    backend_url=backend_url,
                    max_samples=max_samples,
                )
                pr_result = pr_curve_request(
                    backend_url=backend_url,
                    max_samples=max_samples,
                )
                cm_thresh_result = confusion_at_threshold_request(
                    backend_url=backend_url,
                    max_samples=max_samples,
                    threshold=threshold_for_cm,
                )
            except Exception as e:
                st.error(f"Error calling performance endpoints: {e}")
                roc_result = pr_result = cm_thresh_result = None

        # --- ROC PLOT ---
        if roc_result is not None:
            st.markdown("### ROC Curve (phishing as positive class)")
            fpr = np.array(roc_result.get("fpr", []))
            tpr = np.array(roc_result.get("tpr", []))
            auc_val = roc_result.get("auc", 0.0)

            if fpr.size > 0 and tpr.size > 0:
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve (Phishing vs Benign)")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            else:
                st.write("ROC data unavailable.")

        # --- PR PLOT ---
        if pr_result is not None:
            st.markdown("### Precision–Recall Curve (phishing as positive class)")
            precision = np.array(pr_result.get("precision", []))
            recall = np.array(pr_result.get("recall", []))
            auc_pr = pr_result.get("auc_pr", 0.0)

            if precision.size > 0 and recall.size > 0:
                fig_pr, ax_pr = plt.subplots()
                ax_pr.plot(recall, precision, label=f"AUC(PR) = {auc_pr:.3f}")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.set_title("Precision–Recall Curve (Phishing positive)")
                ax_pr.legend(loc="lower left")
                st.pyplot(fig_pr)
            else:
                st.write("PR curve data unavailable.")

        # --- CONFUSION MATRIX HEATMAP @ THRESHOLD ---
        if cm_thresh_result is not None:
            st.markdown(f"### Confusion Matrix @ Threshold {threshold_for_cm:.2f}")

            labels = cm_thresh_result.get("labels", ["benign", "phishing"])
            matrix = cm_thresh_result.get("matrix", [[0, 0], [0, 0]])
            metrics = cm_thresh_result.get("metrics", {})

            cm_arr = np.array(matrix)

            fig_cm, ax_cm = plt.subplots()
            im = ax_cm.imshow(cm_arr)

            ax_cm.set_xticks(np.arange(2))
            ax_cm.set_yticks(np.arange(2))
            ax_cm.set_xticklabels([f"Pred {labels[0]}", f"Pred {labels[1]}"])
            ax_cm.set_yticklabels([f"True {labels[0]}", f"True {labels[1]}"])

            # Annotate each matrix cell with count.
            for i in range(2):
                for j in range(2):
                    ax_cm.text(
                        j,
                        i,
                        str(matrix[i][j]),
                        ha="center",
                        va="center",
                    )

            ax_cm.set_title("Confusion Matrix Heatmap")
            fig_cm.colorbar(im, ax=ax_cm)
            st.pyplot(fig_cm)

            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Precision (phishing)", f"{metrics.get('precision_phishing', 0.0):.3f}")
            with mcol2:
                st.metric("Recall (phishing)", f"{metrics.get('recall_phishing', 0.0):.3f}")
            with mcol3:
                st.metric("F1 (phishing)", f"{metrics.get('f1_phishing', 0.0):.3f}")

            mcol4, mcol5 = st.columns(2)
            with mcol4:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.3f}")
            with mcol5:
                st.metric("FP rate", f"{metrics.get('fp_rate', 0.0):.3f}")

            with st.expander("Raw ROC / PR / Confusion JSON"):
                combined = {
                    "roc": roc_result,
                    "pr": pr_result,
                    "confusion_at_threshold": cm_thresh_result,
                }
                st.code(json.dumps(combined, indent=2), language="json")
    else:
        st.info(
            "Set the maximum samples, choose a threshold, then click "
            "**Generate ROC & PR Curves** to visualize classifier performance."
        )

    # --------- Robustness: Adversarial Mutations ---------
    st.markdown("---")
    st.subheader("Robustness Testing (Adversarial Mutations)")

    st.write(
        "Generate small mutated versions of a base email and see how stable the "
        "classifier is under typos, URL obfuscation, and noise."
    )

    rm_col1, rm_col2 = st.columns(2)
    with rm_col1:
        adv_subject = st.text_input(
            "Base Subject (robustness test)",
            value="Urgent: Verify your account now",
            key="adv_subject",
        )
    with rm_col2:
        adv_strength = st.selectbox(
            "Mutation strength",
            options=["light", "medium", "strong"],
            index=1,
            key="adv_strength",
        )

    adv_body = st.text_area(
        "Base Body (robustness test)",
        height=180,
        value=(
            "Dear user,\n\n"
            "We detected unusual activity on your account. Please log in immediately at "
            "https://secure-bank-support.xyz/login to verify your information.\n\n"
            "Regards,\nFake Bank Security Team"
        ),
        key="adv_body",
    )

    adv_num_variants = st.slider(
        "Number of variants (including original)",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
        key="adv_num_variants",
    )

    run_adv_clicked = st.button("Run Robustness Test")

    adv_result = None
    if run_adv_clicked:
        if not adv_subject.strip() and not adv_body.strip():
            st.error("Please provide at least a subject or body for robustness testing.")
        else:
            with st.spinner("Running adversarial mutation analysis..."):
                try:
                    adv_result = adversarial_mutations_request(
                        backend_url=backend_url,
                        subject=adv_subject,
                        body=adv_body,
                        num_variants=adv_num_variants,
                        strength=adv_strength,
                    )
                except Exception as e:
                    st.error(f"Error calling /adversarial_mutations: {e}")
                    adv_result = None

        if adv_result is not None:
            st.markdown("### Robustness Summary")

            summary = adv_result.get("summary", {})
            base_decision = summary.get("base_decision", "benign")
            max_risk = summary.get("max_risk_score", 0.0)
            min_risk = summary.get("min_risk_score", 0.0)
            flipped = summary.get("num_flipped_decisions", 0)
            decision_counts = summary.get("decision_counts", {})

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Base decision", base_decision)
            with s2:
                st.metric("Max risk score", f"{max_risk:.3f}")
            with s3:
                st.metric("Min risk score", f"{min_risk:.3f}")
            with s4:
                st.metric("Flipped decisions", int(flipped))

            st.write("**Decision distribution across variants:**")
            st.json(decision_counts)

            variants = adv_result.get("variants", [])
            if variants:
                st.markdown("### Variant Risk Scores")

                # Build a table for each variant including risk and TI details.
                table_rows = []
                risks = []
                ids = []
                for v in variants:
                    vid = v.get("variant_id", 0)
                    rs = v.get("risk_score", 0.0)
                    dc = v.get("decision", "benign")
                    risks.append(rs)
                    ids.append(vid)
                    table_rows.append(
                        {
                            "variant_id": vid,
                            "decision": dc,
                            "risk_score": round(rs, 3),
                            "p_phishing": round(v.get("p_phishing", 0.0), 3),
                            "heuristic_score": round(v.get("heuristic_score", 0.0), 3),
                            "num_urls": v.get("num_urls", 0),
                            "num_ti_hits": v.get("num_ti_hits", 0),
                        }
                    )
                st.table(table_rows)

                # Simple line plot: variant_id vs risk_score.
                if ids and risks:
                    fig_risk, ax_risk = plt.subplots()
                    ax_risk.plot(ids, risks, marker="o")
                    ax_risk.set_xlabel("Variant ID")
                    ax_risk.set_ylabel("Risk Score")
                    ax_risk.set_title("Risk Score per Mutated Variant")
                    st.pyplot(fig_risk)

            with st.expander("Raw adversarial JSON"):
                st.code(json.dumps(adv_result, indent=2), language="json")

    # --------- Robustness: Cross-Model Comparison ---------
    st.markdown("---")
    st.subheader("Model Robustness Panel (Cross-Model Comparison)")

    st.write(
        "Compare decisions and risk scores across different local LLMs "
        "used in the LangGraph pipeline."
    )

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        cm_subject = st.text_input(
            "Subject (cross-model)",
            value="Security alert: verify your identity",
            key="cm_subject",
        )
    with cm_col2:
        # Allow selection of any subset of available models for comparison.
        selected_cm_models = st.multiselect(
            "Models to compare",
            options=available_models,
            default=available_models,
            key="cm_models",
        )

    cm_body = st.text_area(
        "Body (cross-model)",
        height=160,
        value=(
            "Dear customer,\n\n"
            "We noticed a sign-in from a new device. Please confirm your identity by visiting "
            "https://secure-bank-support.xyz/verify.\n\n"
            "Thank you,\nSecurity Team"
        ),
        key="cm_body",
    )

    run_cm_clicked = st.button("Run Cross-Model Comparison")

    if run_cm_clicked:
        if not cm_subject.strip() and not cm_body.strip():
            st.error("Please provide at least a subject or body for cross-model comparison.")
        elif not selected_cm_models:
            st.error("Select at least one model to compare.")
        else:
            with st.spinner("Running cross-model analysis..."):
                try:
                    cm_result = cross_model_compare_request(
                        backend_url=backend_url,
                        subject=cm_subject,
                        body=cm_body,
                        model_names=selected_cm_models,
                    )
                except Exception as e:
                    st.error(f"Error calling /cross_model_compare: {e}")
                    cm_result = None

            if cm_result is not None:
                st.markdown("### Cross-Model Summary")

                rows = []
                models = []
                risks = []

                for entry in cm_result.get("results", []):
                    name = entry.get("llm_model_name", "")
                    dec = entry.get("decision", "unknown")
                    rlevel = entry.get("risk_level", "low")
                    rscore = entry.get("risk_score", 0.0)
                    p_ben = entry.get("p_benign", 0.0)
                    p_ph = entry.get("p_phishing", 0.0)
                    err = entry.get("error")

                    rows.append(
                        {
                            "model": name,
                            "decision": dec,
                            "risk_level": rlevel,
                            "risk_score": round(rscore, 3),
                            "p_benign": round(p_ben, 3),
                            "p_phishing": round(p_ph, 3),
                            "error": err,
                        }
                    )
                    models.append(name)
                    risks.append(rscore)

                st.table(rows)

                # Bar chart of risk_score by model for visual comparison.
                if models and risks:
                    fig_cm_risk, ax_cm_risk = plt.subplots()
                    x = np.arange(len(models))
                    ax_cm_risk.bar(x, risks)
                    ax_cm_risk.set_xticks(x)
                    ax_cm_risk.set_xticklabels(models, rotation=30, ha="right")
                    ax_cm_risk.set_ylabel("Risk Score")
                    ax_cm_risk.set_title("Risk Score by LLM Model")
                    st.pyplot(fig_cm_risk)

                with st.expander("Raw cross-model JSON"):
                    st.code(json.dumps(cm_result, indent=2), language="json")
