"""
Graph builder for the LangGraph-based phishing email analysis pipeline.

This module defines:
  • The full sequence of nodes (ingest → filter → threat_intel → explain →
    response → soc → forensics → END)
  • A helper to compile that graph for execution
  • A convenience wrapper (`run_email_analysis`) to run the entire pipeline
    in one call and return the final annotated state.

The graph structure is intentionally linear for v1, making the pipeline
easy to reason about while still modular enough to extend in future versions.
"""

from typing import Dict, Any

from langgraph.graph import StateGraph, END

from .state import EmailAnalysisState
from . import nodes


def build_email_graph() -> StateGraph:
    """
    Construct and return an uncompiled StateGraph for email analysis.

    Nodes registered here correspond to distinct processing stages:
      - ingest:      normalize inputs, extract initial metadata
      - filter:      ML + heuristic classification + risk scoring
      - threat_intel:query TI feeds for URL reputation
      - explain:     generate human-readable explanations via LLM
      - response:    create a safe or helpful email reply
      - soc:         generate SOC-facing recommendations
      - forensics:   record deep-dive artifacts for dashboards/logging

    Edges are wired into a simple linear chain for clarity.
    """
    graph = StateGraph(EmailAnalysisState)

    # Register pipeline nodes (each implemented in backend.graph.nodes)
    graph.add_node("ingest", nodes.ingest_node)
    graph.add_node("filter", nodes.filter_node)
    graph.add_node("threat_intel", nodes.threat_intel_node)
    graph.add_node("explain", nodes.explainability_node)
    graph.add_node("response", nodes.response_node)
    graph.add_node("soc", nodes.soc_node)
    graph.add_node("forensics", nodes.forensics_node)

    # Define deterministic linear flow for v1 pipeline.
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "filter")
    graph.add_edge("filter", "threat_intel")
    graph.add_edge("threat_intel", "explain")
    graph.add_edge("explain", "response")
    graph.add_edge("response", "soc")
    graph.add_edge("soc", "forensics")
    graph.add_edge("forensics", END)

    return graph


def get_compiled_graph():
    """
    Compile the raw StateGraph into an executable graph.

    Compilation performs internal graph validation and returns
    a callable application object with `.invoke()` used for
    synchronous execution.
    """
    graph = build_email_graph()
    return graph.compile()


def run_email_analysis(
    subject: str,
    body: str,
    llm_model_name: str = "qwen2.5:3b",
) -> Dict[str, Any]:
    """
    Execute the email analysis pipeline end-to-end.

    Parameters:
        subject         — email subject line
        body            — email body text
        llm_model_name  — optional override for LLM used in explanation/response

    Returns:
        A plain `dict` representing the final EmailAnalysisState after all nodes
        have executed, suitable for serialization (e.g., JSON to frontend).
    """
    app = get_compiled_graph()

    # Initial state is a plain dict matching EmailAnalysisState keys.
    initial_state: EmailAnalysisState = {
        "subject": subject,
        "body": body,
        "llm_model_name": llm_model_name,
    }

    # Execute the compiled graph synchronously.
    final_state = app.invoke(initial_state)

    # Convert TypedDict → standard dict for easier downstream handling.
    return dict(final_state)
