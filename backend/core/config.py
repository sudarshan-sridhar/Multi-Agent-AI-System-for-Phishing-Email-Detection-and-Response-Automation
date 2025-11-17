"""
Global configuration module for the Multi-Phishing LangChain backend.

This file centralizes all path configurations, classifier settings,
heuristic scoring parameters, LLM backend configuration, and threat-intelligence
feed settings.

Nothing here contains executable logic — only static configuration — ensuring
clean separation of concerns and avoiding hard-coded paths scattered across
the codebase.

All configurations are instantiated at the bottom so they can be imported
anywhere in the backend with consistent values.
"""

import os
from dataclasses import dataclass
from typing import List


# -------------------------------------------------------------
# Path Configuration
# -------------------------------------------------------------
# The following section defines absolute paths used by the backend.
# Paths are resolved relative to the project root so the system works
# even when installed or run from different environments.

# BACKEND_DIR resolves to: <project_root>/backend
BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))          # .../backend

# PROJECT_ROOT resolves to the main project folder that contains /backend and /data.
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)                       # .../phish-lc

# DATA_DIR contains datasets, cached processed data, and TI (threat intelligence) files.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# TI_DIR houses OpenPhish + URLHaus cached feeds.
TI_DIR = os.path.join(DATA_DIR, "ti")

# MODEL_DIR holds trained model artifacts such as joblib classifier files.
MODEL_DIR = os.path.join(BACKEND_DIR, "models")

# Ensure these folders exist so no caller ever fails due to missing directories.
# This is safe because `exist_ok=True` prevents accidental overwrites.
os.makedirs(TI_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------------------------------------------
# Classifier Model Configuration
# -------------------------------------------------------------
# Bundles all model-related configuration into a dataclass to avoid scattering
# magic strings or file paths across multiple modules.

@dataclass
class ClassifierConfig:
    """
    Configuration for the ML classifier used in phishing detection.

    - `model_path`: absolute path to the serialized joblib classifier.
    - `phishing_label` / `benign_label`: the canonical labels the classifier outputs.
    """

    model_path: str = os.path.join(MODEL_DIR, "email_classifier.joblib")

    phishing_label: str = "phishing"
    benign_label: str = "benign"


# -------------------------------------------------------------
# Heuristics + Scoring System Configuration
# -------------------------------------------------------------
# Controls how ML scores, heuristics, and TI-based signals are blended.
# These weights and thresholds govern the risk-scoring pipeline behavior.

@dataclass
class HeuristicConfig:
    """
    Tuning parameters for the heuristic + ML ensemble scoring system.

    - `phishing_threshold`: base probability cutoff for ML classifier.
    - `high_risk_threshold`: severity threshold for SOC escalation.
    - `max_urls_considered`: upper bound for URL inspection to avoid expensive scans.
    - `ml_weight` / `heuristic_weight` / `ti_weight`: ensemble scoring distribution.
    """

    phishing_threshold: float = 0.5
    high_risk_threshold: float = 0.8
    max_urls_considered: int = 20

    ml_weight: float = 0.6
    heuristic_weight: float = 0.2
    ti_weight: float = 0.2


# -------------------------------------------------------------
# Local LLM (Ollama) Configuration
# -------------------------------------------------------------
# Centralizes all configuration for interacting with locally-hosted LLMs via Ollama.

@dataclass
class LLMConfig:
    """
    Configuration for local LLM models served by Ollama.

    - `available_models`: list of identifiers surfaced in the `/models` endpoint.
    - `default_model`: fallback model when none is specified by the frontend.
    - `base_url`: HTTP endpoint to the Ollama server.
    - `timeout`: request timeout applied across downstream LLM calls.
    """

    available_models: List[str] = (
        "qwen2.5:3b",
        "qwen2.5:7b",
        "llama3.1:8b",
    )

    default_model: str = "qwen2.5:3b"

    base_url: str = "http://localhost:11434"
    timeout: int = 60


# -------------------------------------------------------------
# Threat Intelligence Feeds Configuration
# -------------------------------------------------------------
# Defines URLs to external feeds and the corresponding cache file locations.
# These settings support the TI lookup utilities used throughout the analysis pipeline.

@dataclass
class ThreatIntelConfig:
    """
    Configuration for external threat-intelligence sources.

    - `openphish_url` / `urlhaus_url`: external feed endpoints.
    - `cache_path_openphish` / `cache_path_urlhaus`: local cache file paths.
    """

    openphish_url: str = "https://openphish.com/feed.txt"
    urlhaus_url: str = "https://urlhaus.abuse.ch/downloads/hostfile/"

    cache_path_openphish: str = os.path.join(TI_DIR, "openphish_urls.txt")
    cache_path_urlhaus: str = os.path.join(TI_DIR, "urlhaus_hosts.txt")


# -------------------------------------------------------------
# Instantiate Global Configuration Objects
# -------------------------------------------------------------
# These global instances are used throughout the backend via imports.
# Keeping them here ensures all modules read consistent configuration values.

classifier_config = ClassifierConfig()
heuristic_config = HeuristicConfig()
llm_config = LLMConfig()
ti_config = ThreatIntelConfig()
