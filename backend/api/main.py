"""
FastAPI backend for the Multi-Phishing LangChain project.

This service exposes:
- Online email analysis via LLM-powered multi-agent graph (`/analyze_email`).
- Model discovery (`/models`, tied to `llm_config`).
- Offline evaluation utilities (ROC/PR curves, threshold sweeps, confusion matrices).
- Dataset insight endpoints (summary/stats for processed phishing dataset).
- Robustness utilities (adversarial mutations, cross-model comparisons).

All critical business logic lives in the underlying `graph`, `core`, and `robustness`
modules; this file is intentionally a thin, typed HTTP API layer.
"""

from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..graph.graph_builder import run_email_analysis
from ..core.config import llm_config
from ..core.eval_utils import (
    evaluate_model_on_processed,
    sweep_thresholds,
    roc_curve_data,
    pr_curve_data,
    confusion_at_threshold,
)
from ..core.dataset_utils import dataset_summary, dataset_feature_stats
from ..core.robustness_utils import (
    generate_adversarial_mutations,
    cross_model_compare,
)


# --------- Pydantic models ---------


class EmailRequest(BaseModel):
    """
    Request payload for online email analysis.

    This is the primary entry point from the frontend:
    - `subject` and `body` represent the raw email content.
    - `llm_model_name` optionally overrides the backend default LLM.
    """

    subject: str = Field("", description="Email subject")
    body: str = Field("", description="Email body (plain text)")
    llm_model_name: Optional[str] = Field(
        None,
        description=(
            "Optional LLM model name "
            "(e.g., 'qwen2.5:3b', 'qwen2.5:7b', 'llama3.1:8b'). "
            "If omitted, backend default is used."
        ),
    )


class EmailAnalysisResponse(BaseModel):
    """
    Structured response from the email analysis pipeline.

    The response combines:
    - Scalar metrics (risk score/probabilities).
    - Parsed content (normalized subject/body/URLs).
    - Analyst-facing context (explanation, forensic notes).
    - SOC automation hints and an LLM-suggested safe reply.
    """

    decision: str
    risk_level: str
    risk_score: float
    p_benign: float
    p_phishing: float
    heuristic_score: float

    subject: str
    body: str
    urls: List[str]

    explanation: str
    user_guidance: str
    suggested_reply: str
    soc_recommendations: List[str]
    forensic_notes: str

    llm_model_name: str
    full_state: Dict[str, Any]


class ModelsResponse(BaseModel):
    """
    Response for `/models` exposing model discovery to the frontend.

    `available_models` is typically sourced from `llm_config.available_models`,
    while `default_model` indicates the backend's preferred choice.
    """

    available_models: List[str]
    default_model: str


class HealthResponse(BaseModel):
    """
    Minimal health-check response used by k8s/monitoring and the frontend.
    """

    status: str


class EvalRequest(BaseModel):
    """
    Request payload for offline evaluation endpoints.

    `max_samples` is used as a safety cap to avoid pulling the entire dataset
    into memory unintentionally when running experiments from the UI.
    """

    max_samples: int = Field(
        5000,
        ge=1,
        description=(
            "Maximum number of samples to use from the processed dataset "
            "for offline evaluation."
        ),
    )


class EvalResponse(BaseModel):
    """
    Aggregated offline evaluation result.

    - `base_stats`: dataset / split level stats.
    - `report`: metric breakdown per class / overall.
    - `confusion_matrix`: confusion counts and any derived metrics.
    """

    base_stats: Dict[str, Any]
    report: Dict[str, Any]
    confusion_matrix: Dict[str, Any]


class ThresholdSweepRequest(BaseModel):
    """
    Request payload for threshold sweeping on phishing probability.

    This is typically used to select an operating point (threshold) that
    balances FP/FN trade-offs given business constraints.
    """

    max_samples: int = Field(
        5000,
        ge=1,
        description=(
            "Maximum number of samples to use from the processed dataset "
            "for threshold sweep."
        ),
    )
    num_thresholds: int = Field(
        21,
        ge=2,
        le=201,
        description="Number of thresholds to evaluate between 0.0 and 1.0.",
    )


class ThresholdSweepResponse(BaseModel):
    """
    Threshold sweep result for phishing classification.

    All metric lists are aligned index-wise with `thresholds`.
    """

    label_positive: str
    n_samples: int
    label_counts: Dict[str, int]
    thresholds: List[float]
    precision: List[float]
    recall: List[float]
    f1: List[float]
    accuracy: List[float]
    fp_rate: List[float]
    fn_rate: List[float]
    best_by_f1: Dict[str, float]


class RocRequest(BaseModel):
    """
    Request payload for computing ROC curve.

    `max_samples` again acts as a guardrail for dataset size.
    """

    max_samples: int = Field(
        5000,
        ge=1,
        description="Maximum number of samples from processed dataset for ROC.",
    )


class RocResponse(BaseModel):
    """
    ROC curve output for phishing as the positive class.

    - `fpr` / `tpr`: lists of false/true positive rates.
    - `thresholds`: score cutoffs used to compute the points.
    - `auc`: Area Under the ROC Curve.
    """

    label_positive: str
    n_samples: int
    label_counts: Dict[str, int]
    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]
    auc: float


class PrRequest(BaseModel):
    """
    Request payload for Precision-Recall curve computation.
    """

    max_samples: int = Field(
        5000,
        ge=1,
        description="Maximum number of samples from processed dataset for PR curve.",
    )


class PrResponse(BaseModel):
    """
    Precision-Recall curve output for phishing as the positive class.

    - `precision` / `recall`: aligned with `thresholds`.
    - `auc_pr`: Area Under the PR Curve (more informative on imbalanced data).
    """

    label_positive: str
    n_samples: int
    label_counts: Dict[str, int]
    precision: List[float]
    recall: List[float]
    thresholds: List[float]
    auc_pr: float


class ConfusionAtThresholdRequest(BaseModel):
    """
    Request payload for confusion matrix computation at a fixed threshold.
    """

    max_samples: int = Field(
        5000,
        ge=1,
        description="Maximum number of samples from processed dataset.",
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold on p_phishing.",
    )


class ConfusionAtThresholdResponse(BaseModel):
    """
    Confusion matrix and derived metrics at a chosen operating threshold.
    """

    label_positive: str
    threshold: float
    n_samples: int
    label_counts: Dict[str, int]
    labels: List[str]
    matrix: List[List[int]]
    metrics: Dict[str, float]


# --------- Dataset insight models (C9/C10) ---------


class DatasetSummaryRequest(BaseModel):
    """
    Request to compute a high-level summary of the processed dataset.

    Typically used to populate an "Explore Dataset" tab in the UI.
    """

    max_samples: int = Field(
        5000,
        ge=1,
        description="Maximum number of samples for dataset summary.",
    )


class DatasetSummaryResponse(BaseModel):
    """
    High-level dataset summary stats.

    Includes size, label distribution, length statistics, and imbalance indicators.
    """

    n_samples: int
    label_counts: Dict[str, int]
    source_counts: Dict[str, int]
    avg_length: float
    median_length: float
    p95_length: float
    min_length: float
    max_length: float
    imbalance_ratio: float
    imbalance_infinite: bool
    phishing_ratio: float


class DatasetFeatureStatsRequest(BaseModel):
    """
    Request for feature-level statistics, including token distribution.

    `top_k` controls how many class-specific tokens we surface for analysis.
    """

    max_samples: int = Field(
        5000,
        ge=1,
        description="Maximum number of samples for feature statistics.",
    )
    top_k: int = Field(
        30,
        ge=1,
        le=200,
        description="Number of top tokens per class to return.",
    )


class DatasetFeatureStatsResponse(BaseModel):
    """
    Detailed feature statistics for the dataset.

    This is used for:
    - Inspecting token distributions.
    - Comparing phishing vs benign vocabularies.
    """

    n_samples: int
    word_counts: List[int]
    url_counts: List[int]
    vocab_size: int
    top_phishing_tokens: List[Dict[str, Any]]
    top_benign_tokens: List[Dict[str, Any]]


# --------- Robustness models (C11/C12) ---------


class AdversarialMutationsRequest(BaseModel):
    """
    Request payload for generating adversarial-style email variants.

    - `num_variants`: number of mutated samples to generate.
    - `strength`: qualitative control over perturbation magnitude.
    """

    subject: str = ""
    body: str = ""
    num_variants: int = Field(8, ge=1, le=50)
    strength: str = Field("medium", description="light | medium | strong")


class AdversarialMutationsResponse(BaseModel):
    """
    Response structure for adversarial mutation generation.

    - `summary`: aggregate stats about the mutations.
    - `variants`: list of concrete mutated email candidates.
    """

    summary: Dict[str, Any]
    variants: List[Dict[str, Any]]


class CrossModelCompareRequest(BaseModel):
    """
    Request to score the same email across multiple LLM backends.

    Used to detect model-specific blind spots or disagreement.
    """

    subject: str = ""
    body: str = ""
    model_names: List[str]


class CrossModelCompareResponse(BaseModel):
    """
    Response for cross-model comparison.

    Each item in `results` contains per-model outputs/metadata
    as returned by the underlying utility.
    """

    results: List[Dict[str, Any]]


# --------- FastAPI app ---------


app = FastAPI(
    title="Phish Multi-Agent Analyzer",
    version="1.0.0",
    description=(
        "FastAPI backend for multi-agent phishing email analysis "
        "using LangGraph and local LLMs."
    ),
)

# Allow Streamlit / localhost frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # NOTE: Relaxed for local/dev; consider tightening in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Routes ---------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Lightweight health-check endpoint.

    Intended for:
    - Liveness probes.
    - Simple connectivity checks from the frontend.
    """
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """
    List currently configured LLM models and the backend default.

    This allows the frontend to populate a dropdown and keep in sync with
    the server-side configuration without hard-coding model names.
    """
    return ModelsResponse(
        available_models=list(llm_config.available_models),
        default_model=llm_config.default_model,
    )


@app.post("/analyze_email", response_model=EmailAnalysisResponse)
async def analyze_email(req: EmailRequest) -> EmailAnalysisResponse:
    """
    Run the multi-agent phishing analysis graph on a single email.

    - Optionally respects `llm_model_name` to route to a specific backend model.
    - Delegates all heavy lifting to `run_email_analysis` and normalizes
      its result into a stable Pydantic response model for the UI.
    """
    # Decide which LLM model to use (explicit override or config default).
    model_name = req.llm_model_name or llm_config.default_model

    # Core analysis pipeline: graph execution + heuristics + scoring.
    result = run_email_analysis(
        subject=req.subject,
        body=req.body,
        llm_model_name=model_name,
    )

    # Extract with sane defaults to avoid KeyError and ensure type stability.
    decision = str(result.get("decision", "benign"))
    risk_level = str(result.get("risk_level", "low"))
    risk_score = float(result.get("risk_score", 0.0))
    p_benign = float(result.get("p_benign", 0.0))
    p_phishing = float(result.get("p_phishing", 0.0))
    heuristic_score = float(result.get("heuristic_score", 0.0))

    subject = str(result.get("subject", ""))
    body = str(result.get("body", ""))
    urls = list(result.get("urls", []))

    explanation = str(result.get("explanation", ""))
    user_guidance = str(result.get("user_guidance", ""))
    suggested_reply = str(result.get("suggested_reply", ""))
    soc_recommendations = list(result.get("soc_recommendations", []))
    forensic_notes = str(result.get("forensic_notes", ""))

    return EmailAnalysisResponse(
        decision=decision,
        risk_level=risk_level,
        risk_score=risk_score,
        p_benign=p_benign,
        p_phishing=p_phishing,
        heuristic_score=heuristic_score,
        subject=subject,
        body=body,
        urls=urls,
        explanation=explanation,
        user_guidance=user_guidance,
        suggested_reply=suggested_reply,
        soc_recommendations=soc_recommendations,
        forensic_notes=forensic_notes,
        llm_model_name=model_name,
        full_state=result,
    )


@app.post("/eval_summary", response_model=EvalResponse)
async def eval_summary_endpoint(req: EvalRequest) -> EvalResponse:
    """
    Run offline evaluation of the saved classifier on the processed dataset.

    Returns a high-level summary suitable for rendering in an "Evaluation" tab
    (e.g., confusion matrix, macro/micro metrics).
    """
    data = evaluate_model_on_processed(
        max_samples=req.max_samples,
    )
    return EvalResponse(**data)


@app.post("/threshold_sweep", response_model=ThresholdSweepResponse)
async def threshold_sweep_endpoint(req: ThresholdSweepRequest) -> ThresholdSweepResponse:
    """
    Sweep thresholds over the phishing probability (p_phishing) and compute
    precision, recall, F1, accuracy, false positive rate, and false negative rate
    for 'phishing' as the positive class.

    This enables interactive trade-off exploration in the frontend.
    """
    data = sweep_thresholds(
        max_samples=req.max_samples,
        num_thresholds=req.num_thresholds,
    )
    return ThresholdSweepResponse(**data)


@app.post("/roc_curve", response_model=RocResponse)
async def roc_curve_endpoint(req: RocRequest) -> RocResponse:
    """
    Compute ROC curve (FPR, TPR, thresholds, AUC) for phishing as positive class.

    Typically used to visualize classifier ranking performance across thresholds.
    """
    data = roc_curve_data(
        max_samples=req.max_samples,
    )
    return RocResponse(**data)


@app.post("/pr_curve", response_model=PrResponse)
async def pr_curve_endpoint(req: PrRequest) -> PrResponse:
    """
    Compute Precision-Recall curve for phishing as positive class.

    PR curves are especially informative on imbalanced datasets like phishing
    corpora where positives are relatively rare.
    """
    data = pr_curve_data(
        max_samples=req.max_samples,
    )
    return PrResponse(**data)


@app.post(
    "/confusion_at_threshold",
    response_model=ConfusionAtThresholdResponse,
)
async def confusion_at_threshold_endpoint(
    req: ConfusionAtThresholdRequest,
) -> ConfusionAtThresholdResponse:
    """
    Compute confusion matrix and derived metrics at a given threshold on p_phishing.

    Useful for:
    - Validating the chosen production threshold.
    - Running "what if" scenarios from the UI.
    """
    data = confusion_at_threshold(
        threshold=req.threshold,
        max_samples=req.max_samples,
    )
    return ConfusionAtThresholdResponse(**data)


# --------- Dataset insight endpoints (C9/C10) ---------


@app.post("/dataset_summary", response_model=DatasetSummaryResponse)
async def dataset_summary_endpoint(
    req: DatasetSummaryRequest,
) -> DatasetSummaryResponse:
    """
    Return a compact summary of the processed dataset.

    This is used by the analytics UI to provide quick visibility into
    label balance, length distribution, and potential issues like
    extreme imbalance.
    """
    data = dataset_summary(
        max_samples=req.max_samples,
    )
    return DatasetSummaryResponse(**data)


@app.post("/dataset_features", response_model=DatasetFeatureStatsResponse)
async def dataset_features_endpoint(
    req: DatasetFeatureStatsRequest,
) -> DatasetFeatureStatsResponse:
    """
    Return feature-level statistics for the dataset.

    This endpoint powers:
    - Token importance visualizations.
    - Comparative views of phishing vs benign vocabularies.
    """
    data = dataset_feature_stats(
        max_samples=req.max_samples,
        random_state=42,
        top_k=req.top_k,
    )
    return DatasetFeatureStatsResponse(**data)


# --------- Robustness endpoints (C11/C12) ---------


@app.post(
    "/adversarial_mutations",
    response_model=AdversarialMutationsResponse,
)
async def adversarial_mutations_endpoint(
    req: AdversarialMutationsRequest,
) -> AdversarialMutationsResponse:
    """
    Generate adversarial-like variants of a given email.

    Intended for:
    - Stress-testing the classifier and LLM graph.
    - Understanding robustness against minor content mutations.
    """
    data = generate_adversarial_mutations(
        subject=req.subject,
        body=req.body,
        num_variants=req.num_variants,
        strength=req.strength,
    )
    return AdversarialMutationsResponse(**data)


@app.post(
    "/cross_model_compare",
    response_model=CrossModelCompareResponse,
)
async def cross_model_compare_endpoint(
    req: CrossModelCompareRequest,
) -> CrossModelCompareResponse:
    """
    Compare phishing analysis outputs across multiple LLM backends.

    This endpoint is useful for:
    - A/B testing different models on the same input.
    - Detecting divergent behavior or regressions between releases.
    """
    data = cross_model_compare(
        subject=req.subject,
        body=req.body,
        model_names=req.model_names,
    )
    return CrossModelCompareResponse(**data)
