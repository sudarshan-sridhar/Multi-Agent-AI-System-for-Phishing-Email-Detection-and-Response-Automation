"""
Email feature extraction and heuristic scoring utilities.

This module is responsible for:
- Parsing URLs out of raw email text.
- Computing structural + lexical features (URL counts, suspicious TLDs,
  presence of urgent/login/financial phrasing).
- Producing a compact `EmailFeatures` dataclass used by:
    • heuristic (rule-based) scoring
    • ML-based phishing detection pipelines

The goal is to keep all low-level feature engineering logic in one place so
it can be reused consistently across the backend.
"""

import re
from dataclasses import dataclass
from typing import List, Dict

import tldextract


# -------------------------------------------------------------
# URL Extraction Pattern
# -------------------------------------------------------------
# This regex captures HTTP/HTTPS and "www." style URLs from text.
# (?i) makes it case-insensitive.
# It avoids stopping at typical punctuation or HTML characters.
URL_REGEX = re.compile(
    r"""(?i)\b((?:https?://|www\.)[^\s'">]+)"""
)


# -------------------------------------------------------------
# Email Feature Data Structure
# -------------------------------------------------------------
# Encapsulates all extracted structural features from email content.
# These features feed both:
#   - heuristics (rule-based scoring)
#   - ML-based detection
# This structure keeps feature handling consistent across modules.
@dataclass
class EmailFeatures:
    """
    Container for all first-pass structural and lexical features extracted
    from an email.

    Attributes:
        urls: Raw URL strings discovered in the email.
        num_urls: Total number of unique URLs.
        num_suspicious_tlds: Number of URLs whose TLD is in SUSPICIOUS_TLDS.
        has_urgent_words: True if text includes urgency-related keywords.
        has_login_words: True if text includes login/credential-related keywords.
        has_financial_words: True if text includes finance/payment-related keywords.
    """

    urls: List[str]
    num_urls: int
    num_suspicious_tlds: int
    has_urgent_words: bool
    has_login_words: bool
    has_financial_words: bool


# -------------------------------------------------------------
# Keyword / TLD Lists for Heuristic Detection
# -------------------------------------------------------------
# Suspicious TLDs known to be frequently abused by phishing domains.
SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "link", "work", "support", "gq", "tk",
    "ml", "cf", "ru", "cn", "live", "loan", "download", "men",
}

# Terms indicating urgency – common psychological manipulation technique.
URGENT_WORDS = [
    "urgent", "immediately", "asap", "now", "action required", "last warning",
    "account suspended", "verify your account",
]

# Terms associated with credential harvesting.
LOGIN_WORDS = [
    "login", "log in", "sign in", "password", "credentials", "verify",
    "authentication", "2fa", "otp",
]

# Terms associated with financial fraud or payment scams.
FINANCIAL_WORDS = [
    "bank", "paypal", "invoice", "payment", "transaction", "refund",
    "wire transfer", "bitcoin", "crypto", "gift card",
]


# -------------------------------------------------------------
# URL Extraction Logic
# -------------------------------------------------------------
# Pulls raw URLs from email text, removes trailing punctuation,
# and preserves uniqueness while keeping original order.
def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from arbitrary email text.

    Steps:
        1. Use URL_REGEX to find all URL-like substrings.
        2. Strip trailing punctuation that is commonly attached in prose.
        3. Deduplicate while preserving original order.

    Returns:
        List of unique URL strings, or an empty list if none are found.
    """
    if not text:
        return []
    urls = URL_REGEX.findall(text)

    # Clean trailing characters like ")" or "." which often appear after URLs.
    cleaned = []
    for u in urls:
        cleaned.append(u.rstrip(").,;\"'"))

    # dict.fromkeys(...) keeps the first occurrence and removes duplicates.
    return list(dict.fromkeys(cleaned))


# -------------------------------------------------------------
# Suspicious TLD Counter
# -------------------------------------------------------------
# Uses tldextract to accurately parse a URL's top-level domain
# (more reliable than manual string operations).
def count_suspicious_tlds(urls: List[str]) -> int:
    """
    Count how many URLs have TLDs that appear in the SUSPICIOUS_TLDS set.

    Using tldextract ensures robust parsing of host + public suffix, even for
    non-trivial domain structures (e.g., multi-part TLDs).
    """
    count = 0
    for url in urls:
        ext = tldextract.extract(url)
        tld = (ext.suffix or "").lower()
        if tld in SUSPICIOUS_TLDS:
            count += 1
    return count


# -------------------------------------------------------------
# Keyword Presence Checker
# -------------------------------------------------------------
# Generic helper to evaluate whether text contains *any* of
# the keywords from a given list. This avoids repetitive code.
def text_contains_any(text: str, words: List[str]) -> bool:
    """
    Check if the given text contains any of the provided keyword phrases.

    Comparison is done case-insensitively on the raw substring,
    which is sufficient for heuristic flagging.
    """
    if not text:
        return False
    lower = text.lower()
    return any(w in lower for w in words)


# -------------------------------------------------------------
# Feature Extraction Pipeline
# -------------------------------------------------------------
# Aggregates all feature extraction:
#   - URL detection
#   - Suspicious TLD count
#   - Keyword triggers (urgent, login, financial)
#
# Produces an EmailFeatures instance which downstream modules use
# for ML inference + heuristic scoring.
def compute_features(subject: str, body: str) -> EmailFeatures:
    """
    Extract all configured heuristic features from an email.

    Combines subject and body into a single string to search for:
      - URLs and suspicious TLDs.
      - Urgent, login, and financial keywords.

    Returns:
        EmailFeatures: ready-to-use features for scoring pipelines.
    """
    # Combine subject + body into a single searchable text block.
    combined = (subject or "") + "\n\n" + (body or "")
    urls = extract_urls(combined)
    num_suspicious_tlds = count_suspicious_tlds(urls)

    # Keyword checks for behavioral signals.
    has_urgent = text_contains_any(combined, URGENT_WORDS)
    has_login = text_contains_any(combined, LOGIN_WORDS)
    has_financial = text_contains_any(combined, FINANCIAL_WORDS)

    return EmailFeatures(
        urls=urls,
        num_urls=len(urls),
        num_suspicious_tlds=num_suspicious_tlds,
        has_urgent_words=has_urgent,
        has_login_words=has_login,
        has_financial_words=has_financial,
    )


# -------------------------------------------------------------
# Heuristic Risk Scoring
# -------------------------------------------------------------
# Produces a confidence score between 0 and 1 representing how
# "phishy" the email appears based solely on rule-based signals.
#
# ML scoring is done separately; this rule-based score is combined
# with ML + TI weights in the final ensemble.
def heuristic_score(feats: EmailFeatures) -> float:
    """
    Compute a simple heuristic phishing score in [0, 1] based on
    suspicious indicators extracted from the email.

    This intentionally remains interpretable and coarse-grained:
    - URL count influences baseline suspicion.
    - Suspicious TLDs increase risk, capped to avoid dominating.
    - Urgent, login, and financial keywords add additive risk.

    The final score is later combined with ML + threat-intel signals.
    """
    score = 0.0

    # URL count influence.
    if feats.num_urls > 0:
        score += 0.1
    if feats.num_urls > 3:
        score += 0.1

    # Suspicious TLD scaling: capped to avoid overweighting.
    score += min(feats.num_suspicious_tlds * 0.15, 0.4)

    # Keyword flags.
    if feats.has_urgent_words:
        score += 0.2
    if feats.has_login_words:
        score += 0.2
    if feats.has_financial_words:
        score += 0.2

    # Clamp to 0–1 range.
    return max(0.0, min(1.0, score))
