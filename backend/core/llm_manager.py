"""
LLM manager for the Multi-Phishing LangChain backend.

This module encapsulates all interaction with the local Ollama LLM server.
Primary goals:

- Provide a single abstraction (`LLMManager`) for:
    • model selection / identification
    • formatting chat-style requests
    • robust HTTP communication with the Ollama API
- Expose high-level helpers tailored to the phishing use case:
    • `explain_detection` for human-friendly explanations
    • `draft_safe_reply` for safe outbound email responses

By routing all LLM calls through this class, the rest of the system remains
decoupled from Ollama-specific details and can more easily support additional
backends in the future.
"""

from typing import Optional, Dict, Any

import httpx

from .config import llm_config


class LLMManager:
    """
    Manager class that wraps interactions with the local Ollama LLM server.

    Responsibilities:
      • Manage model selection (default or user-specified).
      • Format and send chat-style requests to Ollama's HTTP API.
      • Provide higher-level helper functions for phishing explanations
        and safe email replies.

    This abstraction isolates all LLM communication from the rest of the system,
    making it easier to swap or extend model backends later.
    """

    def __init__(self, model_name: Optional[str] = None):
        # Select a default model if none is provided.
        self.model_name = model_name or llm_config.default_model

        # If the requested model is not in the expected list, warn but still allow.
        # This supports experimental or custom local Ollama models.
        if self.model_name not in llm_config.available_models:
            print(
                f"[LLMManager] Warning: model '{self.model_name}' "
                f"not in configured available_models."
            )

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Internal helper to call Ollama's `/api/chat` endpoint.

        Arguments:
            system_prompt — instructions to guide the model's behavior.
            user_prompt   — user-level content or question.

        Returns:
            The assistant's message content as plain text.

        Notes:
        - Uses a synchronous httpx client (fast and reliable for local requests).
        - Includes robust error handling to prevent the entire pipeline from failing.
        - Falls back to a safe, deterministic string if communication fails.
        """
        # Compose the Ollama chat endpoint URL from configured base_url.
        url = f"{llm_config.base_url}/api/chat"

        # Standard chat-format payload expected by Ollama's HTTP API.
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,  # This service expects a full, consolidated response.
        }

        try:
            # Use a short-lived client for each call; fine for local-only traffic.
            with httpx.Client(timeout=llm_config.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data: Dict[str, Any] = resp.json()
        except Exception as e:
            print(f"[LLMManager] Error calling Ollama: {e}")
            # Fail gracefully – never break detection or analysis flows.
            return "Unable to contact local LLM. Please review this email manually."

        # Extract structured chat message from Ollama response format:
        # { "message": {"role": "...", "content": "..."} }
        message = data.get("message") or {}
        content = message.get("content") or ""

        return str(content).strip()

    # ----------------------------------------------------------------------
    # High-level model-use helpers
    # ----------------------------------------------------------------------

    def explain_detection(self, reasons_text: str) -> str:
        """
        Produce a simple, non-technical explanation of phishing indicators
        for end users.

        Used when presenting classifier + heuristic + TI results
        inside dashboards or automated responses.
        """
        system_prompt = (
            "You are a cybersecurity assistant that explains phishing detections "
            "to end users in simple, non-technical language. Be concise but clear."
        )
        user_prompt = (
            "Explain why this email was classified as phishing or benign.\n\n"
            f"Signals and analysis details:\n{reasons_text}\n\n"
            "Write 3–6 sentences maximum."
        )

        # If reasons_text somehow becomes whitespace, fall back to using user_prompt.
        # (This call delegates the actual LLM interaction to `_chat`.)
        return self._chat(system_prompt, reasons_text if not reasons_text.isspace() else user_prompt)

    def draft_safe_reply(self, context_text: str) -> str:
        """
        Generate a safe email reply based on system analysis.

        Rules enforced by the system prompt:
          • Never include links or attachments.
          • Never ask for sensitive information.
          • If suspicious, encourage the user to contact the official company
            through known channels.
          • If benign, provide a helpful, relevant response.
        """
        system_prompt = (
            "You are a cybersecurity assistant helping users respond safely to "
            "potential phishing emails. You never include links or attachments "
            "in the reply, and you never ask for sensitive data."
        )
        user_prompt = (
            "Given the following analysis of an email, write a safe reply that "
            "the user could send back. If the email is phishing, politely decline "
            "and advise contacting the company through official channels. "
            "If benign, respond helpfully.\n\n"
            f"Analysis context:\n{context_text}\n\n"
            "Reply:\n"
        )
        # Delegate to the shared chat helper for consistent error handling and
        # model invocation behavior.
        return self._chat(system_prompt, user_prompt)
