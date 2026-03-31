from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence
from urllib import error, request


@dataclass(frozen=True)
class ModelMessage:
    role: str
    content: str


@dataclass
class ModelResponse:
    content: str
    raw_payload: Dict[str, Any] | None = None
    finish_reason: str = ""
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when a model-backed request fails."""


class LLMAdapter:
    """Small provider-agnostic interface for model generation."""

    def generate(
        self,
        messages: Sequence[ModelMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
    ) -> ModelResponse:
        raise NotImplementedError


class StaticLLMAdapter(LLMAdapter):
    """Deterministic adapter for tests and dry runs."""

    def __init__(
        self,
        response: str | None = None,
        *,
        handler: Callable[[Sequence[ModelMessage]], str | ModelResponse] | None = None,
    ) -> None:
        self.response = response or '{"assistant_response":"unknown"}'
        self.handler = handler

    def generate(
        self,
        messages: Sequence[ModelMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
    ) -> ModelResponse:
        del model, temperature, max_output_tokens
        if self.handler is not None:
            result = self.handler(messages)
            if isinstance(result, ModelResponse):
                return result
            return ModelResponse(content=result)
        return ModelResponse(content=self.response)


class OpenAICompatibleChatAdapter(LLMAdapter):
    """Minimal HTTP adapter for chat-completions-compatible providers."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.openai.com/v1",
        request_path: str = "/chat/completions",
        timeout_seconds: float = 30.0,
        max_output_tokens_field: str | None = "max_tokens",
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_path = request_path
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens_field = max_output_tokens_field
        self.extra_headers = dict(extra_headers or {})

    def generate(
        self,
        messages: Sequence[ModelMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
    ) -> ModelResponse:
        if not self.api_key:
            raise LLMError(
                "Missing API key for model-backed BrainLayer loop. "
                "Set OPENAI_API_KEY or pass an explicit api_key."
            )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": temperature,
        }
        if self.max_output_tokens_field is not None:
            payload[self.max_output_tokens_field] = max_output_tokens

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        http_request = request.Request(
            f"{self.base_url}{self.request_path}",
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(
                f"Model request failed with HTTP {exc.code}: {error_body[:300]}"
            ) from exc
        except error.URLError as exc:
            raise LLMError(f"Model request failed: {exc.reason}") from exc

        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LLMError("Model provider returned non-JSON output.") from exc

        choice = _extract_first_choice(response_payload)
        message_content = _extract_message_content(choice)
        finish_reason = str(choice.get("finish_reason", ""))
        response_model = str(response_payload.get("model", model))
        usage = response_payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        return ModelResponse(
            content=message_content,
            raw_payload=response_payload,
            finish_reason=finish_reason,
            model=response_model,
            usage={str(key): value for key, value in usage.items()},
        )


def _extract_first_choice(payload: Dict[str, Any]) -> Dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("Model provider response did not include any choices.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMError("Model provider returned an invalid choice payload.")
    return choice


def _extract_message_content(choice: Dict[str, Any]) -> str:
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LLMError("Model provider response did not include a message payload.")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part).strip()
    return str(content)
