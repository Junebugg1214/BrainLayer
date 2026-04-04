from __future__ import annotations

import json
import socket
import time
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


class AnthropicMessagesAdapter(LLMAdapter):
    """Minimal HTTP adapter for Anthropic's Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.anthropic.com",
        request_path: str = "/v1/messages",
        timeout_seconds: float = 30.0,
        anthropic_version: str = "2023-06-01",
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_path = request_path
        self.timeout_seconds = timeout_seconds
        self.anthropic_version = anthropic_version
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
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
                "Set ANTHROPIC_API_KEY or pass an explicit api_key."
            )

        system_parts: list[str] = []
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            role = "assistant" if message.role == "assistant" else "user"
            anthropic_messages.append({"role": role, "content": message.content})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        system_text = "\n\n".join(part.strip() for part in system_parts if part.strip()).strip()
        if system_text:
            payload["system"] = system_text

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        headers.update(self.extra_headers)

        http_request = request.Request(
            f"{self.base_url}{self.request_path}",
            data=body,
            headers=headers,
            method="POST",
        )

        raw_body = ""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    raw_body = response.read().decode("utf-8")
                last_error = None
                break
            except error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if attempt < self.max_retries and exc.code in {408, 429, 500, 502, 503, 504}:
                    self._sleep_before_retry(attempt)
                    continue
                raise LLMError(
                    f"Model request failed with HTTP {exc.code}: {error_body[:300]}"
                ) from exc
            except (error.URLError, socket.timeout, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries and _is_retryable_transport_error(exc):
                    self._sleep_before_retry(attempt)
                    continue
                if isinstance(exc, error.URLError):
                    raise LLMError(f"Model request failed: {exc.reason}") from exc
                raise LLMError(f"Model request failed: {exc}") from exc

        if last_error is not None:
            raise LLMError(f"Model request failed: {last_error}") from last_error

        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LLMError("Model provider returned non-JSON output.") from exc

        content = _extract_anthropic_text(response_payload)
        finish_reason = str(response_payload.get("stop_reason", ""))
        response_model = str(response_payload.get("model", model))
        usage = response_payload.get("usage")
        normalized_usage: Dict[str, Any] = {}
        if isinstance(usage, dict):
            normalized_usage.update({str(key): value for key, value in usage.items()})
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if isinstance(input_tokens, (int, float)):
                normalized_usage["prompt_tokens"] = input_tokens
            if isinstance(output_tokens, (int, float)):
                normalized_usage["completion_tokens"] = output_tokens
            if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
                normalized_usage["total_tokens"] = input_tokens + output_tokens

        return ModelResponse(
            content=content,
            raw_payload=response_payload,
            finish_reason=finish_reason,
            model=response_model,
            usage=normalized_usage,
        )

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_seconds <= 0.0:
            return
        time.sleep(self.retry_backoff_seconds * (2**attempt))


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


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(
                item.get("text"), str
            ):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part).strip()
    raise LLMError("Model provider response did not include any text content.")


def _is_retryable_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return True
        return True
    return isinstance(exc, OSError)
