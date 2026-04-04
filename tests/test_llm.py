import json
import urllib.error
import unittest
from unittest.mock import patch

from brainlayer.llm import AnthropicMessagesAdapter, ModelMessage
from brainlayer.model_eval import build_live_model_eval_adapter


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        return None


class AnthropicAdapterTests(unittest.TestCase):
    def test_build_live_adapter_supports_anthropic_provider(self) -> None:
        adapter = build_live_model_eval_adapter(
            provider_name="anthropic_messages",
            api_key_env="DOES_NOT_MATTER",
        )

        self.assertIsInstance(adapter, AnthropicMessagesAdapter)

    def test_anthropic_messages_adapter_maps_system_messages_and_usage(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(http_request, timeout=0):
            del timeout
            captured["url"] = http_request.full_url
            captured["headers"] = dict(http_request.header_items())
            captured["body"] = json.loads(http_request.data.decode("utf-8"))
            return _FakeResponse(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello there"}],
                    "model": "claude-sonnet-test",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 12, "output_tokens": 5},
                }
            )

        adapter = AnthropicMessagesAdapter(api_key="test-key")
        with patch("brainlayer.llm.request.urlopen", side_effect=fake_urlopen):
            response = adapter.generate(
                [
                    ModelMessage(role="system", content="You are a memory research assistant."),
                    ModelMessage(role="user", content="Summarize the task."),
                ],
                model="claude-sonnet-test",
                temperature=0.0,
                max_output_tokens=128,
            )

        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        body = captured["body"]
        self.assertEqual(body["model"], "claude-sonnet-test")
        self.assertEqual(body["max_tokens"], 128)
        self.assertEqual(body["system"], "You are a memory research assistant.")
        self.assertEqual(body["messages"], [{"role": "user", "content": "Summarize the task."}])
        self.assertEqual(response.content, "hello there")
        self.assertEqual(response.finish_reason, "end_turn")
        self.assertEqual(response.model, "claude-sonnet-test")
        self.assertEqual(response.usage["input_tokens"], 12)
        self.assertEqual(response.usage["output_tokens"], 5)
        self.assertEqual(response.usage["prompt_tokens"], 12)
        self.assertEqual(response.usage["completion_tokens"], 5)
        self.assertEqual(response.usage["total_tokens"], 17)

    def test_anthropic_messages_adapter_retries_retryable_transport_errors(self) -> None:
        attempts = {"count": 0}

        def fake_urlopen(http_request, timeout=0):
            del http_request, timeout
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise urllib.error.URLError("temporary network glitch")
            return _FakeResponse(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "claude-haiku-test",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            )

        adapter = AnthropicMessagesAdapter(
            api_key="test-key",
            max_retries=1,
            retry_backoff_seconds=0.0,
        )
        with patch("brainlayer.llm.request.urlopen", side_effect=fake_urlopen):
            response = adapter.generate(
                [ModelMessage(role="user", content="Ping.")],
                model="claude-haiku-test",
                temperature=0.0,
                max_output_tokens=16,
            )

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(response.content, "ok")


if __name__ == "__main__":
    unittest.main()
