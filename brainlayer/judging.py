from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "by",
    "default",
    "for",
    "i",
    "in",
    "is",
    "it",
    "like",
    "me",
    "now",
    "on",
    "right",
    "should",
    "the",
    "this",
    "to",
    "use",
    "we",
    "with",
    "you",
}
PHRASE_ALIASES = (
    ("keep the citations intact", "preserve citations"),
    ("keep citations intact", "preserve citations"),
    ("preserve source citations", "preserve citations"),
    ("ship the evaluation report", "ship eval report"),
    ("ship evaluation report", "ship eval report"),
    ("ship the eval report", "ship eval report"),
    ("deliver the evaluation report tonight", "ship eval report"),
    ("deliver evaluation report tonight", "ship eval report"),
    ("deliver the evaluation report", "ship eval report"),
    ("deliver evaluation report", "ship eval report"),
    ("ship the evaluation summary", "ship eval summary"),
    ("ship evaluation summary", "ship eval summary"),
    ("ship the eval summary", "ship eval summary"),
    ("shipping the evaluation summary today", "ship eval summary"),
    ("shipping the evaluation summary", "ship eval summary"),
    ("shipping the eval summary today", "ship eval summary"),
    ("shipping the eval summary", "ship eval summary"),
    ("check github auth first", "check authentication"),
    ("check auth first", "check authentication"),
    ("check authentication first", "check authentication"),
    ("verify github auth", "check authentication"),
    ("verify auth", "check authentication"),
    ("confirm github authentication first", "check authentication"),
    ("confirm github authentication", "check authentication"),
    ("verify authentication before retrying the release", "check authentication"),
    ("verify your authentication before retrying the release", "check authentication"),
    ("task runner", "task executor"),
    ("research partnership", "research partner"),
)
TOKEN_ALIASES = {
    "briefly": "brief",
    "concise": "brief",
    "succinct": "brief",
    "short": "brief",
    "shorter": "brief",
    "detailed": "detailed",
    "thorough": "detailed",
    "shipping": "ship",
    "deliver": "ship",
    "delivering": "ship",
    "evaluation": "eval",
    "auth": "authentication",
    "confirm": "check",
    "verify": "check",
    "verifying": "check",
    "authenticate": "authentication",
    "authenticated": "authentication",
    "partnership": "partner",
}


@dataclass(frozen=True)
class ScoreDecision:
    passed: bool
    score: float
    method: str
    reason: str


@dataclass(frozen=True)
class BehaviorJudgeInput:
    scenario_slug: str
    scenario_title: str
    scenario_description: str
    checkpoint: str
    prompt: str
    expected: str
    actual: str


class BehaviorJudge:
    def score(self, payload: BehaviorJudgeInput) -> ScoreDecision:
        raise NotImplementedError


class ExactMatchJudge(BehaviorJudge):
    def score(self, payload: BehaviorJudgeInput) -> ScoreDecision:
        matched = exact_answers_match(payload.expected, payload.actual)
        if matched:
            return ScoreDecision(
                passed=True,
                score=1.0,
                method="exact_match",
                reason="Expected answer matched after normalized token comparison.",
            )
        return ScoreDecision(
            passed=False,
            score=0.0,
            method="exact_match",
            reason="Expected answer did not match after normalized token comparison.",
        )


class HeuristicBehaviorJudge(BehaviorJudge):
    def score(self, payload: BehaviorJudgeInput) -> ScoreDecision:
        exact_decision = ExactMatchJudge().score(payload)
        if exact_decision.passed:
            return exact_decision

        expected_canonical = canonicalize_text(payload.expected)
        actual_canonical = canonicalize_text(payload.actual)
        if not expected_canonical and not actual_canonical:
            return ScoreDecision(
                passed=True,
                score=1.0,
                method="heuristic_semantic_judge",
                reason="Both expected and actual answers were empty after canonicalization.",
            )
        if expected_canonical and _contains_phrase(actual_canonical, expected_canonical):
            return ScoreDecision(
                passed=True,
                score=1.0,
                method="heuristic_semantic_judge",
                reason="Canonicalized answer preserved the expected meaning.",
            )

        expected_tokens = set(content_tokens(payload.expected))
        actual_tokens = set(content_tokens(payload.actual))
        if expected_tokens and expected_tokens.issubset(actual_tokens):
            return ScoreDecision(
                passed=True,
                score=1.0,
                method="heuristic_semantic_judge",
                reason="Canonicalized content tokens covered the expected answer.",
            )

        overlap_ratio = _overlap_ratio(expected_tokens, actual_tokens)
        if expected_tokens and overlap_ratio >= 0.8:
            return ScoreDecision(
                passed=True,
                score=0.8,
                method="heuristic_semantic_judge",
                reason="Canonicalized content-token overlap was strong enough to accept the answer.",
            )

        return ScoreDecision(
            passed=False,
            score=overlap_ratio,
            method="heuristic_semantic_judge",
            reason="Canonicalized answer did not preserve enough of the expected meaning.",
        )


def exact_answers_match(expected: str, actual: str) -> bool:
    normalized_expected = normalize_answer_text(expected)
    normalized_actual = normalize_answer_text(actual)
    if not normalized_expected:
        return normalized_actual == normalized_expected
    if normalized_actual == normalized_expected:
        return True
    return _contains_phrase(normalized_actual, normalized_expected)


def score_structured_value(
    expected: str,
    actual: str,
    *,
    target_layer: str,
    target_key: str,
) -> ScoreDecision:
    exact_payload = BehaviorJudgeInput(
        scenario_slug="",
        scenario_title="",
        scenario_description="",
        checkpoint="",
        prompt="",
        expected=expected,
        actual=actual,
    )
    exact_decision = ExactMatchJudge().score(exact_payload)
    if exact_decision.passed:
        return ScoreDecision(
            passed=True,
            score=1.0,
            method="structural_exact_match",
            reason=(
                f"Stored {target_layer}:{target_key} matched the expected value after normalized comparison."
            ),
        )

    semantic_decision = HeuristicBehaviorJudge().score(exact_payload)
    if semantic_decision.passed:
        return ScoreDecision(
            passed=True,
            score=semantic_decision.score,
            method="structural_semantic_match",
            reason=(
                f"Stored {target_layer}:{target_key} matched semantically after canonicalization."
            ),
        )

    return ScoreDecision(
        passed=False,
        score=semantic_decision.score,
        method="structural_semantic_match",
        reason=(
            f"Stored {target_layer}:{target_key} did not match the expected value."
        ),
    )


def normalize_answer_text(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower()))


def canonicalize_text(value: str) -> str:
    lowered = value.lower().strip()
    for source, replacement in PHRASE_ALIASES:
        lowered = lowered.replace(source, replacement)
    tokens = [TOKEN_ALIASES.get(token, token) for token in TOKEN_RE.findall(lowered)]
    return " ".join(tokens)


def content_tokens(value: str) -> list[str]:
    return [token for token in canonicalize_text(value).split() if token not in STOPWORDS]


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return not text
    return f" {phrase} " in f" {text} "


def _overlap_ratio(expected_tokens: set[str], actual_tokens: set[str]) -> float:
    if not expected_tokens:
        return 1.0 if not actual_tokens else 0.0
    return len(expected_tokens & actual_tokens) / len(expected_tokens)


__all__ = [
    "BehaviorJudge",
    "BehaviorJudgeInput",
    "ExactMatchJudge",
    "HeuristicBehaviorJudge",
    "ScoreDecision",
    "canonicalize_text",
    "content_tokens",
    "exact_answers_match",
    "normalize_answer_text",
    "score_structured_value",
]
