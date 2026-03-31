"""Minimal BrainLayer research prototype."""

from .benchmark_harness import run_suite
from .consolidation import ConsolidationConfig, ConsolidationEngine, ConsolidationReport
from .agents import BrainLayerFeatureConfig
from .llm import LLMAdapter, LLMError, ModelMessage, ModelResponse, OpenAICompatibleChatAdapter
from .runtime import BrainLayerRuntime, BrainLayerRuntimeConfig, ModelTurnResult
from .session import BrainLayerSession
from .storage import load_state, save_state
from .validation import BrainLayerValidationError, validate_state_dict

__all__ = [
    "BrainLayerRuntime",
    "BrainLayerRuntimeConfig",
    "BrainLayerSession",
    "BrainLayerValidationError",
    "BrainLayerFeatureConfig",
    "ConsolidationConfig",
    "ConsolidationEngine",
    "ConsolidationReport",
    "LLMAdapter",
    "LLMError",
    "load_state",
    "ModelMessage",
    "ModelResponse",
    "ModelTurnResult",
    "OpenAICompatibleChatAdapter",
    "run_suite",
    "save_state",
    "validate_state_dict",
]
