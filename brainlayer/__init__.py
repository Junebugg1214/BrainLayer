"""Minimal BrainLayer research prototype."""

from .benchmark_harness import run_suite
from .consolidation import ConsolidationConfig, ConsolidationEngine, ConsolidationReport
from .agents import BrainLayerFeatureConfig
from .eval_support import estimate_usage_cost_usd
from .judging import BehaviorJudge, ExactMatchJudge, HeuristicBehaviorJudge, score_structured_value
from .llm import LLMAdapter, LLMError, ModelMessage, ModelResponse, OpenAICompatibleChatAdapter
from .model_eval import (
    build_live_model_eval_adapter,
    default_model_eval_runtime_config,
    export_model_eval_results,
    render_model_eval_report,
    run_live_model_eval_suite,
    run_model_eval_suite,
)
from .natural_eval import (
    default_natural_eval_runtime_config,
    export_natural_eval_results,
    render_natural_eval_report,
    run_live_natural_eval_suite,
    run_natural_eval_suite,
)
from .model_matrix import (
    ModelMatrixEntry,
    build_matrix_leaderboard,
    export_model_matrix_results,
    load_model_matrix_entries,
    render_model_matrix_report,
    render_model_matrix_x_post,
    run_model_matrix,
)
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
    "BehaviorJudge",
    "ConsolidationConfig",
    "ConsolidationEngine",
    "ConsolidationReport",
    "ExactMatchJudge",
    "HeuristicBehaviorJudge",
    "LLMAdapter",
    "LLMError",
    "load_state",
    "ModelMessage",
    "ModelResponse",
    "ModelTurnResult",
    "ModelMatrixEntry",
    "OpenAICompatibleChatAdapter",
    "build_matrix_leaderboard",
    "build_live_model_eval_adapter",
    "default_model_eval_runtime_config",
    "default_natural_eval_runtime_config",
    "estimate_usage_cost_usd",
    "export_model_eval_results",
    "export_model_matrix_results",
    "export_natural_eval_results",
    "load_model_matrix_entries",
    "render_model_eval_report",
    "render_model_matrix_report",
    "render_model_matrix_x_post",
    "render_natural_eval_report",
    "run_live_model_eval_suite",
    "run_live_natural_eval_suite",
    "run_model_matrix",
    "run_natural_eval_suite",
    "run_suite",
    "run_model_eval_suite",
    "score_structured_value",
    "save_state",
    "validate_state_dict",
]
