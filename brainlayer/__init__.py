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
    get_model_eval_scenarios,
    render_model_eval_report,
    run_live_model_eval_suite,
    run_model_eval_suite,
)
from .natural_eval import (
    default_natural_eval_runtime_config,
    export_natural_eval_results,
    get_natural_eval_scenarios,
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
from .matrix_analysis import (
    build_cost_quality_frontier,
    build_matrix_analysis,
    export_matrix_analysis,
    load_matrix_history,
    render_matrix_analysis_markdown,
    render_matrix_analysis_x_post,
    select_matrix_history_run,
)
from .runtime import BrainLayerRuntime, BrainLayerRuntimeConfig, ModelTurnResult
from .runtime_variants import (
    RUNTIME_PROFILE_DEFAULT,
    RUNTIME_PROFILE_STUDY_V2,
    RuntimeVariantSpec,
    build_runtime_variants,
)
from .session import BrainLayerSession
from .storage import load_state, save_state
from .study_runner import (
    DEFAULT_STUDY_CONFIG,
    DEFAULT_STUDY_EXPORT_ROOT,
    DEFAULT_STUDY_PROTOCOL,
    build_study_aggregate_leaderboard,
    parse_study_scenario_packs,
    render_study_summary_markdown,
    render_study_x_post,
    run_study,
)
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
    "build_cost_quality_frontier",
    "build_matrix_analysis",
    "build_live_model_eval_adapter",
    "default_model_eval_runtime_config",
    "default_natural_eval_runtime_config",
    "estimate_usage_cost_usd",
    "export_matrix_analysis",
    "export_model_eval_results",
    "export_model_matrix_results",
    "export_natural_eval_results",
    "get_model_eval_scenarios",
    "get_natural_eval_scenarios",
    "load_matrix_history",
    "load_model_matrix_entries",
    "render_model_eval_report",
    "render_matrix_analysis_markdown",
    "render_model_matrix_report",
    "render_model_matrix_x_post",
    "render_matrix_analysis_x_post",
    "render_natural_eval_report",
    "render_study_summary_markdown",
    "render_study_x_post",
    "RuntimeVariantSpec",
    "run_live_model_eval_suite",
    "run_live_natural_eval_suite",
    "run_model_matrix",
    "run_natural_eval_suite",
    "run_study",
    "run_suite",
    "run_model_eval_suite",
    "build_runtime_variants",
    "score_structured_value",
    "select_matrix_history_run",
    "save_state",
    "DEFAULT_STUDY_CONFIG",
    "DEFAULT_STUDY_EXPORT_ROOT",
    "DEFAULT_STUDY_PROTOCOL",
    "RUNTIME_PROFILE_DEFAULT",
    "RUNTIME_PROFILE_STUDY_V2",
    "build_study_aggregate_leaderboard",
    "parse_study_scenario_packs",
    "validate_state_dict",
]
