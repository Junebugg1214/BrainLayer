from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .agents import BrainLayerFeatureConfig


RUNTIME_PROFILE_DEFAULT = "default"
RUNTIME_PROFILE_STUDY_V2 = "study_v2"
VALID_RUNTIME_PROFILES = {RUNTIME_PROFILE_DEFAULT, RUNTIME_PROFILE_STUDY_V2}


@dataclass(frozen=True)
class RuntimeVariantSpec:
    name: str
    features: BrainLayerFeatureConfig
    memory_strategy: str = "brainlayer"


def build_runtime_variants(
    *,
    include_ablations: bool = True,
    runtime_profile: str = RUNTIME_PROFILE_DEFAULT,
) -> List[RuntimeVariantSpec]:
    if runtime_profile not in VALID_RUNTIME_PROFILES:
        raise ValueError(
            f"Unsupported runtime profile: {runtime_profile}. "
            f"Expected one of {sorted(VALID_RUNTIME_PROFILES)}."
        )

    if runtime_profile == RUNTIME_PROFILE_STUDY_V2:
        variants = [
            RuntimeVariantSpec(
                name="brainlayer_full",
                features=BrainLayerFeatureConfig(),
                memory_strategy="brainlayer",
            ),
            RuntimeVariantSpec(
                name="context_only",
                features=BrainLayerFeatureConfig(
                    enable_consolidation=False,
                    enable_forgetting=False,
                    enable_autobio=False,
                    enable_working_state=False,
                ),
                memory_strategy="context_only",
            ),
            RuntimeVariantSpec(
                name="naive_retrieval",
                features=BrainLayerFeatureConfig(
                    enable_consolidation=False,
                    enable_forgetting=False,
                    enable_autobio=False,
                    enable_working_state=False,
                ),
                memory_strategy="naive_retrieval",
            ),
            RuntimeVariantSpec(
                name="structured_no_consolidation",
                features=BrainLayerFeatureConfig(enable_consolidation=False, enable_forgetting=False),
                memory_strategy="structured_no_consolidation",
            ),
            RuntimeVariantSpec(
                name="summary_state",
                features=BrainLayerFeatureConfig(
                    enable_consolidation=False,
                    enable_forgetting=False,
                    enable_autobio=False,
                    enable_working_state=False,
                ),
                memory_strategy="summary_state",
            ),
        ]
        return variants

    variants = [
        RuntimeVariantSpec(
            name="model_loop",
            features=BrainLayerFeatureConfig(),
            memory_strategy="brainlayer",
        )
    ]
    if not include_ablations:
        return variants

    variants.extend(
        [
            RuntimeVariantSpec(
                name="model_loop_no_consolidation",
                features=BrainLayerFeatureConfig(enable_consolidation=False),
                memory_strategy="brainlayer",
            ),
            RuntimeVariantSpec(
                name="model_loop_no_forgetting",
                features=BrainLayerFeatureConfig(enable_forgetting=False),
                memory_strategy="brainlayer",
            ),
            RuntimeVariantSpec(
                name="model_loop_no_autobio",
                features=BrainLayerFeatureConfig(enable_autobio=False),
                memory_strategy="brainlayer",
            ),
            RuntimeVariantSpec(
                name="model_loop_no_working_state",
                features=BrainLayerFeatureConfig(enable_working_state=False),
                memory_strategy="brainlayer",
            ),
        ]
    )
    return variants


__all__ = [
    "RUNTIME_PROFILE_DEFAULT",
    "RUNTIME_PROFILE_STUDY_V2",
    "RuntimeVariantSpec",
    "VALID_RUNTIME_PROFILES",
    "build_runtime_variants",
]
