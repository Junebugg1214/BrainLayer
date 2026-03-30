from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set

from .models import BrainLayerState, Episode


def _has_any_tag(episode: Episode, tags: Sequence[str]) -> bool:
    return any(tag in episode.tags for tag in tags)


@dataclass(frozen=True)
class ConsolidationConfig:
    belief_promotion_min_support: float = 0.75
    procedure_promotion_min_support: float = 0.8
    working_item_promotion_min_support: float = 0.75
    autobio_promotion_min_support: float = 0.75
    repeated_signal_min_count: int = 2
    noise_forget_threshold: float = 0.3
    max_active_working_items: int = 4
    working_item_priority_floor: float = 0.35


@dataclass
class ConsolidationReport:
    promoted_belief_keys: List[str] = field(default_factory=list)
    promoted_procedure_triggers: List[str] = field(default_factory=list)
    updated_working_keys: List[str] = field(default_factory=list)
    updated_autobio_keys: List[str] = field(default_factory=list)
    forgotten_episode_ids: List[str] = field(default_factory=list)
    paused_working_item_ids: List[str] = field(default_factory=list)


class ConsolidationEngine:
    def __init__(self, config: ConsolidationConfig | None = None) -> None:
        self.config = config or ConsolidationConfig()

    def run(self, state: BrainLayerState) -> ConsolidationReport:
        report = ConsolidationReport()
        self._consolidate_beliefs(state, report)
        self._consolidate_procedures(state, report)
        self._consolidate_working_state(state, report)
        self._consolidate_autobio(state, report)
        self._pause_stale_working_items(state, report)
        self._forget_low_salience_noise(state, report)
        return report

    def _consolidate_beliefs(
        self,
        state: BrainLayerState,
        report: ConsolidationReport,
    ) -> None:
        grouped = self._group_by_key(
            state.episodes,
            include_tags=("preference", "correction", "preference_hint"),
        )
        for key, episodes in grouped.items():
            candidate = self._select_value_candidate(
                episodes,
                explicit_tags=("preference", "correction"),
                min_support=self.config.belief_promotion_min_support,
            )
            if candidate is None:
                continue
            proposition = candidate["metadata"].get(
                "proposition",
                f"{key} is currently {candidate['value']}.",
            )
            state.upsert_belief(
                key=key,
                proposition=proposition,
                value=candidate["value"],
                confidence=candidate["confidence"],
                evidence_episode_ids=candidate["evidence_episode_ids"],
            )
            self._append_unique(report.promoted_belief_keys, key)

    def _consolidate_procedures(
        self,
        state: BrainLayerState,
        report: ConsolidationReport,
    ) -> None:
        grouped = self._group_by_key(
            state.episodes,
            include_tags=("lesson", "lesson_hint"),
            key_field="trigger",
        )
        for trigger, episodes in grouped.items():
            candidate = self._select_value_candidate(
                episodes,
                explicit_tags=("lesson",),
                min_support=self.config.procedure_promotion_min_support,
                value_field="action",
            )
            if candidate is None:
                continue
            summary = candidate["metadata"].get(
                "summary",
                f"When {trigger}, {candidate['value']}.",
            )
            state.learn_procedure(
                trigger=trigger,
                summary=summary,
                steps=[candidate["value"]],
                confidence=candidate["confidence"],
                derived_from=candidate["evidence_episode_ids"],
            )
            self._append_unique(report.promoted_procedure_triggers, trigger)

    def _consolidate_working_state(
        self,
        state: BrainLayerState,
        report: ConsolidationReport,
    ) -> None:
        grouped = self._group_by_key(
            state.episodes,
            include_tags=("goal", "goal_hint"),
        )
        for key, episodes in grouped.items():
            candidate = self._select_value_candidate(
                episodes,
                explicit_tags=("goal",),
                min_support=self.config.working_item_promotion_min_support,
            )
            if candidate is None:
                continue
            summary = candidate["metadata"].get(
                "summary",
                f"The current {key} is {candidate['value']}.",
            )
            state.upsert_working_item(
                key=key,
                value=candidate["value"],
                content=summary,
                priority=candidate["confidence"],
                source_refs=candidate["evidence_episode_ids"],
            )
            self._append_unique(report.updated_working_keys, key)

    def _consolidate_autobio(
        self,
        state: BrainLayerState,
        report: ConsolidationReport,
    ) -> None:
        grouped = self._group_by_key(
            state.episodes,
            include_tags=("relationship", "relationship_hint"),
        )
        for key, episodes in grouped.items():
            candidate = self._select_value_candidate(
                episodes,
                explicit_tags=("relationship",),
                min_support=self.config.autobio_promotion_min_support,
            )
            if candidate is None:
                continue
            themes = [
                value.strip()
                for value in candidate["metadata"].get("themes", "").split(",")
                if value.strip()
            ] or ["relationship"]
            summary = candidate["metadata"].get(
                "summary",
                f"{key} is currently {candidate['value']}.",
            )
            note = state.upsert_autobio_note(
                key=key,
                value=candidate["value"],
                summary=summary,
                themes=themes,
                supporting_ids=candidate["evidence_episode_ids"],
            )
            state.upsert_working_item(
                key=note.key,
                value=note.value,
                content=note.summary,
                priority=candidate["confidence"],
                source_refs=candidate["evidence_episode_ids"] + [note.id],
            )
            self._append_unique(report.updated_autobio_keys, key)

    def _pause_stale_working_items(
        self,
        state: BrainLayerState,
        report: ConsolidationReport,
    ) -> None:
        active_items = [item for item in state.working_state if item.status == "active"]
        active_items.sort(key=lambda item: (item.priority, item.updated_at), reverse=True)

        keep_count = 0
        for item in active_items:
            if (
                item.priority >= self.config.working_item_priority_floor
                and keep_count < self.config.max_active_working_items
            ):
                keep_count += 1
                continue
            item.status = "paused"
            self._append_unique(report.paused_working_item_ids, item.id)

    def _forget_low_salience_noise(
        self,
        state: BrainLayerState,
        report: ConsolidationReport,
    ) -> None:
        referenced = self._collect_referenced_episode_ids(state)
        forget_ids = [
            episode.id
            for episode in state.episodes
            if "noise" in episode.tags
            and episode.salience < self.config.noise_forget_threshold
            and episode.id not in referenced
        ]
        state.forget_episodes(forget_ids)
        for episode_id in forget_ids:
            self._append_unique(report.forgotten_episode_ids, episode_id)

    def _collect_referenced_episode_ids(self, state: BrainLayerState) -> Set[str]:
        referenced: Set[str] = set()
        for belief in state.beliefs:
            referenced.update(belief.evidence_episode_ids)
        for procedure in state.procedures:
            referenced.update(procedure.derived_from)
        for note in state.autobiographical_state:
            referenced.update(note.supporting_ids)
        for item in state.working_state:
            referenced.update(
                source_ref for source_ref in item.source_refs if source_ref.startswith("episode-")
            )
        return referenced

    def _group_by_key(
        self,
        episodes: Iterable[Episode],
        *,
        include_tags: Sequence[str],
        key_field: str = "key",
    ) -> Dict[str, List[Episode]]:
        grouped: Dict[str, List[Episode]] = defaultdict(list)
        for episode in episodes:
            if not _has_any_tag(episode, include_tags):
                continue
            key = episode.metadata.get(key_field)
            if key:
                grouped[key].append(episode)
        return grouped

    def _select_value_candidate(
        self,
        episodes: Sequence[Episode],
        *,
        explicit_tags: Sequence[str],
        min_support: float,
        value_field: str = "value",
    ) -> Dict[str, object] | None:
        explicit_episodes = [episode for episode in episodes if _has_any_tag(episode, explicit_tags)]
        if explicit_episodes:
            chosen = explicit_episodes[-1]
            value = chosen.metadata.get(value_field)
            if not value:
                return None
            supporting = [
                episode for episode in episodes if episode.metadata.get(value_field) == value
            ]
            return {
                "value": value,
                "confidence": min(1.0, sum(episode.salience for episode in supporting)),
                "evidence_episode_ids": [episode.id for episode in supporting],
                "metadata": dict(chosen.metadata),
            }

        by_value: Dict[str, List[Episode]] = defaultdict(list)
        for episode in episodes:
            value = episode.metadata.get(value_field)
            if value:
                by_value[value].append(episode)
        if not by_value:
            return None

        best_value = ""
        best_support = -1.0
        best_episodes: List[Episode] = []
        for value, grouped_episodes in by_value.items():
            support = sum(episode.salience for episode in grouped_episodes)
            if support > best_support:
                best_value = value
                best_support = support
                best_episodes = grouped_episodes

        if len(best_episodes) < self.config.repeated_signal_min_count:
            return None
        if best_support < min_support:
            return None

        chosen = best_episodes[-1]
        return {
            "value": best_value,
            "confidence": min(1.0, best_support),
            "evidence_episode_ids": [episode.id for episode in best_episodes],
            "metadata": dict(chosen.metadata),
        }

    def _append_unique(self, items: List[str], value: str) -> None:
        if value not in items:
            items.append(value)
