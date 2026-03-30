from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class WorkingItem:
    id: str
    content: str
    priority: float
    status: str = "active"
    source_refs: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class Episode:
    id: str
    scenario: str
    summary: str
    tags: List[str]
    salience: float
    outcome: str = ""
    source_refs: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass
class Belief:
    id: str
    key: str
    proposition: str
    value: str
    confidence: float
    status: str = "active"
    evidence_episode_ids: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class AutobioNote:
    id: str
    summary: str
    themes: List[str]
    supporting_ids: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class Procedure:
    id: str
    trigger: str
    summary: str
    steps: List[str]
    confidence: float
    derived_from: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class BrainLayerState:
    working_state: List[WorkingItem] = field(default_factory=list)
    episodes: List[Episode] = field(default_factory=list)
    beliefs: List[Belief] = field(default_factory=list)
    autobiographical_state: List[AutobioNote] = field(default_factory=list)
    procedures: List[Procedure] = field(default_factory=list)
    _counters: Dict[str, int] = field(default_factory=dict, repr=False)

    def _next_id(self, prefix: str) -> str:
        current = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = current
        return f"{prefix}-{current}"

    def add_working_item(
        self,
        content: str,
        priority: float,
        source_refs: List[str],
    ) -> WorkingItem:
        item = WorkingItem(
            id=self._next_id("working"),
            content=content,
            priority=priority,
            source_refs=list(source_refs),
        )
        self.working_state.append(item)
        return item

    def record_episode(
        self,
        scenario: str,
        summary: str,
        tags: List[str],
        salience: float,
        outcome: str = "",
        source_refs: List[str] | None = None,
    ) -> Episode:
        episode = Episode(
            id=self._next_id("episode"),
            scenario=scenario,
            summary=summary,
            tags=list(tags),
            salience=salience,
            outcome=outcome,
            source_refs=list(source_refs or []),
        )
        self.episodes.append(episode)
        return episode

    def upsert_belief(
        self,
        key: str,
        proposition: str,
        value: str,
        confidence: float,
        evidence_episode_ids: List[str],
    ) -> Belief:
        for belief in self.beliefs:
            if belief.key == key and belief.status == "active" and belief.value != value:
                belief.status = "superseded"
                belief.updated_at = utc_now_iso()

        for belief in self.beliefs:
            if belief.key == key and belief.status == "active" and belief.value == value:
                belief.proposition = proposition
                belief.confidence = confidence
                belief.evidence_episode_ids = list(evidence_episode_ids)
                belief.updated_at = utc_now_iso()
                return belief

        belief = Belief(
            id=self._next_id("belief"),
            key=key,
            proposition=proposition,
            value=value,
            confidence=confidence,
            evidence_episode_ids=list(evidence_episode_ids),
        )
        self.beliefs.append(belief)
        return belief

    def upsert_autobio_note(
        self,
        summary: str,
        themes: List[str],
        supporting_ids: List[str],
    ) -> AutobioNote:
        theme_key = tuple(sorted(themes))
        for note in self.autobiographical_state:
            if tuple(sorted(note.themes)) == theme_key:
                note.summary = summary
                note.supporting_ids = list(supporting_ids)
                note.updated_at = utc_now_iso()
                return note

        note = AutobioNote(
            id=self._next_id("autobio"),
            summary=summary,
            themes=list(themes),
            supporting_ids=list(supporting_ids),
        )
        self.autobiographical_state.append(note)
        return note

    def learn_procedure(
        self,
        trigger: str,
        summary: str,
        steps: List[str],
        confidence: float,
        derived_from: List[str],
    ) -> Procedure:
        for procedure in self.procedures:
            if procedure.trigger == trigger:
                procedure.summary = summary
                procedure.steps = list(steps)
                procedure.confidence = confidence
                procedure.derived_from = list(derived_from)
                procedure.updated_at = utc_now_iso()
                return procedure

        procedure = Procedure(
            id=self._next_id("procedure"),
            trigger=trigger,
            summary=summary,
            steps=list(steps),
            confidence=confidence,
            derived_from=list(derived_from),
        )
        self.procedures.append(procedure)
        return procedure

    def to_dict(self) -> Dict[str, object]:
        return {
            "working_state": [asdict(item) for item in self.working_state],
            "episodes": [asdict(item) for item in self.episodes],
            "beliefs": [asdict(item) for item in self.beliefs],
            "autobiographical_state": [
                asdict(item) for item in self.autobiographical_state
            ],
            "procedures": [asdict(item) for item in self.procedures],
        }
