from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


ID_RE = re.compile(r"^(?P<prefix>[a-z_]+)-(?P<count>\d+)$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class WorkingItem:
    id: str
    key: str
    value: str
    content: str
    priority: float
    status: str = "active"
    source_refs: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "WorkingItem":
        return cls(
            id=str(payload["id"]),
            key=str(payload["key"]),
            value=str(payload["value"]),
            content=str(payload["content"]),
            priority=float(payload["priority"]),
            status=str(payload["status"]),
            source_refs=[str(value) for value in payload["source_refs"]],
            updated_at=str(payload["updated_at"]),
        )


@dataclass
class Episode:
    id: str
    scenario: str
    summary: str
    tags: List[str]
    metadata: Dict[str, str]
    salience: float
    outcome: str = ""
    source_refs: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Episode":
        return cls(
            id=str(payload["id"]),
            scenario=str(payload["scenario"]),
            summary=str(payload["summary"]),
            tags=[str(value) for value in payload["tags"]],
            metadata={str(key): str(value) for key, value in payload["metadata"].items()},
            salience=float(payload["salience"]),
            outcome=str(payload["outcome"]),
            source_refs=[str(value) for value in payload["source_refs"]],
            timestamp=str(payload["timestamp"]),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Belief":
        return cls(
            id=str(payload["id"]),
            key=str(payload["key"]),
            proposition=str(payload["proposition"]),
            value=str(payload["value"]),
            confidence=float(payload["confidence"]),
            status=str(payload["status"]),
            evidence_episode_ids=[str(value) for value in payload["evidence_episode_ids"]],
            updated_at=str(payload["updated_at"]),
        )


@dataclass
class AutobioNote:
    id: str
    key: str
    value: str
    summary: str
    themes: List[str]
    supporting_ids: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "AutobioNote":
        return cls(
            id=str(payload["id"]),
            key=str(payload["key"]),
            value=str(payload["value"]),
            summary=str(payload["summary"]),
            themes=[str(value) for value in payload["themes"]],
            supporting_ids=[str(value) for value in payload["supporting_ids"]],
            updated_at=str(payload["updated_at"]),
        )


@dataclass
class Procedure:
    id: str
    trigger: str
    summary: str
    steps: List[str]
    confidence: float
    derived_from: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Procedure":
        return cls(
            id=str(payload["id"]),
            trigger=str(payload["trigger"]),
            summary=str(payload["summary"]),
            steps=[str(value) for value in payload["steps"]],
            confidence=float(payload["confidence"]),
            derived_from=[str(value) for value in payload["derived_from"]],
            updated_at=str(payload["updated_at"]),
        )


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

    def upsert_working_item(
        self,
        key: str,
        value: str,
        content: str,
        priority: float,
        source_refs: List[str],
    ) -> WorkingItem:
        for item in self.working_state:
            if item.key == key:
                item.value = value
                item.content = content
                item.priority = priority
                item.status = "active"
                item.source_refs = list(source_refs)
                item.updated_at = utc_now_iso()
                return item

        item = WorkingItem(
            id=self._next_id("working"),
            key=key,
            value=value,
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
        metadata: Dict[str, str],
        salience: float,
        outcome: str = "",
        source_refs: List[str] | None = None,
    ) -> Episode:
        episode = Episode(
            id=self._next_id("episode"),
            scenario=scenario,
            summary=summary,
            tags=list(tags),
            metadata=dict(metadata),
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
        key: str,
        value: str,
        summary: str,
        themes: List[str],
        supporting_ids: List[str],
    ) -> AutobioNote:
        for note in self.autobiographical_state:
            if note.key == key:
                note.value = value
                note.summary = summary
                note.themes = list(themes)
                note.supporting_ids = list(supporting_ids)
                note.updated_at = utc_now_iso()
                return note

        note = AutobioNote(
            id=self._next_id("autobio"),
            key=key,
            value=value,
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

    def forget_episodes(self, episode_ids: List[str]) -> int:
        target_ids = set(episode_ids)
        if not target_ids:
            return 0
        before = len(self.episodes)
        self.episodes = [episode for episode in self.episodes if episode.id not in target_ids]
        return before - len(self.episodes)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BrainLayerState":
        state = cls(
            working_state=[
                WorkingItem.from_dict(item) for item in payload.get("working_state", [])
            ],
            episodes=[Episode.from_dict(item) for item in payload.get("episodes", [])],
            beliefs=[Belief.from_dict(item) for item in payload.get("beliefs", [])],
            autobiographical_state=[
                AutobioNote.from_dict(item)
                for item in payload.get("autobiographical_state", [])
            ],
            procedures=[
                Procedure.from_dict(item) for item in payload.get("procedures", [])
            ],
        )
        state._rebuild_counters()
        return state

    def _rebuild_counters(self) -> None:
        counters: Dict[str, int] = {}
        all_ids = [
            item.id for item in self.working_state
        ] + [
            item.id for item in self.episodes
        ] + [
            item.id for item in self.beliefs
        ] + [
            item.id for item in self.autobiographical_state
        ] + [
            item.id for item in self.procedures
        ]
        for item_id in all_ids:
            match = ID_RE.match(item_id)
            if not match:
                continue
            prefix = match.group("prefix")
            count = int(match.group("count"))
            counters[prefix] = max(counters.get(prefix, 0), count)
        self._counters = counters
