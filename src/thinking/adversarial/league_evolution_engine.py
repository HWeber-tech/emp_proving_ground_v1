"""Mini-league coordinator over evolution cycles.

Links the evolution cycle orchestrator with the :class:`MiniLeague` abstraction
so that each generation populates the champion, exploiter, and chaos rosters.
The engine projects ``EvolutionCycleResult`` payloads into league entries and
produces lightweight match results based on per-genome fitness metrics when
explicit self-play results are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.orchestration.evolution_cycle import (
    ChampionRecord,
    EvolutionCycleOrchestrator,
    EvolutionCycleResult,
    EvaluationRecord,
    FitnessReport,
)
from src.thinking.adversarial.mini_league import (
    LeagueEntry,
    LeagueMatchup,
    LeagueResult,
    LeagueSlot,
    MiniLeague,
)

__all__ = [
    "LeagueEvolutionSnapshot",
    "LeagueEvolutionEngine",
]


@dataclass(slots=True)
class LeagueEvolutionSnapshot:
    """State emitted after running a league evolution cycle."""

    cycle: EvolutionCycleResult
    champion: LeagueEntry | None
    matchups: tuple[LeagueMatchup, ...]
    results: tuple[LeagueResult, ...]
    league_snapshot: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "cycle": self.cycle.as_payload(),
            "champion": self.champion.as_dict() if self.champion else None,
            "matchups": [matchup.as_dict() for matchup in self.matchups],
            "results": [result.as_dict() for result in self.results],
            "league": dict(self.league_snapshot),
        }


class LeagueEvolutionEngine:
    """Run evolution cycles while maintaining champion, exploit, and chaos rosters."""

    def __init__(
        self,
        orchestrator: EvolutionCycleOrchestrator,
        *,
        league: MiniLeague | None = None,
        metadata_slot_keys: Sequence[str] = (
            "league_slot",
            "league_role",
            "slot",
            "role",
        ),
        metadata_tags_key: str = "tags",
    ) -> None:
        self._orchestrator = orchestrator
        self._league = league or MiniLeague()
        self._metadata_slot_keys = tuple(metadata_slot_keys)
        self._metadata_tags_key = metadata_tags_key

    @property
    def league(self) -> MiniLeague:
        return self._league

    async def run_cycle(self) -> LeagueEvolutionSnapshot:
        cycle = await self._orchestrator.run_cycle()
        evaluations = cycle.evaluations
        evaluation_index = {record.genome_id: record for record in evaluations}

        champion_entry = self._apply_champion(cycle.champion)
        current_entry = champion_entry

        current_entry = self._apply_population_assignments(
            evaluations, current_entry=current_entry
        )

        matchups = self._league.schedule_round()
        results = self._score_matchups(matchups, evaluation_index)

        snapshot = LeagueEvolutionSnapshot(
            cycle=cycle,
            champion=current_entry,
            matchups=matchups,
            results=results,
            league_snapshot=self._league.snapshot(),
        )
        return snapshot

    def _apply_champion(self, champion: ChampionRecord | None) -> LeagueEntry | None:
        if champion is None:
            return None
        metadata = self._report_metadata(champion.report)
        metadata.update(
            {
                "registered": bool(champion.registered),
                "parent_ids": list(champion.parent_ids),
                "species": champion.species,
                "mutation_history": list(champion.mutation_history),
                "league_slot": LeagueSlot.CURRENT.value,
            }
        )
        entry = LeagueEntry(
            agent_id=champion.genome_id,
            score=champion.fitness,
            tags=("champion",),
            metadata=metadata,
        )
        self._league.register(LeagueSlot.CURRENT, entry)
        return entry

    def _apply_population_assignments(
        self,
        evaluations: Iterable[EvaluationRecord],
        *,
        current_entry: LeagueEntry | None,
    ) -> LeagueEntry | None:
        resolved_current = current_entry
        for record in evaluations:
            if resolved_current is not None and record.genome_id == resolved_current.agent_id:
                continue

            slot = self._resolve_slot(record.report.metadata)
            if slot is None:
                continue

            entry = LeagueEntry(
                agent_id=record.genome_id,
                score=record.fitness,
                tags=(slot.value,),
                metadata=self._entry_metadata(record, slot),
            )

            if slot is LeagueSlot.CURRENT:
                self._league.register(LeagueSlot.CURRENT, entry)
                resolved_current = entry
                continue

            self._league.register(slot, entry)
        return resolved_current

    def _entry_metadata(self, record: EvaluationRecord, slot: LeagueSlot) -> dict[str, object]:
        metadata = self._report_metadata(record.report)
        metadata.setdefault("league_slot", slot.value)
        return metadata

    def _resolve_slot(self, metadata: Mapping[str, object] | MutableMapping[str, object] | None) -> LeagueSlot | None:
        if not isinstance(metadata, Mapping):
            return None

        for key in self._metadata_slot_keys:
            slot = self._slot_from_value(metadata.get(key))
            if slot is not None:
                return slot

        tags_candidate = metadata.get(self._metadata_tags_key)
        if isinstance(tags_candidate, (list, tuple, set)):
            for tag in tags_candidate:
                slot = self._slot_from_value(tag)
                if slot is not None and slot is not LeagueSlot.CURRENT:
                    return slot
        return None

    def _slot_from_value(self, value: object) -> LeagueSlot | None:
        if isinstance(value, LeagueSlot):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for slot in LeagueSlot:
                if normalised in {slot.value, slot.name.lower()}:
                    return slot
        return None

    def _report_metadata(self, report: object) -> dict[str, object]:
        if not isinstance(report, FitnessReport):  # defensive guard - legacy payloads
            return {}

        metadata: dict[str, object] = {}
        raw_meta = report.metadata
        if isinstance(raw_meta, Mapping):
            metadata.update(raw_meta)

        metadata.setdefault("fitness_score", float(report.fitness_score))
        metadata.setdefault("max_drawdown", float(report.max_drawdown))
        metadata.setdefault("sharpe_ratio", float(report.sharpe_ratio))
        metadata.setdefault("total_return", float(report.total_return))
        metadata.setdefault("volatility", float(report.volatility))
        return metadata

    def _score_matchups(
        self,
        matchups: Sequence[LeagueMatchup],
        evaluation_index: Mapping[str, EvaluationRecord],
    ) -> tuple[LeagueResult, ...]:
        results: list[LeagueResult] = []
        for matchup in matchups:
            challenger_record = evaluation_index.get(matchup.challenger.agent_id)
            opponent_record = evaluation_index.get(matchup.opponent.agent_id)

            challenger_score = (
                challenger_record.fitness if challenger_record is not None else matchup.challenger.score or 0.0
            )
            opponent_score = (
                opponent_record.fitness if opponent_record is not None else matchup.opponent.score or 0.0
            )

            metadata: dict[str, object] = {"source": "evolution_cycle"}
            if challenger_record is not None:
                metadata["challenger_report"] = challenger_record.report.as_payload()
            if opponent_record is not None:
                metadata["opponent_report"] = opponent_record.report.as_payload()

            result = LeagueResult(
                matchup=matchup,
                challenger_score=float(challenger_score),
                opponent_score=float(opponent_score),
                metadata=metadata,
            )
            self._league.record_result(result)
            results.append(result)
        return tuple(results)
