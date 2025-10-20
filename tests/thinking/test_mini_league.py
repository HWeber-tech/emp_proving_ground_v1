from __future__ import annotations

import pytest

from src.evolution.mutation_ledger import MutationLedger
from src.thinking.adversarial.mini_league import (
    ExploitabilityObservation,
    LeagueEntry,
    LeagueMatchup,
    LeagueResult,
    LeagueSlot,
    MiniLeague,
)


def _entry(
    agent_id: str,
    sharpe: float,
    turnover: float,
    *,
    sharpness: float = 0.2,
    calibration_brier: float = 0.08,
    turnover_variance: float | None = None,
    inventory_variance: float | None = None,
) -> LeagueEntry:
    metadata = {
        "sharpe": sharpe,
        "turnover": turnover,
        "sharpness": sharpness,
        "calibration_brier": calibration_brier,
    }
    if turnover_variance is not None:
        metadata["turnover_variance"] = turnover_variance
    if inventory_variance is not None:
        metadata["inventory_variance"] = inventory_variance
    return LeagueEntry(
        agent_id=agent_id,
        score=sharpe,
        metadata=metadata,
    )


def test_mini_league_promotes_current_to_best() -> None:
    league = MiniLeague()
    league.register(
        LeagueSlot.CURRENT,
        _entry("agent-cur", sharpe=0.42, turnover=1.0, sharpness=0.25, calibration_brier=0.05),
    )
    league.register(
        LeagueSlot.BEST,
        _entry("agent-best", sharpe=0.42, turnover=1.0, sharpness=0.22, calibration_brier=0.05),
    )
    league.record_exploitability_observation()

    promoted = league.promote_current_to_best()
    assert promoted is not None
    assert promoted.agent_id == "agent-cur"
    assert league.best() is not None
    assert league.best() is not league.current()

    snapshot = league.snapshot()
    assert snapshot["current"][0]["agent_id"] == "agent-cur"
    assert snapshot["best"][0]["agent_id"] == "agent-cur"


def test_mini_league_enforces_exploit_capacity_and_ordering() -> None:
    league = MiniLeague(max_exploit=3)
    scores = [0.2, 0.8, 0.5, 0.9]
    for idx, score in enumerate(scores):
        league.register(
            LeagueSlot.EXPLOIT,
            LeagueEntry(agent_id=f"exploit-{idx}", score=score),
        )

    roster = league.roster(LeagueSlot.EXPLOIT)
    assert len(roster) == 3
    # Highest scores retained in order
    retained = [entry.agent_id for entry in roster]
    assert retained == ["exploit-3", "exploit-1", "exploit-2"]
    assert all(entry.score is not None for entry in roster)


def test_mini_league_schedule_round_and_record_results() -> None:
    league = MiniLeague()
    current = LeagueEntry(agent_id="current", score=0.3)
    best = LeagueEntry(agent_id="best", score=0.6)
    exploit = LeagueEntry(agent_id="exploit", score=0.4)
    chaos = LeagueEntry(agent_id="chaos", score=0.1)

    league.register(LeagueSlot.CURRENT, current)
    league.register(LeagueSlot.BEST, best)
    league.register(LeagueSlot.EXPLOIT, exploit)
    league.register(LeagueSlot.CHAOS, chaos)

    matchups = league.schedule_round()
    assert len(matchups) == 3
    assert all(isinstance(matchup, LeagueMatchup) for matchup in matchups)

    result = LeagueResult(matchups[0], challenger_score=1.0, opponent_score=0.0)
    league.record_result(result)

    history = league.history()
    assert history[-1] is result
    assert league.current() is current
    assert current.games_played == 1
    assert current.score == pytest.approx(1.0)


def test_remove_agent_from_slot() -> None:
    league = MiniLeague()
    league.register(LeagueSlot.EXPLOIT, LeagueEntry(agent_id="to-remove"))

    assert league.remove(LeagueSlot.EXPLOIT, "to-remove") is True
    assert not league.roster(LeagueSlot.EXPLOIT)
    assert league.remove(LeagueSlot.EXPLOIT, "ghost") is False


def test_exploitability_observation_requires_matched_turnover() -> None:
    league = MiniLeague()
    league.register(LeagueSlot.CURRENT, _entry("current", 1.05, 1.0))
    league.register(LeagueSlot.BEST, _entry("best", 1.30, 1.08))
    league.register(LeagueSlot.EXPLOIT, _entry("exploit-a", 1.20, 1.25))
    league.register(LeagueSlot.EXPLOIT, _entry("exploit-b", 1.22, 0.95))

    observation = league.compute_exploitability_observation(turnover_tolerance_pct=10.0)
    assert isinstance(observation, ExploitabilityObservation)
    assert observation.current_agent_id == "current"
    assert observation.current_metric == pytest.approx(1.05)
    assert {comp.agent_id for comp in observation.comparisons} == {"best", "exploit-b"}
    assert observation.selected_gap == pytest.approx(0.25)
    assert observation.selected_slot is LeagueSlot.BEST
    assert observation.selected_agent_id == "best"


def test_promotion_requires_exploitability_gap_to_shrink() -> None:
    league = MiniLeague()
    league.register(
        LeagueSlot.CURRENT,
        _entry("current", 1.0, 1.0, sharpness=0.25, calibration_brier=0.05),
    )
    league.register(LeagueSlot.BEST, _entry("best", 1.05, 1.0))

    first = league.record_exploitability_observation()
    assert first.selected_gap == pytest.approx(0.05)
    assert first.wow_delta is None

    league.register(LeagueSlot.BEST, _entry("best", 1.20, 1.0))
    widened = league.record_exploitability_observation()
    assert widened.selected_gap == pytest.approx(0.20)
    assert widened.wow_delta == pytest.approx(0.15)
    assert league.promote_current_to_best() is None

    league.register(LeagueSlot.BEST, _entry("best", 1.03, 1.0))
    improved = league.record_exploitability_observation()
    assert improved.selected_gap == pytest.approx(0.03)
    assert improved.wow_delta == pytest.approx(-0.17)

    promoted = league.promote_current_to_best()
    assert promoted is not None
    assert promoted.agent_id == "current"


def test_promotion_blocked_when_covenant_metrics_fail() -> None:
    league = MiniLeague()
    league.register(
        LeagueSlot.CURRENT,
        _entry("current", 1.0, 1.0, sharpness=0.1, calibration_brier=0.05),
    )
    league.register(LeagueSlot.BEST, _entry("best", 1.02, 1.0))
    league.record_exploitability_observation()

    assert league.promote_current_to_best() is None

    league.register(
        LeagueSlot.CURRENT,
        _entry("current", 1.0, 1.0, sharpness=0.2, calibration_brier=0.2),
    )

    assert league.promote_current_to_best() is None

    league.register(
        LeagueSlot.CURRENT,
        _entry("current", 1.0, 1.0, sharpness=0.22, calibration_brier=0.05),
    )

    promoted = league.promote_current_to_best()
    assert promoted is not None
    assert promoted.agent_id == "current"


def test_mini_league_records_exploitability_in_ledger() -> None:
    ledger = MutationLedger()
    league = MiniLeague(mutation_ledger=ledger)
    league.register(LeagueSlot.CURRENT, _entry("current", 1.10, 1.0))
    league.register(LeagueSlot.BEST, _entry("best", 1.30, 1.05))
    league.register(LeagueSlot.EXPLOIT, _entry("exploit", 1.25, 1.02))

    observation = league.record_exploitability_observation(turnover_tolerance_pct=10.0)
    assert observation.selected_agent_id is not None

    records = ledger.exploitability_results
    assert len(records) == 1
    payload = records[0].as_dict()
    assert payload["selected_agent_id"] == observation.selected_agent_id
    assert payload["metric"] == observation.metric
    snapshot = league.mutation_ledger.snapshot()
    assert snapshot["exploitability_results"], "snapshot should include recorded observation"


def test_lagrangian_constraints_penalize_variance() -> None:
    league = MiniLeague()
    league.register(
        LeagueSlot.CURRENT,
        _entry(
            "current",
            1.0,
            1.0,
            turnover_variance=0.05,
            inventory_variance=0.02,
        ),
    )
    league.register(
        LeagueSlot.BEST,
        _entry(
            "best",
            1.03,
            1.0,
            turnover_variance=0.05,
            inventory_variance=0.02,
        ),
    )
    league.register(
        LeagueSlot.EXPLOIT,
        _entry(
            "exploit",
            1.05,
            1.02,
            turnover_variance=0.30,
            inventory_variance=0.25,
        ),
    )

    first = league.record_exploitability_observation()
    assert first.selected_agent_id == "exploit"
    assert first.selected_gap == pytest.approx(0.05)
    exploit_first = next(comp for comp in first.comparisons if comp.agent_id == "exploit")
    assert exploit_first.lagrangian_penalty == pytest.approx(0.0)
    assert exploit_first.lagrangian_adjusted_gap == pytest.approx(0.05)

    second = league.record_exploitability_observation()
    assert second.selected_agent_id == "best"
    assert second.selected_gap == pytest.approx(0.03)
    exploit_comp = next(comp for comp in second.comparisons if comp.agent_id == "exploit")
    assert exploit_comp.lagrangian_penalty is not None and exploit_comp.lagrangian_penalty > 0.0
    assert exploit_comp.lagrangian_adjusted_gap == pytest.approx(0.0)
    assert second.selected_penalty == pytest.approx(0.0)
