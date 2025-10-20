from __future__ import annotations

import pytest

from src.thinking.adversarial.mini_league import (
    LeagueEntry,
    LeagueMatchup,
    LeagueResult,
    LeagueSlot,
    MiniLeague,
)


def test_mini_league_promotes_current_to_best() -> None:
    league = MiniLeague()
    league.register(LeagueSlot.CURRENT, LeagueEntry(agent_id="agent-cur", score=0.42))

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
