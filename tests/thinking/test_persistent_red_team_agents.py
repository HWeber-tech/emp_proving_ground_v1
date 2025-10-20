import json

import pytest

from src.operational.state_store.adapters import InMemoryStateStore
from src.thinking.adversarial.red_team_ai import RedTeamAI


@pytest.mark.asyncio()
async def test_persistent_red_team_agents_escalate_intensity() -> None:
    store = InMemoryStateStore()
    red_team = RedTeamAI(store)

    weakness = "trend_reversal_blindness"

    async def _analyze(self, target_strategy: str, test_scenarios):  # type: ignore[override]
        return {"behavior_profile": {}}

    async def _find(self, behavior_profile, known_vulnerabilities):  # type: ignore[override]
        return [weakness]

    async def _create(self, weakness_name: str, target_strategy: str):  # type: ignore[override]
        return {
            "attack_id": f"attack-{weakness_name}",
            "strategy_id": target_strategy,
            "weakness_targeted": weakness_name,
            "attack_type": "trend_reversal",
            "parameters": {"intensity": "low", "duration": "short"},
            "expected_impact": 0.9,
        }

    async def _develop(self, weaknesses_list, target_strategy: str):  # type: ignore[override]
        return []

    async def _execute(self, target_strategy: str, attack):  # type: ignore[override]
        return {
            "attack_id": attack["attack_id"],
            "strategy_id": target_strategy,
            "success": True,
            "impact": 0.9,
            "timestamp": "2024-01-01T00:00:00Z",
            "specialization_level": attack.get("specialization_level"),
            "assigned_agent_id": attack.get("assigned_agent_id"),
            "intensity": attack.get("parameters", {}).get("intensity"),
        }

    red_team.strategy_analyzer.analyze_behavior = _analyze.__get__(
        red_team.strategy_analyzer, type(red_team.strategy_analyzer)
    )
    red_team.weakness_detector.find_weaknesses = _find.__get__(
        red_team.weakness_detector, type(red_team.weakness_detector)
    )
    red_team.attack_generator.create_attack = _create.__get__(
        red_team.attack_generator, type(red_team.attack_generator)
    )
    red_team.exploit_developer.develop_exploits = _develop.__get__(
        red_team.exploit_developer, type(red_team.exploit_developer)
    )
    red_team._execute_attack = _execute.__get__(red_team, RedTeamAI)

    first_report = await red_team.attack_strategy("strategy-alpha")
    assert first_report["attack_results"], "expected attack results for first run"
    first_result = first_report["attack_results"][0]
    assert first_result["specialization_level"] == "novice"
    assert first_result["intensity"] == "medium"

    agent_payload = await store.get(f"emp:red_team_agents:{weakness}")
    assert agent_payload is not None
    agent_record = json.loads(agent_payload)
    assert agent_record["attack_count"] == 1
    assert agent_record["skill_level"] == "seasoned"
    assert agent_record["preferred_intensity"] == "medium"

    second_report = await red_team.attack_strategy("strategy-alpha")
    assert second_report["attack_results"], "expected attack results for second run"
    second_result = second_report["attack_results"][0]
    assert second_result["specialization_level"] == "seasoned"
    assert second_result["intensity"] == "high"

    assert second_report.get("persistent_agents")
    persistent_snapshot = second_report["persistent_agents"][0]
    assert persistent_snapshot["skill_level"] == "seasoned"
    assert persistent_snapshot["attack_count"] == 2

    updated_payload = await store.get(f"emp:red_team_agents:{weakness}")
    assert updated_payload is not None
    updated_record = json.loads(updated_payload)
    assert updated_record["attack_count"] == 2
    assert updated_record["skill_level"] == "seasoned"
    assert updated_record["preferred_intensity"] == "high"

    stats = await red_team.get_red_team_stats()
    assert stats["dedicated_agents"] == 1
    assert stats["agent_skill_distribution"]["seasoned"] == 1
