"""Automate PolicyRouter reflection summaries for reviewer-ready telemetry.

This module builds on the PolicyRouter history to generate automated summaries
that highlight emerging tactics, experiment usage, and regime distribution. The
roadmap calls for delivering reviewer-friendly reflections without manual
telemetry spelunking, so the builder below converts raw digests into Markdown
and machine-readable payloads ready for observability surfaces or CLI exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, MutableMapping, Sequence

from .policy_router import PolicyRouter


def _default_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(slots=True, frozen=True)
class PolicyReflectionArtifacts:
    """Container for automated reflection outputs."""

    digest: Mapping[str, object]
    markdown: str
    payload: Mapping[str, object]


class PolicyReflectionBuilder:
    """Generate reviewer-ready reflections from a PolicyRouter instance."""

    def __init__(
        self,
        policy_router: PolicyRouter,
        *,
        now: Callable[[], datetime] | None = None,
        default_window: int | None = None,
        max_tactics: int = 5,
        max_experiments: int = 5,
        max_headlines: int = 5,
        max_features: int = 5,
    ) -> None:
        self._router = policy_router
        self._now = now or _default_now
        self._default_window = default_window
        self._max_tactics = max(1, max_tactics)
        self._max_experiments = max(1, max_experiments)
        self._max_headlines = max(1, max_headlines)
        self._max_features = max(1, max_features)

    def build(self, *, window: int | None = None) -> PolicyReflectionArtifacts:
        digest = self._router.reflection_digest(window=window or self._default_window)
        generated_at = self._now()

        metadata: MutableMapping[str, object] = {
            "generated_at": generated_at.isoformat(),
            "total_decisions": digest.get("total_decisions", 0),
            "as_of": digest.get("as_of"),
        }
        if window is not None:
            metadata["window"] = window
        elif self._default_window is not None:
            metadata["window"] = self._default_window

        insights = self._derive_insights(digest)
        payload: Mapping[str, object] = {
            "metadata": dict(metadata),
            "digest": digest,
            "insights": insights,
        }
        markdown = self._build_markdown(metadata=metadata, digest=digest, insights=insights)
        return PolicyReflectionArtifacts(digest=digest, markdown=markdown, payload=payload)

    def _derive_insights(self, digest: Mapping[str, object]) -> Sequence[str]:
        total = int(digest.get("total_decisions", 0) or 0)
        if total == 0:
            return ("No PolicyRouter decisions recorded yet; run the understanding loop to collect telemetry.",)

        insights: list[str] = []

        tactics = self._slice_entries(digest.get("tactics", ()), self._max_tactics)
        if tactics:
            top = tactics[0]
            share = self._format_percentage(top.get("share", 0.0))
            insights.append(
                "Leading tactic {} at {} share (avg score {:.3f})".format(
                    top.get("tactic_id", "unknown"),
                    share,
                    float(top.get("avg_score", 0.0)),
                )
            )

        experiments = self._slice_entries(digest.get("experiments", ()), self._max_experiments)
        if experiments:
            top_exp = experiments[0]
            share = self._format_percentage(top_exp.get("share", 0.0))
            gating_bits: list[str] = []
            regimes = top_exp.get("regimes")
            if isinstance(regimes, Sequence) and regimes:
                gating_bits.append("regimes " + ", ".join(str(regime) for regime in regimes))
            min_conf = top_exp.get("min_confidence")
            if isinstance(min_conf, (int, float)) and float(min_conf) > 0:
                gating_bits.append(f"confidence >= {float(min_conf):.2f}")
            insights.append(
                "Top experiment {} applied {} times ({} share{})".format(
                    top_exp.get("experiment_id", "unknown"),
                    int(top_exp.get("count", 0)),
                    share,
                    f"; {'; '.join(gating_bits)}" if gating_bits else "",
                ).rstrip(";")
            )

        tag_entries = self._slice_entries(digest.get("tags", ()), self._max_tactics)
        if tag_entries:
            top_tag = tag_entries[0]
            share = self._format_percentage(top_tag.get("share", 0.0))
            top_tag_tactics = top_tag.get("top_tactics")
            top_tactic = (
                top_tag_tactics[0]
                if isinstance(top_tag_tactics, Sequence) and top_tag_tactics
                else "n/a"
            )
            insights.append(
                "Dominant tag {} at {} (top tactic {})".format(
                    top_tag.get("tag", "unknown"),
                    share,
                    top_tactic,
                )
            )

        objective_entries = self._slice_entries(digest.get("objectives", ()), self._max_tactics)
        if objective_entries:
            top_objective = objective_entries[0]
            share = self._format_percentage(top_objective.get("share", 0.0))
            top_obj_tactics = top_objective.get("top_tactics")
            top_tactic = (
                top_obj_tactics[0]
                if isinstance(top_obj_tactics, Sequence) and top_obj_tactics
                else "n/a"
            )
            insights.append(
                "Leading objective {} across {} share (top tactic {})".format(
                    top_objective.get("objective", "unknown"),
                    share,
                    top_tactic,
                )
            )

        confidence = digest.get("confidence", {})
        if isinstance(confidence, Mapping) and int(confidence.get("count", 0) or 0) > 0:
            avg_conf = float(confidence.get("average", 0.0) or 0.0)
            latest_conf = confidence.get("latest")
            latest_text = (
                f"{float(latest_conf):.2f}" if isinstance(latest_conf, (int, float)) else "n/a"
            )
            change = confidence.get("change")
            change_text = (
                f"{float(change):+.2f}" if isinstance(change, (int, float)) else None
            )
            insight = f"Average confidence {avg_conf:.2f} (latest {latest_text})"
            if change_text is not None:
                insight += f"; change {change_text}"
            insights.append(insight)

        feature_entries = self._slice_entries(digest.get("features", ()), self._max_features)
        if feature_entries:
            top_feature = feature_entries[0]
            name = top_feature.get("feature", "unknown")
            latest = top_feature.get("latest")
            latest_text = f"{float(latest):.2f}" if isinstance(latest, (int, float)) else "n/a"
            avg_value = top_feature.get("average")
            avg_text = f"{float(avg_value):.2f}" if isinstance(avg_value, (int, float)) else "n/a"
            trend = top_feature.get("trend")
            trend_text = (
                f"{float(trend):+.2f}" if isinstance(trend, (int, float)) else None
            )
            insight = f"Feature {name} averaging {avg_text} (latest {latest_text})"
            if trend_text is not None:
                insight += f"; trend {trend_text}"
            insights.append(insight)

        longest = digest.get("longest_streak", {})
        streak_tactic = longest.get("tactic_id")
        streak_len = int(longest.get("length") or 0)
        if streak_tactic and streak_len > 1:
            insights.append(
                "Longest streak: {} selected {} times".format(streak_tactic, streak_len)
            )

        regimes = digest.get("regimes", {})
        if isinstance(regimes, Mapping) and regimes:
            dominant_regime, data = next(iter(regimes.items()))
            share = self._format_percentage(data.get("share", 0.0)) if isinstance(data, Mapping) else "0%"
            insights.append(f"Dominant regime {dominant_regime} at {share}")

        headlines = digest.get("recent_headlines", ())
        if isinstance(headlines, Sequence) and headlines:
            latest = str(headlines[-1])
            insights.append(f"Latest headline: {latest}")

        return tuple(insights)

    def _build_markdown(
        self,
        *,
        metadata: Mapping[str, object],
        digest: Mapping[str, object],
        insights: Sequence[str],
    ) -> str:
        lines: list[str] = []
        lines.append("# PolicyRouter reflection summary")
        lines.append("")
        lines.append(f"Generated at: {metadata['generated_at']}")
        if metadata.get("window") is not None:
            lines.append(f"Window: last {metadata['window']} decisions")
        lines.append(
            "Decisions analysed: {}".format(int(digest.get("total_decisions", 0) or 0))
        )
        if digest.get("as_of"):
            lines.append(f"Latest decision timestamp: {digest['as_of']}")
        lines.append("")

        lines.append("## Key insights")
        if not insights:
            lines.append("- No decisions have been recorded yet.")
        else:
            for insight in insights[: self._max_headlines]:
                lines.append(f"- {insight}")
        lines.append("")

        if int(digest.get("total_decisions", 0) or 0) == 0:
            lines.append("_No decisions available. Run the understanding loop to capture telemetry._")
            return "\n".join(lines)

        confidence = digest.get("confidence", {})
        if isinstance(confidence, Mapping) and int(confidence.get("count", 0) or 0) > 0:
            lines.append("## Confidence summary")
            average = confidence.get("average")
            latest = confidence.get("latest")
            change = confidence.get("change")
            lines.append(
                "- Average: {}".format(
                    f"{float(average):.2f}" if isinstance(average, (int, float)) else "n/a"
                )
            )
            lines.append(
                "- Latest: {}".format(
                    f"{float(latest):.2f}" if isinstance(latest, (int, float)) else "n/a"
                )
            )
            if isinstance(change, (int, float)):
                lines.append(f"- Change: {float(change):+.2f}")
            first_seen = confidence.get("first_seen")
            last_seen = confidence.get("last_seen")
            if first_seen:
                lines.append(f"- First observed: {first_seen}")
            if last_seen:
                lines.append(f"- Last observed: {last_seen}")
            lines.append("")

        feature_entries = self._slice_entries(digest.get("features", ()), self._max_features)
        if feature_entries:
            lines.extend(
                self._render_table(
                    title="Feature highlights",
                    headers=(
                        "Feature",
                        "Avg",
                        "Latest",
                        "Trend",
                        "Min",
                        "Max",
                        "Count",
                        "Last seen",
                    ),
                    rows=[
                        (
                            str(entry.get("feature", "")),
                            (
                                f"{float(entry.get('average', 0.0)):.3f}"
                                if isinstance(entry.get("average"), (int, float))
                                else "n/a"
                            ),
                            (
                                f"{float(entry.get('latest', 0.0)):.3f}"
                                if isinstance(entry.get("latest"), (int, float))
                                else "n/a"
                            ),
                            (
                                f"{float(entry.get('trend', 0.0)):+.3f}"
                                if isinstance(entry.get("trend"), (int, float))
                                else "n/a"
                            ),
                            (
                                f"{float(entry.get('min', 0.0)):.3f}"
                                if isinstance(entry.get("min"), (int, float))
                                else "n/a"
                            ),
                            (
                                f"{float(entry.get('max', 0.0)):.3f}"
                                if isinstance(entry.get("max"), (int, float))
                                else "n/a"
                            ),
                            str(entry.get("count", 0)),
                            str(entry.get("last_seen", "")),
                        )
                        for entry in feature_entries
                    ],
                )
            )

        tactics = self._slice_entries(digest.get("tactics", ()), self._max_tactics)
        if tactics:
            lines.extend(self._render_table(
                title="Top tactics",
                headers=("Tactic", "Decisions", "Share", "Avg score", "Last seen", "Tags", "Objectives"),
                rows=[
                    (
                        str(entry.get("tactic_id", "")),
                        str(entry.get("count", 0)),
                        self._format_percentage(entry.get("share", 0.0)),
                        f"{float(entry.get('avg_score', 0.0)):.3f}",
                        str(entry.get("last_seen", "")),
                        ", ".join(entry.get("tags", ())) if entry.get("tags") else "-",
                        ", ".join(entry.get("objectives", ())) if entry.get("objectives") else "-",
                    )
                    for entry in tactics
                ],
            ))

        tag_entries = self._slice_entries(digest.get("tags", ()), self._max_tactics)
        if tag_entries:
            lines.extend(
                self._render_table(
                    title="Tag spotlight",
                    headers=("Tag", "Decisions", "Share", "Avg score", "Last seen", "Top tactics"),
                    rows=[
                        (
                            str(entry.get("tag", "")),
                            str(entry.get("count", 0)),
                            self._format_percentage(entry.get("share", 0.0)),
                            f"{float(entry.get('avg_score', 0.0)):.3f}",
                            str(entry.get("last_seen", "")),
                            ", ".join(entry.get("top_tactics", ()))
                            if entry.get("top_tactics")
                            else "-",
                        )
                        for entry in tag_entries
                    ],
                )
            )

        objective_entries = self._slice_entries(digest.get("objectives", ()), self._max_tactics)
        if objective_entries:
            lines.extend(
                self._render_table(
                    title="Objective coverage",
                    headers=(
                        "Objective",
                        "Decisions",
                        "Share",
                        "Avg score",
                        "Last seen",
                        "Top tactics",
                    ),
                    rows=[
                        (
                            str(entry.get("objective", "")),
                            str(entry.get("count", 0)),
                            self._format_percentage(entry.get("share", 0.0)),
                            f"{float(entry.get('avg_score', 0.0)):.3f}",
                            str(entry.get("last_seen", "")),
                            ", ".join(entry.get("top_tactics", ()))
                            if entry.get("top_tactics")
                            else "-",
                        )
                        for entry in objective_entries
                    ],
                )
            )

        experiments = self._slice_entries(digest.get("experiments", ()), self._max_experiments)
        if experiments:
            lines.extend(self._render_table(
                title="Active experiments",
                headers=(
                    "Experiment",
                    "Decisions",
                    "Share",
                    "Last seen",
                    "Regimes",
                    "Conf >=",
                    "Rationale",
                    "Top tactic",
                ),
                rows=[
                    (
                        str(entry.get("experiment_id", "")),
                        str(entry.get("count", 0)),
                        self._format_percentage(entry.get("share", 0.0)),
                        str(entry.get("last_seen", "")),
                        ", ".join(str(regime) for regime in entry.get("regimes", ()))
                        if entry.get("regimes")
                        else "-",
                        (
                            f"{float(entry.get('min_confidence', 0.0)):.2f}"
                            if isinstance(entry.get("min_confidence"), (int, float))
                            and float(entry.get("min_confidence", 0.0)) > 0.0
                            else "-"
                        ),
                        str(entry.get("rationale", "")),
                        str(entry.get("most_common_tactic", "")),
                    )
                    for entry in experiments
                ],
            ))

        regimes = digest.get("regimes", {})
        if isinstance(regimes, Mapping) and regimes:
            lines.append("## Regime distribution")
            for regime, data in list(regimes.items())[: self._max_headlines]:
                if isinstance(data, Mapping):
                    share = self._format_percentage(data.get("share", 0.0))
                    count = int(data.get("count", 0))
                else:
                    share = "0%"
                    count = 0
                lines.append(f"- {regime}: {count} decisions ({share})")
            lines.append("")

        current = digest.get("current_streak", {})
        longest = digest.get("longest_streak", {})
        lines.append("## Streaks")
        lines.append(
            "- Current: {} (length {})".format(
                current.get("tactic_id") or "none",
                int(current.get("length", 0) or 0),
            )
        )
        lines.append(
            "- Longest: {} (length {})".format(
                longest.get("tactic_id") or "none",
                int(longest.get("length", 0) or 0),
            )
        )
        lines.append("")

        headlines = digest.get("recent_headlines", ())
        if isinstance(headlines, Sequence) and headlines:
            lines.append("## Recent headlines")
            for headline in headlines[-self._max_headlines :]:
                lines.append(f"- {headline}")
            lines.append("")

        return "\n".join(lines)

    def _slice_entries(
        self,
        entries: object,
        limit: int,
    ) -> list[Mapping[str, object]]:
        if not isinstance(entries, Sequence):
            return []
        sliced: list[Mapping[str, object]] = []
        for entry in entries:
            if isinstance(entry, Mapping):
                sliced.append(entry)
            if len(sliced) >= limit:
                break
        return sliced

    def _render_table(
        self,
        *,
        title: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
    ) -> Sequence[str]:
        if not rows:
            return ()
        lines = [f"## {title}"]
        header_row = " | ".join(headers)
        separator = " | ".join(["---"] * len(headers))
        lines.append(header_row)
        lines.append(separator)
        for row in rows:
            lines.append(" | ".join(row))
        lines.append("")
        return lines

    @staticmethod
    def _format_percentage(value: object) -> str:
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "0.0%"


__all__ = [
    "PolicyReflectionArtifacts",
    "PolicyReflectionBuilder",
]
