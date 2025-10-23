"""Interactive helpers for inspecting multi-objective optimisation results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

__all__ = [
    "ObjectivePoint",
    "ObjectiveSpaceExplorer",
    "TradeoffMetrics",
]

Number = float | int
_LABEL_KEYS = ("label", "name", "id", "identifier", "uuid", "tag")


@dataclass(slots=True)
class ObjectivePoint:
    """Container describing an individual within objective space."""

    index: int
    objectives: dict[str, float]
    label: str
    provided_rank: int | None
    crowding_distance: float | None
    metadata: dict[str, Any]
    source: Any | None
    front_rank: int | None = None

    def value(self, objective: str) -> float:
        return self.objectives[objective]

    def as_dict(self, *, include_metadata: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "label": self.label,
            "objectives": dict(self.objectives),
        }
        if self.front_rank is not None:
            data["front_rank"] = self.front_rank
        if self.provided_rank is not None:
            data["rank"] = self.provided_rank
        if self.crowding_distance is not None:
            data["crowding_distance"] = self.crowding_distance
        if include_metadata and self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(slots=True)
class TradeoffMetrics:
    """Summary of pairwise trade-offs between two objectives."""

    objectives: tuple[str, str]
    correlation: float | None
    slope: float | None
    x_span: float
    y_span: float
    opportunity_cost: float | None
    efficiency_ratio: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "objectives": self.objectives,
            "correlation": self.correlation,
            "slope": self.slope,
            "x_span": self.x_span,
            "y_span": self.y_span,
            "opportunity_cost": self.opportunity_cost,
            "efficiency_ratio": self.efficiency_ratio,
        }


class ObjectiveSpaceExplorer:
    """Explore objective trade-offs and Pareto structure for ranked individuals."""

    def __init__(
        self,
        individuals: Iterable[Any],
        *,
        objective_names: Sequence[str] | None = None,
        maximise_objectives: Mapping[str, bool] | Sequence[bool] | None = None,
    ) -> None:
        raw_individuals = list(individuals)
        inferred_names: tuple[str, ...] | None = tuple(objective_names) if objective_names else None
        points: list[ObjectivePoint] = []
        for index, item in enumerate(raw_individuals):
            parsed = self._parse_individual(item, index, inferred_names)
            objectives, candidate_names, label, rank, crowding, metadata, source = parsed
            if inferred_names is None:
                inferred_names = candidate_names
            else:
                self._validate_objective_set(candidate_names, inferred_names)
                objectives = {name: objectives[name] for name in inferred_names}
            points.append(
                ObjectivePoint(
                    index=index,
                    objectives=objectives,
                    label=label or f"point-{index}",
                    provided_rank=rank,
                    crowding_distance=crowding,
                    metadata=metadata,
                    source=item if source is None else source,
                )
            )
        if inferred_names is None:
            raise ValueError("objective_names could not be inferred; provide them explicitly")
        self._objective_names = inferred_names
        self._points = tuple(points)
        self._fronts: tuple[tuple[ObjectivePoint, ...], ...] | None = None
        self._maximise = self._normalise_maximise(maximise_objectives)

    # ------------------------------------------------------------------
    # Public lifecycle helpers
    # ------------------------------------------------------------------
    @property
    def objective_names(self) -> tuple[str, ...]:
        return self._objective_names

    @property
    def maximise_objectives(self) -> Mapping[str, bool]:
        return self._maximise

    def __iter__(self) -> Iterator[ObjectivePoint]:
        return iter(self._points)

    def __len__(self) -> int:
        return len(self._points)

    @classmethod
    def from_nsga2_result(
        cls,
        result: Any,
        *,
        objective_names: Sequence[str] | None = None,
        maximise_objectives: Mapping[str, bool] | Sequence[bool] | None = None,
    ) -> "ObjectiveSpaceExplorer":
        if not hasattr(result, "ranked_population"):
            raise TypeError("result must expose ranked_population")
        ranked = getattr(result, "ranked_population")
        return cls(ranked, objective_names=objective_names, maximise_objectives=maximise_objectives)

    def points(self) -> tuple[ObjectivePoint, ...]:
        return self._points

    def pareto_front(self) -> tuple[ObjectivePoint, ...]:
        fronts = self.fronts()
        return fronts[0] if fronts else ()

    def fronts(self) -> tuple[tuple[ObjectivePoint, ...], ...]:
        if self._fronts is not None:
            return self._fronts
        remaining = list(self._points)
        fronts: list[tuple[ObjectivePoint, ...]] = []
        rank = 0
        while remaining:
            front: list[ObjectivePoint] = []
            for candidate in remaining:
                dominated = False
                for other in remaining:
                    if candidate is other:
                        continue
                    if self._dominates(other, candidate):
                        dominated = True
                        break
                if not dominated:
                    front.append(candidate)
            if not front:
                break
            for point in front:
                point.front_rank = rank
            fronts.append(tuple(front))
            remaining = [point for point in remaining if point not in front]
            rank += 1
        self._fronts = tuple(fronts)
        return self._fronts

    def tradeoff_matrix(self, *, front_only: bool = False) -> tuple[TradeoffMetrics, ...]:
        points = self.pareto_front() if front_only else self._points
        if len(points) < 2 or len(self._objective_names) < 2:
            return ()
        metrics: list[TradeoffMetrics] = []
        for i, left in enumerate(self._objective_names):
            for right in self._objective_names[i + 1 :]:
                metrics.append(self._build_tradeoff(points, left, right))
        return tuple(metrics)

    def describe_tradeoffs(self, *, front_only: bool = False, precision: int = 3) -> str:
        metrics = self.tradeoff_matrix(front_only=front_only)
        if not metrics:
            return "No objectives to compare."
        lines: list[str] = []
        for item in metrics:
            corr = "n/a" if item.correlation is None else f"{item.correlation:.{precision}f}"
            slope = "n/a" if item.slope is None else f"{item.slope:.{precision}f}"
            cost = "n/a" if item.opportunity_cost is None else f"{item.opportunity_cost:.{precision}f}"
            lines.append(
                f"{item.objectives[0]} ↔ {item.objectives[1]} | corr={corr} | slope={slope} | cost={cost}"
            )
        return "\n".join(lines)

    def build_figure(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        color_by_rank: bool = True,
        highlight_front: bool = True,
    ) -> Any:
        try:
            import plotly.graph_objects as go  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("plotly is required to build figures") from exc

        dimensions = [dim for dim in (x, y, z) if dim is not None]
        if not dimensions:
            dimensions = list(self._objective_names[:3])
        for dim in dimensions:
            if dim not in self._objective_names:
                raise KeyError(f"Unknown objective '{dim}'")
        if len(dimensions) < 2:
            raise ValueError("At least two objectives are required for visualisation")
        if len(dimensions) > 3:
            raise ValueError("A maximum of three dimensions is supported")

        self.fronts()  # ensure ranks are populated
        front_indices = {point.index for point in self.pareto_front()}
        colors = [self._resolve_point_colour(point, color_by_rank) for point in self._points]
        hover_text = [self._build_hover_text(point) for point in self._points]

        axis_data = {dim: [point.value(dim) for point in self._points] for dim in dimensions}
        custom_data = [[point.label, point.front_rank] for point in self._points]

        if len(dimensions) == 2:
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=axis_data[dimensions[0]],
                    y=axis_data[dimensions[1]],
                    mode="markers",
                    marker=dict(color=colors, size=9, opacity=0.8),
                    text=hover_text,
                    customdata=custom_data,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            if highlight_front and front_indices:
                figure.add_trace(
                    go.Scatter(
                        x=[axis_data[dimensions[0]][idx] for idx in front_indices],
                        y=[axis_data[dimensions[1]][idx] for idx in front_indices],
                        mode="markers",
                        marker=dict(color="crimson", size=12, symbol="diamond"),
                        text=[hover_text[idx] for idx in front_indices],
                        name="Pareto front",
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
            figure.update_layout(
                xaxis_title=dimensions[0],
                yaxis_title=dimensions[1],
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            return figure

        figure3d = go.Figure(
            data=[
                go.Scatter3d(
                    x=axis_data[dimensions[0]],
                    y=axis_data[dimensions[1]],
                    z=axis_data[dimensions[2]],
                    mode="markers",
                    marker=dict(color=colors, size=7, opacity=0.7),
                    text=hover_text,
                    customdata=custom_data,
                    hovertemplate="%{text}<extra></extra>",
                    name="Population",
                )
            ]
        )
        if highlight_front and front_indices:
            figure3d.add_trace(
                go.Scatter3d(
                    x=[axis_data[dimensions[0]][idx] for idx in front_indices],
                    y=[axis_data[dimensions[1]][idx] for idx in front_indices],
                    z=[axis_data[dimensions[2]][idx] for idx in front_indices],
                    mode="markers",
                    marker=dict(color="crimson", size=11, symbol="diamond"),
                    text=[hover_text[idx] for idx in front_indices],
                    name="Pareto front",
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        figure3d.update_layout(
            scene=dict(
                xaxis_title=dimensions[0],
                yaxis_title=dimensions[1],
                zaxis_title=dimensions[2],
            )
        )
        return figure3d

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_individual(
        self,
        item: Any,
        index: int,
        objective_names: tuple[str, ...] | None,
    ) -> tuple[dict[str, float], tuple[str, ...], str, int | None, float | None, dict[str, Any], Any | None]:
        if hasattr(item, "objectives") and hasattr(item, "genome"):
            objectives_seq = getattr(item, "objectives")
            objectives, names = self._coerce_objective_container(objectives_seq, objective_names)
            metadata: dict[str, Any] = {"genome": getattr(item, "genome")}
            rank = _coerce_optional_int(getattr(item, "rank", None))
            crowding = _coerce_optional_float(getattr(item, "crowding_distance", None))
            label = _infer_label_from(metadata["genome"]) or _infer_label_from(item) or f"individual-{index}"
            return objectives, names, label, rank, crowding, metadata, item

        if isinstance(item, Mapping):
            metadata = dict(item)
            label = _infer_label_from(metadata)
            crowding = _coerce_optional_float(metadata.pop("crowding_distance", None))
            rank = _coerce_optional_int(metadata.pop("rank", None))
            objectives_container = None
            for key in ("objectives", "metrics", "values"):
                if key in metadata:
                    objectives_container = metadata.pop(key)
                    break
            objectives: dict[str, float]
            names: tuple[str, ...]
            if objectives_container is not None:
                objectives, names = self._coerce_objective_container(objectives_container, objective_names)
            else:
                numeric_values: MutableMapping[str, float] = {}
                to_remove: list[str] = []
                for candidate_key, raw_value in metadata.items():
                    try:
                        numeric_values[candidate_key] = _coerce_to_float(raw_value)
                        to_remove.append(candidate_key)
                    except (TypeError, ValueError):
                        continue
                if not numeric_values:
                    raise ValueError("Mapping must expose objective values")
                for key in to_remove:
                    metadata.pop(key, None)
                objectives, names = self._coerce_objective_container(numeric_values, objective_names)
            return objectives, names, label or f"point-{index}", rank, crowding, metadata, None

        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            objectives, names = self._coerce_objective_container(item, objective_names)
            return objectives, names, f"point-{index}", None, None, {}, None

        raise TypeError("Unsupported individual type for objective exploration")

    def _coerce_objective_container(
        self,
        container: Any,
        objective_names: tuple[str, ...] | None,
    ) -> tuple[dict[str, float], tuple[str, ...]]:
        if isinstance(container, Mapping):
            items = list(container.items())
            if not items:
                raise ValueError("Objective mapping cannot be empty")
            if objective_names is None:
                names = tuple(str(key) for key, _ in items)
            else:
                names = objective_names
                self._validate_objective_set(tuple(str(key) for key, _ in items), names)
            values = {name: _coerce_to_float(container[name]) for name in names}
            return values, names

        if isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)):
            values_list = list(container)
            if not values_list:
                raise ValueError("Objective sequence cannot be empty")
            if objective_names is None:
                names = _generate_default_names(len(values_list))
            else:
                if len(values_list) != len(objective_names):
                    raise ValueError("Objective count mismatch with provided names")
                names = objective_names
            values = {name: _coerce_to_float(value) for name, value in zip(names, values_list)}
            return values, names

        raise TypeError("Objective container must be a mapping or sequence")

    def _validate_objective_set(
        self,
        candidate: tuple[str, ...],
        expected: tuple[str, ...],
    ) -> None:
        if len(candidate) != len(expected):
            raise ValueError("Objective dimensionality mismatch")
        expected_set = set(expected)
        candidate_set = set(candidate)
        if not candidate_set.issuperset(expected_set):
            raise ValueError("Objective keys do not align with explorer configuration")

    def _normalise_maximise(
        self,
        maximise_objectives: Mapping[str, bool] | Sequence[bool] | None,
    ) -> dict[str, bool]:
        maximise: dict[str, bool] = {name: True for name in self._objective_names}
        if maximise_objectives is None:
            return maximise
        if isinstance(maximise_objectives, Mapping):
            for name, flag in maximise_objectives.items():
                if name not in maximise:
                    raise KeyError(f"Unknown objective '{name}' in maximise mapping")
                maximise[name] = bool(flag)
            return maximise
        flags = [bool(flag) for flag in maximise_objectives]
        if len(flags) != len(self._objective_names):
            raise ValueError("Length of maximise_objectives must match objective count")
        for name, flag in zip(self._objective_names, flags):
            maximise[name] = flag
        return maximise

    def _dominates(self, left: ObjectivePoint, right: ObjectivePoint) -> bool:
        better_in_any = False
        for name, maximise in self._maximise.items():
            l_value = left.value(name)
            r_value = right.value(name)
            if _is_nan(l_value) and _is_nan(r_value):
                continue
            if _is_nan(l_value):
                return False
            if _is_nan(r_value):
                better_in_any = True
                continue
            if maximise:
                if l_value < r_value - 1e-12:
                    return False
                if l_value > r_value + 1e-12:
                    better_in_any = True
            else:
                if l_value > r_value + 1e-12:
                    return False
                if l_value < r_value - 1e-12:
                    better_in_any = True
        return better_in_any

    def _build_tradeoff(
        self,
        points: Sequence[ObjectivePoint],
        left: str,
        right: str,
    ) -> TradeoffMetrics:
        x_values = [point.value(left) for point in points]
        y_values = [point.value(right) for point in points]
        correlation = _pearson_correlation(x_values, y_values)
        slope = _least_squares_slope(x_values, y_values)
        x_span = max(x_values) - min(x_values) if x_values else 0.0
        y_span = max(y_values) - min(y_values) if y_values else 0.0
        direction_factor = self._direction_factor(left, right)
        opportunity = slope * direction_factor if slope is not None else None
        efficiency = (y_span / x_span) * direction_factor if x_span and y_span else None
        return TradeoffMetrics(
            objectives=(left, right),
            correlation=correlation,
            slope=slope,
            x_span=x_span,
            y_span=y_span,
            opportunity_cost=opportunity,
            efficiency_ratio=efficiency,
        )

    def _direction_factor(self, x: str, y: str) -> float:
        factor = 1.0
        if not self._maximise.get(x, True):
            factor *= -1.0
        if self._maximise.get(y, True):
            factor *= 1.0
        else:
            factor *= -1.0
        return factor

    def _resolve_point_colour(self, point: ObjectivePoint, color_by_rank: bool) -> float | int:
        if not color_by_rank:
            return 0.5
        rank = point.front_rank if point.front_rank is not None else point.provided_rank
        if rank is None:
            return 0.0
        return float(rank)

    def _build_hover_text(self, point: ObjectivePoint) -> str:
        components = [point.label]
        components.extend(f"{name}: {point.value(name):.4f}" for name in self._objective_names)
        if point.front_rank is not None:
            components.append(f"front: {point.front_rank}")
        elif point.provided_rank is not None:
            components.append(f"rank: {point.provided_rank}")
        if point.crowding_distance is not None:
            components.append(f"crowding: {point.crowding_distance:.4f}")
        return " | ".join(components)


def _generate_default_names(count: int) -> tuple[str, ...]:
    return tuple(f"objective_{index + 1}" for index in range(count))


def _infer_label_from(candidate: Any) -> str | None:
    if isinstance(candidate, Mapping):
        for key in _LABEL_KEYS:
            value = candidate.get(key)
            if value:
                return str(value)
    if hasattr(candidate, "label"):
        label = getattr(candidate, "label")
        if label:
            return str(label)
    if hasattr(candidate, "name"):
        name = getattr(candidate, "name")
        if name:
            return str(name)
    return None


def _coerce_to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid objective magnitudes")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError("Objective values must be numeric") from exc
    if math.isnan(numeric):
        raise ValueError("Objective values cannot be NaN")
    return numeric


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    mean_x = math.fsum(x_values) / len(x_values)
    mean_y = math.fsum(y_values) / len(y_values)
    cov = 0.0
    sq_x = 0.0
    sq_y = 0.0
    for x, y in zip(x_values, y_values):
        dx = x - mean_x
        dy = y - mean_y
        cov += dx * dy
        sq_x += dx * dx
        sq_y += dy * dy
    denom = math.sqrt(sq_x * sq_y)
    if denom == 0.0:
        return 0.0
    correlation = cov / denom
    if correlation > 1.0:
        return 1.0
    if correlation < -1.0:
        return -1.0
    return correlation


def _least_squares_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    mean_x = math.fsum(x_values) / len(x_values)
    mean_y = math.fsum(y_values) / len(y_values)
    numerator = 0.0
    denominator = 0.0
    for x, y in zip(x_values, y_values):
        dx = x - mean_x
        numerator += dx * (y - mean_y)
        denominator += dx * dx
    if denominator == 0.0:
        return None
    return numerator / denominator


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)
