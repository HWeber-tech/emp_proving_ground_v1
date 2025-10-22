"""Objective space exploration utilities for multi-objective optimization results."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class ObjectiveSpaceError(ValueError):
    """Raised when the objective space explorer cannot be constructed."""


def _normalise_objectives(
    frame: pd.DataFrame, objectives: Sequence[str]
) -> pd.DataFrame:
    missing = [name for name in objectives if name not in frame.columns]
    if missing:
        raise ObjectiveSpaceError(
            "Objective columns missing from frame: " + ", ".join(sorted(missing))
        )
    cleaned = frame.dropna(subset=list(objectives)).copy()
    if cleaned.empty:
        raise ObjectiveSpaceError("No rows available after dropping NaN objective values")
    return cleaned.reset_index(drop=True)


def _prepare_orientation(
    objectives: Sequence[str], maximise: Mapping[str, bool] | None
) -> tuple[np.ndarray, dict[str, bool]]:
    maximise_map = {name: True for name in objectives}
    if maximise:
        for name, flag in maximise.items():
            if name in maximise_map:
                maximise_map[name] = bool(flag)
    orientation = np.array([1.0 if not maximise_map[name] else -1.0 for name in objectives])
    return orientation, maximise_map


def _dominates(row_a: np.ndarray, row_b: np.ndarray) -> bool:
    """Return ``True`` when ``row_a`` dominates ``row_b`` in minimisation space."""

    return bool(np.all(row_a <= row_b) and np.any(row_a < row_b))


def _pareto_front(oriented: np.ndarray) -> np.ndarray:
    dominated = np.zeros(oriented.shape[0], dtype=bool)
    for i in range(oriented.shape[0]):
        if dominated[i]:
            continue
        for j in range(oriented.shape[0]):
            if i == j or dominated[j]:
                continue
            if _dominates(oriented[j], oriented[i]):
                dominated[i] = True
                break
            if _dominates(oriented[i], oriented[j]):
                dominated[j] = True
    return ~dominated


@dataclass(frozen=True)
class TradeOffRecord:
    """Summary statistics for a pair of objectives."""

    objective_a: str
    objective_b: str
    correlation: float | None
    pareto_correlation: float | None
    range_ratio: float | None
    orientation: str


def _safe_correlation(series_a: pd.Series, series_b: pd.Series) -> float | None:
    if len(series_a) < 2 or len(series_b) < 2:
        return None
    if series_a.std(ddof=0) == 0 or series_b.std(ddof=0) == 0:
        return None
    value = series_a.corr(series_b)
    if value is None or pd.isna(value):
        return None
    return float(value)


class ObjectiveSpaceExplorer:
    """Analyse and visualise multi-objective optimisation results."""

    def __init__(
        self,
        frame: pd.DataFrame,
        objectives: Sequence[str],
        *,
        maximise: Mapping[str, bool] | None = None,
        metadata: Sequence[str] | None = None,
    ) -> None:
        if not objectives:
            raise ObjectiveSpaceError("At least one objective column must be provided")
        cleaned = _normalise_objectives(frame, objectives)
        orientation, maximise_map = _prepare_orientation(objectives, maximise)

        self._frame = cleaned
        self._objectives = tuple(objectives)
        self._orientation = orientation
        self._maximise = maximise_map
        self._metadata = tuple(col for col in (metadata or ()) if col in cleaned.columns)

    @property
    def frame(self) -> pd.DataFrame:
        """Return a copy of the underlying frame."""

        return self._frame.copy()

    @property
    def objectives(self) -> tuple[str, ...]:
        return self._objectives

    @property
    def maximise(self) -> Mapping[str, bool]:
        return dict(self._maximise)

    def oriented_values(self) -> np.ndarray:
        values = self._frame.loc[:, self._objectives].to_numpy(dtype=float)
        return values * self._orientation

    def pareto_front(self) -> pd.DataFrame:
        mask = _pareto_front(self.oriented_values())
        return self._frame.loc[mask].reset_index(drop=True)

    def pareto_front_indices(self) -> tuple[int, ...]:
        mask = _pareto_front(self.oriented_values())
        return tuple(np.nonzero(mask)[0].tolist())

    def pareto_layers(self) -> list[pd.DataFrame]:
        oriented = self.oriented_values()
        layers: list[pd.DataFrame] = []
        indices = np.arange(oriented.shape[0])
        remaining_indices = indices.copy()
        remaining_oriented = oriented.copy()
        while remaining_oriented.size:
            front_mask = _pareto_front(remaining_oriented)
            current_indices = remaining_indices[front_mask]
            layers.append(self._frame.iloc[current_indices].reset_index(drop=True))
            remaining_oriented = remaining_oriented[~front_mask]
            remaining_indices = remaining_indices[~front_mask]
        return layers

    def plot_objective_space(
        self,
        *,
        title: str | None = None,
        hover_data: Sequence[str] | None = None,
    ) -> go.Figure:
        """Return an interactive figure describing the objective space."""

        hover = set(hover_data or ())
        hover.update(self._metadata)
        hover.update(self._objectives)
        hover_fields = [col for col in hover if col in self._frame.columns]

        front_indices = set(self.pareto_front_indices())
        frame = self._frame
        dims = len(self._objectives)

        figure = go.Figure()
        marker_style = dict(size=9, opacity=0.55, color="#4C78A8")
        pareto_marker = dict(size=11, color="#F58518", opacity=0.9, symbol="diamond")

        if dims == 1:
            objective = self._objectives[0]
            y = frame[objective]
            x = list(range(len(frame)))
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=marker_style,
                    name="Population",
                    hovertext=[
                        "<br>".join(
                            f"{field}: {frame.iloc[i][field]}" for field in hover_fields
                        )
                        for i in range(len(frame))
                    ],
                )
            )
            pareto_x = [idx for idx in range(len(frame)) if idx in front_indices]
            pareto_y = [y.iloc[idx] for idx in pareto_x]
            figure.add_trace(
                go.Scatter(
                    x=pareto_x,
                    y=pareto_y,
                    mode="markers",
                    marker=pareto_marker,
                    name="Pareto front",
                )
            )
            figure.update_layout(
                title=title or f"Objective space — {objective}",
                xaxis_title="Candidate",
                yaxis_title=objective,
            )
            return figure

        if dims == 2:
            x_obj, y_obj = self._objectives
            figure.add_trace(
                go.Scatter(
                    x=frame[x_obj],
                    y=frame[y_obj],
                    mode="markers",
                    marker=marker_style,
                    name="Population",
                    hovertext=[
                        "<br>".join(
                            f"{field}: {frame.iloc[i][field]}" for field in hover_fields
                        )
                        for i in range(len(frame))
                    ],
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[frame.iloc[i][x_obj] for i in front_indices],
                    y=[frame.iloc[i][y_obj] for i in front_indices],
                    mode="markers",
                    marker=pareto_marker,
                    name="Pareto front",
                )
            )
            figure.update_layout(
                title=title or f"Objective space — {x_obj} vs {y_obj}",
                xaxis_title=x_obj,
                yaxis_title=y_obj,
            )
            return figure

        if dims == 3:
            x_obj, y_obj, z_obj = self._objectives
            figure.add_trace(
                go.Scatter3d(
                    x=frame[x_obj],
                    y=frame[y_obj],
                    z=frame[z_obj],
                    mode="markers",
                    marker=marker_style,
                    name="Population",
                    hovertext=[
                        "<br>".join(
                            f"{field}: {frame.iloc[i][field]}" for field in hover_fields
                        )
                        for i in range(len(frame))
                    ],
                )
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[frame.iloc[i][x_obj] for i in front_indices],
                    y=[frame.iloc[i][y_obj] for i in front_indices],
                    z=[frame.iloc[i][z_obj] for i in front_indices],
                    mode="markers",
                    marker=pareto_marker,
                    name="Pareto front",
                )
            )
            figure.update_layout(
                title=title or f"Objective space — {x_obj}, {y_obj}, {z_obj}",
                scene=dict(
                    xaxis_title=x_obj,
                    yaxis_title=y_obj,
                    zaxis_title=z_obj,
                ),
            )
            return figure

        # Higher dimensional objective spaces fall back to parallel coordinates.
        dimensions = [
            dict(label=obj, values=self._frame[obj])
            for obj in self._objectives
        ]
        line_colors = [1 if idx in front_indices else 0 for idx in range(len(frame))]
        figure.add_trace(
            go.Parcoords(
                line=dict(color=line_colors, colorscale=[[0, "#B5C0D0"], [1, "#F58518"]]),
                dimensions=dimensions,
            )
        )
        figure.update_layout(title=title or "Objective space — parallel coordinates")
        return figure

    def trade_off_analysis(self) -> list[TradeOffRecord]:
        """Return pair-wise trade-off diagnostics for the configured objectives."""

        records: list[TradeOffRecord] = []
        front = self.pareto_front()
        for left, right in combinations(self._objectives, 2):
            series_a = self._frame[left]
            series_b = self._frame[right]
            correlation = _safe_correlation(series_a, series_b)
            front_corr = _safe_correlation(front[left], front[right]) if not front.empty else None
            range_a = series_a.max() - series_a.min()
            range_b = series_b.max() - series_b.min()
            range_ratio = None
            if range_a and range_b:
                range_ratio = float(range_a / range_b)
            orientation = "mixed"
            if self._maximise[left] == self._maximise[right]:
                orientation = "maximise" if self._maximise[left] else "minimise"
            records.append(
                TradeOffRecord(
                    objective_a=left,
                    objective_b=right,
                    correlation=correlation,
                    pareto_correlation=front_corr,
                    range_ratio=range_ratio,
                    orientation=orientation,
                )
            )
        return records

    def to_pareto_table(self) -> pd.DataFrame:
        """Return a tidy table combining metadata with Pareto front information."""

        front = self.pareto_front()
        if not self._metadata:
            return front
        return front.loc[:, list(dict.fromkeys(self._metadata + self._objectives))]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, float]],
        *,
        objectives: Sequence[str],
        maximise: Mapping[str, bool] | None = None,
        metadata: Sequence[str] | None = None,
    ) -> "ObjectiveSpaceExplorer":
        frame = pd.DataFrame.from_records(list(records))
        return cls(frame, objectives, maximise=maximise, metadata=metadata)

