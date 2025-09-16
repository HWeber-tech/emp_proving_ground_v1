"""Shared protocol and dataclass definitions used across the system."""

from __future__ import annotations

from .analysis import AnalysisResult, SensorySignal, ThinkingPattern
from .base import (
    Cache,
    ConfigProvider,
    DecisionGenome,
    EventBus,
    IExecutionEngine,
    IMutationStrategy,
    IPopulationManager,
    Logger,
    PopulationManager,
    RiskManager,
    SupportsEventPublish,
)
from .ecosystem import (
    CoordinationResult,
    EcosystemSummary,
    HasSpeciesType,
    ICoordinationEngine,
    IEcosystemOptimizer,
    ISpecialistGenomeFactory,
    MarketContext,
    MetricsSummary,
    TradeIntent,
)
from .metrics import CounterLike, GaugeLike, HistogramLike
from ..market_data import MarketDataGateway
from ..regime import RegimeClassifier, RegimeResult
from ..types import JSONObject

JSONValue = object

__all__ = [
    "Cache",
    "EventBus",
    "SupportsEventPublish",
    "Logger",
    "ConfigProvider",
    "RiskManager",
    "DecisionGenome",
    "IMutationStrategy",
    "IExecutionEngine",
    "PopulationManager",
    "IPopulationManager",
    "TradeIntent",
    "MarketContext",
    "HasSpeciesType",
    "CoordinationResult",
    "ICoordinationEngine",
    "IEcosystemOptimizer",
    "ISpecialistGenomeFactory",
    "MetricsSummary",
    "EcosystemSummary",
    "MarketDataGateway",
    "RegimeClassifier",
    "RegimeResult",
    "CounterLike",
    "GaugeLike",
    "HistogramLike",
    "ThinkingPattern",
    "SensorySignal",
    "AnalysisResult",
    "JSONValue",
    "JSONObject",
]
