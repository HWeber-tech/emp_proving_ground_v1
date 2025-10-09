"""
Orchestration Composition Root
------------------------------

Assembles concrete runtime adapters for core ports while keeping domains
decoupled. All imports are guarded so that missing optional dependencies
degrade gracefully to core no-op implementations.

Provided adapters:
- YahooMarketDataGateway (preferred MarketDataGateway adapter when available)
- MarketDataGatewayAdapter (wraps YahooMarketDataGateway or injected organ)
- AnomalyDetectorAdapter (uses ManipulationDetectionSystem when available)
- RegimeClassifierAdapter (simple heuristic over pandas DataFrame)

Also provides compose_validation_adapters() to wire default instances.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, Dict, List, Optional, TypedDict, cast
from typing import Dict as _Dict

from src.core.adaptation import AdaptationService, NoOpAdaptationService
from src.core.anomaly import AnomalyDetector, AnomalyEvent, NoOpAnomalyDetector
from src.core.config_access import ConfigurationProvider, NoOpConfigurationProvider
from src.core.genome import GenomeProvider, NoOpGenomeProvider
from src.core.market_data import MarketDataGateway, NoOpMarketDataGateway
from src.core.regime import NoOpRegimeClassifier, RegimeClassifier, RegimeResult
from src.config.risk.risk_config import RiskConfig
from src.risk.manager import RiskManager, create_risk_manager
from src.data_foundation.ingest.yahoo_gateway import YahooMarketDataGateway


class ComposeAdaptersTD(TypedDict, total=False):
    market_data_gateway: MarketDataGateway
    anomaly_detector: AnomalyDetector
    regime_classifier: RegimeClassifier
    risk_manager: RiskManager
    adaptation_service: AdaptationService
    configuration_provider: ConfigurationProvider
    genome_provider: GenomeProvider


class MarketDataGatewayAdapter:
    """
    Concrete adapter for ``MarketDataGateway`` using the canonical Yahoo gateway
    when an explicit organ is not provided.

    All methods swallow exceptions and return ``None`` on error, consistent with
    the core port safety contract.
    """

    def __init__(self, organ: Optional[Any] = None) -> None:
        self._gateway: Optional[MarketDataGateway] = None
        self._organ: Optional[Any] = organ
        if organ is None:
            try:
                self._gateway = YahooMarketDataGateway()
            except Exception:
                self._gateway = None

    def fetch_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        """
        Synchronous fetch suitable for CLI/tests.
        Maps to YahooFinanceOrgan.fetch_data; start/end are accepted but ignored.
        """
        try:
            use_period = period or "1d"
            use_interval = interval or "1m"

            if self._gateway is not None:
                fetch_gateway = getattr(self._gateway, "fetch_data", None)
                if callable(fetch_gateway):
                    return fetch_gateway(
                        symbol,
                        period=use_period,
                        interval=use_interval,
                        start=start,
                        end=end,
                    )

            if self._organ is None:
                return None
            fetch = getattr(self._organ, "fetch_data", None)
            if not callable(fetch):
                return None
            return fetch(
                symbol,
                period=use_period,
                interval=use_interval,
                start=start,
                end=end,
            )
        except Exception:
            return None

    async def get_market_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        """
        Async fetch suitable for orchestrated flows.
        Implementation runs the synchronous fetch in a worker thread.
        """
        try:
            return await asyncio.to_thread(
                self.fetch_data,
                symbol,
                period,
                interval,
                start,
                end,
            )
        except Exception:
            return None


class AnomalyDetectorAdapter:
    """
    Concrete adapter for AnomalyDetector using ManipulationDetectionSystem when available.

    Swallows exceptions and returns an empty list on any error.
    """

    def __init__(self, system: Optional[Any] = None) -> None:
        self._system: Optional[Any] = None
        if system is not None:
            self._system = system
            return
        try:
            module = importlib.import_module("src.sensory.enhanced.anomaly.manipulation_detection")
            sys_cls = getattr(module, "ManipulationDetectionSystem", None)
            if callable(sys_cls):
                self._system = sys_cls()
        except Exception:
            self._system = None

    async def detect_manipulation(self, data: Any) -> List[_Dict[str, Any]] | List[AnomalyEvent]:
        try:
            if self._system is None:
                return []
            func = getattr(self._system, "detect_manipulation", None)
            if not callable(func):
                return []
            # If underlying is async, await; otherwise run in a thread.
            if asyncio.iscoroutinefunction(func):
                res = await func(data)
            else:
                res = await asyncio.to_thread(func, data)
            return (
                cast(List[_Dict[str, Any]] | List[AnomalyEvent], res)
                if isinstance(res, list)
                else []
            )
        except Exception:
            return []


class RegimeClassifierAdapter:
    """
    Concrete adapter for RegimeClassifier operating on pandas DataFrame.

    Heuristic:
      - Compute returns from 'close' column.
      - vol = std(returns)
      - mean_delta = mean of last 20 returns
      - Rules:
          vol > 0.03 -> CRISIS
          mean_delta > 0.0005 -> BULLISH
          mean_delta < -0.0005 -> BEARISH
          else -> SIDEWAYS
      - Confidence: clamp to [0.1, 1.0] using normalized vol and abs(mean_delta).
    """

    async def detect_regime(self, data: Any) -> Optional[RegimeResult]:
        try:
            import pandas as pd

            if not isinstance(data, pd.DataFrame) or "close" not in data.columns:
                return RegimeResult(regime="UNKNOWN", confidence=0.0)

            close = data["close"]
            returns = close.pct_change().dropna()

            if returns.empty:
                return RegimeResult(regime="UNKNOWN", confidence=0.0)

            vol = float(returns.std())
            mean_delta = float(returns.tail(20).mean())

            if vol > 0.03:
                regime = "CRISIS"
            else:
                if mean_delta > 0.0005:
                    regime = "BULLISH"
                elif mean_delta < -0.0005:
                    regime = "BEARISH"
                else:
                    regime = "SIDEWAYS"

            # Normalize and clamp confidence
            vol_norm = max(0.0, min(vol / 0.05, 1.0))
            delta_norm = max(0.0, min(abs(mean_delta) / 0.01, 1.0))
            confidence = max(0.1, min((vol_norm + delta_norm) / 2.0, 1.0))

            return RegimeResult(regime=regime, confidence=confidence)
        except Exception:
            return RegimeResult(regime="UNKNOWN", confidence=0.0)


class AdaptationServiceAdapter:
    """
    Adapter that wraps the optional SentientAdaptationEngine and presents the
    AdaptationService Protocol interface. All imports and calls are guarded and
    return safe defaults on error.
    """

    def __init__(self, engine: Optional[Any] = None) -> None:
        self._engine: Optional[Any] = None
        if engine is not None:
            self._engine = engine
            return
        try:
            module = importlib.import_module("src.intelligence.sentient_adaptation")
            engine_cls = getattr(module, "SentientAdaptationEngine", None)
            if callable(engine_cls):
                # The engine constructor takes no required args
                self._engine = engine_cls()
        except Exception:
            self._engine = None

    async def initialize(self) -> bool:
        try:
            if self._engine is None:
                return True
            init = getattr(self._engine, "initialize", None)
            if callable(init):
                if asyncio.iscoroutinefunction(init):
                    await init()
                else:
                    # If a sync initializer exists, run in a thread
                    await asyncio.to_thread(init)
            return True
        except Exception:
            return True

    async def stop(self) -> bool:
        try:
            if self._engine is None:
                return True
            stop = getattr(self._engine, "stop", None)
            if callable(stop):
                if asyncio.iscoroutinefunction(stop):
                    await stop()
                else:
                    await asyncio.to_thread(stop)
            return True
        except Exception:
            return True

    async def adapt_in_real_time(
        self, market_event: Any, strategy_response: Any, outcome: Any
    ) -> Dict[str, Any]:
        try:
            if self._engine is None:
                return {"success": False, "quality": 0.0, "adaptations": [], "confidence": 0.0}

            func = getattr(self._engine, "adapt_in_real_time", None)
            if not callable(func):
                return {"success": False, "quality": 0.0, "adaptations": [], "confidence": 0.0}

            if asyncio.iscoroutinefunction(func):
                res = await func(market_event, strategy_response, outcome)
            else:
                res = await asyncio.to_thread(func, market_event, strategy_response, outcome)

            # Normalize to a dict
            if isinstance(res, dict):
                d: Dict[str, Any] = dict(res)
            else:
                d = {}
                # Extract common attributes if present on dataclass-like result
                for key in (
                    "confidence",
                    "adaptation_strength",
                    "pattern_relevance",
                    "risk_adjustment",
                ):
                    try:
                        d[key] = getattr(res, key)
                    except Exception:
                        pass

            # Derive fields expected by consumers
            confidence_val = d.get("confidence", 0.0)
            try:
                confidence = float(confidence_val)  # best-effort cast
            except Exception:
                confidence = 0.0

            quality_val = d.get("quality", None)
            if not isinstance(quality_val, (int, float)):
                # Derive a proxy quality from available attributes
                parts = []
                for k in ("adaptation_strength", "pattern_relevance", "confidence"):
                    v = d.get(k)
                    if isinstance(v, (int, float)):
                        parts.append(float(v))
                quality = float(sum(parts) / len(parts)) if parts else 0.0
            else:
                quality = float(quality_val)

            return {
                "success": bool(d.get("success", True)),
                "quality": quality,
                "adaptations": d.get("adaptations", []),
                "confidence": confidence,
            }
        except Exception:
            return {"success": False, "quality": 0.0, "adaptations": [], "confidence": 0.0}


class ConfigurationProviderAdapter:
    """
    Adapter that exposes governance.system_config through the ConfigurationProvider port.

    All imports and attribute lookups are guarded; on any error, safe defaults are returned.
    """

    def __init__(self, cfg: Optional[Any] = None) -> None:
        self._cfg: Optional[Any] = None
        if cfg is not None:
            self._cfg = cfg
            return
        try:
            mod = importlib.import_module("src.governance.system_config")
            candidate = getattr(mod, "config", None)
            if candidate is None:
                sys_cls = getattr(mod, "SystemConfig", None)
                if callable(sys_cls):
                    try:
                        from_env = getattr(sys_cls, "from_env", None)
                        candidate = from_env() if callable(from_env) else sys_cls()
                    except Exception:
                        candidate = None
            # Prefer concrete instance if available, else fall back to module for attribute lookups
            self._cfg = candidate if candidate is not None else mod
        except Exception:
            self._cfg = None

    def get_value(self, key: str, default: Any = None) -> Any:
        try:
            obj = self._cfg
            if obj is None:
                return default
            # Mapping-style get
            get_fn = getattr(obj, "get", None)
            if callable(get_fn):
                try:
                    return get_fn(key, default)
                except Exception:
                    pass
            # Item access
            try:
                if hasattr(obj, "__getitem__"):
                    val = obj[key]
                    return default if val is None else val
            except Exception:
                pass
            # Attribute access
            try:
                val = getattr(obj, key, None)
                if val is not None:
                    return val
            except Exception:
                pass
            # Fallback to namespace lookup
            ns = self.get_namespace("system")
            return ns.get(key, default)
        except Exception:
            return default

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        try:
            obj = self._cfg
            if obj is None:
                return {}
            # Mapping-style
            try:
                if hasattr(obj, "__getitem__"):
                    ns_val = obj[namespace]
                    if isinstance(ns_val, dict):
                        return dict(ns_val)
                    to_dict = getattr(ns_val, "to_dict", None)
                    if callable(to_dict):
                        try:
                            d = to_dict()
                            if isinstance(d, dict):
                                return dict(d)
                        except Exception:
                            pass
            except Exception:
                pass
            # Attribute-based
            try:
                ns_attr = getattr(obj, namespace, None)
                if isinstance(ns_attr, dict):
                    return dict(ns_attr)
            except Exception:
                pass
        except Exception:
            return {}
        return {}


class GenomeProviderAdapterFactory:
    """
    Factory to construct a GenomeProvider instance without introducing import cycles.
    Returns GenomeProviderAdapter when available; otherwise falls back to NoOpGenomeProvider
    from src.core.genome. All imports are guarded and return safe defaults.
    """

    def build(self) -> Any:
        # Try concrete adapter from genome package
        try:
            mod = importlib.import_module("src.genome.models.genome_adapter")
            adapter_cls = getattr(mod, "GenomeProviderAdapter", None)
            if callable(adapter_cls):
                return adapter_cls()
        except Exception:
            pass

        # Fallback to core NoOp provider
        try:
            core_mod = importlib.import_module("src.core.genome")
            noop_cls = getattr(core_mod, "NoOpGenomeProvider", None)
            if callable(noop_cls):
                return noop_cls()
        except Exception:
            pass

        # Last-resort local no-op to preserve non-raising contract
        try:
            import time as _time

            class _LocalNoOp:
                def new_genome(
                    self,
                    id: str,
                    parameters: Dict[str, float],
                    generation: int = 0,
                    species_type: Optional[str] = None,
                ) -> Any:
                    return {
                        "id": str(id),
                        "parameters": dict(parameters or {}),
                        "fitness": None,
                        "generation": int(generation) if isinstance(generation, int) else 0,
                        "species_type": species_type,
                        "parent_ids": [],
                        "mutation_history": [],
                        "performance_metrics": {},
                        "created_at": _time.time(),
                    }

                def mutate(self, genome: Any, mutation: str, new_params: Dict[str, float]) -> Any:
                    try:
                        params = dict(getattr(genome, "parameters", {}) or {})
                        for k, v in (new_params or {}).items():
                            try:
                                params[k] = float(v) if v is not None else v
                            except Exception:
                                continue
                        if hasattr(genome, "with_updated"):
                            return genome.with_updated(parameters=params)
                    except Exception:
                        pass
                    return genome

                def from_legacy(self, obj: Any) -> Any:
                    return obj

                def to_legacy_view(self, genome: Any) -> Dict[str, Any] | Any:
                    try:
                        to_dict = getattr(genome, "to_dict", None)
                        if callable(to_dict):
                            d = to_dict()
                            return d if isinstance(d, dict) else {}
                        if isinstance(genome, dict):
                            return dict(genome)
                    except Exception:
                        pass
                    return {}

            return _LocalNoOp()
        except Exception:
            # Extremely defensive final fallback (shape-compatible callable dict)
            return {
                "new_genome": lambda *a, **k: {},
                "mutate": lambda g, *a, **k: g,
                "from_legacy": lambda o: o,
                "to_legacy_view": lambda g: {},
            }


def compose_validation_adapters() -> ComposeAdaptersTD:
    """
    Compose and return default adapters for validation or orchestration flows.

    Returns a dict with:
      - 'market_data_gateway': MarketDataGateway
      - 'anomaly_detector': AnomalyDetector
      - 'regime_classifier': RegimeClassifier
      - 'risk_manager': RiskManagerPort
      - 'adaptation_service': AdaptationService
      - 'configuration_provider': ConfigurationProvider
      - 'genome_provider': GenomeProvider (adapter instance or NoOp)
    """
    adapters: ComposeAdaptersTD = {}

    # Market data gateway (prefer hardened Yahoo gateway)
    try:
        adapters["market_data_gateway"] = YahooMarketDataGateway()
    except Exception:
        # Canonical gateway unavailable; fall back to a no-op implementation.
        adapters["market_data_gateway"] = NoOpMarketDataGateway()

    # Anomaly detector
    try:
        module = importlib.import_module("src.sensory.enhanced.anomaly.manipulation_detection")
        sys_cls = getattr(module, "ManipulationDetectionSystem", None)
        system = sys_cls() if callable(sys_cls) else None
        if system is not None:
            adapters["anomaly_detector"] = AnomalyDetectorAdapter(system=system)
        else:
            adapters["anomaly_detector"] = NoOpAnomalyDetector()
    except Exception:
        adapters["anomaly_detector"] = NoOpAnomalyDetector()

    # Regime classifier
    try:
        adapters["regime_classifier"] = RegimeClassifierAdapter()
    except Exception:
        adapters["regime_classifier"] = NoOpRegimeClassifier()

    # Risk manager (prefer canonical implementation)
    try:
        default_config = RiskConfig()
        adapters["risk_manager"] = create_risk_manager(config=default_config)
    except Exception:
        adapters["risk_manager"] = RiskManager(config=RiskConfig())

    # Adaptation service
    try:
        adapters["adaptation_service"] = AdaptationServiceAdapter()
    except Exception:
        adapters["adaptation_service"] = NoOpAdaptationService()

    # Configuration provider
    try:
        adapters["configuration_provider"] = ConfigurationProviderAdapter()
    except Exception:
        adapters["configuration_provider"] = NoOpConfigurationProvider()

    # Genome provider (DI only; no side effects/registration here)
    try:
        factory = GenomeProviderAdapterFactory()
        adapters["genome_provider"] = cast(GenomeProvider, factory.build())
    except Exception:
        # Fallback to core NoOp if available; else retain a minimal local no-op
        try:
            core_mod = importlib.import_module("src.core.genome")
            noop_cls = getattr(core_mod, "NoOpGenomeProvider", None)
            gp_fallback: GenomeProvider = (
                cast(GenomeProvider, noop_cls()) if callable(noop_cls) else NoOpGenomeProvider()
            )
            adapters["genome_provider"] = gp_fallback
        except Exception:
            adapters["genome_provider"] = NoOpGenomeProvider()

    return adapters


__all__ = [
    "MarketDataGatewayAdapter",
    "AnomalyDetectorAdapter",
    "RegimeClassifierAdapter",
    "AdaptationServiceAdapter",
    "ConfigurationProviderAdapter",
    "GenomeProviderAdapterFactory",
    "compose_validation_adapters",
]
