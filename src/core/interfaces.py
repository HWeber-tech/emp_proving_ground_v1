"""
EMP Core Interfaces v1.1

Defines the interfaces and contracts that enable communication between
layers in the EMP system. These interfaces ensure proper separation
of concerns and enable modular development.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketData:
    """Standardized market data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    source: str
    latency_ms: float


@dataclass
class SensorySignal:
    """Sensory layer output signal."""
    timestamp: datetime
    signal_type: str
    value: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class AnalysisResult:
    """Thinking layer analysis result."""
    timestamp: datetime
    analysis_type: str
    result: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class TradingDecision:
    """Trading layer decision."""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: float
    price: float
    confidence: float
    metadata: Dict[str, Any]


class SensoryOrgan(ABC):
    """Interface for sensory organs."""
    
    @abstractmethod
    def perceive(self, data: MarketData) -> SensorySignal:
        """Process raw data into sensory signals."""
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate the sensory organ."""
        pass


class ThinkingPattern(ABC):
    """Interface for thinking patterns."""
    
    @abstractmethod
    def analyze(self, signals: List[SensorySignal]) -> AnalysisResult:
        """Analyze sensory signals and produce insights."""
        pass
    
    @abstractmethod
    def learn(self, feedback: Dict[str, Any]) -> bool:
        """Learn from feedback and improve analysis."""
        pass


class TradingStrategy(ABC):
    """Interface for trading strategies."""
    
    @abstractmethod
    def decide(self, analysis: AnalysisResult) -> TradingDecision:
        """Make trading decisions based on analysis."""
        pass
    
    @abstractmethod
    def execute(self, decision: TradingDecision) -> bool:
        """Execute trading decisions."""
        pass


class EvolutionEngine(ABC):
    """Interface for evolution engines."""
    
    @abstractmethod
    def evolve(self, population: List[Any], fitness_scores: List[float]) -> List[Any]:
        """Evolve population based on fitness scores."""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, individual: Any) -> float:
        """Evaluate fitness of an individual."""
        pass


class GovernanceInterface(ABC):
    """Interface for governance layer."""
    
    @abstractmethod
    def approve_strategy(self, strategy: TradingStrategy) -> bool:
        """Approve a strategy for deployment."""
        pass
    
    @abstractmethod
    def log_decision(self, decision: TradingDecision) -> bool:
        """Log a trading decision for audit."""
        pass 