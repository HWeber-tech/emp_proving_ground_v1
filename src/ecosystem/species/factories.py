#!/usr/bin/env python3
"""
Specialist Genome Factories
===========================

Creates genomes with species-specific biases for different predator types.
Each factory generates DecisionGenome objects optimized for specific market niches.
"""

import random
from collections.abc import Mapping
from typing import cast

from src.core.interfaces import DecisionGenome as GenomeLike
from src.core.interfaces import ISpecialistGenomeFactory
from src.genome.models.genome import DecisionGenome as CanonDecisionGenome


def _coerce_params(params: Mapping[str, float | int | str]) -> dict[str, float]:
    """Coerce parameter mapping to dict[str, float], dropping non-coercibles."""
    out: dict[str, float] = {}
    for k, v in params.items():
        try:
            out[str(k)] = float(v)  # explicit float coercion
        except Exception:
            continue
    return out


class StalkerFactory(ISpecialistGenomeFactory):
    """Factory for creating Stalker genomes - long-term trend followers."""
    
    def create_genome(self) -> GenomeLike:
        """Create a genome biased for long-term trend following."""
        params = _coerce_params({
            'sma_fast': random.randint(50, 200),
            'sma_slow': random.randint(100, 400),
            'ema_period': random.randint(20, 100),
            'adx_period': random.randint(14, 28),
            'adx_threshold': random.uniform(20.0, 30.0),
            'rsi_period': random.randint(14, 21),
            'rsi_oversold': random.uniform(25.0, 35.0),
            'rsi_overbought': random.uniform(65.0, 75.0),
            'min_trend_duration': random.randint(5, 15),
            'confirmation_bars': random.randint(3, 8),
            'max_volatility': random.uniform(0.02, 0.05),
            'volatility_period': random.randint(20, 50)
        })
        return cast(GenomeLike, CanonDecisionGenome.from_dict({
            "species_type": "stalker",
            "parameters": params,
        }))
    
    def get_species_name(self) -> str:
        return "stalker"
    
    def get_parameter_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            'sma_fast': (50.0, 200.0),
            'sma_slow': (100.0, 400.0),
            'ema_period': (20.0, 100.0),
            'adx_period': (14.0, 28.0),
            'rsi_period': (14.0, 21.0),
            'min_trend_duration': (5.0, 15.0),
            'max_position_size': (0.05, 0.15),
            'stop_loss_pct': (0.02, 0.05),
            'take_profit_pct': (0.05, 0.15),
        }


class AmbusherFactory(ISpecialistGenomeFactory):
    """Factory for creating Ambusher genomes - high-frequency scalpers."""
    
    def create_genome(self) -> GenomeLike:
        """Create a genome biased for high-frequency scalping."""
        params = _coerce_params({
            'sma_fast': random.randint(3, 10),
            'sma_slow': random.randint(5, 20),
            'ema_period': random.randint(3, 12),
            'rsi_period': random.randint(5, 10),
            'rsi_oversold': random.uniform(15.0, 25.0),
            'rsi_overbought': random.uniform(75.0, 85.0),
            'bb_period': random.randint(10, 20),
            'bb_std': random.uniform(1.5, 2.5),
            'stoch_k': random.randint(5, 14),
            'stoch_d': random.randint(3, 5),
            'target_pips': random.uniform(5.0, 20.0),
            'max_hold_time': random.randint(1, 5),
            'volatility_threshold': random.uniform(0.001, 0.005)
        })
        return cast(GenomeLike, CanonDecisionGenome.from_dict({
            "species_type": "ambusher",
            "parameters": params,
        }))
    
    def get_species_name(self) -> str:
        return "ambusher"
    
    def get_parameter_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            'sma_fast': (3.0, 10.0),
            'sma_slow': (5.0, 20.0),
            'rsi_period': (5.0, 10.0),
            'bb_period': (10.0, 20.0),
            'target_pips': (5.0, 20.0),
            'max_hold_time': (1.0, 5.0),
            'max_position_size': (0.01, 0.05),
            'stop_loss_pct': (0.005, 0.015)
        }


class PackHunterFactory(ISpecialistGenomeFactory):
    """Factory for creating Pack Hunter genomes - multi-timeframe coordination."""
    
    def create_genome(self) -> GenomeLike:
        """Create a genome for multi-timeframe strategy coordination."""
        params = _coerce_params({
            'tf1_sma': random.randint(5, 15),
            'tf2_sma': random.randint(15, 30),
            'tf3_sma': random.randint(30, 60),
            'alignment_threshold': random.uniform(0.7, 0.9),
            'confirmation_count': random.randint(2, 4),
            'rsi_tf1': random.randint(7, 14),
            'rsi_tf2': random.randint(14, 21),
            'rsi_tf3': random.randint(21, 28),
            'correlation_period': random.randint(20, 50),
            'min_correlation': random.uniform(0.6, 0.8),
            'entry_delay': random.uniform(0.1, 0.5),
            'exit_delay': random.uniform(0.1, 0.3)
        })
        return cast(GenomeLike, CanonDecisionGenome.from_dict({
            "species_type": "pack_hunter",
            "parameters": params,
        }))
    
    def get_species_name(self) -> str:
        return "pack_hunter"
    
    def get_parameter_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            'tf1_sma': (5.0, 15.0),
            'tf2_sma': (15.0, 30.0),
            'tf3_sma': (30.0, 60.0),
            'alignment_threshold': (0.7, 0.9),
            'confirmation_count': (2.0, 4.0),
            'max_position_size': (0.03, 0.08),
            'stop_loss_pct': (0.01, 0.03)
        }


class ScavengerFactory(ISpecialistGenomeFactory):
    """Factory for creating Scavenger genomes - market inefficiency exploiters."""
    
    def create_genome(self) -> GenomeLike:
        """Create a genome for exploiting market inefficiencies."""
        params = _coerce_params({
            'gap_threshold': random.uniform(0.001, 0.005),
            'gap_fill_target': random.uniform(0.5, 0.8),
            'volume_spike_threshold': random.uniform(2.0, 5.0),
            'price_spike_threshold': random.uniform(0.005, 0.02),
            'correlation_deviation': random.uniform(0.01, 0.03),
            'mean_reversion_speed': random.uniform(0.1, 0.3),
            'liquidity_threshold': random.uniform(1000.0, 5000.0),
            'spread_threshold': random.uniform(0.0001, 0.0005),
            'reaction_time': random.uniform(0.1, 0.5),
            'opportunity_window': random.uniform(1.0, 10.0)
        })
        return cast(GenomeLike, CanonDecisionGenome.from_dict({
            "species_type": "scavenger",
            "parameters": params,
        }))
    
    def get_species_name(self) -> str:
        return "scavenger"
    
    def get_parameter_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            'gap_threshold': (0.001, 0.005),
            'volume_spike_threshold': (2.0, 5.0),
            'correlation_deviation': (0.01, 0.03),
            'liquidity_threshold': (1000.0, 5000.0),
            'max_position_size': (0.02, 0.06),
            'stop_loss_pct': (0.008, 0.02)
        }


class AlphaFactory(ISpecialistGenomeFactory):
    """Factory for creating Alpha genomes - dominant market leaders."""
    
    def create_genome(self) -> GenomeLike:
        """Create a genome for dominant market leadership strategies."""
        params = _coerce_params({
            'market_lead_period': random.randint(10, 30),
            'leadership_threshold': random.uniform(0.6, 0.9),
            'volume_lead_ratio': random.uniform(1.5, 3.0),
            'volume_confirmation': random.uniform(0.8, 1.0),
            'price_lead_distance': random.uniform(0.002, 0.01),
            'momentum_lead': random.uniform(0.01, 0.05),
            'sentiment_threshold': random.uniform(0.7, 0.95),
            'sentiment_period': random.randint(5, 15),
            'leader_stop_distance': random.uniform(0.01, 0.03),
            'follower_confirmation': random.uniform(0.5, 0.8)
        })
        return cast(GenomeLike, CanonDecisionGenome.from_dict({
            "species_type": "alpha",
            "parameters": params,
        }))
    
    def get_species_name(self) -> str:
        return "alpha"
    
    def get_parameter_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            'market_lead_period': (10.0, 30.0),
            'leadership_threshold': (0.6, 0.9),
            'volume_lead_ratio': (1.5, 3.0),
            'price_lead_distance': (0.002, 0.01),
            'max_position_size': (0.08, 0.20),
            'stop_loss_pct': (0.015, 0.04),
            'take_profit_pct': (0.04, 0.12)
        }


# Factory registry
SPECIES_FACTORIES: dict[str, ISpecialistGenomeFactory] = {
    'stalker': StalkerFactory(),
    'ambusher': AmbusherFactory(),
    'pack_hunter': PackHunterFactory(),
    'scavenger': ScavengerFactory(),
    'alpha': AlphaFactory(),
}


def get_factory(species_type: str) -> ISpecialistGenomeFactory | None:
    """Get factory for specific species type."""
    return SPECIES_FACTORIES.get(species_type)


def get_all_factories() -> dict[str, ISpecialistGenomeFactory]:
    """Get all available species factories."""
    return SPECIES_FACTORIES.copy()


__all__ = [
    "ISpecialistGenomeFactory",
    "get_factory",
    "get_all_factories",
    "StalkerFactory",
    "AmbusherFactory",
    "PackHunterFactory",
    "ScavengerFactory",
    "AlphaFactory",
]
