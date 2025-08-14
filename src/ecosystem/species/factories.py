#!/usr/bin/env python3
"""
Specialist Genome Factories
===========================

Creates genomes with species-specific biases for different predator types.
Each factory generates DecisionGenome objects optimized for specific market niches.
"""

import random
from typing import Dict, Tuple

from src.core.interfaces import DecisionGenome, ISpecialistGenomeFactory


class StalkerFactory(ISpecialistGenomeFactory):
    """Factory for creating Stalker genomes - long-term trend followers."""
    
    def create_genome(self) -> DecisionGenome:
        """Create a genome biased for long-term trend following."""
        genome = DecisionGenome(
            species_type="stalker",
            parameters={
                'sma_fast': random.randint(50, 200),
                'sma_slow': random.randint(100, 400),
                'ema_period': random.randint(20, 100),
                'adx_period': random.randint(14, 28),
                'adx_threshold': random.uniform(20, 30),
                'rsi_period': random.randint(14, 21),
                'rsi_oversold': random.uniform(25, 35),
                'rsi_overbought': random.uniform(65, 75),
                'min_trend_duration': random.randint(5, 15),
                'confirmation_bars': random.randint(3, 8),
                'max_volatility': random.uniform(0.02, 0.05),
                'volatility_period': random.randint(20, 50)
            },
            indicators=['SMA', 'EMA', 'ADX', 'RSI', 'ATR'],
            rules={
                'entry': [
                    'sma_crossover_long',
                    'adx_confirmation',
                    'volume_confirmation',
                    'trend_strength_filter'
                ],
                'exit': [
                    'trend_reversal',
                    'volatility_spike',
                    'time_stop',
                    'profit_target'
                ]
            },
            risk_profile={
                'max_position_size': random.uniform(0.05, 0.15),
                'stop_loss_pct': random.uniform(0.02, 0.05),
                'take_profit_pct': random.uniform(0.05, 0.15),
                'max_drawdown': random.uniform(0.05, 0.10),
                'risk_reward_ratio': random.uniform(2.0, 4.0)
            }
        )
        return genome
    
    def get_species_name(self) -> str:
        return "stalker"
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'sma_fast': (50, 200),
            'sma_slow': (100, 400),
            'ema_period': (20, 100),
            'adx_period': (14, 28),
            'rsi_period': (14, 21),
            'min_trend_duration': (5, 15),
            'max_position_size': (0.05, 0.15),
            'stop_loss_pct': (0.02, 0.05),
            'take_profit_pct': (0.05, 0.15)
        }


class AmbusherFactory(ISpecialistGenomeFactory):
    """Factory for creating Ambusher genomes - high-frequency scalpers."""
    
    def create_genome(self) -> DecisionGenome:
        """Create a genome biased for high-frequency scalping."""
        genome = DecisionGenome(
            species_type="ambusher",
            parameters={
                'sma_fast': random.randint(3, 10),
                'sma_slow': random.randint(5, 20),
                'ema_period': random.randint(3, 12),
                'rsi_period': random.randint(5, 10),
                'rsi_oversold': random.uniform(15, 25),
                'rsi_overbought': random.uniform(75, 85),
                'bb_period': random.randint(10, 20),
                'bb_std': random.uniform(1.5, 2.5),
                'stoch_k': random.randint(5, 14),
                'stoch_d': random.randint(3, 5),
                'target_pips': random.uniform(5, 20),
                'max_hold_time': random.randint(1, 5),
                'volatility_threshold': random.uniform(0.001, 0.005)
            },
            indicators=['SMA', 'EMA', 'RSI', 'BOLLINGER', 'STOCHASTIC'],
            rules={
                'entry': [
                    'mean_reversion',
                    'momentum_burst',
                    'volume_spike',
                    'oversold_bounce'
                ],
                'exit': [
                    'quick_profit',
                    'time_stop',
                    'reversal_signal',
                    'volatility_exit'
                ]
            },
            risk_profile={
                'max_position_size': random.uniform(0.01, 0.05),
                'stop_loss_pct': random.uniform(0.005, 0.015),
                'take_profit_pct': random.uniform(0.01, 0.03),
                'max_drawdown': random.uniform(0.02, 0.05),
                'risk_reward_ratio': random.uniform(1.0, 2.0)
            }
        )
        return genome
    
    def get_species_name(self) -> str:
        return "ambusher"
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'sma_fast': (3, 10),
            'sma_slow': (5, 20),
            'rsi_period': (5, 10),
            'bb_period': (10, 20),
            'target_pips': (5, 20),
            'max_hold_time': (1, 5),
            'max_position_size': (0.01, 0.05),
            'stop_loss_pct': (0.005, 0.015)
        }


class PackHunterFactory(ISpecialistGenomeFactory):
    """Factory for creating Pack Hunter genomes - multi-timeframe coordination."""
    
    def create_genome(self) -> DecisionGenome:
        """Create a genome for multi-timeframe strategy coordination."""
        genome = DecisionGenome(
            species_type="pack_hunter",
            parameters={
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
            },
            indicators=['SMA', 'RSI', 'MACD', 'CORRELATION', 'VOLUME_PROFILE'],
            rules={
                'entry': [
                    'multi_tf_alignment',
                    'correlation_confirmation',
                    'volume_consensus',
                    'momentum_agreement'
                ],
                'exit': [
                    'tf_divergence',
                    'correlation_break',
                    'volume_dry_up',
                    'momentum_loss'
                ]
            },
            risk_profile={
                'max_position_size': random.uniform(0.03, 0.08),
                'stop_loss_pct': random.uniform(0.01, 0.03),
                'take_profit_pct': random.uniform(0.02, 0.06),
                'max_drawdown': random.uniform(0.03, 0.08),
                'risk_reward_ratio': random.uniform(1.5, 3.0)
            }
        )
        return genome
    
    def get_species_name(self) -> str:
        return "pack_hunter"
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'tf1_sma': (5, 15),
            'tf2_sma': (15, 30),
            'tf3_sma': (30, 60),
            'alignment_threshold': (0.7, 0.9),
            'confirmation_count': (2, 4),
            'max_position_size': (0.03, 0.08),
            'stop_loss_pct': (0.01, 0.03)
        }


class ScavengerFactory(ISpecialistGenomeFactory):
    """Factory for creating Scavenger genomes - market inefficiency exploiters."""
    
    def create_genome(self) -> DecisionGenome:
        """Create a genome for exploiting market inefficiencies."""
        genome = DecisionGenome(
            species_type="scavenger",
            parameters={
                'gap_threshold': random.uniform(0.001, 0.005),
                'gap_fill_target': random.uniform(0.5, 0.8),
                'volume_spike_threshold': random.uniform(2.0, 5.0),
                'price_spike_threshold': random.uniform(0.005, 0.02),
                'correlation_deviation': random.uniform(0.01, 0.03),
                'mean_reversion_speed': random.uniform(0.1, 0.3),
                'liquidity_threshold': random.uniform(1000, 5000),
                'spread_threshold': random.uniform(0.0001, 0.0005),
                'reaction_time': random.uniform(0.1, 0.5),
                'opportunity_window': random.uniform(1, 10)
            },
            indicators=['VOLUME', 'SPREAD', 'CORRELATION', 'GAP', 'LIQUIDITY'],
            rules={
                'entry': [
                    'gap_detection',
                    'volume_anomaly',
                    'correlation_break',
                    'liquidity_imbalance'
                ],
                'exit': [
                    'gap_fill',
                    'volume_normalization',
                    'correlation_restore',
                    'time_decay'
                ]
            },
            risk_profile={
                'max_position_size': random.uniform(0.02, 0.06),
                'stop_loss_pct': random.uniform(0.008, 0.02),
                'take_profit_pct': random.uniform(0.01, 0.04),
                'max_drawdown': random.uniform(0.02, 0.06),
                'risk_reward_ratio': random.uniform(1.2, 2.5)
            }
        )
        return genome
    
    def get_species_name(self) -> str:
        return "scavenger"
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'gap_threshold': (0.001, 0.005),
            'volume_spike_threshold': (2.0, 5.0),
            'correlation_deviation': (0.01, 0.03),
            'liquidity_threshold': (1000, 5000),
            'max_position_size': (0.02, 0.06),
            'stop_loss_pct': (0.008, 0.02)
        }


class AlphaFactory(ISpecialistGenomeFactory):
    """Factory for creating Alpha genomes - dominant market leaders."""
    
    def create_genome(self) -> DecisionGenome:
        """Create a genome for dominant market leadership strategies."""
        genome = DecisionGenome(
            species_type="alpha",
            parameters={
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
            },
            indicators=['VOLUME', 'PRICE_ACTION', 'MOMENTUM', 'SENTIMENT', 'ORDER_FLOW'],
            rules={
                'entry': [
                    'market_leadership',
                    'volume_dominance',
                    'sentiment_alignment',
                    'follower_confirmation'
                ],
                'exit': [
                    'leadership_loss',
                    'volume_decline',
                    'sentiment_shift',
                    'follower_exodus'
                ]
            },
            risk_profile={
                'max_position_size': random.uniform(0.08, 0.20),
                'stop_loss_pct': random.uniform(0.015, 0.04),
                'take_profit_pct': random.uniform(0.04, 0.12),
                'max_drawdown': random.uniform(0.04, 0.10),
                'risk_reward_ratio': random.uniform(2.5, 5.0)
            }
        )
        return genome
    
    def get_species_name(self) -> str:
        return "alpha"
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'market_lead_period': (10, 30),
            'leadership_threshold': (0.6, 0.9),
            'volume_lead_ratio': (1.5, 3.0),
            'price_lead_distance': (0.002, 0.01),
            'max_position_size': (0.08, 0.20),
            'stop_loss_pct': (0.015, 0.04),
            'take_profit_pct': (0.04, 0.12)
        }


# Factory registry
SPECIES_FACTORIES = {
    'stalker': StalkerFactory(),
    'ambusher': AmbusherFactory(),
    'pack_hunter': PackHunterFactory(),
    'scavenger': ScavengerFactory(),
    'alpha': AlphaFactory()
}


def get_factory(species_type: str) -> ISpecialistGenomeFactory:
    """Get factory for specific species type."""
    return SPECIES_FACTORIES.get(species_type)


def get_all_factories() -> Dict[str, ISpecialistGenomeFactory]:
    """Get all available species factories."""
    return SPECIES_FACTORIES.copy()
