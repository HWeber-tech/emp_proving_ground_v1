"""
Governance Models for EMP System

SQLAlchemy models for strategy registry and governance data.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class StrategyModel(Base):
    """
    SQLAlchemy model for evolved strategies.
    
    Replaces the file-based SQLite database with PostgreSQL for better
    scalability and reliability.
    """
    
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    genome_id = Column(String(64), unique=True, nullable=False, index=True)
    dna = Column(Text, nullable=False)
    fitness_score = Column(Float, default=0.0)
    generation = Column(Integer, default=0)
    parent_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_champion = Column(Boolean, default=False)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'genome_id': self.genome_id,
            'dna': self.dna,
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_champion': self.is_champion,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyModel':
        """Create model from dictionary."""
        return cls(
            genome_id=data['genome_id'],
            dna=data['dna'],
            fitness_score=data.get('fitness_score', 0.0),
            generation=data.get('generation', 0),
            parent_id=data.get('parent_id'),
            is_champion=data.get('is_champion', False),
            metadata=data.get('metadata', {})
        )


class PerformanceModel(Base):
    """
    Model for storing strategy performance metrics.
    """
    
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    trades_count = Column(Integer, default=0)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'strategy_id': self.strategy_id,
            'timestamp': self.timestamp.isoformat(),
            'pnl': self.pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'trades_count': self.trades_count,
            'metadata': self.metadata
        }


class MarketRegimeModel(Base):
    """
    Model for storing market regime classifications.
    """
    
    __tablename__ = 'market_regimes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False, index=True)
    regime = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.0)
    features = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'regime': self.regime,
            'confidence': self.confidence,
            'features': self.features
        }


class DatabaseManager:
    """
    Manages PostgreSQL database connections and operations.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        logger = logging.getLogger(__name__)
        logger.info("DatabaseManager initialized with PostgreSQL")
    
    def get_session(self):
        """Get a new database session."""
        return self.Session()
    
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(self.engine)
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return {
                'healthy': True,
                'message': 'Database connection OK'
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': str(e)
            }


# Example usage
if __name__ == "__main__":
    import os
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Example database URL
    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://emp:emp123@localhost:5432/emp_strategies'
    )
    
    # Create database manager
    db_manager = DatabaseManager(database_url)
    
    # Test connection
    health = db_manager.health_check()
    print("Database health:", health)
    
    # Create sample data
    with db_manager.get_session() as session:
        # Create a sample strategy
        strategy = StrategyModel(
            genome_id="test_genome_001",
            dna="sample_dna_data",
            fitness_score=0.85,
            generation=1,
            is_champion=True
        )
        
        session.add(strategy)
        session.commit()
        
        print("Sample strategy created:", strategy.to_dict())
