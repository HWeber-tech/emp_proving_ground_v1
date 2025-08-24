"""
EMP Strategy Registry v1.1 - SQLite Implementation

Persistent strategy registry using SQLite for champion genome storage.
Implements GOV-02 ticket requirements for database-backed strategy management.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Persistent strategy registry using SQLite database."""

    def __init__(self, db_path: str = "governance.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize SQLite database with required schema."""
        try:
            cursor = self.conn.cursor()

            # Create strategies table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    genome_id TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    dna TEXT NOT NULL,
                    fitness_report TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'evolved',
                    strategy_name TEXT,
                    generation INTEGER,
                    fitness_score REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    total_return REAL,
                    volatility REAL
                )
            """
            )

            # Create indexes for performance
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_genome_id ON strategies(genome_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status ON strategies(status)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fitness_score ON strategies(fitness_score DESC)
            """
            )

            self.conn.commit()
            logger.info(f"Strategy Registry initialized with database: {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def register_champion(self, genome: Any, fitness_report: dict[str, Any]) -> bool:
        """
        Register a champion genome with its fitness report.

        Args:
            genome: The evolved genome object
            fitness_report: Dictionary containing fitness metrics

        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()

            # Serialize genome DNA (decision tree) to JSON
            dna_json = json.dumps(
                genome.decision_tree if hasattr(genome, "decision_tree") else str(genome)
            )

            # Serialize fitness report to JSON
            report_json = json.dumps(fitness_report)

            # Extract key metrics for quick querying
            strategy_name = getattr(
                genome, "name", f"genome_{genome.id if hasattr(genome, 'id') else 'unknown'}"
            )
            generation = getattr(genome, "generation", 0)
            fitness_score = fitness_report.get("fitness_score", 0.0)
            max_drawdown = fitness_report.get("max_drawdown", 0.0)
            sharpe_ratio = fitness_report.get("sharpe_ratio", 0.0)
            total_return = fitness_report.get("total_return", 0.0)
            volatility = fitness_report.get("volatility", 0.0)

            cursor.execute(
                """
                INSERT OR REPLACE INTO strategies 
                (genome_id, created_at, dna, fitness_report, status, strategy_name, 
                 generation, fitness_score, max_drawdown, sharpe_ratio, total_return, volatility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    genome.id if hasattr(genome, "id") else str(genome),
                    datetime.now(),
                    dna_json,
                    report_json,
                    "evolved",
                    strategy_name,
                    generation,
                    fitness_score,
                    max_drawdown,
                    sharpe_ratio,
                    total_return,
                    volatility,
                ),
            )

            self.conn.commit()
            logger.info(
                f"Registered champion genome: {genome.id if hasattr(genome, 'id') else str(genome)}"
            )
            return True

        except Exception as e:
            logger.error(f"Error registering champion: {e}")
            self.conn.rollback()
            return False

    def get_strategy(self, strategy_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve a strategy by its ID.

        Args:
            strategy_id: The genome_id of the strategy

        Returns:
            Dictionary with strategy data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM strategies WHERE genome_id = ?", (strategy_id,))
            row = cursor.fetchone()

            if row:
                return {
                    "id": row["id"],
                    "genome_id": row["genome_id"],
                    "created_at": row["created_at"],
                    "dna": cast(dict[str, Any], json.loads(row["dna"])),
                    "fitness_report": cast(dict[str, Any], json.loads(row["fitness_report"])),
                    "status": row["status"],
                    "strategy_name": row["strategy_name"],
                    "generation": row["generation"],
                    "fitness_score": row["fitness_score"],
                    "max_drawdown": row["max_drawdown"],
                    "sharpe_ratio": row["sharpe_ratio"],
                    "total_return": row["total_return"],
                    "volatility": row["volatility"],
                }
            return None

        except Exception as e:
            logger.error(f"Error retrieving strategy: {e}")
            return None

    def update_strategy_status(self, strategy_id: str, new_status: str) -> bool:
        """
        Update the status of a strategy.

        Args:
            strategy_id: The genome_id of the strategy
            new_status: New status value

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE strategies SET status = ? WHERE genome_id = ?", (new_status, strategy_id)
            )

            if cursor.rowcount > 0:
                self.conn.commit()
                logger.info(f"Updated strategy {strategy_id} status to {new_status}")
                return True
            else:
                logger.warning(f"Strategy {strategy_id} not found")
                return False

        except Exception as e:
            logger.error(f"Error updating strategy status: {e}")
            self.conn.rollback()
            return False

    def get_champion_strategies(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get the top performing strategies.

        Args:
            limit: Maximum number of strategies to return

        Returns:
            List of strategy dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM strategies 
                WHERE status IN ('evolved', 'approved', 'active')
                ORDER BY fitness_score DESC
                LIMIT ?
            """,
                (limit,),
            )

            strategies = []
            for row in cursor.fetchall():
                strategies.append(
                    {
                        "id": row["id"],
                        "genome_id": row["genome_id"],
                        "created_at": row["created_at"],
                        "dna": cast(dict[str, Any], json.loads(row["dna"])),
                        "fitness_report": cast(dict[str, Any], json.loads(row["fitness_report"])),
                        "status": row["status"],
                        "strategy_name": row["strategy_name"],
                        "generation": row["generation"],
                        "fitness_score": row["fitness_score"],
                        "max_drawdown": row["max_drawdown"],
                        "sharpe_ratio": row["sharpe_ratio"],
                        "total_return": row["total_return"],
                        "volatility": row["volatility"],
                    }
                )

            return strategies

        except Exception as e:
            logger.error(f"Error retrieving champion strategies: {e}")
            return []

    def get_strategies_by_status(self, status: str) -> list[dict[str, Any]]:
        """
        Get all strategies with a specific status.

        Args:
            status: Status value to filter by

        Returns:
            List of strategy dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM strategies WHERE status = ?", (status,))

            strategies = []
            for row in cursor.fetchall():
                strategies.append(
                    {
                        "id": row["id"],
                        "genome_id": row["genome_id"],
                        "created_at": row["created_at"],
                        "dna": cast(dict[str, Any], json.loads(row["dna"])),
                        "fitness_report": cast(dict[str, Any], json.loads(row["fitness_report"])),
                        "status": row["status"],
                        "strategy_name": row["strategy_name"],
                        "generation": row["generation"],
                        "fitness_score": row["fitness_score"],
                    }
                )

            return strategies

        except Exception as e:
            logger.error(f"Error retrieving strategies by status: {e}")
            return []

    def get_registry_summary(self) -> dict[str, Any]:
        """
        Get summary statistics of the registry.

        Returns:
            Dictionary with registry statistics
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_strategies,
                    COUNT(CASE WHEN status = 'evolved' THEN 1 END) as evolved_count,
                    COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_count,
                    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count,
                    AVG(fitness_score) as avg_fitness_score,
                    MAX(fitness_score) as max_fitness_score,
                    MIN(fitness_score) as min_fitness_score
                FROM strategies
            """
            )

            row = cursor.fetchone()

            return {
                "total_strategies": row["total_strategies"],
                "evolved_count": row["evolved_count"],
                "approved_count": row["approved_count"],
                "active_count": row["active_count"],
                "avg_fitness_score": row["avg_fitness_score"] or 0.0,
                "max_fitness_score": row["max_fitness_score"] or 0.0,
                "min_fitness_score": row["min_fitness_score"] or 0.0,
                "database_path": str(self.db_path),
            }

        except Exception as e:
            logger.error(f"Error getting registry summary: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Strategy Registry database connection closed")

    def __del__(self) -> None:
        """Cleanup database connection on destruction."""
        self.close()
