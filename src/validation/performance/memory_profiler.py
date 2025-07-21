#!/usr/bin/env python3
"""
Memory Profiling Framework
Memory usage analysis and leak detection for Phase 2 validation
"""

import logging
import psutil
import gc
import tracemalloc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """Memory usage profile"""
    timestamp: datetime
    memory_usage_mb: float
    peak_memory_mb: float
    memory_leaks: List[str]
    allocation_stats: Dict[str, Any]
    garbage_collection_stats: Dict[str, Any]


@dataclass
class MemoryTestResult:
    """Result of memory testing"""
    test_name: str
    baseline_memory_mb: float
    peak_memory_mb: float
    memory_increase_mb: float
    memory_leaks_detected: bool
    leak_details: List[str]
    gc_stats: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryProfiler:
    """Advanced memory profiling and leak detection system"""
    
    def __init__(self):
        self.snapshots: List[MemoryProfile] = []
        self.baseline_memory = None
        
    def start_profiling(self) -> None:
        """Start memory profiling"""
        tracemalloc.start()
        gc.collect()
        self.baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
    def stop_profiling(self) -> None:
        """Stop memory profiling"""
        tracemalloc.stop()
        
    def take_snapshot(self, description: str = "") -> MemoryProfile:
        """Take a memory snapshot"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        
        # Get allocation statistics
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Detect potential memory leaks
        leaks = self._detect_memory_leaks(snapshot)
        
        # Get garbage collection statistics
        gc_stats = {
            'collections': gc.get_count(),
            'thresholds': gc.get_threshold(),
            'garbage': len(gc.garbage)
        }
        
        profile = MemoryProfile(
            timestamp=datetime.now(),
            memory_usage_mb=current_memory,
            peak_memory_mb=peak_memory,
            memory_leaks=leaks,
            allocation_stats={
                'total_allocations': len(top_stats),
                'top_allocations': [str(stat) for stat in top_stats[:10]]
            },
            garbage_collection_stats=gc_stats
        )
        
        self.snapshots.append(profile)
        return profile
    
    def _detect_memory_leaks(self, snapshot) -> List[str]:
        """Detect potential memory leaks from snapshot"""
        leaks = []
        
        # Get top memory consumers
        top_stats = snapshot.statistics('lineno')
        
        # Look for suspicious patterns
        for stat in top_stats[:5]:
            if stat.size_diff > 1024 * 1024:  # 1MB increase
                leaks.append(f"Potential leak: {stat}")
        
        return leaks
    
    async def profile_operation(
        self,
        operation: Callable,
        test_name: str,
        iterations: int = 1000,
        warmup_iterations: int = 100
    ) -> MemoryTestResult:
        """Profile memory usage of a specific operation"""
        
        logger.info(f"Profiling memory for: {test_name}")
        
        # Start profiling
        self.start_profiling()
        
        # Warmup
        for _ in range(warmup_iterations):
            await operation()
        
        # Take baseline snapshot
        baseline_snapshot = self.take_snapshot("baseline")
        
        # Run operation multiple times
        for _ in range(iterations):
            await operation()
        
        # Take final snapshot
        final_snapshot = self.take_snapshot("final")
        
        # Stop profiling
        self.stop_profiling()
        
        # Analyze results
        memory_increase = final_snapshot.memory_usage_mb - baseline_snapshot.memory_usage_mb
        memory_leaks = len(final_snapshot.memory_leaks) > 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            memory_increase, 
            memory_leaks, 
            final_snapshot
        )
        
        return MemoryTestResult(
            test_name=test_name,
            baseline_memory_mb=baseline_snapshot.memory_usage_mb,
            peak_memory_mb=final_snapshot.peak_memory_mb,
            memory_increase_mb=memory_increase,
            memory_leaks_detected=memory_leaks,
            leak_details=final_snapshot.memory_leaks,
            gc_stats=final_snapshot.garbage_collection_stats,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self, 
        memory_increase: float, 
        leaks_detected: bool, 
        snapshot: MemoryProfile
    ) -> List[str]:
        """Generate memory optimization recommendations"""
        
        recommendations = []
        
        if memory_increase > 10:  # 10MB increase
            recommendations.append("Consider memory optimization - significant increase detected")
        
        if leaks_detected:
            recommendations.append("Memory leaks detected - investigate allocation patterns")
        
        if snapshot.garbage_collection_stats['garbage'] > 0:
            recommendations.append("Circular references detected - review object lifecycle")
        
        if snapshot.peak_memory_mb > 100:  # 100MB peak
            recommendations.append("High peak memory usage - consider object pooling")
        
        return recommendations
    
    def monitor_memory_usage(
        self,
        duration: int = 60,
        interval: int = 1
    ) -> List[MemoryProfile]:
        """Monitor memory usage over time"""
        
        logger.info(f"Monitoring memory usage for {duration} seconds")
        
        self.start_profiling()
        snapshots = []
        
        start_time = time.time()
        while time.time() - start_time < duration:
            snapshot = self.take_snapshot("monitoring")
            snapshots.append(snapshot)
            time.sleep(interval)
        
        self.stop_profiling()
        return snapshots
    
    def detect_memory_leaks_long_running(
        self,
        operation: Callable,
        test_name: str,
        duration: int = 300,
        check_interval: int = 30
    ) -> MemoryTestResult:
        """Detect memory leaks in long-running operations"""
        
        logger.info(f"Detecting memory leaks for: {test_name}")
        
        self.start_profiling()
        
        # Take initial snapshot
        initial_snapshot = self.take_snapshot("initial")
        
        # Run operation in background
        async def background_operation():
            while True:
                await operation()
                await asyncio.sleep(0.1)
        
        # Monitor for duration
        start_time = time.time()
        memory_trend = []
        
        while time.time() - start_time < duration:
            snapshot = self.take_snapshot("monitoring")
            memory_trend.append(snapshot.memory_usage_mb)
            time.sleep(check_interval)
        
        # Take final snapshot
        final_snapshot = self.take_snapshot("final")
        
        self.stop_profiling()
        
        # Analyze trend
        memory_increase = final_snapshot.memory_usage_mb - initial_snapshot.memory_usage_mb
        leak_detected = memory_increase > 5  # 5MB threshold
        
        # Check for consistent increase
        if len(memory_trend) > 2:
            trend_slope = (memory_trend[-1] - memory_trend[0]) / len(memory_trend)
            leak_detected = leak_detected or trend_slope > 0.1
        
        return MemoryTestResult(
            test_name=f"{test_name}_long_running",
            baseline_memory_mb=initial_snapshot.memory_usage_mb,
            peak_memory_mb=max(memory_trend) if memory_trend else initial_snapshot.memory_usage_mb,
            memory_increase_mb=memory_increase,
            memory_leaks_detected=leak_detected,
            leak_details=final_snapshot.memory_leaks,
            gc_stats=final_snapshot.garbage_collection_stats,
            recommendations=self._generate_recommendations(
                memory_increase, 
                leak_detected, 
                final_snapshot
            )
        )
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary"""
        
        if not self.snapshots:
            return {"summary": "No memory profiling data available"}
        
        # Calculate statistics
        memory_values = [s.memory_usage_mb for s in self.snapshots]
        peak_memory = max(s.peak_memory_mb for s in self.snapshots)
        
        return {
            "summary": {
                "total_snapshots": len(self.snapshots),
                "min_memory_mb": min(memory_values),
                "max_memory_mb": max(memory_values),
                "avg_memory_mb": sum(memory_values) / len(memory_values),
                "peak_memory_mb": peak_memory
            },
            "snapshots": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "memory_usage_mb": s.memory_usage_mb,
                    "peak_memory_mb": s.peak_memory_mb,
                    "leaks_detected": len(s.memory_leaks) > 0
                }
                for s in self.snapshots
            ]
        }
    
    def export_memory_report(self, filename: str = None) -> str:
        """Export memory profiling results"""
        
        import json
        
        if filename is None:
            filename = f"memory_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = self.get_memory_summary()
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filename
