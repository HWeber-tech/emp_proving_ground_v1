#!/usr/bin/env python3
"""
Load Testing Framework
Stress testing and load analysis for Phase 2 validation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Result of a load test"""
    test_name: str
    load_level: int
    duration: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_response_time: float
    throughput: float
    error_rate: float
    resource_usage: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class LoadTester:
    """Advanced load testing system for Phase 2 validation"""
    
    def __init__(self):
        self.results: List[LoadTestResult] = []
        
    async def run_load_test(
        self,
        test_function: Callable,
        test_name: str,
        load_levels: List[int] = [1, 5, 10, 20, 50, 100],
        duration_per_level: int = 30,
        ramp_up_time: int = 5
    ) -> List[LoadTestResult]:
        """Run comprehensive load testing across multiple load levels"""
        
        logger.info(f"Starting load test: {test_name}")
        results = []
        
        for load_level in load_levels:
            logger.info(f"Testing load level: {load_level}")
            
            # Ramp up
            await asyncio.sleep(ramp_up_time)
            
            # Run load test
            result = await self._execute_load_level(
                test_function, 
                test_name, 
                load_level, 
                duration_per_level
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    async def _execute_load_level(
        self,
        test_function: Callable,
        test_name: str,
        load_level: int,
        duration: int
    ) -> LoadTestResult:
        """Execute a single load level test"""
        
        start_time = time.time()
        end_time = start_time + duration
        
        successful_ops = 0
        failed_ops = 0
        response_times = []
        
        async def worker():
            nonlocal successful_ops, failed_ops
            while time.time() < end_time:
                try:
                    op_start = time.time()
                    await test_function()
                    op_end = time.time()
                    
                    response_times.append(op_end - op_start)
                    successful_ops += 1
                    
                except Exception as e:
                    logger.warning(f"Operation failed: {e}")
                    failed_ops += 1
        
        # Create workers
        workers = [worker() for _ in range(load_level)]
        await asyncio.gather(*workers, return_exceptions=True)
        
        total_ops = successful_ops + failed_ops
        avg_response_time = np.mean(response_times) if response_times else 0
        throughput = successful_ops / duration
        error_rate = failed_ops / total_ops if total_ops > 0 else 0
        
        # Collect resource usage
        import psutil
        process = psutil.Process()
        resource_usage = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'threads': process.num_threads()
        }
        
        return LoadTestResult(
            test_name=test_name,
            load_level=load_level,
            duration=duration,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            average_response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate,
            resource_usage=resource_usage
        )
    
    async def run_stress_test(
        self,
        test_function: Callable,
        test_name: str,
        max_load: int = 1000,
        duration: int = 300,
        step_duration: int = 30
    ) -> List[LoadTestResult]:
        """Run stress test with increasing load until failure"""
        
        logger.info(f"Starting stress test: {test_name}")
        results = []
        
        load_level = 10
        while load_level <= max_load:
            logger.info(f"Stress testing at load: {load_level}")
            
            result = await self._execute_load_level(
                test_function, 
                test_name, 
                load_level, 
                step_duration
            )
            
            results.append(result)
            self.results.append(result)
            
            # Check for failure conditions
            if result.error_rate > 0.1 or result.average_response_time > 5.0:
                logger.warning(f"System failure detected at load {load_level}")
                break
            
            load_level *= 2
        
        return results
    
    def analyze_load_patterns(self) -> Dict[str, Any]:
        """Analyze load test patterns and identify trends"""
        
        if not self.results:
            return {"analysis": "No load test results available"}
        
        # Sort by load level
        sorted_results = sorted(self.results, key=lambda x: x.load_level)
        
        # Calculate trends
        load_levels = [r.load_level for r in sorted_results]
        throughputs = [r.throughput for r in sorted_results]
        response_times = [r.average_response_time for r in sorted_results]
        error_rates = [r.error_rate for r in sorted_results]
        
        # Find optimal load level
        optimal_index = np.argmax(throughputs)
        optimal_load = load_levels[optimal_index]
        max_throughput = throughputs[optimal_index]
        
        # Calculate degradation
        degradation_point = None
        for i, (load, throughput) in enumerate(zip(load_levels, throughputs)):
            if i > 0 and throughput < throughputs[i-1] * 0.9:
                degradation_point = load
                break
        
        return {
            "optimal_load": optimal_load,
            "max_throughput": max_throughput,
            "degradation_point": degradation_point,
            "scalability_trend": {
                "load_levels": load_levels,
                "throughputs": throughputs,
                "response_times": response_times,
                "error_rates": error_rates
            },
            "capacity_analysis": {
                "sustainable_load": optimal_load,
                "peak_capacity": max(load_levels),
                "failure_threshold": degradation_point
            }
        }
    
    def get_load_test_summary(self) -> Dict[str, Any]:
        """Get comprehensive load test summary"""
        
        if not self.results:
            return {"summary": "No load tests run"}
        
        # Find best and worst performing load levels
        best_result = max(self.results, key=lambda x: x.throughput)
        worst_result = min(self.results, key=lambda x: x.throughput)
        
        return {
            "summary": {
                "total_tests": len(self.results),
                "best_throughput": best_result.throughput,
                "worst_throughput": worst_result.throughput,
                "highest_load": max(r.load_level for r in self.results),
                "lowest_error_rate": min(r.error_rate for r in self.results)
            },
            "detailed_results": [
                {
                    "load_level": r.load_level,
                    "throughput": r.throughput,
                    "avg_response_time": r.average_response_time,
                    "error_rate": r.error_rate,
                    "successful_ops": r.successful_operations,
                    "failed_ops": r.failed_operations
                }
                for r in self.results
            ]
        }
