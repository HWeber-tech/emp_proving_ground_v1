#!/usr/bin/env python3
"""
Performance Benchmarking Framework
Comprehensive performance measurement and analysis for Phase 2 validation
"""

import asyncio
import logging
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import gc

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result with statistical analysis"""
    test_name: str
    metric_type: str  # response_time, throughput, memory, concurrent
    measured_value: float
    unit: str
    target_value: float
    passed: bool
    timestamp: datetime = field(default_factory=datetime.now)
    iterations: int = 100
    percentiles: Dict[str, float] = field(default_factory=dict)
    statistical_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    standard_deviation: float = 0.0
    coefficient_of_variation: float = 0.0


class PerformanceBenchmarker:
    """Advanced performance benchmarking system for Phase 2 validation"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_info = self._collect_system_info()
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmarking context"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform
        }
    
    async def run_response_time_benchmark(
        self,
        test_function: Callable,
        test_name: str,
        target_time: float = 1.0,
        iterations: int = 1000,
        warmup_iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark response time with statistical analysis"""
        
        logger.info(f"Running response time benchmark: {test_name}")
        
        # Warmup
        for _ in range(warmup_iterations):
            await test_function()
        
        # Collect measurements
        measurements = []
        for i in range(iterations):
            gc.collect()  # Clean memory before each test
            start_time = time.perf_counter()
            await test_function()
            end_time = time.perf_counter()
            measurements.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Statistical analysis
        avg_time = statistics.mean(measurements)
        std_dev = statistics.stdev(measurements) if len(measurements) > 1 else 0
        percentiles = {
            'p50': np.percentile(measurements, 50),
            'p95': np.percentile(measurements, 95),
            'p99': np.percentile(measurements, 99)
        }
        
        # Confidence interval (95%)
        confidence_interval = self._calculate_confidence_interval(measurements, 0.95)
        cv = std_dev / avg_time if avg_time > 0 else 0
        
        passed = avg_time <= target_time * 1000  # Convert target to ms
        
        result = BenchmarkResult(
            test_name=test_name,
            metric_type="response_time",
            measured_value=avg_time,
            unit="milliseconds",
            target_value=target_time * 1000,
            passed=passed,
            iterations=iterations,
            percentiles=percentiles,
            statistical_data={
                'min': min(measurements),
                'max': max(measurements),
                'median': statistics.median(measurements),
                'std_dev': std_dev,
                'measurements': measurements
            },
            confidence_interval=confidence_interval,
            standard_deviation=std_dev,
            coefficient_of_variation=cv
        )
        
        self.results.append(result)
        return result
    
    async def run_throughput_benchmark(
        self,
        test_function: Callable,
        test_name: str,
        target_throughput: float = 5.0,
        duration: int = 60,
        concurrent_tasks: int = 1
    ) -> BenchmarkResult:
        """Benchmark throughput with concurrent execution"""
        
        logger.info(f"Running throughput benchmark: {test_name}")
        
        async def single_operation():
            start_time = time.perf_counter()
            await test_function()
            end_time = time.perf_counter()
            return end_time - start_time
        
        # Run concurrent operations
        start_time = time.perf_counter()
        operations = 0
        
        async def worker():
            nonlocal operations
            while time.perf_counter() - start_time < duration:
                await single_operation()
                operations += 1
        
        # Create concurrent workers
        workers = [worker() for _ in range(concurrent_tasks)]
        await asyncio.gather(*workers)
        
        end_time = time.perf_counter()
        actual_duration = end_time - start_time
        throughput = operations / actual_duration
        
        passed = throughput >= target_throughput
        
        result = BenchmarkResult(
            test_name=test_name,
            metric_type="throughput",
            measured_value=throughput,
            unit="operations_per_second",
            target_value=target_throughput,
            passed=passed,
            iterations=operations,
            metadata={
                'duration': actual_duration,
                'concurrent_tasks': concurrent_tasks,
                'total_operations': operations
            }
        )
        
        self.results.append(result)
        return result
    
    async def run_memory_benchmark(
        self,
        test_function: Callable,
        test_name: str,
        target_memory: float = 100.0,
        iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark memory usage and efficiency"""
        
        logger.info(f"Running memory benchmark: {test_name}")
        
        # Baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Run test and measure memory
        memory_measurements = []
        for _ in range(iterations):
            gc.collect()
            before_memory = psutil.Process().memory_info().rss / 1024 / 1024
            await test_function()
            after_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = after_memory - before_memory
            memory_measurements.append(max(0, memory_used))
        
        avg_memory = statistics.mean(memory_measurements)
        max_memory = max(memory_measurements)
        
        passed = avg_memory <= target_memory
        
        result = BenchmarkResult(
            test_name=test_name,
            metric_type="memory",
            measured_value=avg_memory,
            unit="megabytes",
            target_value=target_memory,
            passed=passed,
            iterations=iterations,
            metadata={
                'baseline_memory': baseline_memory,
                'max_memory': max_memory,
                'memory_measurements': memory_measurements
            }
        )
        
        self.results.append(result)
        return result
    
    async def run_concurrent_benchmark(
        self,
        test_function: Callable,
        test_name: str,
        target_concurrent: float = 5.0,
        max_concurrent: int = 50,
        step_size: int = 5
    ) -> BenchmarkResult:
        """Benchmark concurrent performance scaling"""
        
        logger.info(f"Running concurrent benchmark: {test_name}")
        
        results = []
        
        for concurrent in range(step_size, max_concurrent + 1, step_size):
            start_time = time.perf_counter()
            operations = 0
            
            async def worker():
                nonlocal operations
                for _ in range(100):  # Each worker does 100 operations
                    await test_function()
                    operations += 1
            
            workers = [worker() for _ in range(concurrent)]
            await asyncio.gather(*workers)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            throughput = operations / duration
            
            results.append({
                'concurrent': concurrent,
                'throughput': throughput,
                'operations': operations,
                'duration': duration
            })
        
        # Find maximum sustainable throughput
        max_throughput = max(r['throughput'] for r in results)
        optimal_concurrent = next(r['concurrent'] for r in results 
                                if r['throughput'] == max_throughput)
        
        passed = max_throughput >= target_concurrent
        
        result = BenchmarkResult(
            test_name=test_name,
            metric_type="concurrent",
            measured_value=max_throughput,
            unit="operations_per_second",
            target_value=target_concurrent,
            passed=passed,
            iterations=len(results),
            metadata={
                'scaling_results': results,
                'optimal_concurrent': optimal_concurrent,
                'max_throughput': max_throughput
            }
        )
        
        self.results.append(result)
        return result
    
    async def run_scalability_benchmark(
        self,
        test_function: Callable,
        test_name: str,
        load_levels: List[int] = [1, 5, 10, 20, 50],
        duration_per_level: int = 10
    ) -> BenchmarkResult:
        """Analyze performance scaling across load levels"""
        
        logger.info(f"Running scalability benchmark: {test_name}")
        
        scaling_results = []
        
        for load in load_levels:
            start_time = time.perf_counter()
            operations = 0
            
            async def worker():
                nonlocal operations
                end_time = start_time + duration_per_level
                while time.perf_counter() < end_time:
                    await test_function()
                    operations += 1
            
            workers = [worker() for _ in range(load)]
            await asyncio.gather(*workers)
            
            end_time = time.perf_counter()
            actual_duration = end_time - start_time
            throughput = operations / actual_duration
            
            scaling_results.append({
                'load': load,
                'throughput': throughput,
                'efficiency': throughput / load,
                'operations': operations
            })
        
        # Calculate scaling efficiency
        base_throughput = scaling_results[0]['throughput']
        scaling_efficiency = []
        
        for result in scaling_results:
            ideal_throughput = base_throughput * result['load']
            actual_throughput = result['throughput']
            efficiency = actual_throughput / ideal_throughput if ideal_throughput > 0 else 0
            scaling_efficiency.append(efficiency)
        
        avg_efficiency = statistics.mean(scaling_efficiency)
        
        result = BenchmarkResult(
            test_name=test_name,
            metric_type="scalability",
            measured_value=avg_efficiency,
            unit="efficiency_ratio",
            target_value=0.8,  # 80% efficiency target
            passed=avg_efficiency >= 0.8,
            iterations=len(load_levels),
            metadata={
                'scaling_results': scaling_results,
                'efficiency_curve': scaling_efficiency,
                'bottleneck_analysis': self._identify_bottlenecks(scaling_results)
            }
        )
        
        self.results.append(result)
        return result
    
    def _calculate_confidence_interval(
        self, 
        measurements: List[float], 
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for measurements"""
        if len(measurements) < 2:
            return (0.0, 0.0)
        
        mean = statistics.mean(measurements)
        std_err = statistics.stdev(measurements) / (len(measurements) ** 0.5)
        
        # Use t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(measurements) - 1)
        
        margin = t_value * std_err
        return (mean - margin, mean + margin)
    
    def _identify_bottlenecks(self, scaling_results: List[Dict]) -> Dict[str, Any]:
        """Identify performance bottlenecks from scaling analysis"""
        if len(scaling_results) < 2:
            return {"bottleneck_detected": False}
        
        # Look for efficiency drop-off
        efficiencies = [r['efficiency'] for r in scaling_results]
        
        # Find where efficiency drops below 80%
        bottleneck_point = None
        for i, efficiency in enumerate(efficiencies):
            if efficiency < 0.8:
                bottleneck_point = scaling_results[i]['load']
                break
        
        return {
            "bottleneck_detected": bottleneck_point is not None,
            "bottleneck_load": bottleneck_point,
            "efficiency_trend": efficiencies,
            "recommendation": "Consider optimization" if bottleneck_point else "Scaling well"
        }
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get comprehensive benchmark summary"""
        if not self.results:
            return {"summary": "No benchmarks run"}
        
        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        
        return {
            "summary": {
                "total_benchmarks": total_tests,
                "passed_benchmarks": passed_tests,
                "failed_benchmarks": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "system_info": self.system_info,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "metric_type": r.metric_type,
                    "measured_value": r.measured_value,
                    "target_value": r.target_value,
                    "passed": r.passed,
                    "unit": r.unit,
                    "percentiles": r.percentiles,
                    "confidence_interval": r.confidence_interval
                }
                for r in self.results
            ]
        }
    
    def export_results(self, filename: str = None) -> str:
        """Export benchmark results to JSON"""
        import json
        
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = self.get_benchmark_summary()
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filename


# Example usage and testing
async def example_benchmark():
    """Example usage of the performance benchmarker"""
    
    async def example_operation():
        """Example operation to benchmark"""
        await asyncio.sleep(0.001)  # Simulate 1ms operation
    
    benchmarker = PerformanceBenchmarker()
    
    # Run various benchmarks
    await benchmarker.run_response_time_benchmark(
        example_operation, 
        "example_response_time",
        target_time=0.01
    )
    
    await benchmarker.run_throughput_benchmark(
        example_operation,
        "example_throughput",
        target_throughput=100
    )
    
    await benchmarker.run_memory_benchmark(
        example_operation,
        "example_memory",
        target_memory=1.0
    )
    
    # Print summary
    summary = benchmarker.get_benchmark_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import asyncio
    import json
    import sys
    
    asyncio.run(example_benchmark())
