#!/usr/bin/env python3
"""
Performance Benchmarker
=======================

Benchmarks system performance against Phase 2 requirements.
"""

import time
from dataclasses import dataclass
from typing import Dict

import psutil


@dataclass
class BenchmarkResult:
    """Performance benchmark result"""
    metric: str
    value: float
    threshold: float
    unit: str
    passed: bool
    details: str = ""


class PerformanceBenchmarker:
    """System performance benchmarking"""
    
    def __init__(self):
        self.benchmarks = {
            'response_time': 1.0,  # seconds
            'throughput': 100,      # ops/sec
            'memory_usage': 500,    # MB
            'cpu_usage': 80         # percentage
        }
    
    def benchmark_response_time(self, func, *args, **kwargs) -> BenchmarkResult:
        """Benchmark response time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        return BenchmarkResult(
            metric="response_time",
            value=elapsed,
            threshold=self.benchmarks['response_time'],
            unit="seconds",
            passed=elapsed < self.benchmarks['response_time'],
            details=f"Function completed in {elapsed:.3f}s"
        )
    
    def benchmark_throughput(self, func, iterations: int = 100) -> BenchmarkResult:
        """Benchmark throughput"""
        start_time = time.time()
        for _ in range(iterations):
            func()
        elapsed = time.time() - start_time
        
        throughput = iterations / elapsed
        
        return BenchmarkResult(
            metric="throughput",
            value=throughput,
            threshold=self.benchmarks['throughput'],
            unit="ops/sec",
            passed=throughput > self.benchmarks['throughput'],
            details=f"Processed {iterations} operations in {elapsed:.3f}s"
        )
    
    def benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return BenchmarkResult(
            metric="memory_usage",
            value=memory_mb,
            threshold=self.benchmarks['memory_usage'],
            unit="MB",
            passed=memory_mb < self.benchmarks['memory_usage'],
            details=f"Current memory usage: {memory_mb:.1f}MB"
        )
    
    def benchmark_cpu_usage(self, interval: float = 1.0) -> BenchmarkResult:
        """Benchmark CPU usage"""
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=interval)
        
        return BenchmarkResult(
            metric="cpu_usage",
            value=cpu_percent,
            threshold=self.benchmarks['cpu_usage'],
            unit="percentage",
            passed=cpu_percent < self.benchmarks['cpu_usage'],
            details=f"CPU usage: {cpu_percent:.1f}%"
        )
    
    def run_all_benchmarks(self, test_func) -> Dict[str, BenchmarkResult]:
        """Run all performance benchmarks"""
        return {
            'response_time': self.benchmark_response_time(test_func),
            'throughput': self.benchmark_throughput(test_func),
            'memory_usage': self.benchmark_memory_usage(),
            'cpu_usage': self.benchmark_cpu_usage()
        }
