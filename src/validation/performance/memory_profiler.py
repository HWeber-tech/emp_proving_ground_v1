#!/usr/bin/env python3
"""
Memory Profiler
===============

Profiles memory usage for Phase 2 validation.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class MemoryProfile:
    """Memory usage profile"""

    current_mb: float
    peak_mb: float
    increase_mb: float
    objects_count: int
    details: dict[str, Any]


class MemoryProfiler:
    """Memory usage profiling"""

    def __init__(self) -> None:
        self.initial_memory: float | None = None

    def start_profiling(self) -> None:
        """Start memory profiling"""
        gc.collect()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

    def get_memory_profile(self) -> MemoryProfile:
        """Get current memory profile"""
        gc.collect()
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024

        return MemoryProfile(
            current_mb=current_memory,
            peak_mb=current_memory,
            increase_mb=current_memory - (self.initial_memory or 0),
            objects_count=len(gc.get_objects()),
            details={
                "gc_collections": gc.get_count(),
                "memory_percent": psutil.Process().memory_percent(),
            },
        )

    def check_memory_leaks(self, test_func: Any, iterations: int = 100) -> dict[str, Any]:
        """Check for memory leaks"""
        self.start_profiling()

        initial_profile = self.get_memory_profile()

        for _ in range(iterations):
            test_func()

        final_profile = self.get_memory_profile()

        return {
            "memory_leak_detected": final_profile.increase_mb > 10,
            "memory_increase_mb": final_profile.increase_mb,
            "initial_memory_mb": initial_profile.current_mb,
            "final_memory_mb": final_profile.current_mb,
            "objects_increase": final_profile.objects_count - initial_profile.objects_count,
        }
