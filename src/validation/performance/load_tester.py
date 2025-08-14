#!/usr/bin/env python3
"""
Load Tester
===========

Tests system performance under load conditions.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class LoadTestResult:
    """Load test result"""
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    throughput: float
    errors: list


class LoadTester:
    """System load testing"""
    
    def __init__(self):
        self.max_concurrent = 100
    
    async def load_test(self, 
                     test_function: Callable, 
                     concurrent_users: int = 10, 
                     requests_per_user: int = 10) -> LoadTestResult:
        """Run load test"""
        
        total_requests = concurrent_users * requests_per_user
        successful_requests = 0
        failed_requests = 0
        total_response_time = 0
        errors = []
        
        async def user_session():
            nonlocal successful_requests, failed_requests, total_response_time
            
            for _ in range(requests_per_user):
                try:
                    start_time = time.time()
                    await test_function()
                    elapsed = time.time() - start_time
                    
                    successful_requests += 1
                    total_response_time += elapsed
                    
                except Exception as e:
                    failed_requests += 1
                    errors.append(str(e))
        
        # Run concurrent users
        start_time = time.time()
        tasks = [user_session() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Calculate metrics
        throughput = total_requests / total_time
        avg_response_time = total_response_time / successful_requests if successful_requests > 0 else 0
        
        return LoadTestResult(
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            throughput=throughput,
            errors=errors
        )
    
    async def stress_test(self, test_function: Callable) -> Dict[str, Any]:
        """Run stress test with increasing load"""
        results = {}
        
        for users in [1, 5, 10, 25, 50, 100]:
            result = await self.load_test(test_function, concurrent_users=users)
            results[f"{users}_users"] = {
                'throughput': result.throughput,
                'avg_response_time': result.average_response_time,
                'success_rate': result.successful_requests / result.total_requests,
                'errors': len(result.errors)
            }
        
        return results
