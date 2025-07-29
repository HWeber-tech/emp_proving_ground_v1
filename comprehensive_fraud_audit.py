#!/usr/bin/env python3
"""
Comprehensive Fraud Detection Audit for EMP Proving Ground
Identifies hardcoded returns, stub implementations, and non-functional logic
"""

import os
import re
import ast
import sys
from pathlib import Path
from collections import defaultdict

class FraudDetector:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.fraud_patterns = {
            'hardcoded_returns': [
                r'return\s+True\s*$',
                r'return\s+False\s*$', 
                r'return\s+None\s*$',
                r'return\s+\[\]\s*$',
                r'return\s+\{\}\s*$',
                r'return\s+"[^"]*"\s*$',
                r'return\s+\d+\s*$',
                r'return\s+0\.0\s*$'
            ],
            'stub_implementations': [
                r'^\s*pass\s*$',
                r'^\s*\.\.\.\s*$',
                r'raise\s+NotImplementedError',
                r'TODO:',
                r'FIXME:',
                r'STUB:',
                r'# TODO',
                r'# FIXME',
                r'# STUB'
            ],
            'mock_patterns': [
                r'mock_',
                r'fake_',
                r'dummy_',
                r'test_data\s*=',
                r'sample_data\s*=',
                r'placeholder'
            ],
            'suspicious_logic': [
                r'if\s+True:',
                r'if\s+False:',
                r'while\s+True:.*break',
                r'for.*in\s+\[\]:',
                r'random\.choice\(',
                r'time\.sleep\(\d+\)'
            ]
        }
        
    def analyze_file(self, file_path):
        """Analyze a single Python file for fraud patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            results = {
                'file': str(file_path),
                'total_lines': len(lines),
                'issues': [],
                'severity_score': 0
            }
            
            # Check each line for fraud patterns
            for line_num, line in enumerate(lines, 1):
                for category, patterns in self.fraud_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            severity = self.calculate_severity(category, line, file_path)
                            results['issues'].append({
                                'line': line_num,
                                'content': line.strip(),
                                'category': category,
                                'pattern': pattern,
                                'severity': severity
                            })
                            results['severity_score'] += severity
                            
            return results
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'issues': [],
                'severity_score': 0
            }
    
    def calculate_severity(self, category, line, file_path):
        """Calculate severity score for fraud patterns"""
        severity_map = {
            'hardcoded_returns': 6,
            'stub_implementations': 8,
            'mock_patterns': 4,
            'suspicious_logic': 5
        }
        
        base_severity = severity_map.get(category, 3)
        
        # Increase severity for core components
        if 'src/core/' in str(file_path):
            base_severity += 2
        elif 'src/trading/' in str(file_path):
            base_severity += 2
        elif 'src/evolution/' in str(file_path):
            base_severity += 1
            
        # Decrease severity for test files and interfaces
        if 'test_' in str(file_path) or '/tests/' in str(file_path):
            base_severity = max(1, base_severity - 3)
        elif 'interface' in str(file_path).lower() and 'pass' in line:
            base_severity = max(1, base_severity - 4)  # Abstract interfaces are OK
            
        return min(10, base_severity)
    
    def scan_directory(self):
        """Scan entire directory for fraud patterns"""
        results = []
        python_files = list(self.root_dir.rglob('*.py'))
        
        print(f"Scanning {len(python_files)} Python files...")
        
        for file_path in python_files:
            # Skip certain directories
            if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.pytest_cache']):
                continue
                
            result = self.analyze_file(file_path)
            if result['issues'] or 'error' in result:
                results.append(result)
                
        return results

if __name__ == "__main__":
    detector = FraudDetector('.')
    results = detector.scan_directory()
    print(f"Found {len(results)} files with fraud patterns")
