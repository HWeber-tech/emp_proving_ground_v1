"""
Automated Stub Detection System
Identifies all stubs, mocks, and placeholder implementations in the codebase
"""

import ast
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class StubLocation:
    """Location and details of a stub/mock implementation"""
    file_path: str
    line_number: int
    function_name: str
    class_name: str
    stub_type: str  # 'pass', 'mock', 'placeholder', 'not_implemented', 'todo'
    criticality: str  # 'critical', 'high', 'medium', 'low'
    complexity_estimate: int  # hours to implement
    description: str = ""

class StubDetector:
    """Detects stub implementations across the codebase"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.stub_patterns = {
            'pass_statement': r'^\s*pass\s*$',
            'not_implemented': r'raise\s+NotImplementedError|NotImplementedError\(\)',
            'todo_comment': r'#\s*TODO|#\s*FIXME|#\s*HACK',
            'mock_object': r'Mock[A-Z]\w*|mock_\w+|Mock\(\)',
            'placeholder': r'placeholder|PLACEHOLDER|stub|STUB|TODO',
            'hardcoded': r'return\s+(0\.0|1\.0|True|False|\[\]|\{\}|None)\s*#.*placeholder',
            'empty_function': r'def\s+\w+\([^)]*\):\s*\n\s*pass',
            'async_empty': r'async\s+def\s+\w+\([^)]*\):\s*\n\s*pass'
        }
        
        self.critical_paths = [
            'src/core/',
            'src/trading/',
            'src/sensory/',
            'src/evolution/',
            'src/risk/',
            'src/portfolio/'
        ]
        
        self.critical_functions = [
            'generate_signal',
            'process',
            'evolve',
            'evaluate_fitness',
            'calculate_risk',
            'execute_trade',
            'get_population',
            'assess_portfolio_risk',
            'calculate_position_size'
        ]
    
    def scan_repository(self) -> List[StubLocation]:
        """Scan entire repository for stubs and mocks"""
        stubs = []
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip test directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and 'test' not in d.lower()]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    stubs.extend(self.scan_file(str(file_path)))
        
        return self.prioritize_stubs(stubs)
    
    def scan_file(self, file_path: str) -> List[StubLocation]:
        """Scan individual file for stubs"""
        stubs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Parse AST for structural analysis
            try:
                tree = ast.parse(content)
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, using regex fallback")
                tree = None
            
            # Find stubs using AST
            if tree:
                stubs.extend(self._find_ast_stubs(tree, file_path))
            
            # Find stubs using regex
            stubs.extend(self._find_regex_stubs(content, file_path))
            
            # Find mock objects
            stubs.extend(self._find_mock_objects(content, file_path))
            
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return stubs
    
    def _find_ast_stubs(self, tree: ast.AST, file_path: str) -> List[StubLocation]:
        """Find stubs using AST analysis"""
        stubs = []
        
        class StubVisitor(ast.NodeVisitor):
            def __init__(self):
                self.class_name = ""
                self.function_stack = []
            
            def visit_ClassDef(self, node):
                old_class = self.class_name
                self.class_name = node.name
                self.generic_visit(node)
                self.class_name = old_class
            
            def visit_FunctionDef(self, node):
                self.function_stack.append(node.name)
                self._check_function_stub(node, file_path)
                self.generic_visit(node)
                self.function_stack.pop()
            
            def visit_AsyncFunctionDef(self, node):
                self.function_stack.append(node.name)
                self._check_function_stub(node, file_path)
                self.generic_visit(node)
                self.function_stack.pop()
            
            def _check_function_stub(self, node, file_path):
                # Check for pass statements only
                if (len(node.body) == 1 and 
                    isinstance(node.body[0], ast.Pass)):
                    stub = StubLocation(
                        file_path=file_path,
                        line_number=node.lineno,
                        function_name=node.name,
                        class_name=self.class_name,
                        stub_type='pass',
                        criticality=self._assess_criticality(file_path, node.name),
                        complexity_estimate=self._estimate_complexity(node, file_path),
                        description="Empty function with pass statement"
                    )
                    stubs.append(stub)
                
                # Check for NotImplementedError
                for stmt in node.body:
                    if (isinstance(stmt, ast.Raise) and
                        isinstance(stmt.exc, ast.Name) and
                        stmt.exc.id == 'NotImplementedError'):
                        stub = StubLocation(
                            file_path=file_path,
                            line_number=node.lineno,
                            function_name=node.name,
                            class_name=self.class_name,
                            stub_type='not_implemented',
                            criticality=self._assess_criticality(file_path, node.name),
                            complexity_estimate=self._estimate_complexity(node, file_path),
                            description="Function raises NotImplementedError"
                        )
                        stubs.append(stub)
        
        visitor = StubVisitor()
        visitor._assess_criticality = self._assess_criticality
        visitor._estimate_complexity = self._estimate_complexity
        visitor.stubs = stubs
        
        # Monkey patch the methods
        def assess_criticality(file_path, function_name):
            return self._assess_criticality(file_path, function_name)
        
        def estimate_complexity(node, file_path):
            return self._estimate_complexity(node, file_path)
        
        visitor._assess_criticality = assess_criticality
        visitor._estimate_complexity = estimate_complexity
        
        visitor.visit(tree)
        return visitor.stubs
    
    def _find_regex_stubs(self, content: str, file_path: str) -> List[StubLocation]:
        """Find stubs using regex patterns"""
        stubs = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern_name, pattern in self.stub_patterns.items():
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Find function context
                    function_name = self._extract_function_name(lines, i)
                    class_name = self._extract_class_name(lines, i)
                    
                    stub = StubLocation(
                        file_path=file_path,
                        line_number=i,
                        function_name=function_name,
                        class_name=class_name,
                        stub_type=pattern_name,
                        criticality=self._assess_criticality(file_path, function_name),
                        complexity_estimate=self._estimate_complexity_from_context(lines, i),
                        description=f"Found {pattern_name} pattern"
                    )
                    stubs.append(stub)
        
        return stubs
    
    def _find_mock_objects(self, content: str, file_path: str) -> List[StubLocation]:
        """Find mock objects in production code"""
        stubs = []
        lines = content.split('\n')
        
        # Skip test files
        if 'test' in file_path.lower():
            return stubs
        
        mock_patterns = [
            r'class\s+Mock[A-Z]\w*',
            r'Mock[A-Z]\w*\s*\(',
            r'mock_\w+\s*=',
            r'return\s+Mock\(\)',
            r'MagicMock\(\)',
            r'patch\('
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in mock_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    function_name = self._extract_function_name(lines, i)
                    class_name = self._extract_class_name(lines, i)
                    
                    stub = StubLocation(
                        file_path=file_path,
                        line_number=i,
                        function_name=function_name,
                        class_name=class_name,
                        stub_type='mock',
                        criticality=self._assess_criticality(file_path, function_name),
                        complexity_estimate=2,  # Mock objects are usually simple to replace
                        description="Mock object in production code"
                    )
                    stubs.append(stub)
        
        return stubs
    
    def _assess_criticality(self, file_path: str, function_name: str) -> str:
        """Assess criticality based on file path and function name"""
        # Check critical paths
        for path in self.critical_paths:
            if path in file_path:
                # Check critical functions
                for func in self.critical_functions:
                    if func in function_name.lower():
                        return 'critical'
                return 'high'
        
        # Check if it's in main src directory
        if 'src/' in file_path and not any(skip in file_path for skip in ['test', '__pycache__']):
            return 'medium'
        
        return 'low'
    
    def _estimate_complexity(self, node: ast.AST, file_path: str) -> int:
        """Estimate implementation complexity"""
        # Simple heuristic based on function complexity
        complexity = 1
        
        # Count parameters
        if hasattr(node, 'args'):
            complexity += len(node.args.args)
        
        # Count decorators
        if hasattr(node, 'decorator_list'):
            complexity += len(node.decorator_list)
        
        # Adjust based on file type
        if 'evolution' in file_path:
            complexity *= 4  # Evolution components are complex
        elif 'risk' in file_path:
            complexity *= 3  # Risk management is moderately complex
        elif 'trading' in file_path:
            complexity *= 2  # Trading components are moderately complex
        
        return min(complexity, 40)  # Cap at 40 hours
    
    def _estimate_complexity_from_context(self, lines: List[str], line_num: int) -> int:
        """Estimate complexity from context"""
        # Look at surrounding lines for context
        start = max(0, line_num - 5)
        end = min(len(lines), line_num + 5)
        
        context = '\n'.join(lines[start:end])
        
        # Simple heuristics
        if 'evolution' in context.lower():
            return 20
        elif 'risk' in context.lower():
            return 15
        elif 'trading' in context.lower():
            return 10
        else:
            return 5
    
    def _extract_function_name(self, lines: List[str], line_num: int) -> str:
        """Extract function name from context"""
        for i in range(line_num - 1, max(0, line_num - 10), -1):
            line = lines[i-1]  # Convert to 0-based indexing
            match = re.search(r'def\s+(\w+)', line)
            if match:
                return match.group(1)
        
        return "unknown_function"
    
    def _extract_class_name(self, lines: List[str], line_num: int) -> str:
        """Extract class name from context"""
        for i in range(line_num - 1, max(0, line_num - 10), -1):
            line = lines[i-1]  # Convert to 0-based indexing
            match = re.search(r'class\s+(\w+)', line)
            if match:
                return match.group(1)
        
        return ""
    
    def prioritize_stubs(self, stubs: List[StubLocation]) -> List[StubLocation]:
        """Prioritize stubs by criticality and dependencies"""
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        return sorted(stubs, key=lambda x: (
            priority_order[x.criticality],
            -x.complexity_estimate,  # Higher complexity first
            x.file_path
        ))
    
    def generate_report(self, stubs: List[StubLocation]) -> Dict:
        """Generate comprehensive stub report"""
        report = {
            'scan_timestamp': str(datetime.now()),
            'total_stubs': len(stubs),
            'summary': {
                'critical': len([s for s in stubs if s.criticality == 'critical']),
                'high': len([s for s in stubs if s.criticality == 'high']),
                'medium': len([s for s in stubs if s.criticality == 'medium']),
                'low': len([s for s in stubs if s.criticality == 'low'])
            },
            'by_type': {},
            'by_file': {},
            'detailed_list': []
        }
        
        # Group by type
        for stub in stubs:
            if stub.stub_type not in report['by_type']:
                report['by_type'][stub.stub_type] = []
            report['by_type'][stub.stub_type].append(stub.__dict__)
        
        # Group by file
        for stub in stubs:
            if stub.file_path not in report['by_file']:
                report['by_file'][stub.file_path] = []
            report['by_file'][stub.file_path].append(stub.__dict__)
        
        # Detailed list
        report['detailed_list'] = [stub.__dict__ for stub in stubs]
        
        return report

def main():
    """Main function to run stub detection"""
    import sys
    
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = "."
    
    detector = StubDetector(repo_path)
    stubs = detector.scan_repository()
    
    # Generate report
    report = detector.generate_report(stubs)
    
    # Save report
    with open('stub_detection_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n=== STUB DETECTION REPORT ===")
    print(f"Total stubs found: {len(stubs)}")
    print(f"Critical: {report['summary']['critical']}")
    print(f"High: {report['summary']['high']}")
    print(f"Medium: {report['summary']['medium']}")
    print(f"Low: {report['summary']['low']}")
    
    # Print critical stubs
    critical_stubs = [s for s in stubs if s.criticality == 'critical']
    if critical_stubs:
        print("\n=== CRITICAL STUBS ===")
        for stub in critical_stubs[:10]:  # Show top 10
            print(f"- {stub.file_path}:{stub.line_number} - {stub.function_name}")
            print(f"  Type: {stub.stub_type}, Estimate: {stub.complexity_estimate}h")
    
    print(f"\nDetailed report saved to: stub_detection_report.json")

if __name__ == '__main__':
    main()
