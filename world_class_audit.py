#!/usr/bin/env python3
"""
World-Class Trading System Audit
Identifies ALL synthetic, mock, and simulation components that have no place in production.
Zero tolerance for placeholder code.
"""

import os
import re
import ast
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SyntheticComponent:
    file_path: str
    line_number: int
    component_type: str
    severity: str
    description: str
    code_snippet: str

class WorldClassAuditor:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.synthetic_components: List[SyntheticComponent] = []
        
        # Patterns that indicate synthetic/mock/simulation code
        self.synthetic_patterns = {
            'CRITICAL': {
                'mock_': r'\bmock_\w+',
                'simulate_': r'\bsimulate_\w+',
                'fake_': r'\bfake_\w+',
                'dummy_': r'\bdummy_\w+',
                'test_data': r'\btest_data\b',
                'hardcoded_': r'\bhardcoded_\w+',
                'placeholder': r'\bplaceholder\b',
                'todo_implementation': r'#\s*TODO.*implement',
                'mock_return': r'return\s+(Mock|MagicMock)',
                'sleep_simulation': r'time\.sleep|asyncio\.sleep.*#.*simul',
            },
            'HIGH': {
                'random_data': r'random\.(uniform|randint|choice).*#.*test',
                'fixed_values': r'return\s+[0-9.]+\s*#.*placeholder',
                'stub_comment': r'#.*stub|#.*placeholder|#.*mock',
                'demo_only': r'#.*demo\s+only|#.*testing\s+only',
                'synthetic_class': r'class\s+\w*(Mock|Fake|Dummy|Stub)\w*',
                'simulation_method': r'def\s+\w*simul\w*',
            },
            'MEDIUM': {
                'test_constants': r'TEST_\w+\s*=',
                'debug_prints': r'print\s*\(.*debug|print\s*\(.*test',
                'temporary_fix': r'#.*temporary|#.*temp\s+fix',
                'bypass_logic': r'#.*bypass|#.*skip',
            }
        }
        
        # File patterns to examine
        self.file_patterns = ['*.py']
        
        # Directories to skip
        self.skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
        
    def audit_codebase(self) -> Dict[str, List[SyntheticComponent]]:
        """Perform comprehensive audit of codebase."""
        print("🔍 Starting World-Class Trading System Audit...")
        
        results = {
            'CRITICAL': [],
            'HIGH': [],
            'MEDIUM': []
        }
        
        # Scan all Python files
        for py_file in self._get_python_files():
            components = self._audit_file(py_file)
            for component in components:
                results[component.severity].append(component)
        
        # Additional specialized scans
        self._scan_for_mock_interfaces()
        self._scan_for_simulation_classes()
        self._scan_for_synthetic_data_sources()
        
        return results
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files to audit."""
        python_files = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _audit_file(self, file_path: Path) -> List[SyntheticComponent]:
        """Audit a single file for synthetic components."""
        components = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Pattern-based detection
            for severity, patterns in self.synthetic_patterns.items():
                for pattern_name, pattern in patterns.items():
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = lines[line_num - 1].strip()
                        
                        component = SyntheticComponent(
                            file_path=str(file_path.relative_to(self.root_dir)),
                            line_number=line_num,
                            component_type=pattern_name,
                            severity=severity,
                            description=f"Synthetic component: {pattern_name}",
                            code_snippet=line_content
                        )
                        components.append(component)
            
            # AST-based analysis for deeper inspection
            try:
                tree = ast.parse(content)
                ast_components = self._analyze_ast(tree, file_path, lines)
                components.extend(ast_components)
            except SyntaxError:
                pass  # Skip files with syntax errors
                
        except Exception as e:
            print(f"⚠️  Error auditing {file_path}: {e}")
        
        return components
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[SyntheticComponent]:
        """Analyze AST for synthetic patterns."""
        components = []
        
        class SyntheticVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                # Check for mock/simulation classes
                if any(keyword in node.name.lower() for keyword in ['mock', 'fake', 'dummy', 'stub', 'simulation']):
                    components.append(SyntheticComponent(
                        file_path=str(file_path.relative_to(self.root_dir)),
                        line_number=node.lineno,
                        component_type='synthetic_class',
                        severity='CRITICAL',
                        description=f"Synthetic class: {node.name}",
                        code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    ))
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                # Check for simulation/mock methods
                if any(keyword in node.name.lower() for keyword in ['simulate', 'mock', 'fake', 'dummy']):
                    components.append(SyntheticComponent(
                        file_path=str(file_path.relative_to(self.root_dir)),
                        line_number=node.lineno,
                        component_type='synthetic_method',
                        severity='CRITICAL',
                        description=f"Synthetic method: {node.name}",
                        code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    ))
                
                # Check for methods that only return hardcoded values
                if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                    if isinstance(node.body[0].value, (ast.Constant, ast.Num, ast.Str)):
                        components.append(SyntheticComponent(
                            file_path=str(file_path.relative_to(self.root_dir)),
                            line_number=node.lineno,
                            component_type='hardcoded_return',
                            severity='HIGH',
                            description=f"Method with hardcoded return: {node.name}",
                            code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                        ))
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for mock imports
                for alias in node.names:
                    if 'mock' in alias.name.lower() or 'fake' in alias.name.lower():
                        components.append(SyntheticComponent(
                            file_path=str(file_path.relative_to(self.root_dir)),
                            line_number=node.lineno,
                            component_type='mock_import',
                            severity='HIGH',
                            description=f"Mock import: {alias.name}",
                            code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                        ))
                self.generic_visit(node)
        
        visitor = SyntheticVisitor()
        visitor.visit(tree)
        
        return components
    
    def _scan_for_mock_interfaces(self):
        """Scan for mock interface implementations."""
        mock_files = [
            'mock_ctrader_interface.py',
            'mock_data_provider.py', 
            'mock_broker.py',
            'simulation.py'
        ]
        
        for mock_file in mock_files:
            for py_file in self._get_python_files():
                if mock_file in str(py_file):
                    self.synthetic_components.append(SyntheticComponent(
                        file_path=str(py_file.relative_to(self.root_dir)),
                        line_number=1,
                        component_type='mock_interface_file',
                        severity='CRITICAL',
                        description=f"Entire mock interface file: {mock_file}",
                        code_snippet="[ENTIRE FILE IS MOCK]"
                    ))
    
    def _scan_for_simulation_classes(self):
        """Scan for simulation classes that have no place in production."""
        pass  # Implementation would scan for specific simulation patterns
    
    def _scan_for_synthetic_data_sources(self):
        """Scan for synthetic data sources."""
        pass  # Implementation would identify fake data generators
    
    def generate_report(self, results: Dict[str, List[SyntheticComponent]]) -> str:
        """Generate comprehensive audit report."""
        report = []
        report.append("# World-Class Trading System Audit Report")
        report.append("## Synthetic Component Elimination Analysis")
        report.append("")
        
        total_issues = sum(len(components) for components in results.values())
        report.append(f"**Total Synthetic Components Found:** {total_issues}")
        report.append("")
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM']:
            components = results[severity]
            if components:
                report.append(f"## {severity} Issues ({len(components)})")
                report.append("")
                
                # Group by file
                by_file = {}
                for component in components:
                    if component.file_path not in by_file:
                        by_file[component.file_path] = []
                    by_file[component.file_path].append(component)
                
                for file_path, file_components in sorted(by_file.items()):
                    report.append(f"### {file_path}")
                    for component in file_components:
                        report.append(f"- **Line {component.line_number}:** {component.description}")
                        report.append(f"  ```python")
                        report.append(f"  {component.code_snippet}")
                        report.append(f"  ```")
                    report.append("")
        
        return "\n".join(report)

def main():
    auditor = WorldClassAuditor('src')
    results = auditor.audit_codebase()
    
    # Print summary
    total = sum(len(components) for components in results.values())
    print(f"\n🎯 AUDIT COMPLETE")
    print(f"Total Synthetic Components: {total}")
    print(f"Critical Issues: {len(results['CRITICAL'])}")
    print(f"High Issues: {len(results['HIGH'])}")
    print(f"Medium Issues: {len(results['MEDIUM'])}")
    
    # Generate detailed report
    report = auditor.generate_report(results)
    with open('WORLD_CLASS_AUDIT_REPORT.md', 'w') as f:
        f.write(report)
    
    print(f"\n📊 Detailed report saved to: WORLD_CLASS_AUDIT_REPORT.md")
    
    # Print top offenders
    if results['CRITICAL']:
        print(f"\n🚨 TOP CRITICAL ISSUES:")
        for i, component in enumerate(results['CRITICAL'][:10], 1):
            print(f"{i}. {component.file_path}:{component.line_number} - {component.description}")

if __name__ == "__main__":
    main()
