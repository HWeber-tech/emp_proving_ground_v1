#!/usr/bin/env python3
"""
TRUTH-01: Overhaul the Validation Framework
===========================================

A ruthless, honest validator that detects stubs, mocks, and false implementations.
This is the foundation of truth-first development.

Usage:
    python tools/truth_validator.py --scan
    python tools/truth_validator.py --validate
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict


class StubDetector(ast.NodeVisitor):
    """AST visitor to detect pass statements and NotImplementedError"""
    
    def __init__(self):
        self.pass_statements = []
        self.not_implemented_errors = []
        self.mock_objects = []
        
    def visit_Pass(self, node):
        """Detect pass statements"""
        self.pass_statements.append({
            'line': node.lineno,
            'type': 'pass_statement',
            'context': self._get_context(node)
        })
        
    def visit_Raise(self, node):
        """Detect NotImplementedError"""
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            if node.exc.func.id == 'NotImplementedError':
                self.not_implemented_errors.append({
                    'line': node.lineno,
                    'type': 'not_implemented_error',
                    'context': self._get_context(node)
                })
                
    def visit_Name(self, node):
        """Detect mock objects"""
        if isinstance(node, ast.Name) and 'mock' in str(node.id).lower():
            self.mock_objects.append({
                'line': node.lineno,
                'type': 'mock_object',
                'name': node.id,
                'context': self._get_context(node)
            })
            
    def _get_context(self, node):
        """Get the function/class context for a node"""
        current = node
        while current:
            if isinstance(current, ast.FunctionDef):
                return f"function {current.name}"
            elif isinstance(current, ast.ClassDef):
                return f"class {current.name}"
            current = getattr(current, 'parent', None)
        return "global"


class TruthValidator:
    """Main validator for truth-first development"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.results = {
            'pass_statements': [],
            'not_implemented_errors': [],
            'mock_objects': [],
            'summary': {}
        }
        
    def scan_directory(self, directory: str = "src") -> Dict[str, Any]:
        """Scan directory for stubs, mocks, and false implementations"""
        target_dir = self.root_path / directory
        
        if not target_dir.exists():
            print(f"❌ Directory {target_dir} does not exist")
            return self.results
            
        python_files = list(target_dir.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                # Add parent references for context
                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        child.parent = node
                
                detector = StubDetector()
                detector.visit(tree)
                
                # Add file context to results
                for item in detector.pass_statements:
                    item['file'] = str(file_path.relative_to(self.root_path))
                    self.results['pass_statements'].append(item)
                    
                for item in detector.not_implemented_errors:
                    item['file'] = str(file_path.relative_to(self.root_path))
                    self.results['not_implemented_errors'].append(item)
                    
                for item in detector.mock_objects:
                    item['file'] = str(file_path.relative_to(self.root_path))
                    self.results['mock_objects'].append(item)
                    
            except Exception as e:
                print(f"⚠️ Error scanning {file_path}: {e}")
                
        self._generate_summary()
        return self.results
        
    def _generate_summary(self):
        """Generate summary statistics"""
        self.results['summary'] = {
            'total_pass_statements': len(self.results['pass_statements']),
            'total_not_implemented_errors': len(self.results['not_implemented_errors']),
            'total_mock_objects': len(self.results['mock_objects']),
            'total_issues': (
                len(self.results['pass_statements']) +
                len(self.results['not_implemented_errors']) +
                len(self.results['mock_objects'])
            ),
            'status': 'FAILED' if (
                len(self.results['pass_statements']) > 0 or
                len(self.results['not_implemented_errors']) > 0 or
                len(self.results['mock_objects']) > 0
            ) else 'PASSED'
        }
        
    def validate(self) -> bool:
        """Validate the codebase for truth-first compliance"""
        results = self.scan_directory()
        
        print("\n" + "="*80)
        print("TRUTH-FIRST VALIDATION RESULTS")
        print("="*80)
        
        print(f"📊 Summary:")
        print(f"   Pass Statements: {results['summary']['total_pass_statements']}")
        print(f"   NotImplementedError: {results['summary']['total_not_implemented_errors']}")
        print(f"   Mock Objects: {results['summary']['total_mock_objects']}")
        print(f"   Total Issues: {results['summary']['total_issues']}")
        print(f"   Status: {results['summary']['status']}")
        
        if results['summary']['total_issues'] > 0:
            print("\n❌ ISSUES FOUND:")
            
            if results['pass_statements']:
                print("\n🔍 Pass Statements:")
                for item in results['pass_statements']:
                    print(f"   {item['file']}:{item['line']} - {item['context']}")
                    
            if results['not_implemented_errors']:
                print("\n🔍 NotImplementedError:")
                for item in results['not_implemented_errors']:
                    print(f"   {item['file']}:{item['line']} - {item['context']}")
                    
            if results['mock_objects']:
                print("\n🔍 Mock Objects:")
                for item in results['mock_objects']:
                    print(f"   {item['file']}:{item['line']} - {item['name']} in {item['context']}")
                    
        return results['summary']['status'] == 'PASSED'
        
    def save_report(self, filename: str = "truth_validation_report.json"):
        """Save validation results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n📄 Report saved to: {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Truth-first validation framework')
    parser.add_argument('--scan', action='store_true', help='Scan for stubs and mocks')
    parser.add_argument('--validate', action='store_true', help='Validate codebase')
    parser.add_argument('--directory', default='src', help='Directory to scan')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    
    args = parser.parse_args()
    
    validator = TruthValidator()
    
    if args.scan or args.validate:
        success = validator.validate()
        if args.save:
            validator.save_report()
            
        if not success:
            sys.exit(1)
    else:
        print("Use --scan or --validate to run validation")
        print("Example: python tools/truth_validator.py --scan --save")


if __name__ == "__main__":
    main()
