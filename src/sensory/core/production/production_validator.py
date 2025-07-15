"""
Production Validator - Anti-Simulation Enforcement

This module implements comprehensive validation to detect and prevent
any simulation, mock, or fake code from running in production.
ZERO TOLERANCE FOR SIMULATION.
"""

import inspect
import logging
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import ast
import importlib

logger = logging.getLogger(__name__)

class ViolationType(Enum):
    """Types of simulation violations"""
    SIMULATION_METHOD = "simulation_method"
    MOCK_DATA = "mock_data"
    RANDOM_GENERATION = "random_generation"
    FAKE_API = "fake_api"
    SYNTHETIC_DATA = "synthetic_data"
    TEST_CODE_IN_PROD = "test_code_in_production"

@dataclass
class SimulationViolation:
    """Detected simulation violation"""
    violation_type: ViolationType
    description: str
    location: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    detected_at: datetime
    stack_trace: Optional[str] = None

class ProductionError(Exception):
    """Raised when simulation code is detected in production"""
    pass

class ProductionValidator:
    """
    Comprehensive validator to prevent simulation code in production
    Implements multiple layers of detection and prevention
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize production validator
        
        Args:
            strict_mode: If True, any simulation detection raises exception
        """
        self.strict_mode = strict_mode
        self.violations = []
        
        # Forbidden patterns that indicate simulation
        self.forbidden_patterns = {
            'method_names': [
                r'.*simulate.*',
                r'.*mock.*',
                r'.*fake.*',
                r'.*dummy.*',
                r'.*synthetic.*',
                r'.*generate.*random.*',
                r'.*random.*data.*',
                r'.*test.*data.*',
                r'.*artificial.*'
            ],
            'variable_names': [
                r'.*simulated.*',
                r'.*mocked.*',
                r'.*fake.*',
                r'.*dummy.*',
                r'.*synthetic.*',
                r'.*random.*',
                r'.*test.*'
            ],
            'string_literals': [
                r'.*simulation.*',
                r'.*mock.*',
                r'.*fake.*',
                r'.*dummy.*',
                r'.*synthetic.*',
                r'.*test.*data.*',
                r'.*demo.*api.*'
            ],
            'imports': [
                'random',
                'faker',
                'mock',
                'unittest.mock',
                'pytest-mock'
            ]
        }
        
        # Real data source indicators
        self.real_data_indicators = [
            'fred.stlouisfed.org',
            'api.dukascopy.com',
            'api.ibkr.com',
            'bloomberg.com',
            'reuters.com',
            'marketdata.tradermade.com'
        ]
        
        # Production environment indicators
        self.production_indicators = [
            'prod',
            'production',
            'live',
            'real'
        ]
    
    def validate_function(self, func: callable) -> List[SimulationViolation]:
        """
        Validate a function for simulation code
        
        Args:
            func: Function to validate
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Get function source code
        try:
            source = inspect.getsource(func)
            func_name = func.__name__
            module_name = func.__module__
            location = f"{module_name}.{func_name}"
            
        except (OSError, TypeError):
            # Can't get source code, skip validation
            return violations
        
        # Check function name
        for pattern in self.forbidden_patterns['method_names']:
            if re.match(pattern, func_name, re.IGNORECASE):
                violations.append(SimulationViolation(
                    violation_type=ViolationType.SIMULATION_METHOD,
                    description=f"Function name indicates simulation: {func_name}",
                    location=location,
                    severity="CRITICAL",
                    detected_at=datetime.now()
                ))
        
        # Parse source code for violations
        try:
            tree = ast.parse(source)
            violations.extend(self._analyze_ast(tree, location))
        except SyntaxError:
            # Can't parse source, log warning
            logger.warning(f"Could not parse source code for {location}")
        
        return violations
    
    def validate_class(self, cls: type) -> List[SimulationViolation]:
        """
        Validate a class for simulation code
        
        Args:
            cls: Class to validate
            
        Returns:
            List of violations found
        """
        violations = []
        
        class_name = cls.__name__
        module_name = cls.__module__
        location = f"{module_name}.{class_name}"
        
        # Check class name
        for pattern in self.forbidden_patterns['method_names']:
            if re.match(pattern, class_name, re.IGNORECASE):
                violations.append(SimulationViolation(
                    violation_type=ViolationType.SIMULATION_METHOD,
                    description=f"Class name indicates simulation: {class_name}",
                    location=location,
                    severity="CRITICAL",
                    detected_at=datetime.now()
                ))
        
        # Validate all methods
        for method_name in dir(cls):
            if not method_name.startswith('_'):  # Skip private methods
                try:
                    method = getattr(cls, method_name)
                    if callable(method):
                        method_violations = self.validate_function(method)
                        violations.extend(method_violations)
                except AttributeError:
                    continue
        
        return violations
    
    def validate_module(self, module_name: str) -> List[SimulationViolation]:
        """
        Validate an entire module for simulation code
        
        Args:
            module_name: Name of module to validate
            
        Returns:
            List of violations found
        """
        violations = []
        
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            logger.error(f"Could not import module: {module_name}")
            return violations
        
        # Check module-level attributes
        for attr_name in dir(module):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr = getattr(module, attr_name)
                    
                    if callable(attr):
                        # Validate functions
                        attr_violations = self.validate_function(attr)
                        violations.extend(attr_violations)
                    
                    elif inspect.isclass(attr):
                        # Validate classes
                        class_violations = self.validate_class(attr)
                        violations.extend(class_violations)
                        
                except AttributeError:
                    continue
        
        return violations
    
    def _analyze_ast(self, tree: ast.AST, location: str) -> List[SimulationViolation]:
        """
        Analyze AST for simulation patterns
        
        Args:
            tree: AST to analyze
            location: Location identifier
            
        Returns:
            List of violations found
        """
        violations = []
        
        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                violations.extend(self._check_function_call(node, location))
            
            # Check imports
            elif isinstance(node, ast.Import):
                violations.extend(self._check_import(node, location))
            
            elif isinstance(node, ast.ImportFrom):
                violations.extend(self._check_import_from(node, location))
            
            # Check string literals
            elif isinstance(node, ast.Str):
                violations.extend(self._check_string_literal(node, location))
            
            # Check variable assignments
            elif isinstance(node, ast.Assign):
                violations.extend(self._check_assignment(node, location))
        
        return violations
    
    def _check_function_call(self, node: ast.Call, location: str) -> List[SimulationViolation]:
        """Check function call for simulation patterns"""
        violations = []
        
        # Get function name
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        # Check for forbidden function names
        for pattern in self.forbidden_patterns['method_names']:
            if re.match(pattern, func_name, re.IGNORECASE):
                violations.append(SimulationViolation(
                    violation_type=ViolationType.SIMULATION_METHOD,
                    description=f"Simulation function call detected: {func_name}",
                    location=location,
                    severity="CRITICAL",
                    detected_at=datetime.now()
                ))
        
        # Check for random module usage
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in ['random', 'np'] and 'random' in func_name:
                    violations.append(SimulationViolation(
                        violation_type=ViolationType.RANDOM_GENERATION,
                        description=f"Random data generation detected: {node.func.value.id}.{func_name}",
                        location=location,
                        severity="HIGH",
                        detected_at=datetime.now()
                    ))
        
        return violations
    
    def _check_import(self, node: ast.Import, location: str) -> List[SimulationViolation]:
        """Check import statement for forbidden modules"""
        violations = []
        
        for alias in node.names:
            if alias.name in self.forbidden_patterns['imports']:
                violations.append(SimulationViolation(
                    violation_type=ViolationType.SIMULATION_METHOD,
                    description=f"Forbidden import detected: {alias.name}",
                    location=location,
                    severity="HIGH",
                    detected_at=datetime.now()
                ))
        
        return violations
    
    def _check_import_from(self, node: ast.ImportFrom, location: str) -> List[SimulationViolation]:
        """Check from-import statement for forbidden modules"""
        violations = []
        
        if node.module in self.forbidden_patterns['imports']:
            violations.append(SimulationViolation(
                violation_type=ViolationType.SIMULATION_METHOD,
                description=f"Forbidden from-import detected: from {node.module}",
                location=location,
                severity="HIGH",
                detected_at=datetime.now()
            ))
        
        return violations
    
    def _check_string_literal(self, node: ast.Str, location: str) -> List[SimulationViolation]:
        """Check string literal for simulation indicators"""
        violations = []
        
        string_value = node.s.lower()
        
        for pattern in self.forbidden_patterns['string_literals']:
            if re.search(pattern, string_value, re.IGNORECASE):
                violations.append(SimulationViolation(
                    violation_type=ViolationType.MOCK_DATA,
                    description=f"Simulation indicator in string: {node.s[:50]}...",
                    location=location,
                    severity="MEDIUM",
                    detected_at=datetime.now()
                ))
        
        return violations
    
    def _check_assignment(self, node: ast.Assign, location: str) -> List[SimulationViolation]:
        """Check variable assignment for simulation patterns"""
        violations = []
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                
                for pattern in self.forbidden_patterns['variable_names']:
                    if re.match(pattern, var_name, re.IGNORECASE):
                        violations.append(SimulationViolation(
                            violation_type=ViolationType.SIMULATION_METHOD,
                            description=f"Simulation variable detected: {var_name}",
                            location=location,
                            severity="MEDIUM",
                            detected_at=datetime.now()
                        ))
        
        return violations
    
    def validate_data_source(self, data_source: Any, source_name: str) -> List[SimulationViolation]:
        """
        Validate data source for real vs simulation
        
        Args:
            data_source: Data source object to validate
            source_name: Name of the data source
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Check source name for simulation indicators
        source_lower = source_name.lower()
        for pattern in self.forbidden_patterns['string_literals']:
            if re.search(pattern, source_lower):
                violations.append(SimulationViolation(
                    violation_type=ViolationType.FAKE_API,
                    description=f"Simulation indicator in data source name: {source_name}",
                    location=f"data_source.{source_name}",
                    severity="CRITICAL",
                    detected_at=datetime.now()
                ))
        
        # Check for real data source indicators
        has_real_indicator = False
        for indicator in self.real_data_indicators:
            if indicator in source_lower:
                has_real_indicator = True
                break
        
        if not has_real_indicator:
            violations.append(SimulationViolation(
                violation_type=ViolationType.FAKE_API,
                description=f"No real data source indicators found in: {source_name}",
                location=f"data_source.{source_name}",
                severity="HIGH",
                detected_at=datetime.now()
            ))
        
        # Check data source object for simulation methods
        if hasattr(data_source, '__dict__'):
            for attr_name in dir(data_source):
                for pattern in self.forbidden_patterns['method_names']:
                    if re.match(pattern, attr_name, re.IGNORECASE):
                        violations.append(SimulationViolation(
                            violation_type=ViolationType.SIMULATION_METHOD,
                            description=f"Simulation method in data source: {attr_name}",
                            location=f"data_source.{source_name}.{attr_name}",
                            severity="CRITICAL",
                            detected_at=datetime.now()
                        ))
        
        return violations
    
    def validate_environment(self) -> List[SimulationViolation]:
        """
        Validate environment for production readiness
        
        Returns:
            List of violations found
        """
        violations = []
        
        # Check environment variables
        import os
        
        env_vars = dict(os.environ)
        
        # Look for test/dev environment indicators
        for key, value in env_vars.items():
            key_lower = key.lower()
            value_lower = value.lower()
            
            if any(indicator in key_lower for indicator in ['test', 'dev', 'mock', 'fake']):
                violations.append(SimulationViolation(
                    violation_type=ViolationType.TEST_CODE_IN_PROD,
                    description=f"Test environment variable detected: {key}",
                    location="environment",
                    severity="HIGH",
                    detected_at=datetime.now()
                ))
            
            if any(indicator in value_lower for indicator in ['test', 'dev', 'mock', 'fake', 'demo']):
                violations.append(SimulationViolation(
                    violation_type=ViolationType.TEST_CODE_IN_PROD,
                    description=f"Test value in environment variable {key}: {value}",
                    location="environment",
                    severity="MEDIUM",
                    detected_at=datetime.now()
                ))
        
        # Check for production indicators
        has_prod_indicator = False
        for key, value in env_vars.items():
            if any(indicator in key.lower() for indicator in self.production_indicators):
                has_prod_indicator = True
                break
            if any(indicator in value.lower() for indicator in self.production_indicators):
                has_prod_indicator = True
                break
        
        if not has_prod_indicator:
            violations.append(SimulationViolation(
                violation_type=ViolationType.TEST_CODE_IN_PROD,
                description="No production environment indicators found",
                location="environment",
                severity="MEDIUM",
                detected_at=datetime.now()
            ))
        
        return violations
    
    def enforce_production_mode(self, violations: List[SimulationViolation]):
        """
        Enforce production mode by raising exceptions for violations
        
        Args:
            violations: List of violations to enforce
        """
        if not violations:
            return
        
        # Categorize violations by severity
        critical_violations = [v for v in violations if v.severity == "CRITICAL"]
        high_violations = [v for v in violations if v.severity == "HIGH"]
        
        if critical_violations:
            violation_details = "\n".join([
                f"- {v.description} at {v.location}" 
                for v in critical_violations
            ])
            
            raise ProductionError(
                f"CRITICAL simulation violations detected in production:\n{violation_details}"
            )
        
        if self.strict_mode and high_violations:
            violation_details = "\n".join([
                f"- {v.description} at {v.location}" 
                for v in high_violations
            ])
            
            raise ProductionError(
                f"HIGH severity simulation violations detected in strict mode:\n{violation_details}"
            )
    
    def generate_violation_report(self, violations: List[SimulationViolation]) -> str:
        """
        Generate detailed violation report
        
        Args:
            violations: List of violations to report
            
        Returns:
            Formatted violation report
        """
        if not violations:
            return "✅ NO SIMULATION VIOLATIONS DETECTED - PRODUCTION READY"
        
        report = "🚨 SIMULATION VIOLATIONS DETECTED\n"
        report += "=" * 50 + "\n\n"
        
        # Group by severity
        by_severity = {}
        for violation in violations:
            if violation.severity not in by_severity:
                by_severity[violation.severity] = []
            by_severity[violation.severity].append(violation)
        
        # Report by severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                report += f"{severity} VIOLATIONS ({len(by_severity[severity])}):\n"
                report += "-" * 30 + "\n"
                
                for violation in by_severity[severity]:
                    report += f"• {violation.description}\n"
                    report += f"  Location: {violation.location}\n"
                    report += f"  Type: {violation.violation_type.value}\n"
                    report += f"  Detected: {violation.detected_at}\n\n"
        
        # Summary
        report += f"TOTAL VIOLATIONS: {len(violations)}\n"
        report += f"PRODUCTION STATUS: {'❌ BLOCKED' if violations else '✅ APPROVED'}\n"
        
        return report
    
    def validate_production_readiness(self, target_object: Any, object_name: str) -> bool:
        """
        Comprehensive production readiness validation
        
        Args:
            target_object: Object to validate
            object_name: Name of the object
            
        Returns:
            True if production ready, False otherwise
        """
        all_violations = []
        
        # Validate the object itself
        if callable(target_object):
            all_violations.extend(self.validate_function(target_object))
        elif inspect.isclass(target_object):
            all_violations.extend(self.validate_class(target_object))
        
        # Validate environment
        all_violations.extend(self.validate_environment())
        
        # Store violations
        self.violations.extend(all_violations)
        
        # Generate report
        report = self.generate_violation_report(all_violations)
        logger.info(f"Production validation report for {object_name}:\n{report}")
        
        # Enforce production mode
        if self.strict_mode:
            self.enforce_production_mode(all_violations)
        
        return len(all_violations) == 0

# Global production validator instance
production_validator = ProductionValidator(strict_mode=True)

def validate_production_ready(func):
    """
    Decorator to validate function is production ready
    
    Args:
        func: Function to validate
        
    Returns:
        Wrapped function that validates before execution
    """
    def wrapper(*args, **kwargs):
        # Validate function before execution
        violations = production_validator.validate_function(func)
        
        if violations:
            production_validator.enforce_production_mode(violations)
        
        return func(*args, **kwargs)
    
    return wrapper

def validate_data_source_real(source_name: str, source_obj: Any):
    """
    Validate that data source is real and not simulated
    
    Args:
        source_name: Name of the data source
        source_obj: Data source object
        
    Raises:
        ProductionError: If simulation detected
    """
    violations = production_validator.validate_data_source(source_obj, source_name)
    
    if violations:
        production_validator.enforce_production_mode(violations)

# Example usage
if __name__ == "__main__":
    # Test the validator
    validator = ProductionValidator(strict_mode=True)
    
    # This would trigger violations
    def simulate_market_data():
        import random
        return random.uniform(1.0, 2.0)
    
    violations = validator.validate_function(simulate_market_data)
    report = validator.generate_violation_report(violations)
    print(report)

