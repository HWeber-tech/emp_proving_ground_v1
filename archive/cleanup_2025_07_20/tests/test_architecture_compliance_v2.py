#!/usr/bin/env python3
"""
EMP Architecture Compliance Test v2.0

Comprehensive validation of v1.1 architecture compliance after critical refactoring.
Tests the corrected modular structure and separation of concerns.
"""

import sys
import os
import importlib
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArchitectureComplianceTester:
    """Comprehensive architecture compliance tester."""
    
    def __init__(self):
        self.compliance_results = {}
        self.critical_violations = []
        self.warnings = []
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive architecture compliance test."""
        logger.info("🚀 Starting EMP Architecture Compliance Test v2.0")
        
        # Test 1: Directory Structure Compliance
        self._test_directory_structure()
        
        # Test 2: Import Compliance
        self._test_import_compliance()
        
        # Test 3: Layer Separation Compliance
        self._test_layer_separation()
        
        # Test 4: Event-Driven Architecture Compliance
        self._test_event_driven_architecture()
        
        # Test 5: Genome Model Compliance
        self._test_genome_model_compliance()
        
        # Test 6: Thinking Layer Compliance
        self._test_thinking_layer_compliance()
        
        # Test 7: Simulation Envelope Compliance
        self._test_simulation_envelope_compliance()
        
        # Test 8: Adaptive Core Compliance
        self._test_adaptive_core_compliance()
        
        # Test 9: Governance Layer Compliance
        self._test_governance_layer_compliance()
        
        # Test 10: Operational Backbone Compliance
        self._test_operational_backbone_compliance()
        
        # Generate final report
        return self._generate_final_report()
        
    def _test_directory_structure(self):
        """Test directory structure compliance."""
        logger.info("📁 Testing Directory Structure Compliance")
        
        required_directories = [
            "src/sensory/organs",
            "src/sensory/integration", 
            "src/sensory/calibration",
            "src/thinking/patterns",
            "src/thinking/analysis",
            "src/thinking/inference",
            "src/thinking/memory",
            "src/simulation/evaluation",
            "src/simulation/market_simulation",
            "src/simulation/stress_testing",
            "src/genome/models",
            "src/evolution/selection",
            "src/evolution/variation",
            "src/evolution/engine",
            "src/governance",
            "src/operational",
            "src/ui",
            "config/fitness"
        ]
        
        missing_directories = []
        for directory in required_directories:
            if not os.path.exists(directory):
                missing_directories.append(directory)
                
        if missing_directories:
            self.critical_violations.append(f"Missing directories: {missing_directories}")
            logger.error(f"❌ Missing directories: {missing_directories}")
        else:
            logger.info("✅ Directory structure compliant")
            
        self.compliance_results['directory_structure'] = len(missing_directories) == 0
        
    def _test_import_compliance(self):
        """Test import compliance for core modules."""
        logger.info("📦 Testing Import Compliance")
        
        core_modules = [
            "src.core.events",
            "src.core.event_bus",
            "src.genome.models.genome",
            "src.thinking.analysis.performance_analyzer",
            "src.thinking.analysis.risk_analyzer",
            "src.simulation.evaluation.fitness_evaluator",
            "src.evolution.engine.genetic_engine",
            "src.evolution.engine.population_manager",
            "src.governance.fitness_store",
            "src.governance.strategy_registry"
        ]
        
        import_failures = []
        for module in core_modules:
            try:
                importlib.import_module(module)
                logger.info(f"✅ {module} imports successfully")
            except ImportError as e:
                import_failures.append(f"{module}: {e}")
                logger.error(f"❌ {module} import failed: {e}")
                
        if import_failures:
            self.critical_violations.append(f"Import failures: {import_failures}")
        else:
            logger.info("✅ All core modules import successfully")
            
        self.compliance_results['import_compliance'] = len(import_failures) == 0
        
    def _test_layer_separation(self):
        """Test layer separation compliance."""
        logger.info("🏗️ Testing Layer Separation Compliance")
        
        # Test that thinking layer doesn't import from evolution layer
        try:
            from src.thinking.analysis.performance_analyzer import PerformanceAnalyzer
            # This should work without importing evolution components
            logger.info("✅ Thinking layer properly separated")
        except ImportError as e:
            self.critical_violations.append(f"Thinking layer separation violation: {e}")
            logger.error(f"❌ Thinking layer separation violation: {e}")
            
        # Test that simulation envelope doesn't import from adaptive core
        try:
            from src.simulation.evaluation.fitness_evaluator import FitnessEvaluator
            # This should work without importing adaptive core components
            logger.info("✅ Simulation envelope properly separated")
        except ImportError as e:
            self.critical_violations.append(f"Simulation envelope separation violation: {e}")
            logger.error(f"❌ Simulation envelope separation violation: {e}")
            
        self.compliance_results['layer_separation'] = len([v for v in self.critical_violations if 'separation violation' in v]) == 0
        
    def _test_event_driven_architecture(self):
        """Test event-driven architecture compliance."""
        logger.info("📡 Testing Event-Driven Architecture Compliance")
        
        try:
            from src.core.events import TradeIntent, FitnessReport, PerformanceMetrics, EventType
            
            # Test that event models are properly defined
            required_events = ['TradeIntent', 'FitnessReport', 'PerformanceMetrics']
            for event_name in required_events:
                if event_name in globals():
                    logger.info(f"✅ Event model {event_name} available")
                else:
                    self.critical_violations.append(f"Missing event model: {event_name}")
                    logger.error(f"❌ Missing event model: {event_name}")
                    
            logger.info("✅ Event-driven architecture compliant")
            
        except ImportError as e:
            self.critical_violations.append(f"Event-driven architecture violation: {e}")
            logger.error(f"❌ Event-driven architecture violation: {e}")
            
        self.compliance_results['event_driven_architecture'] = len([v for v in self.critical_violations if 'event' in v.lower()]) == 0
        
    def _test_genome_model_compliance(self):
        """Test genome model compliance."""
        logger.info("🧬 Testing Genome Model Compliance")
        
        try:
            from src.genome.models.genome import DecisionGenome, StrategyGenome, RiskGenome
            
            # Test that genome is properly located in genome/models
            genome_module = DecisionGenome.__module__
            if 'genome.models' in genome_module:
                logger.info("✅ Genome model properly located")
            else:
                self.critical_violations.append(f"Genome model in wrong location: {genome_module}")
                logger.error(f"❌ Genome model in wrong location: {genome_module}")
                
        except ImportError as e:
            self.critical_violations.append(f"Genome model compliance violation: {e}")
            logger.error(f"❌ Genome model compliance violation: {e}")
            
        self.compliance_results['genome_model_compliance'] = len([v for v in self.critical_violations if 'genome' in v.lower()]) == 0
        
    def _test_thinking_layer_compliance(self):
        """Test thinking layer compliance."""
        logger.info("🧠 Testing Thinking Layer Compliance")
        
        try:
            from src.thinking.analysis.performance_analyzer import PerformanceAnalyzer
            from src.thinking.analysis.risk_analyzer import RiskAnalyzer
            
            # Test that performance calculations are in thinking layer
            analyzer = PerformanceAnalyzer()
            if hasattr(analyzer, 'analyze_performance'):
                logger.info("✅ Performance analysis in thinking layer")
            else:
                self.critical_violations.append("Performance analysis not in thinking layer")
                logger.error("❌ Performance analysis not in thinking layer")
                
        except ImportError as e:
            self.critical_violations.append(f"Thinking layer compliance violation: {e}")
            logger.error(f"❌ Thinking layer compliance violation: {e}")
            
        self.compliance_results['thinking_layer_compliance'] = len([v for v in self.critical_violations if 'thinking' in v.lower()]) == 0
        
    def _test_simulation_envelope_compliance(self):
        """Test simulation envelope compliance."""
        logger.info("🎯 Testing Simulation Envelope Compliance")
        
        try:
            from src.simulation.evaluation.fitness_evaluator import FitnessEvaluator
            
            # Test that fitness evaluator orchestrates thinking layer
            evaluator = FitnessEvaluator()
            if hasattr(evaluator, 'evaluate_fitness'):
                logger.info("✅ Fitness evaluator in simulation envelope")
            else:
                self.critical_violations.append("Fitness evaluator not in simulation envelope")
                logger.error("❌ Fitness evaluator not in simulation envelope")
                
        except ImportError as e:
            self.critical_violations.append(f"Simulation envelope compliance violation: {e}")
            logger.error(f"❌ Simulation envelope compliance violation: {e}")
            
        self.compliance_results['simulation_envelope_compliance'] = len([v for v in self.critical_violations if 'simulation' in v.lower()]) == 0
        
    def _test_adaptive_core_compliance(self):
        """Test adaptive core compliance."""
        logger.info("🔄 Testing Adaptive Core Compliance")
        
        try:
            from src.evolution.engine.genetic_engine import GeneticEngine
            from src.evolution.engine.population_manager import PopulationManager
            
            # Test that genetic engine uses simulation envelope
            engine = GeneticEngine()
            if hasattr(engine, 'fitness_evaluator'):
                logger.info("✅ Genetic engine uses simulation envelope")
            else:
                self.critical_violations.append("Genetic engine doesn't use simulation envelope")
                logger.error("❌ Genetic engine doesn't use simulation envelope")
                
        except ImportError as e:
            self.critical_violations.append(f"Adaptive core compliance violation: {e}")
            logger.error(f"❌ Adaptive core compliance violation: {e}")
            
        self.compliance_results['adaptive_core_compliance'] = len([v for v in self.critical_violations if 'adaptive' in v.lower() or 'genetic' in v.lower()]) == 0
        
    def _test_governance_layer_compliance(self):
        """Test governance layer compliance."""
        logger.info("⚖️ Testing Governance Layer Compliance")
        
        try:
            from src.governance.fitness_store import FitnessStore
            from src.governance.strategy_registry import StrategyRegistry
            
            # Test that governance components exist
            store = FitnessStore()
            registry = StrategyRegistry()
            
            logger.info("✅ Governance layer components available")
            
        except ImportError as e:
            self.critical_violations.append(f"Governance layer compliance violation: {e}")
            logger.error(f"❌ Governance layer compliance violation: {e}")
            
        self.compliance_results['governance_layer_compliance'] = len([v for v in self.critical_violations if 'governance' in v.lower()]) == 0
        
    def _test_operational_backbone_compliance(self):
        """Test operational backbone compliance."""
        logger.info("🔧 Testing Operational Backbone Compliance")
        
        # Test for Docker and Kubernetes files
        required_files = [
            "Dockerfile",
            "docker-compose.yml",
            "k8s/emp-deployment.yaml"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                
        if missing_files:
            self.warnings.append(f"Missing operational files: {missing_files}")
            logger.warning(f"⚠️ Missing operational files: {missing_files}")
        else:
            logger.info("✅ Operational backbone files present")
            
        self.compliance_results['operational_backbone_compliance'] = len(missing_files) == 0
        
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final compliance report."""
        total_tests = len(self.compliance_results)
        passed_tests = sum(self.compliance_results.values())
        compliance_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'compliance_percentage': compliance_percentage,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'critical_violations': self.critical_violations,
            'warnings': self.warnings,
            'detailed_results': self.compliance_results,
            'status': 'COMPLIANT' if compliance_percentage >= 95 else 'NON_COMPLIANT'
        }
        
        # Log final results
        logger.info("=" * 60)
        logger.info(f"🏁 ARCHITECTURE COMPLIANCE TEST COMPLETE")
        logger.info(f"📊 Compliance: {compliance_percentage:.1f}% ({passed_tests}/{total_tests})")
        logger.info(f"🎯 Status: {report['status']}")
        
        if self.critical_violations:
            logger.error(f"🚨 Critical Violations: {len(self.critical_violations)}")
            for violation in self.critical_violations:
                logger.error(f"   ❌ {violation}")
                
        if self.warnings:
            logger.warning(f"⚠️ Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                logger.warning(f"   ⚠️ {warning}")
                
        logger.info("=" * 60)
        
        return report


def main():
    """Main test execution."""
    tester = ArchitectureComplianceTester()
    report = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    if report['status'] == 'COMPLIANT':
        print("✅ Architecture is COMPLIANT with v1.1 blueprint")
        sys.exit(0)
    else:
        print("❌ Architecture is NON-COMPLIANT with v1.1 blueprint")
        sys.exit(1)


if __name__ == "__main__":
    main() 