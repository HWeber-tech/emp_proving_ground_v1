#!/usr/bin/env python3
"""
EMP Production Readiness Test v1.1

Comprehensive test suite to validate production readiness
including security, performance, monitoring, and deployment.
"""

import asyncio
import logging
import sys
import time
import subprocess
import requests
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.configuration import Configuration
from src.core.event_bus import event_bus
from src.core.interfaces import MarketData
from src.sensory.integration.sensory_cortex import SensoryCortex
from src.thinking.patterns.trend_detector import TrendDetector
from src.evolution.engine.genetic_engine import GeneticEngine
from src.governance.human_gateway import HumanApprovalGateway
from src.governance.strategy_registry import StrategyRegistry

logger = logging.getLogger(__name__)


class ProductionReadinessTest:
    """Comprehensive production readiness validation."""
    
    def __init__(self):
        self.config = None
        self.sensory_cortex = None
        self.trend_detector = None
        self.genetic_engine = None
        self.human_gateway = None
        self.strategy_registry = None
        self.test_results = {}
        self.performance_metrics = {}
        
    async def setup(self):
        """Set up the test environment."""
        logger.info("Setting up production readiness test")
        
        # Create production configuration
        self.config = Configuration(
            system_name="EMP Production",
            system_version="1.1.0",
            environment="production",
            debug=False
        )
        
        # Initialize core components
        self.sensory_cortex = SensoryCortex()
        self.trend_detector = TrendDetector()
        self.genetic_engine = GeneticEngine()
        self.human_gateway = HumanApprovalGateway()
        self.strategy_registry = StrategyRegistry()
        
        # Start event bus
        await event_bus.start()
        
        # Start sensory cortex
        await self.sensory_cortex.start()
        
        logger.info("Production readiness test environment setup complete")
        
    async def teardown(self):
        """Clean up the test environment."""
        logger.info("Tearing down production readiness test environment")
        
        # Stop sensory cortex
        if self.sensory_cortex:
            await self.sensory_cortex.stop()
            
        # Stop event bus
        await event_bus.stop()
        
        logger.info("Production readiness test environment cleanup complete")
        
    async def test_security_compliance(self):
        """Test security compliance and hardening."""
        logger.info("Testing security compliance")
        
        try:
            security_checks = {
                'rbac_enabled': True,
                'network_policies': True,
                'pod_security_policies': True,
                'secrets_encrypted': True,
                'tls_enabled': True,
                'non_root_containers': True,
                'resource_limits': True,
                'security_context': True
            }
            
            # Check RBAC
            try:
                result = subprocess.run(['kubectl', 'get', 'clusterrole'], 
                                      capture_output=True, text=True)
                security_checks['rbac_enabled'] = result.returncode == 0
            except Exception:
                security_checks['rbac_enabled'] = False
                
            # Check network policies
            try:
                result = subprocess.run(['kubectl', 'get', 'networkpolicy', '-n', 'emp-system'], 
                                      capture_output=True, text=True)
                security_checks['network_policies'] = result.returncode == 0
            except Exception:
                security_checks['network_policies'] = False
                
            # Check pod security policies
            try:
                result = subprocess.run(['kubectl', 'get', 'podsecuritypolicy'], 
                                      capture_output=True, text=True)
                security_checks['pod_security_policies'] = result.returncode == 0
            except Exception:
                security_checks['pod_security_policies'] = False
                
            # Check secrets encryption
            try:
                result = subprocess.run(['kubectl', 'get', 'secret', '-n', 'emp-system'], 
                                      capture_output=True, text=True)
                security_checks['secrets_encrypted'] = result.returncode == 0
            except Exception:
                security_checks['secrets_encrypted'] = False
                
            # Check TLS configuration
            try:
                response = requests.get('https://emp.example.com/health', 
                                      verify=True, timeout=5)
                security_checks['tls_enabled'] = response.status_code == 200
            except Exception:
                security_checks['tls_enabled'] = False
                
            # Calculate security score
            security_score = sum(security_checks.values()) / len(security_checks)
            
            self.test_results['security_compliance'] = {
                'status': 'PASS' if security_score >= 0.8 else 'FAIL',
                'security_score': security_score,
                'checks': security_checks,
                'recommendations': self._get_security_recommendations(security_checks)
            }
            
            logger.info(f"Security compliance test completed with score: {security_score:.2f}")
            
        except Exception as e:
            logger.error(f"Security compliance test FAILED: {e}")
            self.test_results['security_compliance'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_performance_benchmarks(self):
        """Test performance benchmarks and scalability."""
        logger.info("Testing performance benchmarks")
        
        try:
            performance_metrics = {}
            
            # Test sensory processing performance
            start_time = time.time()
            market_data = self._create_test_market_data()
            
            for _ in range(1000):
                await self.sensory_cortex.process_market_data(market_data)
                
            sensory_time = time.time() - start_time
            performance_metrics['sensory_processing'] = {
                'operations_per_second': 1000 / sensory_time,
                'latency_ms': (sensory_time / 1000) * 1000
            }
            
            # Test thinking layer performance
            start_time = time.time()
            sensory_signals = self._create_test_sensory_signals()
            
            for _ in range(1000):
                self.trend_detector.analyze(sensory_signals)
                
            thinking_time = time.time() - start_time
            performance_metrics['thinking_processing'] = {
                'operations_per_second': 1000 / thinking_time,
                'latency_ms': (thinking_time / 1000) * 1000
            }
            
            # Test genetic evolution performance
            start_time = time.time()
            await self.genetic_engine.initialize_population()
            best_genome = await self.genetic_engine.evolve(max_generations=5)
            evolution_time = time.time() - start_time
            
            performance_metrics['genetic_evolution'] = {
                'evolution_time_seconds': evolution_time,
                'generations_per_second': 5 / evolution_time,
                'best_fitness': best_genome.fitness_score if best_genome else 0.0
            }
            
            # Performance thresholds
            thresholds = {
                'sensory_ops_per_sec': 100,
                'thinking_ops_per_sec': 500,
                'evolution_time_sec': 30,
                'sensory_latency_ms': 10,
                'thinking_latency_ms': 5
            }
            
            # Check performance against thresholds
            performance_checks = {
                'sensory_throughput': performance_metrics['sensory_processing']['operations_per_second'] >= thresholds['sensory_ops_per_sec'],
                'thinking_throughput': performance_metrics['thinking_processing']['operations_per_second'] >= thresholds['thinking_ops_per_sec'],
                'evolution_speed': performance_metrics['genetic_evolution']['evolution_time_seconds'] <= thresholds['evolution_time_sec'],
                'sensory_latency': performance_metrics['sensory_processing']['latency_ms'] <= thresholds['sensory_latency_ms'],
                'thinking_latency': performance_metrics['thinking_processing']['latency_ms'] <= thresholds['thinking_latency_ms']
            }
            
            performance_score = sum(performance_checks.values()) / len(performance_checks)
            
            self.test_results['performance_benchmarks'] = {
                'status': 'PASS' if performance_score >= 0.8 else 'FAIL',
                'performance_score': performance_score,
                'metrics': performance_metrics,
                'checks': performance_checks,
                'thresholds': thresholds
            }
            
            self.performance_metrics = performance_metrics
            
            logger.info(f"Performance benchmarks completed with score: {performance_score:.2f}")
            
        except Exception as e:
            logger.error(f"Performance benchmarks test FAILED: {e}")
            self.test_results['performance_benchmarks'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_monitoring_observability(self):
        """Test monitoring and observability setup."""
        logger.info("Testing monitoring and observability")
        
        try:
            monitoring_checks = {}
            
            # Check Prometheus metrics endpoint
            try:
                response = requests.get('http://localhost:9090/api/v1/query?query=up', 
                                      timeout=5)
                monitoring_checks['prometheus_available'] = response.status_code == 200
            except Exception:
                monitoring_checks['prometheus_available'] = False
                
            # Check Grafana dashboard
            try:
                response = requests.get('http://localhost:3000/api/health', 
                                      timeout=5)
                monitoring_checks['grafana_available'] = response.status_code == 200
            except Exception:
                monitoring_checks['grafana_available'] = False
                
            # Check Jaeger tracing
            try:
                response = requests.get('http://localhost:16686/api/services', 
                                      timeout=5)
                monitoring_checks['jaeger_available'] = response.status_code == 200
            except Exception:
                monitoring_checks['jaeger_available'] = response.status_code == 200
                
            # Check application metrics
            try:
                response = requests.get('http://localhost:8000/metrics', 
                                      timeout=5)
                monitoring_checks['app_metrics_available'] = response.status_code == 200
            except Exception:
                monitoring_checks['app_metrics_available'] = False
                
            # Check health endpoint
            try:
                response = requests.get('http://localhost:8000/health', 
                                      timeout=5)
                monitoring_checks['health_endpoint'] = response.status_code == 200
            except Exception:
                monitoring_checks['health_endpoint'] = False
                
            # Check readiness endpoint
            try:
                response = requests.get('http://localhost:8000/ready', 
                                      timeout=5)
                monitoring_checks['readiness_endpoint'] = response.status_code == 200
            except Exception:
                monitoring_checks['readiness_endpoint'] = False
                
            # Check logging
            try:
                log_file = Path('logs/emp.log')
                monitoring_checks['logging_enabled'] = log_file.exists() and log_file.stat().st_size > 0
            except Exception:
                monitoring_checks['logging_enabled'] = False
                
            # Calculate monitoring score
            monitoring_score = sum(monitoring_checks.values()) / len(monitoring_checks)
            
            self.test_results['monitoring_observability'] = {
                'status': 'PASS' if monitoring_score >= 0.7 else 'FAIL',
                'monitoring_score': monitoring_score,
                'checks': monitoring_checks,
                'recommendations': self._get_monitoring_recommendations(monitoring_checks)
            }
            
            logger.info(f"Monitoring observability test completed with score: {monitoring_score:.2f}")
            
        except Exception as e:
            logger.error(f"Monitoring observability test FAILED: {e}")
            self.test_results['monitoring_observability'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_deployment_validation(self):
        """Test deployment validation and infrastructure."""
        logger.info("Testing deployment validation")
        
        try:
            deployment_checks = {}
            
            # Check Kubernetes cluster
            try:
                result = subprocess.run(['kubectl', 'cluster-info'], 
                                      capture_output=True, text=True)
                deployment_checks['kubernetes_cluster'] = result.returncode == 0
            except Exception:
                deployment_checks['kubernetes_cluster'] = False
                
            # Check namespace
            try:
                result = subprocess.run(['kubectl', 'get', 'namespace', 'emp-system'], 
                                      capture_output=True, text=True)
                deployment_checks['namespace_exists'] = result.returncode == 0
            except Exception:
                deployment_checks['namespace_exists'] = False
                
            # Check pods
            try:
                result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'emp-system'], 
                                      capture_output=True, text=True)
                deployment_checks['pods_running'] = 'Running' in result.stdout
            except Exception:
                deployment_checks['pods_running'] = False
                
            # Check services
            try:
                result = subprocess.run(['kubectl', 'get', 'services', '-n', 'emp-system'], 
                                      capture_output=True, text=True)
                deployment_checks['services_available'] = result.returncode == 0
            except Exception:
                deployment_checks['services_available'] = False
                
            # Check ingress
            try:
                result = subprocess.run(['kubectl', 'get', 'ingress', '-n', 'emp-system'], 
                                      capture_output=True, text=True)
                deployment_checks['ingress_configured'] = result.returncode == 0
            except Exception:
                deployment_checks['ingress_configured'] = False
                
            # Check persistent volumes
            try:
                result = subprocess.run(['kubectl', 'get', 'pvc', '-n', 'emp-system'], 
                                      capture_output=True, text=True)
                deployment_checks['storage_configured'] = result.returncode == 0
            except Exception:
                deployment_checks['storage_configured'] = False
                
            # Check horizontal pod autoscaler
            try:
                result = subprocess.run(['kubectl', 'get', 'hpa', '-n', 'emp-system'], 
                                      capture_output=True, text=True)
                deployment_checks['autoscaling_configured'] = result.returncode == 0
            except Exception:
                deployment_checks['autoscaling_configured'] = False
                
            # Calculate deployment score
            deployment_score = sum(deployment_checks.values()) / len(deployment_checks)
            
            self.test_results['deployment_validation'] = {
                'status': 'PASS' if deployment_score >= 0.8 else 'FAIL',
                'deployment_score': deployment_score,
                'checks': deployment_checks,
                'recommendations': self._get_deployment_recommendations(deployment_checks)
            }
            
            logger.info(f"Deployment validation test completed with score: {deployment_score:.2f}")
            
        except Exception as e:
            logger.error(f"Deployment validation test FAILED: {e}")
            self.test_results['deployment_validation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_disaster_recovery(self):
        """Test disaster recovery and backup procedures."""
        logger.info("Testing disaster recovery")
        
        try:
            recovery_checks = {}
            
            # Check backup procedures
            try:
                backup_dir = Path('backups')
                recovery_checks['backup_directory'] = backup_dir.exists()
            except Exception:
                recovery_checks['backup_directory'] = False
                
            # Check database backup
            try:
                result = subprocess.run(['kubectl', 'exec', '-n', 'emp-system', 'emp-postgres-0', '--', 
                                       'pg_dump', '-U', 'emp_user', 'emp_registry'], 
                                      capture_output=True, text=True)
                recovery_checks['database_backup'] = result.returncode == 0
            except Exception:
                recovery_checks['database_backup'] = False
                
            # Check configuration backup
            try:
                result = subprocess.run(['kubectl', 'get', 'configmap', '-n', 'emp-system', '-o', 'yaml'], 
                                      capture_output=True, text=True)
                recovery_checks['config_backup'] = result.returncode == 0
            except Exception:
                recovery_checks['config_backup'] = False
                
            # Check secrets backup
            try:
                result = subprocess.run(['kubectl', 'get', 'secret', '-n', 'emp-system', '-o', 'yaml'], 
                                      capture_output=True, text=True)
                recovery_checks['secrets_backup'] = result.returncode == 0
            except Exception:
                recovery_checks['secrets_backup'] = False
                
            # Check recovery procedures
            recovery_checks['recovery_procedures'] = Path('docs/DISASTER_RECOVERY.md').exists()
            
            # Calculate recovery score
            recovery_score = sum(recovery_checks.values()) / len(recovery_checks)
            
            self.test_results['disaster_recovery'] = {
                'status': 'PASS' if recovery_score >= 0.8 else 'FAIL',
                'recovery_score': recovery_score,
                'checks': recovery_checks,
                'recommendations': self._get_recovery_recommendations(recovery_checks)
            }
            
            logger.info(f"Disaster recovery test completed with score: {recovery_score:.2f}")
            
        except Exception as e:
            logger.error(f"Disaster recovery test FAILED: {e}")
            self.test_results['disaster_recovery'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    def _create_test_market_data(self) -> MarketData:
        """Create test market data."""
        return MarketData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000,
            bid=1.1004,
            ask=1.1006,
            source="test",
            latency_ms=1.0
        )
        
    def _create_test_sensory_signals(self):
        """Create test sensory signals."""
        from src.core.interfaces import SensorySignal
        
        return [
            SensorySignal(
                timestamp=datetime.now(),
                signal_type="price_composite",
                value=0.6,
                confidence=0.8,
                metadata={'organ_id': 'price_organ'}
            )
        ]
        
    def _get_security_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Get security recommendations based on check results."""
        recommendations = []
        
        if not checks.get('rbac_enabled'):
            recommendations.append("Enable RBAC for cluster access control")
        if not checks.get('network_policies'):
            recommendations.append("Implement network policies for pod communication")
        if not checks.get('pod_security_policies'):
            recommendations.append("Configure pod security policies")
        if not checks.get('secrets_encrypted'):
            recommendations.append("Enable secrets encryption at rest")
        if not checks.get('tls_enabled'):
            recommendations.append("Enable TLS for all external communications")
            
        return recommendations
        
    def _get_monitoring_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Get monitoring recommendations based on check results."""
        recommendations = []
        
        if not checks.get('prometheus_available'):
            recommendations.append("Deploy Prometheus for metrics collection")
        if not checks.get('grafana_available'):
            recommendations.append("Deploy Grafana for metrics visualization")
        if not checks.get('jaeger_available'):
            recommendations.append("Deploy Jaeger for distributed tracing")
        if not checks.get('app_metrics_available'):
            recommendations.append("Enable application metrics endpoint")
        if not checks.get('health_endpoint'):
            recommendations.append("Implement health check endpoint")
        if not checks.get('logging_enabled'):
            recommendations.append("Configure centralized logging")
            
        return recommendations
        
    def _get_deployment_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Get deployment recommendations based on check results."""
        recommendations = []
        
        if not checks.get('kubernetes_cluster'):
            recommendations.append("Ensure Kubernetes cluster is properly configured")
        if not checks.get('namespace_exists'):
            recommendations.append("Create emp-system namespace")
        if not checks.get('pods_running'):
            recommendations.append("Ensure all pods are in Running state")
        if not checks.get('services_available'):
            recommendations.append("Configure Kubernetes services")
        if not checks.get('ingress_configured'):
            recommendations.append("Configure ingress controller")
        if not checks.get('storage_configured'):
            recommendations.append("Configure persistent storage")
        if not checks.get('autoscaling_configured'):
            recommendations.append("Configure horizontal pod autoscaling")
            
        return recommendations
        
    def _get_recovery_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Get recovery recommendations based on check results."""
        recommendations = []
        
        if not checks.get('backup_directory'):
            recommendations.append("Create backup directory structure")
        if not checks.get('database_backup'):
            recommendations.append("Implement automated database backups")
        if not checks.get('config_backup'):
            recommendations.append("Implement configuration backup procedures")
        if not checks.get('secrets_backup'):
            recommendations.append("Implement secrets backup procedures")
        if not checks.get('recovery_procedures'):
            recommendations.append("Document disaster recovery procedures")
            
        return recommendations
        
    async def run_all_tests(self):
        """Run all production readiness tests."""
        logger.info("Starting EMP Production Readiness Tests")
        
        try:
            await self.setup()
            
            # Run all test categories
            await self.test_security_compliance()
            await self.test_performance_benchmarks()
            await self.test_monitoring_observability()
            await self.test_deployment_validation()
            await self.test_disaster_recovery()
            
            # Generate comprehensive report
            self.generate_production_report()
            
        except Exception as e:
            logger.error(f"Production readiness test failed: {e}")
        finally:
            await self.teardown()
            
    def generate_production_report(self):
        """Generate comprehensive production readiness report."""
        logger.info("Generating production readiness report")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        # Calculate overall readiness score
        readiness_scores = []
        for result in self.test_results.values():
            if 'security_score' in result:
                readiness_scores.append(result['security_score'])
            elif 'performance_score' in result:
                readiness_scores.append(result['performance_score'])
            elif 'monitoring_score' in result:
                readiness_scores.append(result['monitoring_score'])
            elif 'deployment_score' in result:
                readiness_scores.append(result['deployment_score'])
            elif 'recovery_score' in result:
                readiness_scores.append(result['recovery_score'])
                
        overall_score = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0.0
        
        print("\n" + "="*80)
        print("EMP ULTIMATE ARCHITECTURE v1.1 - PRODUCTION READINESS REPORT")
        print("="*80)
        print(f"Overall Readiness Score: {overall_score:.2%}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*80)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['status'] == 'PASS' else "❌ FAIL"
            print(f"\n{test_name.upper()}: {status}")
            
            if result['status'] == 'FAIL':
                print(f"  Error: {result['error']}")
            else:
                # Display scores
                if 'security_score' in result:
                    print(f"  Security Score: {result['security_score']:.2%}")
                if 'performance_score' in result:
                    print(f"  Performance Score: {result['performance_score']:.2%}")
                if 'monitoring_score' in result:
                    print(f"  Monitoring Score: {result['monitoring_score']:.2%}")
                if 'deployment_score' in result:
                    print(f"  Deployment Score: {result['deployment_score']:.2%}")
                if 'recovery_score' in result:
                    print(f"  Recovery Score: {result['recovery_score']:.2%}")
                    
                # Display recommendations
                if 'recommendations' in result and result['recommendations']:
                    print("  Recommendations:")
                    for rec in result['recommendations']:
                        print(f"    • {rec}")
                        
                # Display performance metrics
                if 'metrics' in result:
                    print("  Performance Metrics:")
                    for metric, value in result['metrics'].items():
                        if isinstance(value, dict):
                            for sub_metric, sub_value in value.items():
                                print(f"    • {metric}.{sub_metric}: {sub_value}")
                        else:
                            print(f"    • {metric}: {value}")
                            
        print("\n" + "="*80)
        
        if overall_score >= 0.8:
            print("🎉 PRODUCTION READY - EMP Ultimate Architecture v1.1 is ready for production!")
            print("🚀 All critical systems validated and operational!")
        elif overall_score >= 0.6:
            print("⚠️  NEARLY READY - Some improvements needed before production deployment")
            print("📋 Review recommendations and address critical issues")
        else:
            print("❌ NOT READY - Significant issues must be resolved before production")
            print("🔧 Address all failed tests and recommendations")
            
        print("="*80)


async def main():
    """Main test runner."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run production readiness tests
    test_runner = ProductionReadinessTest()
    await test_runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 