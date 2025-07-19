"""
EMP Architecture Validation v1.1

Simplified validation script to check EMP Ultimate Architecture v1.1 compliance
without importing legacy components.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

def validate_directory_structure() -> Dict[str, Any]:
    """Validate the v1.1 directory structure."""
    print("Validating Directory Structure...")
    
    required_directories = [
        "src/sensory/organs",
        "src/sensory/integration", 
        "src/sensory/calibration",
        "src/sensory/models",
        "src/thinking/patterns",
        "src/thinking/analysis",
        "src/thinking/inference",
        "src/thinking/memory",
        "src/thinking/models",
        "src/simulation/market_simulator",
        "src/simulation/stress_tester",
        "src/simulation/adversarial",
        "src/simulation/validators",
        "src/genome/encoders",
        "src/genome/decoders",
        "src/genome/models",
        "src/evolution/engine",
        "src/evolution/selection",
        "src/evolution/variation",
        "src/evolution/evaluation",
        "src/evolution/meta",
        "src/trading/strategies",
        "src/trading/execution",
        "src/trading/risk",
        "src/trading/monitoring",
        "src/trading/integration",
        "src/ui/web",
        "src/ui/cli",
        "src/ui/models",
        "src/governance",
        "src/operational",
        "config/fitness",
        "config/governance",
        "config/operational",
        "k8s"
    ]
    
    results = {}
    missing_dirs = []
    existing_dirs = []
    
    for directory in required_directories:
        if os.path.exists(directory):
            existing_dirs.append(directory)
            results[directory] = "EXISTS"
        else:
            missing_dirs.append(directory)
            results[directory] = "MISSING"
    
    structure_score = (len(existing_dirs) / len(required_directories)) * 100
    
    return {
        "total_directories": len(required_directories),
        "existing_directories": len(existing_dirs),
        "missing_directories": len(missing_dirs),
        "structure_score": structure_score,
        "results": results,
        "missing": missing_dirs
    }

def validate_core_files() -> Dict[str, Any]:
    """Validate core v1.1 files."""
    print("Validating Core Files...")
    
    required_files = [
        "src/core/events.py",
        "src/core/event_bus.py",
        "src/thinking/patterns/regime_classifier.py",
        "src/thinking/analysis/performance_analyzer.py",
        "src/thinking/analysis/risk_analyzer.py",
        "src/simulation/evaluation/fitness_evaluator.py",
        "src/evolution/selection/selection_strategies.py",
        "src/evolution/variation/variation_strategies.py",
        "src/evolution/engine/population_manager.py",
        "src/evolution/engine/genetic_engine.py",
        "src/sensory/integration/sensory_cortex.py",
        "src/governance/fitness_store.py",
        "src/governance/strategy_registry.py",
        "config/fitness/default_v1.yaml",
        "Dockerfile",
        "docker-compose.yml",
        "k8s/emp-deployment.yaml"
    ]
    
    results = {}
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            results[file_path] = "EXISTS"
        else:
            missing_files.append(file_path)
            results[file_path] = "MISSING"
    
    files_score = (len(existing_files) / len(required_files)) * 100
    
    return {
        "total_files": len(required_files),
        "existing_files": len(existing_files),
        "missing_files": len(missing_files),
        "files_score": files_score,
        "results": results,
        "missing": missing_files
    }

def validate_layer_separation() -> Dict[str, Any]:
    """Validate layer separation compliance."""
    print("Validating Layer Separation...")
    
    # Check for violations of layer separation
    violations = []
    
    # Check if thinking functions are in sensory layer
    sensory_files = [
        "src/sensory/regime_detector.py",  # Should be in thinking layer
        "src/sensory/analyzers/",  # Should be organs/
    ]
    
    for file_path in sensory_files:
        if os.path.exists(file_path):
            violations.append(f"Thinking function in sensory layer: {file_path}")
    
    # Check if performance calculations are in evolution layer
    evolution_files = [
        "src/evolution/fitness.py",  # Should be in thinking layer
    ]
    
    for file_path in evolution_files:
        if os.path.exists(file_path):
            violations.append(f"Performance calculation in evolution layer: {file_path}")
    
    # Check if genome is properly structured
    genome_files = [
        "src/core/decision_tree_element.py",  # Should be in genome/models/
    ]
    
    for file_path in genome_files:
        if os.path.exists(file_path):
            violations.append(f"Genome model in wrong location: {file_path}")
    
    separation_score = 100 - (len(violations) * 10)  # -10 points per violation
    separation_score = max(0, separation_score)
    
    return {
        "violations": violations,
        "separation_score": separation_score,
        "total_violations": len(violations)
    }

def validate_event_driven_architecture() -> Dict[str, Any]:
    """Validate event-driven architecture."""
    print("Validating Event-Driven Architecture...")
    
    # Check for event bus implementation
    event_bus_exists = os.path.exists("src/core/event_bus.py")
    
    # Check for event models
    events_exists = os.path.exists("src/core/events.py")
    
    # Check for event publishing in sensory cortex
    sensory_cortex_exists = os.path.exists("src/sensory/integration/sensory_cortex.py")
    
    # Check for event handling in thinking layer
    thinking_components = [
        "src/thinking/patterns/regime_classifier.py",
        "src/thinking/analysis/performance_analyzer.py",
        "src/thinking/analysis/risk_analyzer.py"
    ]
    
    thinking_exists = all(os.path.exists(f) for f in thinking_components)
    
    event_score = 0
    if event_bus_exists:
        event_score += 25
    if events_exists:
        event_score += 25
    if sensory_cortex_exists:
        event_score += 25
    if thinking_exists:
        event_score += 25
    
    return {
        "event_bus_exists": event_bus_exists,
        "events_exists": events_exists,
        "sensory_cortex_exists": sensory_cortex_exists,
        "thinking_components_exist": thinking_exists,
        "event_score": event_score
    }

def validate_governance_layer() -> Dict[str, Any]:
    """Validate governance layer implementation."""
    print("Validating Governance Layer...")
    
    # Check governance components
    fitness_store_exists = os.path.exists("src/governance/fitness_store.py")
    strategy_registry_exists = os.path.exists("src/governance/strategy_registry.py")
    fitness_config_exists = os.path.exists("config/fitness/default_v1.yaml")
    
    governance_score = 0
    if fitness_store_exists:
        governance_score += 33
    if strategy_registry_exists:
        governance_score += 33
    if fitness_config_exists:
        governance_score += 34
    
    return {
        "fitness_store_exists": fitness_store_exists,
        "strategy_registry_exists": strategy_registry_exists,
        "fitness_config_exists": fitness_config_exists,
        "governance_score": governance_score
    }

def validate_operational_backbone() -> Dict[str, Any]:
    """Validate operational backbone."""
    print("Validating Operational Backbone...")
    
    # Check operational components
    dockerfile_exists = os.path.exists("Dockerfile")
    docker_compose_exists = os.path.exists("docker-compose.yml")
    k8s_exists = os.path.exists("k8s/emp-deployment.yaml")
    
    operational_score = 0
    if dockerfile_exists:
        operational_score += 33
    if docker_compose_exists:
        operational_score += 33
    if k8s_exists:
        operational_score += 34
    
    return {
        "dockerfile_exists": dockerfile_exists,
        "docker_compose_exists": docker_compose_exists,
        "k8s_exists": k8s_exists,
        "operational_score": operational_score
    }

def generate_compliance_report() -> Dict[str, Any]:
    """Generate comprehensive compliance report."""
    print("Generating Compliance Report...")
    
    # Run all validations
    structure_validation = validate_directory_structure()
    files_validation = validate_core_files()
    separation_validation = validate_layer_separation()
    event_validation = validate_event_driven_architecture()
    governance_validation = validate_governance_layer()
    operational_validation = validate_operational_backbone()
    
    # Calculate overall compliance
    overall_score = (
        structure_validation["structure_score"] * 0.25 +
        files_validation["files_score"] * 0.25 +
        separation_validation["separation_score"] * 0.20 +
        event_validation["event_score"] * 0.15 +
        governance_validation["governance_score"] * 0.10 +
        operational_validation["operational_score"] * 0.05
    )
    
    # Determine compliance level
    if overall_score >= 95:
        compliance_level = "FULLY_COMPLIANT"
    elif overall_score >= 85:
        compliance_level = "HIGHLY_COMPLIANT"
    elif overall_score >= 70:
        compliance_level = "MOSTLY_COMPLIANT"
    elif overall_score >= 50:
        compliance_level = "PARTIALLY_COMPLIANT"
    else:
        compliance_level = "NON_COMPLIANT"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.1.0",
        "overall_compliance": {
            "score": overall_score,
            "level": compliance_level
        },
        "validations": {
            "directory_structure": structure_validation,
            "core_files": files_validation,
            "layer_separation": separation_validation,
            "event_driven_architecture": event_validation,
            "governance_layer": governance_validation,
            "operational_backbone": operational_validation
        },
        "summary": {
            "total_directories": structure_validation["total_directories"],
            "existing_directories": structure_validation["existing_directories"],
            "total_files": files_validation["total_files"],
            "existing_files": files_validation["existing_files"],
            "separation_violations": separation_validation["total_violations"]
        }
    }
    
    return report

def main():
    """Main validation execution."""
    print("="*60)
    print("EMP ULTIMATE ARCHITECTURE v1.1 COMPLIANCE VALIDATION")
    print("="*60)
    
    report = generate_compliance_report()
    
    # Print results
    print("\nCOMPLIANCE SUMMARY:")
    print(f"Overall Score: {report['overall_compliance']['score']:.1f}%")
    print(f"Compliance Level: {report['overall_compliance']['level']}")
    
    print("\nDETAILED RESULTS:")
    
    validations = report["validations"]
    
    print(f"\nDirectory Structure: {validations['directory_structure']['structure_score']:.1f}%")
    print(f"  Existing: {validations['directory_structure']['existing_directories']}/{validations['directory_structure']['total_directories']}")
    
    print(f"\nCore Files: {validations['core_files']['files_score']:.1f}%")
    print(f"  Existing: {validations['core_files']['existing_files']}/{validations['core_files']['total_files']}")
    
    print(f"\nLayer Separation: {validations['layer_separation']['separation_score']:.1f}%")
    print(f"  Violations: {validations['layer_separation']['total_violations']}")
    
    print(f"\nEvent-Driven Architecture: {validations['event_driven_architecture']['event_score']:.1f}%")
    print(f"  Event Bus: {'✓' if validations['event_driven_architecture']['event_bus_exists'] else '✗'}")
    print(f"  Event Models: {'✓' if validations['event_driven_architecture']['events_exists'] else '✗'}")
    print(f"  Sensory Cortex: {'✓' if validations['event_driven_architecture']['sensory_cortex_exists'] else '✗'}")
    print(f"  Thinking Components: {'✓' if validations['event_driven_architecture']['thinking_components_exist'] else '✗'}")
    
    print(f"\nGovernance Layer: {validations['governance_layer']['governance_score']:.1f}%")
    print(f"  Fitness Store: {'✓' if validations['governance_layer']['fitness_store_exists'] else '✗'}")
    print(f"  Strategy Registry: {'✓' if validations['governance_layer']['strategy_registry_exists'] else '✗'}")
    print(f"  Fitness Config: {'✓' if validations['governance_layer']['fitness_config_exists'] else '✗'}")
    
    print(f"\nOperational Backbone: {validations['operational_backbone']['operational_score']:.1f}%")
    print(f"  Dockerfile: {'✓' if validations['operational_backbone']['dockerfile_exists'] else '✗'}")
    print(f"  Docker Compose: {'✓' if validations['operational_backbone']['docker_compose_exists'] else '✗'}")
    print(f"  Kubernetes: {'✓' if validations['operational_backbone']['k8s_exists'] else '✗'}")
    
    # Print violations if any
    if validations['layer_separation']['violations']:
        print(f"\nLAYER SEPARATION VIOLATIONS:")
        for violation in validations['layer_separation']['violations']:
            print(f"  - {violation}")
    
    # Print missing directories if any
    if validations['directory_structure']['missing']:
        print(f"\nMISSING DIRECTORIES:")
        for directory in validations['directory_structure']['missing']:
            print(f"  - {directory}")
    
    # Print missing files if any
    if validations['core_files']['missing']:
        print(f"\nMISSING FILES:")
        for file_path in validations['core_files']['missing']:
            print(f"  - {file_path}")
    
    print("\n" + "="*60)
    
    # Save report
    with open("ARCHITECTURE_VALIDATION_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print("Validation report saved to ARCHITECTURE_VALIDATION_REPORT.json")
    
    return report

if __name__ == "__main__":
    main() 