"""
Quick Progress Check

Simple script to check implementation progress without Unicode issues.
"""

import sys
import os
import inspect
import importlib
from typing import Dict, List, Set, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def check_progress():
    """Check implementation progress"""
    print("PROGRESS CHECK")
    print("=" * 50)
    
    # Check old files
    old_files = [
        'src/sensory/dimensions/enhanced_how_dimension.py',
        'src/sensory/dimensions/enhanced_what_dimension.py', 
        'src/sensory/dimensions/enhanced_when_dimension.py',
        'src/sensory/dimensions/enhanced_why_dimension.py',
        'src/sensory/dimensions/enhanced_anomaly_dimension.py'
    ]
    
    old_functions = 0
    for file_path in old_files:
        try:
            if os.path.exists(file_path):
                module_name = file_path.replace('/', '.').replace('.py', '')
                module = importlib.import_module(module_name)
                
                functions = 0
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj):
                        functions += 1
                    elif inspect.isclass(obj):
                        for method_name, method_obj in inspect.getmembers(obj):
                            if inspect.isfunction(method_obj) and not method_name.startswith('_'):
                                functions += 1
                
                old_functions += functions
                print(f"Old {file_path}: {functions} functions")
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
    
    # Check new files
    new_modules = [
        'src.sensory.dimensions.how.how_engine',
        'src.sensory.dimensions.how.indicators',
        'src.sensory.dimensions.how.patterns',
        'src.sensory.dimensions.how.order_flow',
        'src.sensory.dimensions.what.what_engine',
        'src.sensory.dimensions.what.price_action',
        'src.sensory.dimensions.when.when_engine',
        'src.sensory.dimensions.when.regime_detection',
        'src.sensory.dimensions.why.why_engine',
        'src.sensory.dimensions.anomaly.anomaly_engine',
        'src.sensory.dimensions.compatibility'
    ]
    
    new_functions = 0
    for module_name in new_modules:
        try:
            module = importlib.import_module(module_name)
            
            functions = 0
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    functions += 1
                elif inspect.isclass(obj):
                    for method_name, method_obj in inspect.getmembers(obj):
                        if inspect.isfunction(method_obj) and not method_name.startswith('_'):
                            functions += 1
            
            new_functions += functions
            print(f"New {module_name}: {functions} functions")
        except Exception as e:
            print(f"Error checking {module_name}: {e}")
    
    print(f"\nSUMMARY:")
    print(f"Old functions: {old_functions}")
    print(f"New functions: {new_functions}")
    print(f"Coverage: {(new_functions/old_functions)*100:.1f}%" if old_functions > 0 else "N/A")


if __name__ == "__main__":
    check_progress() 
