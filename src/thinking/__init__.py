"""
EMP Thinking Layer v1.1

The Thinking Layer is responsible for cognitive processing, inference, and higher-order analysis.
It receives sensory signals and produces inferences, conclusions, and derived metrics that
inform decision-making in the Adaptive Core.

Architecture:
- patterns/: Pattern detection and classification
- analysis/: Risk, performance, and market analysis
- inference/: Probability and prediction engines
- memory/: Experience and pattern memory
- models/: Data models for analysis and inference
"""

from .patterns import *
from .analysis import *
from .inference import *
from .memory import *

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Thinking Layer - Cognitive Processing and Inference" 