"""
Prompt Templates for Financial Analysis

This module contains carefully engineered prompt templates
for various trading-related tasks.
"""

from .sentiment import SENTIMENT_PROMPTS
from .signals import SIGNAL_PROMPTS
from .regime import REGIME_PROMPTS

__all__ = ["SENTIMENT_PROMPTS", "SIGNAL_PROMPTS", "REGIME_PROMPTS"]
