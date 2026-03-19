"""
Generative LOB (Limit Order Book) Trading

This module implements generative models for limit order book simulation,
including VAE encoding/decoding, GAN-based generation, diffusion models,
statistical validation, and Bybit API integration.
"""

from .lob_snapshot import LobSnapshot, PriceLevel
from .vae import LobVae
from .gan import LobGan
from .diffusion import LobDiffusion
from .generator import LobGenerator
from .validator import LobValidator
from .bybit_client import BybitClient

__version__ = "0.1.0"
__all__ = [
    "LobSnapshot",
    "PriceLevel",
    "LobVae",
    "LobGan",
    "LobDiffusion",
    "LobGenerator",
    "LobValidator",
    "BybitClient",
]
