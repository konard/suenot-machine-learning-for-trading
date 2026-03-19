"""
Bybit V5 API client for fetching BTC options data.

Fetches live implied volatility data from Bybit's options market
and constructs volatility surfaces from the raw ticker data.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .vol_surface import VolSurface


@dataclass
class ParsedOption:
    """Parsed option data with numeric fields."""

    symbol: str
    strike: float
    expiry: str
    is_call: bool
    mark_iv: float
    underlying: float
    moneyness: float


class BybitClient:
    """Client for fetching BTC options data from Bybit V5 API."""

    BASE_URL = "https://api.bybit.com"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or self.BASE_URL

    def get_option_tickers(self) -> List[dict]:
        """Fetch BTC option tickers from Bybit V5 API.

        Returns:
            List of raw ticker dictionaries.

        Raises:
            RuntimeError: If the API returns an error or requests is not installed.
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library required. Install with: pip install requests")
        url = f"{self.base_url}/v5/market/tickers"
        params = {"category": "option", "baseCoin": "BTC"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode", -1) != 0:
            raise RuntimeError(f"Bybit API error: retCode={data.get('retCode')}")
        return data.get("result", {}).get("list", [])

    @staticmethod
    def parse_symbol(symbol: str) -> Optional[Tuple[str, float, bool]]:
        """Parse a Bybit option symbol into (expiry, strike, is_call).

        Format: "BTC-28MAR25-90000-C"
        """
        parts = symbol.split("-")
        if len(parts) != 4:
            return None
        try:
            expiry = parts[1]
            strike = float(parts[2])
            is_call = parts[3] == "C"
            return expiry, strike, is_call
        except (ValueError, IndexError):
            return None

    @classmethod
    def parse_tickers(cls, tickers: List[dict]) -> List[ParsedOption]:
        """Convert raw tickers into parsed option data."""
        results: List[ParsedOption] = []
        for t in tickers:
            parsed = cls.parse_symbol(t.get("symbol", ""))
            if parsed is None:
                continue
            expiry, strike, is_call = parsed
            try:
                mark_iv = float(t.get("markIv", "0"))
                underlying = float(t.get("underlyingPrice", "0"))
            except (ValueError, TypeError):
                continue
            if mark_iv <= 0.0 or underlying <= 0.0:
                continue
            moneyness = strike / underlying
            results.append(ParsedOption(
                symbol=t["symbol"],
                strike=strike,
                expiry=expiry,
                is_call=is_call,
                mark_iv=mark_iv,
                underlying=underlying,
                moneyness=moneyness,
            ))
        return results

    @staticmethod
    def build_surface(
        options: List[ParsedOption],
        moneyness_grid: np.ndarray,
        maturity_days: np.ndarray,
    ) -> VolSurface:
        """Build a volatility surface from parsed options.

        Uses a fixed moneyness grid and the nearest available data points.
        """
        n_mat = len(maturity_days)
        n_mon = len(moneyness_grid)
        ivols = np.full((n_mat, n_mon), 0.3)  # default 30% vol

        # Group by expiry
        by_expiry: dict[str, List[ParsedOption]] = {}
        for opt in options:
            by_expiry.setdefault(opt.expiry, []).append(opt)

        expiries = list(by_expiry.keys())
        for t_idx in range(n_mat):
            if t_idx >= len(expiries):
                break
            expiry = expiries[t_idx]
            opts = by_expiry[expiry]
            for k_idx, target_m in enumerate(moneyness_grid):
                nearest = min(opts, key=lambda o: abs(o.moneyness - target_m), default=None)
                if nearest is not None and abs(nearest.moneyness - target_m) < 0.15:
                    ivols[t_idx, k_idx] = nearest.mark_iv

        maturities = maturity_days / 365.0
        return VolSurface(
            moneyness=moneyness_grid.copy(),
            maturities=maturities,
            ivols=ivols,
        )
