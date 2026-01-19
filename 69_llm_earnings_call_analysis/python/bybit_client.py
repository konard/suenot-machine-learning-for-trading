"""
Bybit API Client for Cryptocurrency Market Data

This module provides a client for fetching market data from Bybit exchange,
which can be used for analyzing crypto project announcements similar to
earnings calls.
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Ticker:
    """Current ticker information"""
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    price_change_24h: float


class BybitClient:
    """
    Client for fetching market data from Bybit exchange

    Supports fetching candlestick data, ticker information, and
    other market data useful for crypto trading analysis.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit client

        Args:
            testnet: If True, use testnet instead of mainnet
        """
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = self.BASE_URL

        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

    def get_klines(self,
                   symbol: str,
                   interval: str,
                   limit: int = 200,
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None) -> List[Candle]:
        """
        Get candlestick data from Bybit

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candlestick interval:
                - "1", "3", "5", "15", "30" (minutes)
                - "60", "120", "240", "360", "720" (minutes)
                - "D" (daily), "W" (weekly), "M" (monthly)
            limit: Number of candles to fetch (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of Candle objects in chronological order
        """
        endpoint = f"{self.base_url}/v5/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        candles = []
        for item in data['result']['list']:
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(int(item[0]) / 1000),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5])
            ))

        return candles[::-1]  # Reverse to chronological order

    def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker information

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Ticker object with current market data
        """
        endpoint = f"{self.base_url}/v5/market/tickers"

        params = {
            "category": "linear",
            "symbol": symbol
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        ticker = data['result']['list'][0]
        return Ticker(
            symbol=ticker['symbol'],
            last_price=float(ticker['lastPrice']),
            bid=float(ticker['bid1Price']),
            ask=float(ticker['ask1Price']),
            volume_24h=float(ticker['volume24h']),
            price_change_24h=float(ticker['price24hPcnt'])
        )

    def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """
        Get order book data

        Args:
            symbol: Trading pair
            limit: Depth of orderbook (max 500)

        Returns:
            Dictionary with bids and asks
        """
        endpoint = f"{self.base_url}/v5/market/orderbook"

        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": min(limit, 500)
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        result = data['result']
        return {
            'symbol': result['s'],
            'timestamp': datetime.fromtimestamp(int(result['ts']) / 1000),
            'bids': [(float(b[0]), float(b[1])) for b in result['b']],
            'asks': [(float(a[0]), float(a[1])) for a in result['a']]
        }

    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get recent trades

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)

        Returns:
            List of recent trades
        """
        endpoint = f"{self.base_url}/v5/market/recent-trade"

        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": min(limit, 1000)
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        trades = []
        for trade in data['result']['list']:
            trades.append({
                'id': trade['execId'],
                'price': float(trade['price']),
                'size': float(trade['size']),
                'side': trade['side'],
                'timestamp': datetime.fromtimestamp(int(trade['time']) / 1000)
            })

        return trades

    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Get funding rate for perpetual contracts

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with funding rate information
        """
        endpoint = f"{self.base_url}/v5/market/tickers"

        params = {
            "category": "linear",
            "symbol": symbol
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        ticker = data['result']['list'][0]
        return {
            'symbol': symbol,
            'funding_rate': float(ticker.get('fundingRate', 0)),
            'next_funding_time': datetime.fromtimestamp(
                int(ticker.get('nextFundingTime', 0)) / 1000
            ) if ticker.get('nextFundingTime') else None
        }

    def calculate_returns(self, candles: List[Candle]) -> List[float]:
        """
        Calculate returns from candle data

        Args:
            candles: List of Candle objects

        Returns:
            List of returns (percentage change)
        """
        returns = []
        for i in range(1, len(candles)):
            ret = (candles[i].close - candles[i-1].close) / candles[i-1].close
            returns.append(ret)
        return returns

    def calculate_volatility(self, candles: List[Candle], window: int = 20) -> float:
        """
        Calculate historical volatility

        Args:
            candles: List of Candle objects
            window: Rolling window size

        Returns:
            Annualized volatility
        """
        import numpy as np

        returns = self.calculate_returns(candles)

        if len(returns) < window:
            return 0.0

        # Use most recent window
        recent_returns = returns[-window:]
        daily_vol = np.std(recent_returns)

        # Annualize (assuming crypto trades 365 days)
        return daily_vol * np.sqrt(365)


def fetch_crypto_data_for_analysis(symbol: str = "BTCUSDT",
                                   interval: str = "D",
                                   days: int = 30) -> Dict:
    """
    Fetch cryptocurrency data for analysis

    This function fetches market data that can be used alongside
    crypto project announcement analysis (similar to earnings calls).

    Args:
        symbol: Trading pair
        interval: Candle interval
        days: Number of days of history

    Returns:
        Dictionary with market data and metrics
    """
    client = BybitClient()

    # Fetch candles
    candles = client.get_klines(symbol, interval, limit=days)

    # Get current ticker
    ticker = client.get_ticker(symbol)

    # Calculate metrics
    returns = client.calculate_returns(candles)
    volatility = client.calculate_volatility(candles)

    # Calculate simple statistics
    import numpy as np

    return {
        'symbol': symbol,
        'current_price': ticker.last_price,
        'price_change_24h': ticker.price_change_24h,
        'volume_24h': ticker.volume_24h,
        'candles': candles,
        'returns': returns,
        'volatility': volatility,
        'avg_return': np.mean(returns) if returns else 0,
        'max_return': max(returns) if returns else 0,
        'min_return': min(returns) if returns else 0,
        'positive_days': sum(1 for r in returns if r > 0),
        'negative_days': sum(1 for r in returns if r < 0)
    }


if __name__ == "__main__":
    # Example usage
    print("Fetching Bybit data...")

    try:
        data = fetch_crypto_data_for_analysis("BTCUSDT", "D", 30)

        print(f"\n=== {data['symbol']} Market Data ===")
        print(f"Current Price: ${data['current_price']:,.2f}")
        print(f"24h Change: {data['price_change_24h']*100:.2f}%")
        print(f"24h Volume: ${data['volume_24h']:,.0f}")
        print(f"\n=== Statistics (Last 30 Days) ===")
        print(f"Volatility (Annualized): {data['volatility']*100:.1f}%")
        print(f"Average Daily Return: {data['avg_return']*100:.3f}%")
        print(f"Best Day: {data['max_return']*100:.2f}%")
        print(f"Worst Day: {data['min_return']*100:.2f}%")
        print(f"Positive Days: {data['positive_days']}")
        print(f"Negative Days: {data['negative_days']}")

    except Exception as e:
        print(f"Error fetching data: {e}")
        print("\nNote: This example requires internet connection to Bybit API.")
