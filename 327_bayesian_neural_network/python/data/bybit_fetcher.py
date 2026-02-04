"""
Bybit data fetcher using CCXT.

Fetches OHLCV data from Bybit exchange for cryptocurrency trading.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from loguru import logger
import time


@dataclass
class OHLCVData:
    """Container for OHLCV data."""
    symbol: str
    timeframe: str
    df: pd.DataFrame
    fetched_at: datetime

    def __post_init__(self):
        """Ensure proper column types."""
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])


class BybitFetcher:
    """
    Fetches cryptocurrency market data from Bybit using CCXT.

    Supports:
    - OHLCV (candlestick) data
    - Multiple timeframes
    - Multiple symbols
    - Automatic pagination for large date ranges

    Example:
        fetcher = BybitFetcher()
        data = fetcher.fetch_ohlcv('BTC/USDT', '1h', days=30)
        print(data.df.head())
    """

    # Supported timeframes on Bybit
    TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']

    # Default symbols for crypto trading
    DEFAULT_SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT'
    ]

    def __init__(
        self,
        exchange_id: str = 'bybit',
        rate_limit: bool = True,
        sandbox: bool = False
    ):
        """
        Initialize the Bybit fetcher.

        Args:
            exchange_id: CCXT exchange ID (default: 'bybit')
            rate_limit: Enable rate limiting to avoid API bans
            sandbox: Use testnet instead of mainnet
        """
        self.exchange_id = exchange_id

        # Initialize CCXT exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': rate_limit,
            'options': {
                'defaultType': 'linear',  # Use linear perpetuals
            }
        })

        if sandbox:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"Initialized {exchange_id} fetcher")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        days: Optional[int] = None,
        limit: int = 1000
    ) -> OHLCVData:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (e.g., '1h', '4h', '1d')
            since: Start datetime (optional)
            until: End datetime (optional)
            days: Number of days to fetch (alternative to since/until)
            limit: Maximum candles per request

        Returns:
            OHLCVData object with DataFrame
        """
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {self.TIMEFRAMES}")

        # Determine date range
        if days is not None:
            until = datetime.utcnow()
            since = until - timedelta(days=days)
        elif since is None:
            since = datetime.utcnow() - timedelta(days=30)

        if until is None:
            until = datetime.utcnow()

        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)

        logger.info(f"Fetching {symbol} {timeframe} from {since} to {until}")

        # Fetch data with pagination
        all_ohlcv = []
        current_since = since_ms

        while current_since < until_ms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_since,
                    limit=limit
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Update since to last timestamp + 1
                last_timestamp = ohlcv[-1][0]
                if last_timestamp <= current_since:
                    break
                current_since = last_timestamp + 1

                # Small delay to respect rate limits
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['datetime'] = df['timestamp']
            df = df.set_index('datetime')

            # Filter to requested range
            df = df[df['timestamp'] >= since]
            df = df[df['timestamp'] <= until]

            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

        logger.info(f"Fetched {len(df)} candles for {symbol}")

        return OHLCVData(
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            fetched_at=datetime.utcnow()
        )

    def fetch_multiple_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = '1h',
        days: int = 30
    ) -> Dict[str, OHLCVData]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pairs (default: DEFAULT_SYMBOLS)
            timeframe: Candlestick timeframe
            days: Number of days to fetch

        Returns:
            Dictionary mapping symbol to OHLCVData
        """
        if symbols is None:
            symbols = self.DEFAULT_SYMBOLS

        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    days=days
                )
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        return data

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']

    def get_order_book(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict:
        """
        Fetch order book for a symbol.

        Args:
            symbol: Trading pair
            limit: Number of levels to fetch

        Returns:
            Order book dictionary with 'bids' and 'asks'
        """
        return self.exchange.fetch_order_book(symbol, limit)

    def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent trades for a symbol.

        Args:
            symbol: Trading pair
            limit: Number of trades to fetch

        Returns:
            DataFrame with trade data
        """
        trades = self.exchange.fetch_trades(symbol, limit=limit)
        df = pd.DataFrame(trades)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df


def prepare_training_data(
    ohlcv_data: OHLCVData,
    feature_window: int = 20,
    prediction_horizon: int = 1,
    return_threshold: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare OHLCV data for training.

    Creates features from past prices and labels based on future returns.

    Args:
        ohlcv_data: OHLCV data object
        feature_window: Number of past candles for features
        prediction_horizon: Candles ahead to predict
        return_threshold: Threshold for up/down classification

    Returns:
        X: Feature array (n_samples, n_features)
        y_class: Class labels (0=up, 1=neutral, 2=down)
        y_return: Continuous return values
    """
    df = ohlcv_data.df.copy()

    # Calculate returns
    df['return'] = df['close'].pct_change()
    df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1

    # Simple features: past returns and volume changes
    features = []
    for i in range(1, feature_window + 1):
        df[f'return_lag_{i}'] = df['return'].shift(i)
        df[f'volume_change_lag_{i}'] = df['volume'].pct_change().shift(i)
        features.extend([f'return_lag_{i}', f'volume_change_lag_{i}'])

    # Price-based features
    df['close_to_high'] = df['close'] / df['high']
    df['close_to_low'] = df['close'] / df['low']
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    features.extend(['close_to_high', 'close_to_low', 'high_low_range'])

    # Drop NaN rows
    df = df.dropna()

    # Create labels
    y_return = df['future_return'].values
    y_class = np.where(
        y_return > return_threshold, 0,  # Up
        np.where(y_return < -return_threshold, 2, 1)  # Down or Neutral
    )

    # Extract features
    X = df[features].values

    return X.astype(np.float32), y_class.astype(np.int64), y_return.astype(np.float32)
