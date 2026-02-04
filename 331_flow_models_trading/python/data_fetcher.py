#!/usr/bin/env python3
"""
Data Fetcher for Flow Models Trading

Fetches cryptocurrency market data from Bybit exchange using CCXT library.
Provides OHLCV data, order book snapshots, and trade data for flow model training.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BybitDataFetcher:
    """Fetch market data from Bybit exchange via CCXT"""

    # Default symbols for trading
    DEFAULT_SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT'
    ]

    def __init__(self, sandbox: bool = False):
        """
        Initialize Bybit data fetcher

        Args:
            sandbox: Use Bybit testnet if True
        """
        self.exchange = ccxt.bybit({
            'sandbox': sandbox,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        logger.info(f"Initialized Bybit connection (sandbox={sandbox})")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 500,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max 1000)
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol

            # Calculate additional features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_returns'].rolling(window=20).std()
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict:
        """
        Fetch current order book snapshot

        Args:
            symbol: Trading pair
            limit: Number of levels to fetch

        Returns:
            Dictionary with bids, asks, and computed metrics
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)

            bids = np.array(order_book['bids'])
            asks = np.array(order_book['asks'])

            # Compute order book metrics
            if len(bids) > 0 and len(asks) > 0:
                mid_price = (bids[0][0] + asks[0][0]) / 2
                spread_bps = (asks[0][0] - bids[0][0]) / mid_price * 10000

                bid_depth = bids[:, 0] @ bids[:, 1]  # Price-weighted depth
                ask_depth = asks[:, 0] @ asks[:, 1]
                depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

                # Volume-weighted average prices
                bid_vwap = (bids[:, 0] * bids[:, 1]).sum() / bids[:, 1].sum() if bids[:, 1].sum() > 0 else 0
                ask_vwap = (asks[:, 0] * asks[:, 1]).sum() / asks[:, 1].sum() if asks[:, 1].sum() > 0 else 0
            else:
                mid_price = spread_bps = depth_imbalance = bid_vwap = ask_vwap = 0

            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'bids': bids.tolist(),
                'asks': asks.tolist(),
                'mid_price': mid_price,
                'spread_bps': spread_bps,
                'depth_imbalance': depth_imbalance,
                'bid_vwap': bid_vwap,
                'ask_vwap': ask_vwap
            }

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise

    def fetch_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent trades

        Args:
            symbol: Trading pair
            limit: Number of trades to fetch

        Returns:
            DataFrame with trade data
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)

            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(t['timestamp'], unit='ms'),
                'symbol': symbol,
                'side': t['side'],
                'price': t['price'],
                'amount': t['amount'],
                'cost': t['cost']
            } for t in trades])

            # Compute trade flow metrics
            if len(df) > 0:
                df['is_buy'] = df['side'] == 'buy'
                df['signed_volume'] = df['amount'] * np.where(df['is_buy'], 1, -1)

            logger.info(f"Fetched {len(df)} trades for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise

    def fetch_tickers(
        self,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch current tickers for multiple symbols

        Args:
            symbols: List of trading pairs (default: DEFAULT_SYMBOLS)

        Returns:
            DataFrame with ticker data
        """
        symbols = symbols or self.DEFAULT_SYMBOLS

        try:
            tickers = self.exchange.fetch_tickers(symbols)

            data = []
            for symbol, ticker in tickers.items():
                data.append({
                    'symbol': symbol,
                    'last': ticker['last'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'high': ticker['high'],
                    'low': ticker['low'],
                    'volume': ticker['baseVolume'],
                    'quote_volume': ticker['quoteVolume'],
                    'change_pct': ticker['percentage'],
                    'timestamp': pd.to_datetime(ticker['timestamp'], unit='ms')
                })

            df = pd.DataFrame(data)
            logger.info(f"Fetched tickers for {len(df)} symbols")
            return df

        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            raise

    def fetch_multi_symbol_ohlcv(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = '1h',
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols

        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            limit: Number of candles per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        symbols = symbols or self.DEFAULT_SYMBOLS
        result = {}

        for symbol in symbols:
            try:
                result[symbol] = self.fetch_ohlcv(symbol, timeframe, limit)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

        return result

    def prepare_flow_model_features(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Prepare features suitable for flow model training

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            limit: Number of data points

        Returns:
            DataFrame with normalized features for flow model
        """
        # Fetch OHLCV data
        df = self.fetch_ohlcv(symbol, timeframe, limit)

        # Compute additional features
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)

        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility_20'] = df['log_returns'].rolling(20).std()
        df['volatility_50'] = df['log_returns'].rolling(50).std()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands position
        bb_mean = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - bb_mean) / (2 * bb_std)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Select feature columns
        feature_columns = [
            'return_1', 'return_5', 'return_10',
            'volume_ratio', 'volatility_20', 'volatility_50',
            'rsi', 'bb_position', 'macd_diff'
        ]

        # Drop NaN rows
        features_df = df[['timestamp', 'close'] + feature_columns].dropna()

        logger.info(f"Prepared {len(features_df)} samples with {len(feature_columns)} features")
        return features_df


def main():
    """Example usage of the data fetcher"""
    print("=" * 60)
    print("Flow Models Trading - Data Fetcher")
    print("=" * 60)

    # Initialize fetcher
    fetcher = BybitDataFetcher(sandbox=False)

    # Fetch tickers
    print("\n1. Fetching current tickers...")
    tickers = fetcher.fetch_tickers()
    print(tickers[['symbol', 'last', 'change_pct', 'volume']].to_string())

    # Fetch OHLCV for BTC
    print("\n2. Fetching BTC/USDT OHLCV data...")
    btc_ohlcv = fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=100)
    print(f"   Latest candles:")
    print(btc_ohlcv[['timestamp', 'close', 'volume', 'returns']].tail())

    # Fetch order book
    print("\n3. Fetching BTC/USDT order book...")
    order_book = fetcher.fetch_order_book('BTC/USDT', limit=10)
    print(f"   Mid price: ${order_book['mid_price']:,.2f}")
    print(f"   Spread: {order_book['spread_bps']:.2f} bps")
    print(f"   Depth imbalance: {order_book['depth_imbalance']:.4f}")

    # Fetch recent trades
    print("\n4. Fetching recent trades...")
    trades = fetcher.fetch_trades('BTC/USDT', limit=20)
    print(f"   Recent trades:")
    print(trades[['timestamp', 'side', 'price', 'amount']].tail())

    # Prepare flow model features
    print("\n5. Preparing flow model features...")
    features = fetcher.prepare_flow_model_features('BTC/USDT', '1h', limit=500)
    print(f"   Feature columns: {list(features.columns)}")
    print(f"   Sample features:")
    print(features.iloc[-5:, 2:].to_string())

    print("\n" + "=" * 60)
    print("Data fetching complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
