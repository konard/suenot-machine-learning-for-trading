"""
Data Preparation Module for Mediation Analysis

This module provides utilities for preparing financial data
for mediation analysis, including stock and crypto data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import warnings


def prepare_stock_data(
    treatment_ticker: str,
    mediator_indicator: str,
    outcome_ticker: str,
    start_date: str,
    end_date: str,
    lag_treatment: int = 1,
    lag_mediator: int = 1
) -> pd.DataFrame:
    """
    Prepare data for mediation analysis with stock market data.

    Parameters:
    -----------
    treatment_ticker : Ticker symbol for treatment variable (e.g., 'SPY')
    mediator_indicator : Type of mediator ('volume', 'volatility', 'vix')
    outcome_ticker : Ticker symbol for outcome variable (e.g., 'AAPL')
    start_date : Start date in 'YYYY-MM-DD' format
    end_date : End date in 'YYYY-MM-DD' format
    lag_treatment : Number of periods to lag treatment variable
    lag_mediator : Number of periods to lag mediator variable

    Returns:
    --------
    DataFrame with columns: X (treatment), M (mediator), Y (outcome)

    Example:
    --------
    >>> df = prepare_stock_data(
    ...     treatment_ticker='SPY',
    ...     mediator_indicator='vix',
    ...     outcome_ticker='AAPL',
    ...     start_date='2023-01-01',
    ...     end_date='2024-01-01'
    ... )
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    # Download treatment data
    treatment_data = yf.download(treatment_ticker, start=start_date, end=end_date, progress=False)
    if treatment_data.empty:
        raise ValueError(f"No data found for {treatment_ticker}")

    # Download outcome data
    outcome_data = yf.download(outcome_ticker, start=start_date, end=end_date, progress=False)
    if outcome_data.empty:
        raise ValueError(f"No data found for {outcome_ticker}")

    # Get mediator data based on type
    mediator_indicator = mediator_indicator.lower()

    if mediator_indicator == 'vix' or mediator_indicator == 'volatility':
        mediator_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if mediator_data.empty:
            # Fall back to calculated volatility
            warnings.warn("VIX data not available, using calculated volatility")
            mediator = treatment_data['Adj Close'].pct_change().rolling(20).std() * np.sqrt(252)
        else:
            mediator = mediator_data['Close']

    elif mediator_indicator == 'volume':
        mediator = treatment_data['Volume']

    elif mediator_indicator == 'spread':
        # Bid-ask spread proxy using high-low range
        mediator = (treatment_data['High'] - treatment_data['Low']) / treatment_data['Close']

    elif mediator_indicator == 'momentum':
        # Short-term momentum as mediator
        mediator = treatment_data['Adj Close'].pct_change(5)

    else:
        raise ValueError(f"Unknown mediator_indicator: {mediator_indicator}")

    # Calculate returns
    treatment_returns = treatment_data['Adj Close'].pct_change()
    outcome_returns = outcome_data['Adj Close'].pct_change()

    # Align indices
    common_idx = treatment_returns.index.intersection(outcome_returns.index)
    common_idx = common_idx.intersection(mediator.index)

    # Combine into DataFrame with lags
    df = pd.DataFrame({
        'X': treatment_returns.loc[common_idx].shift(lag_treatment),
        'M': mediator.loc[common_idx].shift(lag_mediator),
        'Y': outcome_returns.loc[common_idx],
    })

    # Add raw values for reference
    df['treatment_raw'] = treatment_returns.loc[common_idx]
    df['mediator_raw'] = mediator.loc[common_idx]
    df['outcome_raw'] = outcome_returns.loc[common_idx]

    return df.dropna()


def prepare_crypto_data_bybit(
    treatment_symbol: str,
    mediator_type: str,
    outcome_symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1h',
    lag_treatment: int = 1,
    lag_mediator: int = 1
) -> pd.DataFrame:
    """
    Prepare crypto data from Bybit for mediation analysis.

    Parameters:
    -----------
    treatment_symbol : Bybit symbol for treatment (e.g., 'BTCUSDT')
    mediator_type : Type of mediator ('funding_rate', 'open_interest', 'volume')
    outcome_symbol : Bybit symbol for outcome (e.g., 'ETHUSDT')
    start_date : Start date in 'YYYY-MM-DD' format
    end_date : End date in 'YYYY-MM-DD' format
    interval : Kline interval ('1h', '4h', '1d')
    lag_treatment : Number of periods to lag treatment
    lag_mediator : Number of periods to lag mediator

    Returns:
    --------
    DataFrame with columns: X (treatment), M (mediator), Y (outcome)

    Example:
    --------
    >>> df = prepare_crypto_data_bybit(
    ...     treatment_symbol='BTCUSDT',
    ...     mediator_type='funding_rate',
    ...     outcome_symbol='ETHUSDT',
    ...     start_date='2024-01-01',
    ...     end_date='2024-06-01'
    ... )
    """
    try:
        from pybit.unified_trading import HTTP
    except ImportError:
        raise ImportError("pybit is required. Install with: pip install pybit")

    # Initialize Bybit session (public endpoints)
    session = HTTP(testnet=False)

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    def fetch_klines(symbol: str) -> pd.DataFrame:
        """Fetch kline data for a symbol"""
        all_data = []
        current_start = start_ts

        while current_start < end_ts:
            try:
                response = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    start=current_start,
                    end=end_ts,
                    limit=1000
                )

                if response['retCode'] != 0:
                    warnings.warn(f"API error: {response['retMsg']}")
                    break

                data = response['result']['list']
                if not data:
                    break

                all_data.extend(data)

                # Update start time for next batch
                last_ts = int(data[-1][0])
                if last_ts <= current_start:
                    break
                current_start = last_ts

            except Exception as e:
                warnings.warn(f"Error fetching data: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df = df.set_index('timestamp').sort_index()

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        return df

    # Fetch treatment and outcome klines
    treatment_df = fetch_klines(treatment_symbol)
    outcome_df = fetch_klines(outcome_symbol)

    if treatment_df.empty or outcome_df.empty:
        raise ValueError("Failed to fetch kline data")

    # Get mediator data based on type
    mediator_type = mediator_type.lower()

    if mediator_type == 'funding_rate':
        # Fetch funding rate history
        all_funding = []
        current_start = start_ts

        while current_start < end_ts:
            try:
                response = session.get_funding_rate_history(
                    category="linear",
                    symbol=treatment_symbol,
                    startTime=current_start,
                    endTime=end_ts,
                    limit=200
                )

                if response['retCode'] != 0:
                    break

                data = response['result']['list']
                if not data:
                    break

                all_funding.extend(data)

                # Update start time
                last_ts = int(data[-1]['fundingRateTimestamp'])
                if last_ts <= current_start:
                    break
                current_start = last_ts

            except Exception as e:
                warnings.warn(f"Error fetching funding rate: {e}")
                break

        if all_funding:
            mediator_df = pd.DataFrame(all_funding)
            mediator_df['timestamp'] = pd.to_datetime(
                mediator_df['fundingRateTimestamp'].astype(int), unit='ms'
            )
            mediator_df['M'] = mediator_df['fundingRate'].astype(float)
            mediator_df = mediator_df.set_index('timestamp')[['M']].sort_index()
        else:
            # Fall back to volume
            warnings.warn("Funding rate not available, using volume")
            mediator_df = treatment_df[['volume']].rename(columns={'volume': 'M'})

    elif mediator_type == 'open_interest':
        # Fetch open interest
        all_oi = []
        current_start = start_ts

        while current_start < end_ts:
            try:
                response = session.get_open_interest(
                    category="linear",
                    symbol=treatment_symbol,
                    intervalTime="1h" if interval in ['1h', '4h'] else "1d",
                    startTime=current_start,
                    endTime=end_ts,
                    limit=200
                )

                if response['retCode'] != 0:
                    break

                data = response['result']['list']
                if not data:
                    break

                all_oi.extend(data)

                # Update start time
                last_ts = int(data[-1]['timestamp'])
                if last_ts <= current_start:
                    break
                current_start = last_ts

            except Exception as e:
                warnings.warn(f"Error fetching open interest: {e}")
                break

        if all_oi:
            mediator_df = pd.DataFrame(all_oi)
            mediator_df['timestamp'] = pd.to_datetime(
                mediator_df['timestamp'].astype(int), unit='ms'
            )
            mediator_df['M'] = mediator_df['openInterest'].astype(float)
            mediator_df = mediator_df.set_index('timestamp')[['M']].sort_index()
        else:
            warnings.warn("Open interest not available, using volume")
            mediator_df = treatment_df[['volume']].rename(columns={'volume': 'M'})

    elif mediator_type == 'volume':
        mediator_df = treatment_df[['volume']].rename(columns={'volume': 'M'})

    elif mediator_type == 'volatility':
        # Realized volatility
        returns = treatment_df['close'].pct_change()
        volatility = returns.rolling(24 if interval == '1h' else 7).std()
        mediator_df = pd.DataFrame({'M': volatility})

    else:
        raise ValueError(f"Unknown mediator_type: {mediator_type}")

    # Calculate returns
    treatment_returns = treatment_df['close'].pct_change()
    outcome_returns = outcome_df['close'].pct_change()

    # Combine data
    combined = pd.DataFrame({
        'treatment_return': treatment_returns,
        'outcome_return': outcome_returns
    })

    # Merge mediator (may have different frequency)
    combined = combined.join(mediator_df, how='left')
    combined['M'] = combined['M'].ffill()

    # Apply lags
    combined['X'] = combined['treatment_return'].shift(lag_treatment)
    combined['M'] = combined['M'].shift(lag_mediator)
    combined['Y'] = combined['outcome_return']

    # Keep only needed columns
    result = combined[['X', 'M', 'Y']].copy()
    result['treatment_raw'] = combined['treatment_return']
    result['mediator_raw'] = mediator_df['M'].reindex(result.index)
    result['outcome_raw'] = combined['outcome_return']

    return result.dropna()


def prepare_custom_data(
    treatment: pd.Series,
    mediator: pd.Series,
    outcome: pd.Series,
    lag_treatment: int = 1,
    lag_mediator: int = 1,
    covariates: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Prepare custom data for mediation analysis.

    Parameters:
    -----------
    treatment : Treatment variable series
    mediator : Mediator variable series
    outcome : Outcome variable series
    lag_treatment : Lag for treatment variable
    lag_mediator : Lag for mediator variable
    covariates : Optional DataFrame of control variables

    Returns:
    --------
    DataFrame ready for mediation analysis
    """
    # Align indices
    common_idx = treatment.index.intersection(mediator.index).intersection(outcome.index)

    if covariates is not None:
        common_idx = common_idx.intersection(covariates.index)

    # Create base DataFrame
    df = pd.DataFrame({
        'X': treatment.loc[common_idx].shift(lag_treatment),
        'M': mediator.loc[common_idx].shift(lag_mediator),
        'Y': outcome.loc[common_idx]
    })

    # Add covariates if provided
    if covariates is not None:
        for col in covariates.columns:
            df[f'C_{col}'] = covariates[col].loc[common_idx]

    return df.dropna()


def calculate_mediation_features(
    df: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate additional features useful for mediation analysis.

    Parameters:
    -----------
    df : DataFrame with X, M, Y columns
    window : Rolling window size

    Returns:
    --------
    DataFrame with additional features
    """
    result = df.copy()

    # Rolling correlations
    result['corr_XM'] = df['X'].rolling(window).corr(df['M'])
    result['corr_MY'] = df['M'].rolling(window).corr(df['Y'])
    result['corr_XY'] = df['X'].rolling(window).corr(df['Y'])

    # Rolling standard deviations
    result['std_X'] = df['X'].rolling(window).std()
    result['std_M'] = df['M'].rolling(window).std()
    result['std_Y'] = df['Y'].rolling(window).std()

    # Rolling means
    result['mean_X'] = df['X'].rolling(window).mean()
    result['mean_M'] = df['M'].rolling(window).mean()
    result['mean_Y'] = df['Y'].rolling(window).mean()

    # Standardized values
    result['X_std'] = (df['X'] - result['mean_X']) / (result['std_X'] + 1e-10)
    result['M_std'] = (df['M'] - result['mean_M']) / (result['std_M'] + 1e-10)
    result['Y_std'] = (df['Y'] - result['mean_Y']) / (result['std_Y'] + 1e-10)

    # Interaction term
    result['XM_interaction'] = result['X_std'] * result['M_std']

    return result
