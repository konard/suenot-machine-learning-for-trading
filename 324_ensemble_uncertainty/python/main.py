#!/usr/bin/env python3
"""
Main entry point for the Ensemble Uncertainty Trading System.

This script demonstrates the complete pipeline:
1. Fetch market data from Bybit via CCXT
2. Compute technical features
3. Train ensemble models
4. Generate predictions with uncertainty
5. Make trading decisions based on uncertainty
6. Run backtest to evaluate performance
"""

import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import logging

# Support both module import and direct script execution
try:
    from .data_fetcher import BybitDataFetcher
    from .features import FeatureEngineer
    from .ensemble import RandomForestEnsemble, GradientBoostingEnsemble, StackingEnsemble
    from .uncertainty import UncertaintyQuantifier, UncertaintyAnalyzer
    from .calibration import Calibrator
    from .strategy import UncertaintyStrategy, PortfolioManager
    from .backtest import Backtester, print_backtest_report
except ImportError:
    from data_fetcher import BybitDataFetcher
    from features import FeatureEngineer
    from ensemble import RandomForestEnsemble, GradientBoostingEnsemble, StackingEnsemble
    from uncertainty import UncertaintyQuantifier, UncertaintyAnalyzer
    from calibration import Calibrator
    from strategy import UncertaintyStrategy, PortfolioManager
    from backtest import Backtester, print_backtest_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_data(symbol: str = 'BTC/USDT', timeframe: str = '1h', limit: int = 1000):
    """Fetch market data from Bybit."""
    logger.info(f"Fetching data for {symbol} ({timeframe})...")

    try:
        fetcher = BybitDataFetcher()
        df = fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        logger.info(f"Fetched {len(df)} candles")
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch live data: {e}")
        logger.info("Generating synthetic data for demonstration...")
        return generate_synthetic_data(limit, symbol)


def generate_synthetic_data(n: int, symbol: str) -> pd.DataFrame:
    """Generate synthetic price data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    price = 50000 + np.cumsum(np.random.randn(n) * 100)
    volume = np.random.uniform(100, 1000, n)

    return pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(n) * 50,
        'high': price + np.abs(np.random.randn(n) * 100),
        'low': price - np.abs(np.random.randn(n) * 100),
        'close': price,
        'volume': volume,
        'symbol': symbol
    })


def prepare_features(df: pd.DataFrame):
    """Compute features from price data."""
    logger.info("Computing features...")
    fe = FeatureEngineer()
    df_features = fe.compute_all_features(df)

    X, y, feature_names = fe.prepare_training_data(df_features, lookforward=5)
    logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")

    return X, y, feature_names, df_features


def train_ensemble(X_train: np.ndarray, y_train: np.ndarray):
    """Train ensemble models."""
    logger.info("Training ensemble models...")

    # Random Forest
    rf = RandomForestEnsemble(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)

    # Gradient Boosting with quantile regression
    gb = GradientBoostingEnsemble(n_estimators=100, use_quantile=True)
    gb.fit(X_train, y_train)

    # Stacking ensemble
    stack = StackingEnsemble(
        base_models=[
            RandomForestEnsemble(n_estimators=50),
            GradientBoostingEnsemble(n_estimators=50, use_quantile=False),
        ]
    )
    stack.fit(X_train, y_train, cv=3)

    return {'rf': rf, 'gb': gb, 'stack': stack}


def generate_predictions(models: dict, X: np.ndarray):
    """Generate predictions and uncertainties from ensemble."""
    logger.info("Generating predictions with uncertainty...")

    predictions = {}
    uncertainties = {}

    for name, model in models.items():
        pred, unc = model.predict_with_uncertainty(X)
        predictions[name] = pred
        uncertainties[name] = unc

    # Combine predictions from all models
    all_preds = np.array(list(predictions.values()))
    combined_pred = np.mean(all_preds, axis=0)
    combined_unc = np.std(all_preds, axis=0)

    # Also add individual model uncertainty
    model_uncertainties = np.array(list(uncertainties.values()))
    avg_model_unc = np.mean(model_uncertainties, axis=0)
    combined_unc = np.sqrt(combined_unc**2 + avg_model_unc**2)

    return combined_pred, combined_unc, predictions, uncertainties


def analyze_uncertainty(predictions: np.ndarray, uncertainties: np.ndarray, actuals: np.ndarray):
    """Analyze uncertainty quality."""
    logger.info("Analyzing uncertainty...")

    quantifier = UncertaintyQuantifier()
    analyzer = UncertaintyAnalyzer(quantifier)

    # Reshape for quantifier (expects [n_models, n_samples])
    preds_reshaped = np.expand_dims(predictions, 0)

    summary = analyzer.uncertainty_summary(preds_reshaped, actuals)
    print("\nUncertainty Summary:")
    print(summary.to_string())

    # Check calibration
    calibrator = Calibrator()
    calib_result = calibrator.regression_calibration_error(actuals, predictions, uncertainties)
    print(f"\nCalibration Error: {calib_result['mean_calibration_error']:.4f}")

    return summary, calib_result


def run_trading_strategy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    uncertainties: np.ndarray
):
    """Run trading strategy with uncertainty-aware decisions."""
    logger.info("Running trading strategy...")

    strategy = UncertaintyStrategy(
        min_prediction_threshold=0.001,
        max_uncertainty_threshold=0.03,
        min_confidence_threshold=0.6,
        use_kelly=True
    )

    # Generate signals
    signals = []
    for i in range(len(predictions)):
        signal = strategy.generate_signal(
            symbol=df['symbol'].iloc[i] if 'symbol' in df.columns else 'BTC/USDT',
            prediction=predictions[i],
            uncertainty=uncertainties[i],
            current_price=df['close'].iloc[i],
            timestamp=df['timestamp'].iloc[i]
        )
        signals.append(signal)

    # Filter and display top signals
    filtered = strategy.filter_signals(signals, max_positions=10)
    print(f"\nTop {len(filtered)} Trading Signals:")
    for signal in filtered[:5]:
        print(f"  {signal}")

    return signals


def run_backtest(
    df: pd.DataFrame,
    predictions: np.ndarray,
    uncertainties: np.ndarray
):
    """Run backtest on the strategy."""
    logger.info("Running backtest...")

    # Prepare price data
    prices = df[['timestamp', 'close']].copy()
    prices['symbol'] = df['symbol'] if 'symbol' in df.columns else 'BTC/USDT'

    # Ensure alignment
    min_len = min(len(prices), len(predictions), len(uncertainties))
    prices = prices.iloc[:min_len].reset_index(drop=True)
    predictions = predictions[:min_len]
    uncertainties = uncertainties[:min_len]

    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )

    # Compare with and without uncertainty
    results = backtester.compare_with_without_uncertainty(
        prices, predictions, uncertainties, hold_periods=5
    )

    print_backtest_report(results['with_uncertainty'], "Strategy WITH Uncertainty")
    print_backtest_report(results['without_uncertainty'], "Strategy WITHOUT Uncertainty")

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Ensemble Uncertainty Trading System')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Candle timeframe')
    parser.add_argument('--limit', type=int, default=1000, help='Number of candles')
    parser.add_argument('--demo', action='store_true', help='Use synthetic data')

    args = parser.parse_args()

    print("=" * 70)
    print("ENSEMBLE UNCERTAINTY TRADING SYSTEM")
    print("=" * 70)
    print(f"\nSymbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Data points: {args.limit}")

    # Step 1: Fetch data
    if args.demo:
        df = generate_synthetic_data(args.limit, args.symbol)
    else:
        df = fetch_data(args.symbol, args.timeframe, args.limit)

    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 2: Prepare features
    X, y, feature_names, df_features = prepare_features(df)

    # Split data
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Step 3: Train ensemble
    models = train_ensemble(X_train, y_train)

    # Step 4: Generate predictions with uncertainty
    combined_pred, combined_unc, all_preds, all_uncs = generate_predictions(models, X_test)

    print(f"\nPrediction Statistics:")
    print(f"  Mean prediction: {combined_pred.mean():.6f}")
    print(f"  Mean uncertainty: {combined_unc.mean():.6f}")
    print(f"  Uncertainty range: [{combined_unc.min():.6f}, {combined_unc.max():.6f}]")

    # Step 5: Analyze uncertainty
    summary, calib = analyze_uncertainty(combined_pred, combined_unc, y_test)

    # Step 6: Run trading strategy
    df_test = df_features.iloc[train_size:train_size + len(combined_pred)].reset_index(drop=True)
    signals = run_trading_strategy(df_test, combined_pred, combined_unc)

    # Step 7: Run backtest
    results = run_backtest(df_test, combined_pred, combined_unc)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    r_with = results['with_uncertainty']
    r_without = results['without_uncertainty']

    improvement = {
        'return': r_with.total_return - r_without.total_return,
        'sharpe': r_with.sharpe_ratio - r_without.sharpe_ratio,
        'drawdown': r_with.max_drawdown - r_without.max_drawdown,  # Less negative is better
        'win_rate': r_with.win_rate - r_without.win_rate,
    }

    print(f"\nImprovements from uncertainty-aware trading:")
    print(f"  Return:      {improvement['return']:+.2%}")
    print(f"  Sharpe:      {improvement['sharpe']:+.2f}")
    print(f"  Max DD:      {improvement['drawdown']:+.2%} (less negative is better)")
    print(f"  Win Rate:    {improvement['win_rate']:+.2%}")

    if improvement['sharpe'] > 0:
        print("\n[SUCCESS] Uncertainty-aware trading improved risk-adjusted returns!")
    else:
        print("\n[NOTE] Further tuning may be needed for this dataset")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
