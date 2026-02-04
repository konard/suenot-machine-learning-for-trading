#!/usr/bin/env python3
"""
Conformal Prediction Trading - Main Entry Point

This script demonstrates the full pipeline:
1. Fetch data from Bybit via CCXT
2. Engineer features
3. Train base model and calibrate conformal predictor
4. Generate trading signals with uncertainty quantification
5. Run backtest and evaluate performance
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

from sklearn.ensemble import GradientBoostingRegressor

from data_fetcher import BybitDataFetcher, DEFAULT_SYMBOLS
from features import FeatureEngineer
from conformal_predictor import SplitConformalPredictor, compute_coverage_metrics
from models import get_default_model, create_quantile_pair
from trading_strategy import ConformalTradingStrategy, signals_to_dataframe
from backtest import ConformalBacktester, print_backtest_summary

warnings.filterwarnings('ignore')


def fetch_data(args):
    """Fetch market data from Bybit."""
    print("\n" + "=" * 60)
    print("FETCHING MARKET DATA")
    print("=" * 60)

    fetcher = BybitDataFetcher()

    symbol = args.symbol
    timeframe = args.timeframe
    days = args.days

    print(f"\nSymbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Days: {days}")

    df = fetcher.fetch_historical_data(symbol, timeframe, days)

    if df.empty:
        print("Error: No data fetched")
        return None

    # Save to CSV
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    filename = f"{symbol.replace('/', '_').replace(':', '_')}_{timeframe}_{days}d.csv"
    output_path = output_dir / filename
    df.to_csv(output_path)

    print(f"\nData saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def train_model(args):
    """Train model and create conformal predictor."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    # Load or fetch data
    data_dir = Path(__file__).parent / "data"
    symbol_clean = args.symbol.replace('/', '_').replace(':', '_')
    data_files = list(data_dir.glob(f"{symbol_clean}_*.csv"))

    if data_files:
        df = pd.read_csv(data_files[0], index_col=0, parse_dates=True)
        print(f"Loaded data from: {data_files[0]}")
    else:
        print("No saved data found, fetching...")
        df = fetch_data(args)
        if df is None:
            return

    # Create features
    print("\nEngineering features...")
    engineer = FeatureEngineer(prediction_horizon=args.prediction_horizon)
    df_features = engineer.create_features(df)

    print(f"Features created: {len(engineer.get_feature_columns())}")
    print(f"Samples after feature engineering: {len(df_features)}")

    # Prepare data
    X_train, y_train, X_cal, y_cal, X_test, y_test, test_index = engineer.prepare_data(
        df_features,
        train_ratio=0.6,
        cal_ratio=0.2
    )

    print(f"\nData splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Calibration: {len(X_cal)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Train base model
    print(f"\nTraining base model ({args.model_type})...")
    base_model = get_default_model(
        args.model_type,
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # Create conformal predictor
    cp = SplitConformalPredictor(
        base_model,
        alpha=args.alpha,
        score_function='absolute'
    )

    # Fit and calibrate
    cp.fit_calibrate(X_train, y_train, X_cal, y_cal)

    print(f"\nConformal predictor calibrated!")
    print(f"  Alpha (miscoverage rate): {args.alpha}")
    print(f"  Target coverage: {(1 - args.alpha) * 100:.0f}%")
    print(f"  Calibration quantile: {cp.quantile:.6f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    coverage, avg_width = cp.evaluate_coverage(X_test, y_test)

    print(f"\nTest Set Results:")
    print(f"  Coverage: {coverage * 100:.2f}% (target: {(1 - args.alpha) * 100:.0f}%)")
    print(f"  Average interval width: {avg_width:.6f}")

    # Save results
    output_dir = Path(__file__).parent / "models"
    output_dir.mkdir(exist_ok=True)

    # Store model info (in practice, use joblib to save model)
    model_info = {
        'symbol': args.symbol,
        'alpha': args.alpha,
        'coverage': coverage,
        'avg_width': avg_width,
        'calibration_quantile': cp.quantile
    }

    print(f"\nModel training complete!")

    return cp, X_test, y_test, df_features['close'].values[-len(X_test):], test_index


def run_prediction(args):
    """Generate predictions with conformal intervals."""
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    # This would load the saved model in practice
    # For demo, we train a new one
    result = train_model(args)
    if result is None:
        return

    cp, X_test, y_test, prices, timestamps = result

    # Generate predictions
    point_preds, lower_bounds, upper_bounds = cp.predict(X_test[:10])

    print("\nSample Predictions:")
    print("-" * 80)
    print(f"{'Timestamp':<20} {'Point Pred':>12} {'Lower':>12} {'Upper':>12} {'Actual':>12} {'Covered':>8}")
    print("-" * 80)

    for i in range(min(10, len(point_preds))):
        covered = lower_bounds[i] <= y_test[i] <= upper_bounds[i]
        print(f"{str(timestamps[i])[:19]:<20} "
              f"{point_preds[i]:>12.6f} "
              f"{lower_bounds[i]:>12.6f} "
              f"{upper_bounds[i]:>12.6f} "
              f"{y_test[i]:>12.6f} "
              f"{'Yes' if covered else 'No':>8}")

    # Coverage summary
    all_point, all_lower, all_upper = cp.predict(X_test)
    metrics = compute_coverage_metrics(y_test, all_lower, all_upper, args.alpha)

    print("\nCoverage Summary:")
    print(f"  Empirical coverage: {metrics['coverage'] * 100:.2f}%")
    print(f"  Target coverage: {metrics['target_coverage'] * 100:.2f}%")
    print(f"  Mean interval width: {metrics['mean_width']:.6f}")


def run_backtest(args):
    """Run full backtest of conformal trading strategy."""
    print("\n" + "=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)

    # Train model
    result = train_model(args)
    if result is None:
        return

    cp, X_test, y_test, prices, timestamps = result

    # Load full data for y_train
    data_dir = Path(__file__).parent / "data"
    symbol_clean = args.symbol.replace('/', '_').replace(':', '_')
    data_files = list(data_dir.glob(f"{symbol_clean}_*.csv"))

    if data_files:
        df = pd.read_csv(data_files[0], index_col=0, parse_dates=True)
        engineer = FeatureEngineer(prediction_horizon=args.prediction_horizon)
        df_features = engineer.create_features(df)
        X_train, y_train, _, _, _, _, _ = engineer.prepare_data(df_features)
    else:
        y_train = y_test[:100]  # Fallback

    # Create backtester
    backtester = ConformalBacktester(
        cp,
        strategy_config={
            'max_position': args.max_position,
            'min_confidence': args.min_confidence,
            'volatility_threshold': 0.05,
            'position_scaling': 'inverse_width'
        },
        portfolio_config={
            'initial_capital': args.initial_capital,
            'transaction_cost': 0.001,
            'slippage': 0.0005
        }
    )

    # Run backtest
    result = backtester.run_backtest(
        X_test, y_test, prices, timestamps, y_train
    )

    # Print results
    print_backtest_summary(result)

    # Save results
    signals_df = signals_to_dataframe(result.signals)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    signals_path = output_dir / f"signals_{symbol_clean}.csv"
    signals_df.to_csv(signals_path)
    print(f"\nSignals saved to: {signals_path}")

    # Generate plot if requested
    if args.plot:
        try:
            from backtest import plot_backtest_results
            plot_path = output_dir / f"backtest_{symbol_clean}.png"
            plot_backtest_results(result, save_path=str(plot_path))
        except Exception as e:
            print(f"Could not generate plot: {e}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Conformal Prediction Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data
  python main.py --fetch-data --symbol BTC/USDT:USDT --days 30

  # Train model
  python main.py --train --symbol BTC/USDT:USDT

  # Generate predictions
  python main.py --predict --symbol BTC/USDT:USDT

  # Run backtest
  python main.py --backtest --symbol BTC/USDT:USDT --plot
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--fetch-data', action='store_true',
                           help='Fetch market data from Bybit')
    mode_group.add_argument('--train', action='store_true',
                           help='Train model and calibrate conformal predictor')
    mode_group.add_argument('--predict', action='store_true',
                           help='Generate predictions with intervals')
    mode_group.add_argument('--backtest', action='store_true',
                           help='Run full backtest')

    # Data parameters
    parser.add_argument('--symbol', type=str, default='BTC/USDT:USDT',
                       help='Trading symbol (default: BTC/USDT:USDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Candlestick timeframe (default: 1h)')
    parser.add_argument('--days', type=int, default=30,
                       help='Days of historical data (default: 30)')

    # Model parameters
    parser.add_argument('--model-type', type=str, default='gradient_boosting',
                       choices=['gradient_boosting', 'random_forest', 'ridge', 'ensemble'],
                       help='Base model type (default: gradient_boosting)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Miscoverage rate (default: 0.1 for 90%% coverage)')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                       help='Prediction horizon in periods (default: 5)')

    # Trading parameters
    parser.add_argument('--max-position', type=float, default=1.0,
                       help='Maximum position size (default: 1.0)')
    parser.add_argument('--min-confidence', type=float, default=0.4,
                       help='Minimum confidence to trade (default: 0.4)')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                       help='Initial capital (default: 10000)')

    # Output
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CONFORMAL PREDICTION TRADING SYSTEM")
    print("=" * 60)
    print(f"\nSymbol: {args.symbol}")
    print(f"Alpha: {args.alpha} (Target coverage: {(1-args.alpha)*100:.0f}%)")

    if args.fetch_data:
        fetch_data(args)

    elif args.train:
        train_model(args)

    elif args.predict:
        run_prediction(args)

    elif args.backtest:
        run_backtest(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
