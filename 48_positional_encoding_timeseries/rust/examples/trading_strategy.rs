//! Example: Trading Strategy with Positional Encoding
//!
//! This example shows how to combine positional encodings with trading
//! strategies and backtesting for financial time series.

use positional_encoding::{
    // Data
    generate_synthetic_data, FeaturePreparator, SequenceCreator, Candle,
    // Encodings
    CalendarEncoding, MarketSessionEncoding, MultiScaleTemporalEncoding, MarketType,
    SinusoidalEncoding, PositionalEncoding,
    // Strategy
    TradingStrategy, Signal, run_backtest, calculate_buy_and_hold, compare_strategies,
};

fn main() {
    println!("Trading Strategy Example");
    println!("========================\n");

    // 1. Generate or load data
    println!("1. Data Preparation");
    println!("-------------------");

    // Generate synthetic hourly data (1 year = 8760 hours)
    let candles = generate_synthetic_data(2000, 42);
    println!("Generated {} hourly candles", candles.len());

    // Extract price information
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let timestamps: Vec<i64> = candles.iter().map(|c| c.timestamp).collect();

    // Calculate returns
    let returns = FeaturePreparator::calculate_returns(&closes);
    println!("Calculated {} return observations", returns.len());

    // Show statistics
    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_return: f64 = (returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
        / returns.len() as f64).sqrt();
    println!("Mean return: {:.6}", mean_return);
    println!("Std return: {:.6}", std_return);

    // 2. Feature Engineering with Positional Encoding
    println!("\n2. Feature Engineering");
    println!("----------------------");

    // Prepare base features from candles
    let base_features = FeaturePreparator::prepare_features(&candles);
    println!("Base features shape: {:?}", base_features.shape());
    println!("Features: return, log_volume, range, body_ratio, direction, return_pct");

    // Add positional encoding
    let pos_encoding = SinusoidalEncoding::new(16, 2000);
    let positions: Vec<usize> = (0..candles.len()).collect();
    let pos_features = pos_encoding.encode(&positions);
    println!("Position encoding shape: {:?}", pos_features.shape());

    // Add calendar encoding
    let calendar = CalendarEncoding::new();
    let cal_features = calendar.encode_timestamps(&timestamps);
    println!("Calendar encoding shape: {:?}", cal_features.shape());

    // Add market session encoding
    let session = MarketSessionEncoding::new(MarketType::Crypto);
    let session_features = session.encode_timestamps(&timestamps);
    println!("Session encoding shape: {:?}", session_features.shape());

    // Total feature dimension
    let total_dim = base_features.ncols() + pos_features.ncols()
        + cal_features.ncols() + session_features.ncols();
    println!("Total feature dimension: {}", total_dim);

    // 3. Create Training Sequences
    println!("\n3. Sequence Creation");
    println!("--------------------");

    let targets: Vec<f64> = candles.iter().map(|c| c.return_pct()).collect();
    let creator = SequenceCreator::new(24, 1);  // 24-hour lookback, 1-hour prediction

    let (sequences, target_arrays, _) = creator.create_sequences(&base_features, &targets, &timestamps);
    println!("Created {} sequences", sequences.len());
    println!("Sequence shape: {:?}", sequences[0].shape());

    // Train/test split
    let (train_seq, test_seq) = SequenceCreator::train_test_split(&sequences, 0.8);
    let (train_targets, test_targets) = SequenceCreator::train_test_split(&target_arrays, 0.8);
    println!("Train: {} sequences, Test: {} sequences", train_seq.len(), test_seq.len());

    // 4. Trading Strategy
    println!("\n4. Trading Strategy");
    println!("-------------------");

    // Create a simple momentum strategy
    // (In practice, this would use model predictions)
    let strategy = TradingStrategy::new(0.001)
        .with_max_position(1.0)
        .with_transaction_cost(0.001);

    println!("Strategy configuration:");
    println!("  Threshold: {:.4}", strategy.threshold);
    println!("  Max Position: {:.1}", strategy.max_position);
    println!("  Transaction Cost: {:.4}", strategy.transaction_cost);

    // Generate signals from a simple moving average crossover
    // (simulating model predictions)
    let window = 24;
    let mut predictions = vec![0.0; returns.len()];
    for i in window..returns.len() {
        let ma_short: f64 = returns[i-12..i].iter().sum::<f64>() / 12.0;
        let ma_long: f64 = returns[i-24..i].iter().sum::<f64>() / 24.0;
        predictions[i] = ma_short - ma_long;  // Momentum signal
    }

    // Generate signals
    let signals = strategy.generate_signals(&predictions);
    let long_count = signals.iter().filter(|&&s| s == Signal::Long).count();
    let short_count = signals.iter().filter(|&&s| s == Signal::Short).count();
    let neutral_count = signals.iter().filter(|&&s| s == Signal::Neutral).count();

    println!("\nSignal distribution:");
    println!("  Long: {} ({:.1}%)", long_count, 100.0 * long_count as f64 / signals.len() as f64);
    println!("  Short: {} ({:.1}%)", short_count, 100.0 * short_count as f64 / signals.len() as f64);
    println!("  Neutral: {} ({:.1}%)", neutral_count, 100.0 * neutral_count as f64 / signals.len() as f64);

    // 5. Backtesting
    println!("\n5. Backtesting");
    println!("--------------");

    let backtest_result = run_backtest(&predictions, &returns, &strategy, 100000.0, 8760);

    println!("\nStrategy Performance:");
    println!("  Total Return: {:.2}%", backtest_result.total_return * 100.0);
    println!("  Annual Return: {:.2}%", backtest_result.metrics.annual_return * 100.0);
    println!("  Volatility: {:.2}%", backtest_result.metrics.volatility * 100.0);
    println!("  Sharpe Ratio: {:.2}", backtest_result.sharpe_ratio);
    println!("  Sortino Ratio: {:.2}", backtest_result.sortino_ratio);
    println!("  Max Drawdown: {:.2}%", backtest_result.max_drawdown * 100.0);
    println!("  Calmar Ratio: {:.2}", backtest_result.metrics.calmar_ratio);
    println!("  Win Rate: {:.2}%", backtest_result.win_rate * 100.0);
    println!("  Number of Trades: {}", backtest_result.n_trades);

    // 6. Benchmark Comparison
    println!("\n6. Benchmark Comparison");
    println!("-----------------------");

    let bh_result = calculate_buy_and_hold(&returns, 100000.0, 8760);

    println!("\nBuy & Hold Performance:");
    println!("  Total Return: {:.2}%", bh_result.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", bh_result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", bh_result.max_drawdown * 100.0);

    println!("\nStrategy vs Buy & Hold:");
    let excess_return = backtest_result.total_return - bh_result.total_return;
    let sharpe_diff = backtest_result.sharpe_ratio - bh_result.sharpe_ratio;
    println!("  Excess Return: {:.2}%", excess_return * 100.0);
    println!("  Sharpe Difference: {:.2}", sharpe_diff);

    // 7. Threshold Comparison
    println!("\n7. Threshold Optimization");
    println!("-------------------------");

    let thresholds = vec![0.0005, 0.001, 0.002, 0.005];
    let comparison = compare_strategies(&predictions, &returns, &thresholds, 8760);

    println!("\n| Threshold | Return   | Sharpe | MaxDD   | Trades |");
    println!("|-----------|----------|--------|---------|--------|");
    for (threshold, result) in comparison {
        println!("| {:.4}    | {:>7.2}% | {:>6.2} | {:>6.2}% | {:>6} |",
            threshold,
            result.total_return * 100.0,
            result.sharpe_ratio,
            result.max_drawdown * 100.0,
            result.n_trades
        );
    }

    // 8. Summary
    println!("\n8. Integration with Positional Encoding");
    println!("---------------------------------------");

    println!("\nHow positional encoding helps trading strategies:");
    println!();
    println!("1. Sequence Position (Sinusoidal/RoPE):");
    println!("   - Helps model understand temporal order of observations");
    println!("   - Enables attention to focus on recent vs distant past");
    println!();
    println!("2. Calendar Features:");
    println!("   - Captures day-of-week effects (Monday dip, Friday close)");
    println!("   - Month-end rebalancing patterns");
    println!("   - Seasonal/quarterly patterns");
    println!();
    println!("3. Market Session:");
    println!("   - Regional activity patterns (Asia/Europe/Americas)");
    println!("   - Session overlap volatility");
    println!("   - Pre/post market behavior");
    println!();
    println!("4. Multi-Scale Temporal:");
    println!("   - Combines all temporal features");
    println!("   - Captures patterns at hourly, daily, weekly scales");
    println!();

    println!("Recommended Pipeline:");
    println!("  1. Prepare base features (OHLCV + technical indicators)");
    println!("  2. Add positional encoding (RoPE for transformers)");
    println!("  3. Add calendar encoding (for time-aware predictions)");
    println!("  4. Add market session encoding (for intraday)");
    println!("  5. Train transformer model with combined features");
    println!("  6. Generate predictions and signals");
    println!("  7. Backtest with realistic transaction costs");
    println!("  8. Optimize threshold and position sizing");

    println!("\nExample completed!");
}
