//! Demo: Positional Encoding for Time Series
//!
//! This demo shows how to use various positional encodings
//! for financial time series data.

use positional_encoding::{
    // Encoding types
    SinusoidalEncoding, TimeSeriesSinusoidalEncoding, LearnedEncoding,
    RelativeEncoding, RotaryEncoding, PositionalEncoding,
    // Calendar encodings
    CalendarEncoding, MarketSessionEncoding, MultiScaleTemporalEncoding, MarketType,
    // Data utilities
    generate_synthetic_data, FeaturePreparator, SequenceCreator, BybitDataLoader,
    // Strategy
    TradingStrategy, run_backtest, calculate_buy_and_hold,
};

fn print_separator(title: &str) {
    println!("\n{}", "=".repeat(60));
    println!("{}", title);
    println!("{}", "=".repeat(60));
}

fn main() {
    println!("Positional Encoding for Time Series - Demo");
    println!("Chapter 48: Machine Learning for Trading\n");

    // ===== 1. Sinusoidal Encoding =====
    print_separator("1. Sinusoidal Positional Encoding");

    let sinusoidal = SinusoidalEncoding::new(64, 1000);
    let positions = vec![0, 1, 2, 10, 100];
    let encoded = sinusoidal.encode(&positions);

    println!("Encoding dimension: {}", sinusoidal.dim());
    println!("Input positions: {:?}", positions);
    println!("Output shape: {:?}", encoded.shape());
    println!("\nFirst 4 dimensions of position 0: [{:.4}, {:.4}, {:.4}, {:.4}]",
        encoded[[0, 0]], encoded[[0, 1]], encoded[[0, 2]], encoded[[0, 3]]);
    println!("First 4 dimensions of position 1: [{:.4}, {:.4}, {:.4}, {:.4}]",
        encoded[[1, 0]], encoded[[1, 1]], encoded[[1, 2]], encoded[[1, 3]]);

    // ===== 2. Time Series Sinusoidal Encoding =====
    print_separator("2. Time Series Sinusoidal Encoding");

    let ts_encoding = TimeSeriesSinusoidalEncoding::new(64);
    let hourly_positions: Vec<usize> = (0..168).collect(); // One week of hourly data
    let ts_encoded = ts_encoding.encode(&hourly_positions);

    println!("Multi-scale encoding for hourly data");
    println!("Scales: 1h, 24h (daily), 168h (weekly), 720h (monthly)");
    println!("Output shape: {:?}", ts_encoded.shape());

    // ===== 3. Learned Encoding =====
    print_separator("3. Learned Positional Encoding");

    let learned = LearnedEncoding::new(64, 100);
    let learned_encoded = learned.encode(&[5, 10, 15]);

    println!("Learned (randomly initialized) encoding");
    println!("Output shape: {:?}", learned_encoded.shape());
    println!("Embedding norm at pos 5: {:.4}",
        learned_encoded.row(0).iter().map(|x| x * x).sum::<f64>().sqrt());

    // ===== 4. Relative Encoding =====
    print_separator("4. Relative Positional Encoding");

    let relative = RelativeEncoding::new(32, 50);
    let rel_5_to_10 = relative.encode_relative(5, 10);
    let rel_10_to_5 = relative.encode_relative(10, 5);

    println!("Relative encoding between positions");
    println!("Distance 5->10 (forward): first 4 dims = [{:.4}, {:.4}, {:.4}, {:.4}]",
        rel_5_to_10[0], rel_5_to_10[1], rel_5_to_10[2], rel_5_to_10[3]);
    println!("Distance 10->5 (backward): first 4 dims = [{:.4}, {:.4}, {:.4}, {:.4}]",
        rel_10_to_5[0], rel_10_to_5[1], rel_10_to_5[2], rel_10_to_5[3]);

    // ===== 5. Rotary Encoding (RoPE) =====
    print_separator("5. Rotary Positional Encoding (RoPE)");

    let rope = RotaryEncoding::new(64, 1000);
    let x = ndarray::Array1::from_vec(vec![1.0; 64]);
    let rotated_0 = rope.apply_rotation(&x, 0);
    let rotated_1 = rope.apply_rotation(&x, 1);
    let rotated_100 = rope.apply_rotation(&x, 100);

    println!("RoPE rotates vectors based on position");
    let norm_0: f64 = rotated_0.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_100: f64 = rotated_100.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("Original vector norm: {:.4}", x.iter().map(|v| v * v).sum::<f64>().sqrt());
    println!("Rotated (pos 0) norm: {:.4}", norm_0);
    println!("Rotated (pos 100) norm: {:.4}", norm_100);
    println!("Norm is preserved (rotation property)");

    // ===== 6. Calendar Encoding =====
    print_separator("6. Calendar Encoding");

    let calendar = CalendarEncoding::new();

    // Test different timestamps
    let timestamps = vec![
        1704067200,  // 2024-01-01 00:00:00 UTC (Monday)
        1704153600,  // 2024-01-02 00:00:00 UTC (Tuesday)
        1704412800,  // 2024-01-05 00:00:00 UTC (Friday)
        1704499200,  // 2024-01-06 00:00:00 UTC (Saturday)
    ];

    println!("Calendar features for different days:");
    for &ts in &timestamps {
        let encoded = calendar.encode_timestamp(ts);
        let dt = chrono::DateTime::from_timestamp(ts, 0).unwrap();
        println!("  {} - Weekend flag: {:.0}", dt.format("%Y-%m-%d %A"), encoded[12]);
    }

    // ===== 7. Market Session Encoding =====
    print_separator("7. Market Session Encoding");

    let crypto_session = MarketSessionEncoding::new(MarketType::Crypto);
    let stock_session = MarketSessionEncoding::new(MarketType::Stock);

    // Test different hours
    let test_hours = vec![
        1704070800,  // 01:00 UTC (Asia)
        1704099600,  // 10:00 UTC (Europe)
        1704121200,  // 16:00 UTC (Americas)
    ];

    println!("Crypto session encoding:");
    for &ts in &test_hours {
        let encoded = crypto_session.encode_timestamp(ts);
        let dt = chrono::DateTime::from_timestamp(ts, 0).unwrap();
        println!("  {} UTC - Asia: {:.0}, Europe: {:.0}, Americas: {:.0}",
            dt.format("%H:%M"), encoded[0], encoded[1], encoded[2]);
    }

    // ===== 8. Data Loading (Synthetic) =====
    print_separator("8. Data Loading and Feature Preparation");

    let candles = generate_synthetic_data(500, 42);
    println!("Generated {} synthetic candles", candles.len());
    println!("First candle: timestamp={}, open={:.2}, close={:.2}",
        candles[0].timestamp, candles[0].open, candles[0].close);
    println!("Last candle: timestamp={}, open={:.2}, close={:.2}",
        candles[499].timestamp, candles[499].open, candles[499].close);

    // Prepare features
    let features = FeaturePreparator::prepare_features(&candles);
    println!("\nFeature matrix shape: {:?}", features.shape());
    println!("Features: [return, log_volume, range, body_ratio, direction, return_pct]");

    // Calculate returns
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let returns = FeaturePreparator::calculate_returns(&closes);
    let (normalized, mean, std) = FeaturePreparator::normalize_zscore(&returns);
    println!("\nReturns: mean={:.6}, std={:.6}", mean, std);

    // ===== 9. Sequence Creation =====
    print_separator("9. Sequence Creation for Training");

    let targets: Vec<f64> = candles.iter().map(|c| c.return_pct()).collect();
    let timestamps: Vec<i64> = candles.iter().map(|c| c.timestamp).collect();

    let creator = SequenceCreator::new(24, 1); // 24-hour lookback, 1-hour prediction
    let (sequences, target_arrays, ts_arrays) = creator.create_sequences(&features, &targets, &timestamps);

    println!("Sequence length: 24, Target length: 1");
    println!("Number of sequences: {}", sequences.len());
    println!("Sequence shape: {:?}", sequences[0].shape());
    println!("Target shape: {:?}", target_arrays[0].shape());

    // Train/test split
    let (train_seq, test_seq) = SequenceCreator::train_test_split(&sequences, 0.8);
    println!("Train sequences: {}, Test sequences: {}", train_seq.len(), test_seq.len());

    // ===== 10. Trading Strategy and Backtesting =====
    print_separator("10. Trading Strategy and Backtesting");

    // Use synthetic predictions (slightly correlated with actual returns)
    let actual_returns: Vec<f64> = candles.iter().skip(1).map(|c| c.return_pct()).collect();
    let predictions: Vec<f64> = actual_returns.iter()
        .map(|&r| r * 0.3 + (rand::random::<f64>() - 0.5) * 0.01)
        .collect();

    let strategy = TradingStrategy::new(0.001)
        .with_max_position(1.0)
        .with_transaction_cost(0.001);

    let result = run_backtest(&predictions, &actual_returns, &strategy, 100000.0, 8760);

    println!("Strategy Performance:");
    println!("  Total Return: {:.2}%", result.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Sortino Ratio: {:.2}", result.sortino_ratio);
    println!("  Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("  Win Rate: {:.2}%", result.win_rate * 100.0);
    println!("  Number of Trades: {}", result.n_trades);

    // Compare with buy and hold
    let bh_result = calculate_buy_and_hold(&actual_returns, 100000.0, 8760);
    println!("\nBuy & Hold Benchmark:");
    println!("  Total Return: {:.2}%", bh_result.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", bh_result.sharpe_ratio);

    // ===== 11. Multi-Scale Temporal Encoding =====
    print_separator("11. Multi-Scale Temporal Encoding");

    let multi_scale = MultiScaleTemporalEncoding::new(MarketType::Crypto);
    let ts_encoded = multi_scale.encode_timestamps(&timestamps[..10]);

    println!("Combined calendar + session encoding");
    println!("Total dimension: {}", multi_scale.dim());
    println!("Encoded shape: {:?}", ts_encoded.shape());

    // ===== Summary =====
    print_separator("Summary");

    println!("Positional encodings available:");
    println!("  - Sinusoidal: Classic sine/cosine based (dimension: {})", sinusoidal.dim());
    println!("  - Time Series Sinusoidal: Multi-scale (dimension: {})", ts_encoding.dim());
    println!("  - Learned: Trainable embeddings (dimension: 64)");
    println!("  - Relative: Distance-based (dimension: {})", relative.encode_relative(0, 1).len());
    println!("  - Rotary (RoPE): Rotation-based (dimension: 64)");
    println!("  - Calendar: Temporal features (dimension: {})", calendar.dim());
    println!("  - Market Session: Trading hours (dimension: {})", crypto_session.dim());
    println!("  - Multi-Scale: Combined temporal (dimension: {})", multi_scale.dim());

    println!("\nDemo completed successfully!");
}
