//! Backtesting Example
//!
//! This example demonstrates how to backtest a prototypical network
//! trading strategy on historical data.
//!
//! Run with: cargo run --example backtest

use chrono::{TimeZone, Utc};
use ndarray::{Array1, Array2};
use prototypical_networks_finance::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

fn main() {
    println!("=== Prototypical Network Backtest Example ===\n");

    // Configuration
    let config = BacktestConfig {
        initial_capital: 10000.0,
        support_set_size: 100, // Samples per regime for training
        lookback_window: 50,   // Bars for feature extraction
        rebalance_interval: 24, // Hours between rebalancing
    };

    // Run backtest
    let results = run_backtest(&config);

    // Print results
    print_results(&results);
}

struct BacktestConfig {
    initial_capital: f64,
    support_set_size: usize,
    lookback_window: usize,
    rebalance_interval: usize,
}

struct BacktestResults {
    portfolio_values: Vec<f64>,
    trades: Vec<Trade>,
    regime_history: Vec<(chrono::DateTime<Utc>, MarketRegime, f64)>,
    metrics: PerformanceMetrics,
}

struct Trade {
    entry_time: chrono::DateTime<Utc>,
    exit_time: chrono::DateTime<Utc>,
    side: PositionSide,
    entry_price: f64,
    exit_price: f64,
    pnl: f64,
    regime_at_entry: MarketRegime,
}

fn run_backtest(config: &BacktestConfig) -> BacktestResults {
    println!("1. Setting up backtest environment...");

    // Initialize feature extractor
    let feature_config = FeatureConfig {
        ma_windows: vec![5, 10, 20],
        volatility_window: 14,
        rsi_window: 14,
        include_orderbook: false,
        include_funding: false,
        normalize: true,
    };
    let feature_extractor = FeatureExtractor::with_config(feature_config);

    // Initialize embedding network
    let embedding_config = EmbeddingConfig {
        input_dim: 15,
        hidden_dims: vec![32, 16],
        output_dim: 8,
        normalize_embeddings: true,
        dropout_rate: 0.0,
        activation: ActivationType::ReLU,
    };
    let embedding_network = EmbeddingNetwork::new(embedding_config);

    // Initialize classifier
    let mut classifier = RegimeClassifier::new(embedding_network, DistanceFunction::Euclidean);

    // Initialize signal generator
    let signal_config = SignalConfig {
        min_confidence: 0.5,
        strong_signal_threshold: 0.7,
        allow_uncertain: false,
        max_position_size: 1.0,
        base_position_size: 0.5,
    };
    let mut signal_generator = SignalGenerator::with_config(signal_config);

    // Initialize position manager
    let execution_config = ExecutionConfig {
        max_position_size: config.initial_capital,
        min_position_size: 100.0,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.05,
        use_trailing_stop: true,
        trailing_activation_pct: 0.02,
        trailing_distance_pct: 0.01,
        max_drawdown_pct: 0.10,
    };
    let mut position_manager = PositionManager::new(execution_config);

    // Generate historical data
    println!("2. Generating historical market data...");
    let historical_data = generate_historical_data(1000);
    println!("   Generated {} hourly bars\n", historical_data.len());

    // Split into training and testing periods
    let train_end = 200;
    let train_data = &historical_data[..train_end];

    // Create support set from training data
    println!("3. Building support set from training data...");
    let (support_features, support_labels) = extract_labeled_samples(
        train_data,
        &feature_extractor,
        config.support_set_size,
        config.lookback_window,
    );
    println!("   Support set: {} samples\n", support_features.nrows());

    // Initialize prototypes
    classifier.initialize_prototypes(&support_features, &support_labels);
    println!("4. Running backtest on out-of-sample data...");

    // Backtest on remaining data
    let test_data = &historical_data[train_end..];
    let mut portfolio_values = vec![config.initial_capital];
    let mut current_capital = config.initial_capital;
    let mut trades = Vec::new();
    let mut regime_history = Vec::new();
    let mut pending_trade: Option<(chrono::DateTime<Utc>, PositionSide, f64, MarketRegime)> = None;

    for (i, window_end) in (config.lookback_window..test_data.len()).enumerate() {
        let window = &test_data[window_end - config.lookback_window..window_end];
        let current_bar = &test_data[window_end - 1];
        let current_price = current_bar.close;

        // Extract features
        let features = feature_extractor.extract_from_klines(window);
        let feature_vec = pad_features(&features.features, 15);

        // Classify regime
        let classification = classifier.classify(&feature_vec);
        regime_history.push((current_bar.timestamp, classification.regime, classification.confidence));

        // Generate signal
        let signal = signal_generator.generate(&classification);

        // Process signal
        let orders = position_manager.process_signal(&signal, current_price, current_capital);

        // Execute orders and track trades
        for order in &orders {
            // Close pending trade if we're closing position
            if matches!(order.order_type, OrderType::MarketClose | OrderType::StopLoss | OrderType::TakeProfit) {
                if let Some((entry_time, side, entry_price, regime)) = pending_trade.take() {
                    let pnl = match side {
                        PositionSide::Long => (current_price - entry_price) * order.size,
                        PositionSide::Short => (entry_price - current_price) * order.size,
                        PositionSide::Flat => 0.0,
                    };
                    trades.push(Trade {
                        entry_time,
                        exit_time: current_bar.timestamp,
                        side,
                        entry_price,
                        exit_price: current_price,
                        pnl,
                        regime_at_entry: regime,
                    });
                }
            }

            // Track new trade
            if matches!(order.order_type, OrderType::MarketOpen) {
                pending_trade = Some((
                    current_bar.timestamp,
                    order.side,
                    current_price,
                    classification.regime,
                ));
            }

            position_manager.execute_order(order, current_price);
        }

        // Update portfolio value
        let position_value = if position_manager.position().is_open() {
            position_manager.position().unrealized_pnl
        } else {
            0.0
        };
        current_capital = config.initial_capital + position_manager.realized_pnl() + position_value;
        portfolio_values.push(current_capital);

        // Progress update
        if i % 100 == 0 && i > 0 {
            println!("   Processed {} bars, current capital: ${:.2}", i, current_capital);
        }
    }

    // Calculate metrics
    let calculator = MetricsCalculator::hourly();
    let trade_pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
    let trade_durations: Vec<f64> = trades.iter().map(|t| {
        (t.exit_time - t.entry_time).num_hours() as f64
    }).collect();
    let metrics = calculator.calculate_with_trades(&portfolio_values, &trade_pnls, &trade_durations);

    println!("   Backtest complete.\n");

    BacktestResults {
        portfolio_values,
        trades,
        regime_history,
        metrics,
    }
}

fn print_results(results: &BacktestResults) {
    println!("5. Backtest Results");
    println!("{:=<60}", "");

    // Portfolio performance
    println!("\n   PORTFOLIO PERFORMANCE");
    println!("   {:-<50}", "");
    println!("   Starting Capital:    ${:>12.2}", 10000.0);
    println!("   Ending Capital:      ${:>12.2}", results.portfolio_values.last().unwrap_or(&10000.0));
    println!("   Total Return:        {:>12.2}%", results.metrics.total_return * 100.0);
    println!("   Annualized Return:   {:>12.2}%", results.metrics.annualized_return * 100.0);
    println!("   Maximum Drawdown:    {:>12.2}%", results.metrics.max_drawdown * 100.0);

    // Risk metrics
    println!("\n   RISK METRICS");
    println!("   {:-<50}", "");
    println!("   Sharpe Ratio:        {:>12.2}", results.metrics.sharpe_ratio);
    println!("   Sortino Ratio:       {:>12.2}", results.metrics.sortino_ratio);
    println!("   Calmar Ratio:        {:>12.2}", results.metrics.calmar_ratio);
    println!("   Volatility (Ann.):   {:>12.2}%", results.metrics.volatility * 100.0);

    // Trading statistics
    println!("\n   TRADING STATISTICS");
    println!("   {:-<50}", "");
    println!("   Total Trades:        {:>12}", results.trades.len());
    println!("   Winning Trades:      {:>12}", results.trades.iter().filter(|t| t.pnl > 0.0).count());
    println!("   Losing Trades:       {:>12}", results.trades.iter().filter(|t| t.pnl <= 0.0).count());
    println!("   Win Rate:            {:>12.1}%", results.metrics.win_rate * 100.0);
    println!("   Profit Factor:       {:>12.2}", results.metrics.profit_factor);
    println!("   Avg Trade Duration:  {:>12.1} hours", results.metrics.avg_trade_duration);

    // Regime analysis
    println!("\n   REGIME ANALYSIS");
    println!("   {:-<50}", "");

    let mut regime_counts: HashMap<MarketRegime, usize> = HashMap::new();
    for (_, regime, _) in &results.regime_history {
        *regime_counts.entry(*regime).or_insert(0) += 1;
    }

    for regime in MarketRegime::all() {
        let count = regime_counts.get(&regime).unwrap_or(&0);
        let pct = (*count as f64 / results.regime_history.len() as f64) * 100.0;
        println!("   {:20}: {:>5} ({:>5.1}%)", regime.name(), count, pct);
    }

    // Trades by regime
    println!("\n   TRADES BY ENTRY REGIME");
    println!("   {:-<50}", "");

    let mut trades_by_regime: HashMap<MarketRegime, Vec<&Trade>> = HashMap::new();
    for trade in &results.trades {
        trades_by_regime.entry(trade.regime_at_entry).or_insert_with(Vec::new).push(trade);
    }

    for regime in MarketRegime::all() {
        if let Some(trades) = trades_by_regime.get(&regime) {
            let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
            let win_rate = trades.iter().filter(|t| t.pnl > 0.0).count() as f64 / trades.len() as f64;
            println!("   {:20}: {:>3} trades, PnL: ${:>8.2}, Win Rate: {:>5.1}%",
                regime.name(), trades.len(), total_pnl, win_rate * 100.0);
        }
    }

    // Summary
    println!("\n{:=<60}", "");
    let status = if results.metrics.total_return > 0.0 { "PROFITABLE" } else { "UNPROFITABLE" };
    let risk_status = if results.metrics.sharpe_ratio > 1.0 { "ACCEPTABLE" } else { "HIGH" };
    println!("   Strategy Status: {} | Risk Level: {}", status, risk_status);
    println!("{:=<60}", "");
}

/// Extract labeled samples for support set
fn extract_labeled_samples(
    data: &[Kline],
    extractor: &FeatureExtractor,
    samples_per_regime: usize,
    lookback: usize,
) -> (Array2<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(42);
    let feature_dim = 15;

    // Label data based on forward returns
    let mut labeled_windows: Vec<(usize, Vec<f64>, usize)> = Vec::new();

    for i in lookback..(data.len() - 10) {
        let window = &data[i - lookback..i];
        let features = extractor.extract_from_klines(window);

        // Calculate forward return for labeling
        let current_price = data[i - 1].close;
        let future_price = data[i + 10 - 1].close; // 10-bar forward
        let forward_return = (future_price / current_price - 1.0) * 100.0;

        // Determine regime based on forward return
        let regime_idx = if forward_return > 3.0 {
            0 // StrongUptrend
        } else if forward_return > 1.0 {
            1 // WeakUptrend
        } else if forward_return > -1.0 {
            2 // Sideways
        } else if forward_return > -3.0 {
            3 // WeakDowntrend
        } else {
            4 // StrongDowntrend
        };

        let mut feature_vec = vec![0.0; feature_dim];
        for (j, &val) in features.features.iter().enumerate() {
            if j < feature_dim {
                feature_vec[j] = val;
            }
        }

        labeled_windows.push((i, feature_vec, regime_idx));
    }

    // Sample balanced set
    let mut selected: Vec<(Vec<f64>, usize)> = Vec::new();

    for regime_idx in 0..MarketRegime::count() {
        let regime_samples: Vec<_> = labeled_windows
            .iter()
            .filter(|(_, _, r)| *r == regime_idx)
            .collect();

        if regime_samples.is_empty() {
            // Generate synthetic samples if none available
            for _ in 0..samples_per_regime {
                selected.push((generate_regime_sample(regime_idx, feature_dim, &mut rng), regime_idx));
            }
        } else {
            let sample_count = samples_per_regime.min(regime_samples.len());
            let indices: Vec<usize> = (0..regime_samples.len())
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, sample_count)
                .cloned()
                .collect();

            for idx in indices {
                selected.push((regime_samples[idx].1.clone(), regime_idx));
            }
        }
    }

    // Convert to arrays
    let n_samples = selected.len();
    let mut features = Array2::zeros((n_samples, feature_dim));
    let mut labels = Vec::with_capacity(n_samples);

    for (i, (feat, label)) in selected.into_iter().enumerate() {
        for (j, &val) in feat.iter().enumerate() {
            features[[i, j]] = val;
        }
        labels.push(label);
    }

    (features, labels)
}

/// Generate synthetic sample for a regime
fn generate_regime_sample<R: Rng>(regime_idx: usize, dim: usize, rng: &mut R) -> Vec<f64> {
    let (bias, scale) = match regime_idx {
        0 => (0.3, 0.1),  // StrongUptrend
        1 => (0.1, 0.08), // WeakUptrend
        2 => (0.0, 0.05), // Sideways
        3 => (-0.1, 0.08),// WeakDowntrend
        4 => (-0.3, 0.1), // StrongDowntrend
        _ => (0.0, 0.05),
    };

    (0..dim)
        .map(|_| bias + (rng.gen::<f64>() - 0.5) * scale)
        .collect()
}

/// Generate historical market data with realistic regime transitions
fn generate_historical_data(n_bars: usize) -> Vec<Kline> {
    let mut rng = StdRng::seed_from_u64(456);
    let mut klines = Vec::with_capacity(n_bars);
    let mut price = 50000.0;

    // Define regime sequence
    let regime_durations = vec![
        (MarketRegime::Sideways, 50),
        (MarketRegime::WeakUptrend, 80),
        (MarketRegime::StrongUptrend, 60),
        (MarketRegime::Sideways, 40),
        (MarketRegime::WeakDowntrend, 70),
        (MarketRegime::StrongDowntrend, 50),
        (MarketRegime::Sideways, 60),
        (MarketRegime::WeakUptrend, 90),
        (MarketRegime::StrongUptrend, 70),
        (MarketRegime::WeakDowntrend, 80),
        (MarketRegime::Sideways, 100),
        (MarketRegime::StrongDowntrend, 60),
        (MarketRegime::WeakUptrend, 90),
        (MarketRegime::Sideways, 100),
    ];

    let mut current_regime_idx = 0;
    let mut bars_in_regime = 0;

    for i in 0..n_bars {
        // Check regime transition
        if current_regime_idx < regime_durations.len()
            && bars_in_regime >= regime_durations[current_regime_idx].1
        {
            current_regime_idx += 1;
            bars_in_regime = 0;
        }

        let regime = if current_regime_idx < regime_durations.len() {
            regime_durations[current_regime_idx].0
        } else {
            MarketRegime::Sideways
        };

        // Generate price movement
        let (drift, volatility) = match regime {
            MarketRegime::StrongUptrend => (0.002, 0.012),
            MarketRegime::WeakUptrend => (0.0008, 0.008),
            MarketRegime::Sideways => (0.0, 0.006),
            MarketRegime::WeakDowntrend => (-0.0008, 0.008),
            MarketRegime::StrongDowntrend => (-0.002, 0.015),
        };

        let return_val = drift + (rng.gen::<f64>() - 0.5) * volatility * 2.0;
        let open = price;
        price *= 1.0 + return_val;
        let close = price;

        let intrabar_vol = volatility * 0.5;
        let high = open.max(close) * (1.0 + rng.gen::<f64>() * intrabar_vol);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * intrabar_vol);
        let volume = 100.0 + rng.gen::<f64>() * 300.0;

        let timestamp = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
            + chrono::Duration::hours(i as i64);

        klines.push(Kline {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            quote_volume: volume * close,
            trade_count: Some(rng.gen_range(50..300)),
        });

        bars_in_regime += 1;
    }

    klines
}

fn pad_features(features: &Array1<f64>, target_dim: usize) -> Array1<f64> {
    let mut padded = Array1::zeros(target_dim);
    let copy_len = features.len().min(target_dim);
    for i in 0..copy_len {
        padded[i] = features[i];
    }
    padded
}
