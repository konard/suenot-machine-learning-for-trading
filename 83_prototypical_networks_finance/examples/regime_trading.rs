//! Market Regime Trading Example
//!
//! This example demonstrates how to use prototypical networks
//! for real-time market regime classification and trading signal generation.
//!
//! Run with: cargo run --example regime_trading

use chrono::{TimeZone, Utc};
use ndarray::{Array1, Array2};
use prototypical_networks_finance::prelude::*;
use rand::prelude::*;

#[tokio::main]
async fn main() {
    println!("=== Market Regime Trading Example ===\n");

    // Step 1: Initialize components
    println!("1. Initializing components...");

    // Feature extractor
    let feature_config = FeatureConfig {
        ma_windows: vec![5, 10, 20],
        volatility_window: 14,
        rsi_window: 14,
        include_orderbook: false,
        include_funding: false,
        normalize: true,
    };
    let feature_extractor = FeatureExtractor::with_config(feature_config);

    // Embedding network
    let embedding_config = EmbeddingConfig {
        input_dim: 15, // Expected feature dimension
        hidden_dims: vec![32, 16],
        output_dim: 8,
        normalize_embeddings: true,
        dropout_rate: 0.0,
        activation: ActivationType::ReLU,
    };
    let embedding_network = EmbeddingNetwork::new(embedding_config);

    // Regime classifier
    let mut classifier = RegimeClassifier::new(embedding_network, DistanceFunction::Euclidean);

    // Signal generator
    let signal_config = SignalConfig {
        min_confidence: 0.5,
        strong_signal_threshold: 0.75,
        allow_uncertain: false,
        max_position_size: 1.0,
        base_position_size: 0.5,
    };
    let mut signal_generator = SignalGenerator::with_config(signal_config);

    // Position manager
    let execution_config = ExecutionConfig {
        max_position_size: 10000.0,
        min_position_size: 100.0,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.05,
        use_trailing_stop: true,
        trailing_activation_pct: 0.02,
        trailing_distance_pct: 0.01,
        max_drawdown_pct: 0.10,
    };
    let mut position_manager = PositionManager::new(execution_config);

    println!("   Components initialized.\n");

    // Step 2: Initialize prototypes from historical data
    println!("2. Building support set from historical data...");

    let (support_features, support_labels) = create_historical_support_set(&feature_extractor);
    println!("   - {} support samples across {} regimes",
        support_features.nrows(),
        MarketRegime::count()
    );

    classifier.initialize_prototypes(&support_features, &support_labels);
    println!("   - Prototypes computed\n");

    // Step 3: Simulate real-time trading
    println!("3. Simulating real-time trading...\n");
    println!("{:=<80}", "");
    println!("{:<20} {:>10} {:>10} {:>12} {:>10} {:>15}",
        "Time", "Price", "Regime", "Confidence", "Signal", "Position"
    );
    println!("{:=<80}", "");

    // Generate simulated market data
    let market_data = generate_simulated_market_data(100);
    let mut portfolio_values = vec![10000.0]; // Starting capital
    let mut current_capital = 10000.0;

    for (i, klines) in market_data.windows(50).enumerate() {
        if klines.len() < 50 {
            continue;
        }

        let current_price = klines.last().unwrap().close;
        let current_time = klines.last().unwrap().timestamp;

        // Extract features
        let features = feature_extractor.extract_from_klines(klines);

        // Pad features if needed
        let feature_vec = pad_features(&features.features, 15);

        // Classify regime
        let classification = classifier.classify(&feature_vec);

        // Generate trading signal
        let signal = signal_generator.generate(&classification);

        // Process signal and get orders
        let orders = position_manager.process_signal(&signal, current_price, current_capital);

        // Execute orders
        for order in &orders {
            position_manager.execute_order(order, current_price);
        }

        // Update portfolio value
        let position_value = if position_manager.position().is_open() {
            position_manager.position().unrealized_pnl
        } else {
            0.0
        };
        current_capital = 10000.0 + position_manager.realized_pnl() + position_value;
        portfolio_values.push(current_capital);

        // Print status every 10 periods
        if i % 10 == 0 {
            let position_str = match position_manager.position().side {
                PositionSide::Long => format!("LONG {:.0}", position_manager.position().size),
                PositionSide::Short => format!("SHORT {:.0}", position_manager.position().size),
                PositionSide::Flat => "FLAT".to_string(),
            };

            println!("{:<20} {:>10.2} {:>10} {:>11.1}% {:>10} {:>15}",
                current_time.format("%Y-%m-%d %H:%M"),
                current_price,
                classification.regime.name(),
                classification.confidence * 100.0,
                format!("{:?}", signal.signal_type),
                position_str
            );
        }
    }

    println!("{:=<80}\n", "");

    // Step 4: Calculate performance metrics
    println!("4. Performance Summary:\n");

    let calculator = MetricsCalculator::hourly();
    let metrics = calculator.calculate(&portfolio_values);

    println!("   Portfolio Performance:");
    println!("   - Starting Capital: $10,000.00");
    println!("   - Ending Capital:   ${:.2}", current_capital);
    println!("   - Total Return:     {:.2}%", metrics.total_return * 100.0);
    println!("   - Max Drawdown:     {:.2}%", metrics.max_drawdown * 100.0);
    println!("   - Sharpe Ratio:     {:.2}", metrics.sharpe_ratio);
    println!("   - Volatility:       {:.2}%", metrics.volatility * 100.0);

    println!("\n   Trading Statistics:");
    println!("   - Total Trades:     {}", position_manager.trade_count());
    println!("   - Win Rate:         {:.1}%", position_manager.win_rate() * 100.0);
    println!("   - Realized PnL:     ${:.2}", position_manager.realized_pnl());

    println!("\n=== Example Complete ===");
}

/// Create historical support set for regime classification
fn create_historical_support_set(extractor: &FeatureExtractor) -> (Array2<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(42);
    let samples_per_regime = 20;
    let n_regimes = MarketRegime::count();
    let feature_dim = 15;

    let mut features = Array2::zeros((samples_per_regime * n_regimes, feature_dim));
    let mut labels = Vec::new();

    for regime_idx in 0..n_regimes {
        let regime = MarketRegime::from_index(regime_idx).unwrap();

        for i in 0..samples_per_regime {
            let row_idx = regime_idx * samples_per_regime + i;

            // Generate regime-specific features
            let sample_features = generate_regime_features(regime, &mut rng, feature_dim);
            for j in 0..feature_dim {
                features[[row_idx, j]] = sample_features[j];
            }
            labels.push(regime_idx);
        }
    }

    (features, labels)
}

/// Generate features characteristic of a specific regime
fn generate_regime_features<R: Rng>(regime: MarketRegime, rng: &mut R, dim: usize) -> Vec<f64> {
    let (return_mean, vol_mean, momentum_mean) = match regime {
        MarketRegime::StrongUptrend => (0.03, 0.02, 0.05),
        MarketRegime::WeakUptrend => (0.01, 0.015, 0.02),
        MarketRegime::Sideways => (0.0, 0.01, 0.0),
        MarketRegime::WeakDowntrend => (-0.01, 0.015, -0.02),
        MarketRegime::StrongDowntrend => (-0.03, 0.04, -0.05),
    };

    let noise_scale = 0.01;

    vec![
        return_mean + rng.gen::<f64>() * noise_scale - noise_scale / 2.0, // return_1
        return_mean * 5.0 + rng.gen::<f64>() * noise_scale,                // return_5
        return_mean * 10.0 + rng.gen::<f64>() * noise_scale,               // return_10
        return_mean * 20.0 + rng.gen::<f64>() * noise_scale,               // return_20
        momentum_mean + rng.gen::<f64>() * noise_scale,                    // price_rel_ma_5
        momentum_mean * 0.8 + rng.gen::<f64>() * noise_scale,              // price_rel_ma_10
        momentum_mean * 0.6 + rng.gen::<f64>() * noise_scale,              // price_rel_ma_20
        momentum_mean * 0.4 + rng.gen::<f64>() * noise_scale,              // ma_crossover
        vol_mean + rng.gen::<f64>() * 0.005,                               // volatility
        (rng.gen::<f64>() - 0.5) * 0.3,                                    // volatility_change
        0.5 + return_mean * 5.0 + rng.gen::<f64>() * 0.1,                  // rsi
        vol_mean * 0.5 + rng.gen::<f64>() * 0.002,                         // atr_pct
        rng.gen::<f64>() * 0.5 - 0.25,                                     // relative_volume
        0.5 + return_mean * 3.0 + rng.gen::<f64>() * 0.1,                  // price_position
        momentum_mean * 0.5 + rng.gen::<f64>() * noise_scale,              // momentum_10
    ]
}

/// Generate simulated market data with regime transitions
fn generate_simulated_market_data(n_periods: usize) -> Vec<Kline> {
    let mut rng = StdRng::seed_from_u64(123);
    let mut klines = Vec::with_capacity(n_periods);
    let mut price = 50000.0; // Starting price (e.g., BTC)

    let regimes = [
        (MarketRegime::Sideways, 20),
        (MarketRegime::WeakUptrend, 15),
        (MarketRegime::StrongUptrend, 10),
        (MarketRegime::Sideways, 10),
        (MarketRegime::WeakDowntrend, 15),
        (MarketRegime::StrongDowntrend, 10),
        (MarketRegime::Sideways, 20),
    ];

    let mut current_idx = 0;
    let mut periods_in_regime = 0;
    let mut regime_idx = 0;

    for i in 0..n_periods {
        // Determine current regime
        if regime_idx < regimes.len() && periods_in_regime >= regimes[regime_idx].1 {
            regime_idx += 1;
            periods_in_regime = 0;
        }

        let (regime, _) = if regime_idx < regimes.len() {
            regimes[regime_idx]
        } else {
            (MarketRegime::Sideways, 100)
        };

        // Generate price movement based on regime
        let (drift, vol) = match regime {
            MarketRegime::StrongUptrend => (0.003, 0.015),
            MarketRegime::WeakUptrend => (0.001, 0.01),
            MarketRegime::Sideways => (0.0, 0.008),
            MarketRegime::WeakDowntrend => (-0.001, 0.01),
            MarketRegime::StrongDowntrend => (-0.003, 0.02),
        };

        let return_val = drift + (rng.gen::<f64>() - 0.5) * vol * 2.0;
        let open = price;
        price *= 1.0 + return_val;
        let close = price;

        let high = open.max(close) * (1.0 + rng.gen::<f64>() * vol);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * vol);
        let volume = 100.0 + rng.gen::<f64>() * 200.0;

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
            trade_count: Some(100 + rng.gen_range(0..200)),
        });

        periods_in_regime += 1;
    }

    klines
}

/// Pad features to expected dimension
fn pad_features(features: &Array1<f64>, target_dim: usize) -> Array1<f64> {
    let mut padded = Array1::zeros(target_dim);
    let copy_len = features.len().min(target_dim);
    for i in 0..copy_len {
        padded[i] = features[i];
    }
    padded
}
