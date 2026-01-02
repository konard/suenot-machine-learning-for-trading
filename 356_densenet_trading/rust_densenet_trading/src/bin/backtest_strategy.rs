//! Backtest DenseNet Trading Strategy
//!
//! Runs a backtest on historical data.
//!
//! Usage:
//!   cargo run --bin backtest_strategy

use rust_densenet_trading::data::candle::{Candle, CandleSeries};
use rust_densenet_trading::densenet::{DenseNet, DenseNetConfig};
use rust_densenet_trading::features::extractor::FeatureExtractor;
use rust_densenet_trading::trading::backtester::{BacktestConfig, Backtester};
use rust_densenet_trading::trading::signal::{SignalDirection, TradingSignal};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          DenseNet Trading - Backtest Demo                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let model_config = DenseNetConfig::tiny(); // Fast for demo
    let backtest_config = BacktestConfig {
        initial_capital: 10000.0,
        fee_rate: 0.0004,    // Bybit taker fee
        slippage: 0.0001,    // 0.01% slippage
        max_position_size: 0.25,
        min_confidence: 0.35,
        use_stop_loss: true,
        stop_loss_atr: 2.0,
        use_take_profit: true,
        take_profit_rr: 2.0,
    };

    println!("Backtest Configuration:");
    println!("  Initial Capital: ${:.2}", backtest_config.initial_capital);
    println!("  Fee Rate: {:.4}%", backtest_config.fee_rate * 100.0);
    println!("  Max Position Size: {:.0}%", backtest_config.max_position_size * 100.0);
    println!("  Min Confidence: {:.0}%", backtest_config.min_confidence * 100.0);
    println!("  Stop Loss ATR: {:.1}x", backtest_config.stop_loss_atr);
    println!("  Take Profit R:R: {:.1}:1", backtest_config.take_profit_rr);
    println!();

    // Generate synthetic data for backtest
    println!("Generating synthetic market data...");
    let candles = generate_market_scenario(1000);
    println!("  Generated {} candles", candles.len());

    // Create model
    println!("Creating DenseNet model...");
    let model = DenseNet::new(model_config.clone());
    println!("  Parameters: {}", model.num_parameters());

    // Extract features
    println!("Extracting features...");
    let extractor = FeatureExtractor::new();
    let features = extractor.extract_from_candles(&candles);

    // Generate trading signals
    println!("Generating trading signals...");
    let mut signals = Vec::new();

    let seq_len = model_config.sequence_length;
    for i in 0..candles.len() {
        if i < seq_len {
            // Not enough history, generate neutral signal
            signals.push(TradingSignal::new(
                SignalDirection::Neutral,
                0.0,
                candles.candles[i].timestamp,
                candles.candles[i].close,
                vec![0.33, 0.34, 0.33],
            ));
            continue;
        }

        // Extract feature window
        let mut feature_window =
            ndarray::Array2::zeros((extractor.num_features(), seq_len));
        for j in 0..seq_len {
            for f in 0..extractor.num_features().min(features.dim().1) {
                feature_window[[f, j]] = features[[i - seq_len + j, f]];
            }
        }

        // Get prediction
        let prediction = model.forward(&feature_window, false);

        // Create signal
        let signal = TradingSignal::from_prediction(
            &prediction.probabilities,
            prediction.confidence,
            candles.candles[i].timestamp,
            candles.candles[i].close,
            backtest_config.min_confidence,
        )
        .with_position_size(prediction.confidence);

        signals.push(signal);
    }

    // Count signals
    let long_count = signals.iter().filter(|s| s.direction == SignalDirection::Long).count();
    let short_count = signals.iter().filter(|s| s.direction == SignalDirection::Short).count();
    let neutral_count = signals.iter().filter(|s| s.direction == SignalDirection::Neutral).count();

    println!("  Long signals: {}", long_count);
    println!("  Short signals: {}", short_count);
    println!("  Neutral signals: {}", neutral_count);
    println!();

    // Run backtest
    println!("Running backtest...");
    let mut backtester = Backtester::new(backtest_config);
    let result = backtester.run(&candles, &signals);

    // Display results
    println!();
    println!("{}", result.summary());

    // Additional analysis
    if !result.trades.is_empty() {
        println!();
        println!("Trade Analysis:");
        println!("{:-<50}", "");

        // Exit reasons
        let mut stop_losses = 0;
        let mut take_profits = 0;
        let mut signal_exits = 0;

        for trade in &result.trades {
            match trade.exit_reason {
                rust_densenet_trading::trading::position::ExitReason::StopLoss => stop_losses += 1,
                rust_densenet_trading::trading::position::ExitReason::TakeProfit => take_profits += 1,
                rust_densenet_trading::trading::position::ExitReason::Signal => signal_exits += 1,
                _ => {}
            }
        }

        println!("Exit Reasons:");
        println!("  Stop Loss: {}", stop_losses);
        println!("  Take Profit: {}", take_profits);
        println!("  Signal: {}", signal_exits);

        // Best and worst trades
        let mut sorted_trades = result.trades.clone();
        sorted_trades.sort_by(|a, b| b.realized_pnl.partial_cmp(&a.realized_pnl).unwrap());

        println!();
        println!("Best Trade: ${:.2}", sorted_trades.first().map(|t| t.realized_pnl).unwrap_or(0.0));
        println!("Worst Trade: ${:.2}", sorted_trades.last().map(|t| t.realized_pnl).unwrap_or(0.0));

        // Average holding period
        let avg_hold: u64 = result.trades.iter().map(|t| t.holding_period()).sum::<u64>()
            / result.trades.len().max(1) as u64;
        println!("Avg Holding Period: {} min", avg_hold / 60000);
    }

    // Equity curve summary
    if !result.equity_curve.is_empty() {
        println!();
        println!("Equity Curve:");
        println!("{:-<50}", "");

        let equity_values: Vec<f64> = result.equity_curve.iter().map(|(_, e)| *e).collect();
        let min_equity = equity_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_equity = equity_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("  Start: ${:.2}", equity_values.first().unwrap_or(&0.0));
        println!("  End: ${:.2}", equity_values.last().unwrap_or(&0.0));
        println!("  Min: ${:.2}", min_equity);
        println!("  Max: ${:.2}", max_equity);
    }

    println!();
    println!("Backtest complete!");
}

/// Generate synthetic market scenario with trends and volatility
fn generate_market_scenario(n: usize) -> CandleSeries {
    let mut candles = CandleSeries::new();
    let mut price = 45000.0;
    let mut trend = 0.0;
    let mut volatility = 150.0;
    let mut rng_seed = 42u64;

    for i in 0..n {
        // Update trend (regime switching)
        if i % 100 == 0 {
            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let rand = (rng_seed % 1000) as f64 / 1000.0;
            trend = (rand - 0.5) * 100.0; // New trend direction
            volatility = 100.0 + rand * 200.0; // New volatility regime
        }

        // Random components
        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let rand1 = (rng_seed % 1000) as f64 / 1000.0 - 0.5;

        rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        let rand2 = (rng_seed % 1000) as f64 / 1000.0;

        // Price evolution
        let change = trend * 0.01 + rand1 * volatility;
        price = (price + change).max(30000.0).min(60000.0);

        // Generate candle
        let open = price;
        let noise = rand1 * volatility * 0.5;
        let close = price + noise;
        let high = open.max(close) + rand2 * volatility * 0.3;
        let low = open.min(close) - rand2 * volatility * 0.3;
        let volume = 50.0 + rand2 * 300.0;

        candles.push(Candle::new(
            (i as u64) * 900_000, // 15-minute candles
            open,
            high,
            low.max(1.0),
            close,
            volume,
            volume * close,
        ));
    }

    candles
}
