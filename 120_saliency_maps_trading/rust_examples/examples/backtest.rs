//! Example: Backtesting saliency-based trading strategies
//!
//! This example demonstrates how to backtest trading strategies
//! that use saliency maps for signal filtering.

use ndarray::Array2;
use saliency_maps_trading::{
    models::network::TradingNetwork,
    saliency::gradient::SaliencyTrader,
};

/// Simple backtest result
struct BacktestResult {
    total_return: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    num_trades: usize,
    win_rate: f64,
}

fn main() {
    env_logger::init();

    println!("Saliency Maps Trading - Backtest");
    println!("=================================\n");

    // Configuration
    let sequence_length = 30;
    let num_features = 5;
    let num_samples = 200;

    // Generate synthetic price data
    println!("Generating synthetic market data...");
    let (samples, prices) = generate_synthetic_data(num_samples, sequence_length, num_features);
    println!("  Generated {} samples", samples.len());

    // Test different strategies
    println!("\n{:-<70}", "");
    println!("Strategy Comparison:");
    println!("{:-<70}\n", "");

    // Strategy 1: Baseline (no saliency filtering)
    let baseline_network = TradingNetwork::new(sequence_length, num_features, &[64, 32]);
    let baseline_result = run_backtest(
        &baseline_network,
        &samples,
        &prices,
        false, // no saliency filter
        0.0,   // no concentration threshold
    );
    print_result("Baseline (No Filter)", &baseline_result);

    // Strategy 2: Saliency concentration filter
    let saliency_network = TradingNetwork::new(sequence_length, num_features, &[64, 32]);
    let saliency_result = run_backtest(
        &saliency_network,
        &samples,
        &prices,
        true,  // use saliency filter
        0.3,   // concentration threshold
    );
    print_result("Saliency Filtered", &saliency_result);

    // Strategy 3: Higher concentration threshold
    let strict_network = TradingNetwork::new(sequence_length, num_features, &[64, 32]);
    let strict_result = run_backtest(
        &strict_network,
        &samples,
        &prices,
        true,
        0.5,  // stricter concentration
    );
    print_result("Strict Saliency", &strict_result);

    // Summary comparison
    println!("\n{:-<70}", "");
    println!("Summary:");
    println!("{:-<70}\n", "");

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "Strategy", "Return", "Sharpe", "Max DD", "Trades", "Win Rate"
    );
    println!("{:-<70}", "");

    for (name, result) in [
        ("Baseline", &baseline_result),
        ("Saliency", &saliency_result),
        ("Strict", &strict_result),
    ] {
        println!(
            "{:<20} {:>11.2}% {:>12.2} {:>11.2}% {:>10} {:>9.1}%",
            name,
            result.total_return * 100.0,
            result.sharpe_ratio,
            result.max_drawdown * 100.0,
            result.num_trades,
            result.win_rate * 100.0
        );
    }

    println!("\n✓ Backtest complete!");
}

/// Generate synthetic market data for testing
fn generate_synthetic_data(
    num_samples: usize,
    sequence_length: usize,
    num_features: usize,
) -> (Vec<Array2<f64>>, Vec<f64>) {
    let mut samples = Vec::new();
    let mut prices = Vec::new();

    let mut price = 100.0;

    for _ in 0..num_samples {
        // Generate sequence
        let mut sequence = Array2::zeros((sequence_length, num_features));

        for t in 0..sequence_length {
            // Random walk with mean reversion
            let change = (rand::random::<f64>() - 0.5) * 2.0 - (price - 100.0) * 0.01;
            price = (price + change).max(50.0).min(150.0);

            sequence[[t, 0]] = price - 0.5 + rand::random::<f64>(); // Open
            sequence[[t, 1]] = price + rand::random::<f64>() * 2.0;  // High
            sequence[[t, 2]] = price - rand::random::<f64>() * 2.0;  // Low
            sequence[[t, 3]] = price;  // Close
            sequence[[t, 4]] = 1000.0 + rand::random::<f64>() * 500.0;  // Volume
        }

        // Normalize
        for j in 0..num_features {
            let col_mean: f64 = (0..sequence_length)
                .map(|i| sequence[[i, j]])
                .sum::<f64>() / sequence_length as f64;
            let col_std: f64 = {
                let variance: f64 = (0..sequence_length)
                    .map(|i| (sequence[[i, j]] - col_mean).powi(2))
                    .sum::<f64>() / sequence_length as f64;
                variance.sqrt().max(1e-10)
            };

            for i in 0..sequence_length {
                sequence[[i, j]] = (sequence[[i, j]] - col_mean) / col_std;
            }
        }

        samples.push(sequence);
        prices.push(price);
    }

    (samples, prices)
}

/// Run a simple backtest
fn run_backtest(
    network: &TradingNetwork,
    samples: &[Array2<f64>],
    prices: &[f64],
    use_saliency: bool,
    concentration_threshold: f64,
) -> BacktestResult {
    let mut equity = 100000.0;
    let initial_equity = equity;
    let mut position = 0;  // 1 = long, -1 = short, 0 = flat
    let mut entry_price = 0.0;

    let mut trades: Vec<f64> = Vec::new();
    let mut equity_history = vec![equity];
    let mut max_equity = equity;
    let mut max_drawdown = 0.0;

    let trader = if use_saliency {
        Some(SaliencyTrader::new(
            network.clone(),
            0.55,
            concentration_threshold,
            vec![3, 4],  // Close, Volume
        ))
    } else {
        None
    };

    for i in 0..samples.len().saturating_sub(1) {
        let current_price = prices[i];
        let next_price = prices[i + 1];

        // Generate signal
        let (signal, _size) = if let Some(ref t) = trader {
            t.generate_signal(&samples[i])
        } else {
            // Simple prediction without filtering
            let pred = network.forward(&samples[i]);
            if pred > 0.55 {
                (1, 1.0)
            } else if pred < 0.45 {
                (-1, 1.0)
            } else {
                (0, 0.0)
            }
        };

        // Close existing position if signal changes
        if position != 0 && signal != position {
            let pnl = if position == 1 {
                (current_price - entry_price) / entry_price
            } else {
                (entry_price - current_price) / entry_price
            };
            equity *= 1.0 + pnl;
            trades.push(pnl);
            position = 0;
        }

        // Open new position
        if position == 0 && signal != 0 {
            position = signal;
            entry_price = current_price;
        }

        // Update equity tracking
        equity_history.push(equity);
        if equity > max_equity {
            max_equity = equity;
        }
        let drawdown = (max_equity - equity) / max_equity;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    // Close final position
    if position != 0 && !prices.is_empty() {
        let final_price = prices[prices.len() - 1];
        let pnl = if position == 1 {
            (final_price - entry_price) / entry_price
        } else {
            (entry_price - final_price) / entry_price
        };
        equity *= 1.0 + pnl;
        trades.push(pnl);
    }

    // Calculate metrics
    let total_return = (equity - initial_equity) / initial_equity;

    let win_rate = if trades.is_empty() {
        0.0
    } else {
        trades.iter().filter(|&&t| t > 0.0).count() as f64 / trades.len() as f64
    };

    let sharpe_ratio = if trades.len() > 1 {
        let mean: f64 = trades.iter().sum::<f64>() / trades.len() as f64;
        let variance: f64 = trades.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / trades.len() as f64;
        let std = variance.sqrt();
        if std > 1e-10 {
            (mean / std) * (252.0_f64).sqrt()  // Annualized
        } else {
            0.0
        }
    } else {
        0.0
    };

    BacktestResult {
        total_return,
        sharpe_ratio,
        max_drawdown,
        num_trades: trades.len(),
        win_rate,
    }
}

fn print_result(name: &str, result: &BacktestResult) {
    println!("{}", name);
    println!("  Total Return: {:.2}%", result.total_return * 100.0);
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("  Number of Trades: {}", result.num_trades);
    println!("  Win Rate: {:.1}%\n", result.win_rate * 100.0);
}
