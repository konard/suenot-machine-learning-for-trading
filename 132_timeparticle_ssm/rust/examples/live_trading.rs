//! Example: Live trading simulation with TimeParticle SSM
//!
//! This example demonstrates a complete trading workflow including
//! data fetching, feature computation, signal generation, and
//! simulated order execution.

use std::collections::VecDeque;
use timeparticle_ssm::api::bybit::BybitClient;
use timeparticle_ssm::data::features::FeatureEngine;
use timeparticle_ssm::data::loader::{generate_simulated_data, OhlcvData};
use timeparticle_ssm::model::timeparticle::{TimeParticleSSM, TradingSignal};

/// Simple position tracker
struct Position {
    quantity: f64,
    entry_price: f64,
    direction: String, // "long", "short", or "flat"
}

impl Position {
    fn new() -> Self {
        Self {
            quantity: 0.0,
            entry_price: 0.0,
            direction: "flat".to_string(),
        }
    }

    fn is_flat(&self) -> bool {
        self.direction == "flat"
    }

    fn is_long(&self) -> bool {
        self.direction == "long"
    }

    fn pnl(&self, current_price: f64) -> f64 {
        if self.is_flat() {
            0.0
        } else if self.is_long() {
            self.quantity * (current_price - self.entry_price)
        } else {
            self.quantity * (self.entry_price - current_price)
        }
    }
}

fn main() {
    println!("TimeParticle SSM - Live Trading Simulation\n");
    println!("==========================================\n");

    // Configuration
    let initial_capital = 100_000.0;
    let position_size_pct = 0.1; // 10% of capital per trade
    let confidence_threshold = 0.55;
    let lookback = 100;

    println!("Configuration:");
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Position Size: {:.0}%", position_size_pct * 100.0);
    println!("  Confidence Threshold: {:.0}%", confidence_threshold * 100.0);
    println!("  Lookback Period: {} bars\n", lookback);

    // Initialize components
    println!("Initializing...");
    let client = BybitClient::new();
    let engine = FeatureEngine::new();

    // Try to fetch real data, fall back to simulated
    let data = match client.fetch_klines("BTCUSDT", "60", 500) {
        Ok(klines) => {
            println!("  Fetched {} real klines from Bybit", klines.len());
            OhlcvData::from_klines(&klines)
        }
        Err(e) => {
            println!("  Could not fetch real data: {}", e);
            println!("  Using simulated data instead");
            generate_simulated_data(500, 42)
        }
    };

    // Compute features
    let features = engine.compute_features(&data);
    println!("  Computed {} features for {} bars", features.ncols(), features.nrows());

    // Create model
    let mut model = TimeParticleSSM::new(features.ncols(), 64, 32, 4, 3);
    model.set_confidence_threshold(confidence_threshold);
    println!("  Created TimeParticle model\n");

    // Trading simulation
    println!("Starting Trading Simulation");
    println!("===========================\n");

    let mut capital = initial_capital;
    let mut position = Position::new();
    let mut trades: Vec<(usize, String, f64, f64)> = Vec::new(); // (index, action, price, pnl)
    let mut equity_curve: Vec<f64> = vec![capital];

    // Simulate trading over the data
    let start_idx = lookback;
    let end_idx = data.len();

    for i in start_idx..end_idx {
        let current_price = data.close[i];

        // Get features for this timestep
        let start = i.saturating_sub(lookback);
        let input = features.slice(ndarray::s![start..i, ..]).to_owned();

        if input.nrows() < 10 {
            continue;
        }

        // Generate signal
        let signal = model.predict(&input);

        // Execute trading logic
        match &signal {
            TradingSignal::Buy(conf) if *conf > confidence_threshold => {
                if position.is_flat() {
                    // Open long position
                    let trade_size = capital * position_size_pct;
                    let quantity = trade_size / current_price;

                    position = Position {
                        quantity,
                        entry_price: current_price,
                        direction: "long".to_string(),
                    };

                    if i < start_idx + 10 || i > end_idx - 10 {
                        println!(
                            "Bar {}: BUY @ ${:.2} (conf: {:.1}%) - Size: {:.4} units",
                            i, current_price, conf * 100.0, quantity
                        );
                    }
                }
            }
            TradingSignal::Sell(conf) if *conf > confidence_threshold => {
                if position.is_long() {
                    // Close long position
                    let pnl = position.pnl(current_price);
                    capital += pnl;

                    trades.push((i, "CLOSE LONG".to_string(), current_price, pnl));

                    if i < start_idx + 10 || i > end_idx - 10 {
                        println!(
                            "Bar {}: SELL @ ${:.2} (conf: {:.1}%) - P&L: ${:.2}",
                            i, current_price, conf * 100.0, pnl
                        );
                    }

                    position = Position::new();
                }
            }
            _ => {}
        }

        // Calculate current equity
        let equity = capital + position.pnl(current_price);
        equity_curve.push(equity);
    }

    // Close any remaining position
    if !position.is_flat() {
        let final_price = data.close[end_idx - 1];
        let pnl = position.pnl(final_price);
        capital += pnl;
        trades.push((end_idx - 1, "FINAL CLOSE".to_string(), final_price, pnl));
        println!("\nFinal close @ ${:.2} - P&L: ${:.2}", final_price, pnl);
    }

    // Calculate performance metrics
    println!("\n==========================================");
    println!("Performance Summary");
    println!("==========================================\n");

    let final_equity = capital;
    let total_return = (final_equity / initial_capital - 1.0) * 100.0;
    let total_trades = trades.len();

    let winning_trades: Vec<_> = trades.iter().filter(|(_, _, _, pnl)| *pnl > 0.0).collect();
    let losing_trades: Vec<_> = trades.iter().filter(|(_, _, _, pnl)| *pnl <= 0.0).collect();

    let win_rate = if total_trades > 0 {
        winning_trades.len() as f64 / total_trades as f64 * 100.0
    } else {
        0.0
    };

    let gross_profit: f64 = winning_trades.iter().map(|(_, _, _, pnl)| pnl).sum();
    let gross_loss: f64 = losing_trades.iter().map(|(_, _, _, pnl)| pnl.abs()).sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        gross_profit
    };

    // Calculate max drawdown
    let mut peak = initial_capital;
    let mut max_drawdown = 0.0;
    for &equity in &equity_curve {
        if equity > peak {
            peak = equity;
        }
        let drawdown = (peak - equity) / peak;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    // Calculate Sharpe ratio (simplified, assumes 0 risk-free rate)
    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();
    let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
        / returns.len() as f64;
    let std_dev = variance.sqrt();
    let sharpe_ratio = if std_dev > 0.0 {
        (avg_return / std_dev) * (252.0_f64).sqrt() // Annualized
    } else {
        0.0
    };

    println!("Initial Capital:    ${:>12.2}", initial_capital);
    println!("Final Capital:      ${:>12.2}", final_equity);
    println!("Total Return:       {:>12.2}%", total_return);
    println!();
    println!("Total Trades:       {:>12}", total_trades);
    println!("Winning Trades:     {:>12}", winning_trades.len());
    println!("Losing Trades:      {:>12}", losing_trades.len());
    println!("Win Rate:           {:>12.2}%", win_rate);
    println!();
    println!("Gross Profit:       ${:>12.2}", gross_profit);
    println!("Gross Loss:         ${:>12.2}", gross_loss);
    println!("Profit Factor:      {:>12.2}", profit_factor);
    println!();
    println!("Max Drawdown:       {:>12.2}%", max_drawdown * 100.0);
    println!("Sharpe Ratio:       {:>12.2}", sharpe_ratio);

    println!("\n==========================================");
    println!("Live trading simulation complete!");
    println!("\nNote: This is a simulation using random model weights.");
    println!("Real trading requires properly trained model parameters.");
}
