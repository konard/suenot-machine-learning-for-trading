//! SNN Trading Strategy Example
//!
//! This example demonstrates a complete trading strategy using
//! Spiking Neural Networks with Bybit cryptocurrency data.

use snn_trading::{
    trading::{SNNTradingStrategy, TradingStrategy, TradingSignal, StrategyParams},
    data::{BybitClient, Candle, generate_simulated_candles, generate_simulated_orderbook},
};

#[tokio::main]
async fn main() {
    println!("=== SNN Trading Strategy Backtest ===\n");

    // Fetch or simulate data
    let candles = match fetch_market_data().await {
        Ok(data) => {
            println!("Using {} candles from Bybit BTCUSDT\n", data.len());
            data
        }
        Err(e) => {
            println!("Bybit API error: {}", e);
            println!("Using simulated data\n");
            generate_simulated_candles(1000, 50000.0)
        }
    };

    if candles.len() < 100 {
        println!("Not enough data for backtest");
        return;
    }

    // Initialize strategy
    let params = StrategyParams {
        min_confidence: 0.5,
        max_position: 1.0,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.04,
        learning_rate: 0.001,
        learning_enabled: true,
    };

    let mut strategy = SNNTradingStrategy::new(5, 20)
        .with_params(params)
        .with_learning();

    // Backtest parameters
    let initial_capital = 10000.0;
    let mut capital = initial_capital;
    let mut position = 0.0;  // BTC held
    let mut entry_price = 0.0;

    // Statistics
    let mut trades: Vec<TradeRecord> = Vec::new();
    let mut equity_curve: Vec<f64> = Vec::new();

    println!("--- Running Backtest ---\n");
    println!("Initial capital: ${:.2}", initial_capital);
    println!("Strategy: {}", strategy.name());
    println!();

    // Skip first few candles for warm-up
    let warmup = 10;

    for (i, candle) in candles.iter().enumerate().skip(warmup) {
        // Get trading decision
        let decision = strategy.process(candle);

        // Current equity
        let current_equity = if position != 0.0 {
            capital + position * candle.close
        } else {
            capital
        };
        equity_curve.push(current_equity);

        // Check stop loss / take profit for existing position
        if position != 0.0 {
            let pnl_pct = (candle.close - entry_price) / entry_price;

            // Stop loss
            if (position > 0.0 && pnl_pct < -0.02) || (position < 0.0 && pnl_pct > 0.02) {
                let pnl = position * (candle.close - entry_price);
                capital += position * candle.close;

                trades.push(TradeRecord {
                    entry_price,
                    exit_price: candle.close,
                    pnl,
                    pnl_pct: pnl_pct * position.signum(),
                    exit_reason: "Stop Loss".to_string(),
                });

                strategy.update_with_result(pnl, entry_price * position.abs() * 0.02);
                position = 0.0;
                continue;
            }

            // Take profit
            if (position > 0.0 && pnl_pct > 0.04) || (position < 0.0 && pnl_pct < -0.04) {
                let pnl = position * (candle.close - entry_price);
                capital += position * candle.close;

                trades.push(TradeRecord {
                    entry_price,
                    exit_price: candle.close,
                    pnl,
                    pnl_pct: pnl_pct * position.signum(),
                    exit_reason: "Take Profit".to_string(),
                });

                strategy.update_with_result(pnl, entry_price * position.abs() * 0.02);
                position = 0.0;
                continue;
            }
        }

        // Execute new signal if no position
        if position == 0.0 && decision.is_actionable() {
            match decision.signal {
                TradingSignal::Buy | TradingSignal::StrongBuy => {
                    let size = capital * decision.position_size;
                    position = size / candle.close;
                    capital -= size;
                    entry_price = candle.close;
                }
                TradingSignal::Sell | TradingSignal::StrongSell => {
                    // Short position (simplified)
                    let size = capital * decision.position_size;
                    position = -size / candle.close;
                    capital += size;  // Receive cash for short
                    entry_price = candle.close;
                }
                TradingSignal::Hold => {}
            }
        }

        // Progress update
        if i % 200 == 0 && i > warmup {
            let equity = if position != 0.0 {
                capital + position * candle.close
            } else {
                capital
            };
            let ret = (equity - initial_capital) / initial_capital * 100.0;
            println!("  Candle {}: Equity ${:.2} ({:+.2}%)", i, equity, ret);
        }
    }

    // Close any remaining position
    if position != 0.0 {
        let last_price = candles.last().unwrap().close;
        let pnl = position * (last_price - entry_price);
        capital += position * last_price;

        trades.push(TradeRecord {
            entry_price,
            exit_price: last_price,
            pnl,
            pnl_pct: (last_price - entry_price) / entry_price * position.signum(),
            exit_reason: "End of Backtest".to_string(),
        });
    }

    // Calculate statistics
    println!("\n--- Backtest Results ---\n");

    let final_equity = capital;
    let total_return = (final_equity - initial_capital) / initial_capital * 100.0;

    println!("Performance:");
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Final Equity: ${:.2}", final_equity);
    println!("  Total Return: {:+.2}%", total_return);

    if !trades.is_empty() {
        let wins: Vec<_> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losses: Vec<_> = trades.iter().filter(|t| t.pnl < 0.0).collect();

        let win_rate = wins.len() as f64 / trades.len() as f64 * 100.0;

        let avg_win = if !wins.is_empty() {
            wins.iter().map(|t| t.pnl_pct).sum::<f64>() / wins.len() as f64 * 100.0
        } else { 0.0 };

        let avg_loss = if !losses.is_empty() {
            losses.iter().map(|t| t.pnl_pct).sum::<f64>() / losses.len() as f64 * 100.0
        } else { 0.0 };

        let profit_factor = if !losses.is_empty() && !wins.is_empty() {
            let total_wins: f64 = wins.iter().map(|t| t.pnl).sum();
            let total_losses: f64 = losses.iter().map(|t| t.pnl.abs()).sum();
            if total_losses > 0.0 { total_wins / total_losses } else { 0.0 }
        } else { 0.0 };

        println!("\nTrade Statistics:");
        println!("  Total Trades: {}", trades.len());
        println!("  Winning Trades: {}", wins.len());
        println!("  Losing Trades: {}", losses.len());
        println!("  Win Rate: {:.1}%", win_rate);
        println!("  Average Win: {:+.2}%", avg_win);
        println!("  Average Loss: {:.2}%", avg_loss);
        println!("  Profit Factor: {:.2}", profit_factor);

        // Calculate max drawdown
        let max_drawdown = calculate_max_drawdown(&equity_curve);
        println!("\nRisk Metrics:");
        println!("  Max Drawdown: {:.2}%", max_drawdown * 100.0);

        // Sharpe ratio (simplified, using daily returns)
        if equity_curve.len() > 1 {
            let returns: Vec<f64> = equity_curve.windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();
            let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 0.0 {
                let sharpe = avg_return / std_dev * (252.0_f64).sqrt();  // Annualized
                println!("  Sharpe Ratio: {:.2}", sharpe);
            }
        }

        // Sample trades
        println!("\n--- Sample Trades ---");
        println!("{:<12} {:<12} {:<12} {:<15}",
            "Entry", "Exit", "PnL %", "Reason");
        println!("{}", "-".repeat(52));

        for trade in trades.iter().take(10) {
            println!("{:<12.2} {:<12.2} {:+<12.2}% {:<15}",
                trade.entry_price,
                trade.exit_price,
                trade.pnl_pct * 100.0,
                trade.exit_reason
            );
        }
    } else {
        println!("\nNo trades executed");
    }

    // Strategy learning stats
    let (total_pnl, trade_count, avg_pnl) = strategy.stats();
    println!("\n--- Strategy Learning ---");
    println!("  Trades processed for learning: {}", trade_count);
    println!("  Total PnL from learning: ${:.2}", total_pnl);
    println!("  Average PnL per trade: ${:.2}", avg_pnl);
}

async fn fetch_market_data() -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "15", 1000).await?;
    Ok(candles)
}

#[derive(Debug)]
struct TradeRecord {
    entry_price: f64,
    exit_price: f64,
    pnl: f64,
    pnl_pct: f64,
    exit_reason: String,
}

fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut max_equity = equity_curve[0];
    let mut max_drawdown = 0.0;

    for &equity in equity_curve {
        max_equity = max_equity.max(equity);
        let drawdown = (max_equity - equity) / max_equity;
        max_drawdown = max_drawdown.max(drawdown);
    }

    max_drawdown
}
