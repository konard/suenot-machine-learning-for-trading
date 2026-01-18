//! Example: Strategy Backtesting
//!
//! Simulates the HOT trading strategy on historical data.

use anyhow::Result;
use hot_crypto::data::{BybitClient, Features, PriceSeries};
use hot_crypto::model::{HOTPredictor, MovementClass};
use hot_crypto::strategy::{SignalGenerator, Signal, RiskManager};
use hot_crypto::utils::Config;

#[tokio::main]
async fn main() -> Result<()> {
    println!("HOT Crypto - Strategy Backtest");
    println!("==============================\n");

    // Configuration
    let config = Config::default();
    let initial_capital = config.portfolio.initial_capital;

    // Initialize components
    let client = BybitClient::new();
    let predictor = HOTPredictor::default_crypto();
    let signal_gen = SignalGenerator::new();
    let risk_mgr = RiskManager::default();

    // Symbols to backtest
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    println!("Fetching historical data...\n");

    // Fetch 180 days of data
    let mut all_series = Vec::new();
    for symbol in &symbols {
        let series = client.get_klines(symbol, "D", None, None, Some(180)).await?;
        println!("  {} - {} candles loaded", symbol, series.len());
        all_series.push(series);
    }

    println!("\nRunning backtest simulation...\n");

    // Backtest parameters
    let lookback = 60;
    let test_start = 90; // Start testing from day 90

    // Track performance
    let mut portfolio_value = initial_capital;
    let mut cash = initial_capital;
    let mut positions: Vec<Position> = Vec::new();
    let mut trade_history: Vec<Trade> = Vec::new();
    let mut daily_returns: Vec<f64> = Vec::new();
    let mut peak_value = initial_capital;
    let mut max_drawdown = 0.0;

    // Get the minimum series length
    let min_len = all_series.iter().map(|s| s.len()).min().unwrap_or(0);

    // Simulate day by day
    for day in test_start..min_len - 1 {
        let prev_value = portfolio_value;

        // Update position values
        for pos in &mut positions {
            if let Some(series) = all_series.iter().find(|s| s.symbol == pos.symbol) {
                let current_price = series.candles[day].close;
                pos.current_price = current_price;
                pos.unrealized_pnl = (current_price - pos.entry_price) / pos.entry_price
                    * pos.size * (if pos.is_long { 1.0 } else { -1.0 });
            }
        }

        // Calculate portfolio value
        let position_value: f64 = positions.iter().map(|p| p.size + p.unrealized_pnl).sum();
        portfolio_value = cash + position_value;

        // Track drawdown
        if portfolio_value > peak_value {
            peak_value = portfolio_value;
        }
        let current_dd = (peak_value - portfolio_value) / peak_value;
        if current_dd > max_drawdown {
            max_drawdown = current_dd;
        }

        // Daily return
        let daily_return = (portfolio_value - prev_value) / prev_value;
        daily_returns.push(daily_return);

        // Check exits (stop loss / take profit)
        let mut closed_positions = Vec::new();
        for (idx, pos) in positions.iter().enumerate() {
            let should_exit = risk_mgr.is_stop_loss(pos.entry_price, pos.current_price, pos.is_long)
                || risk_mgr.is_take_profit(pos.entry_price, pos.current_price, pos.is_long);

            if should_exit {
                closed_positions.push(idx);

                let pnl = pos.unrealized_pnl;
                cash += pos.size + pnl;

                trade_history.push(Trade {
                    symbol: pos.symbol.clone(),
                    is_long: pos.is_long,
                    entry_price: pos.entry_price,
                    exit_price: pos.current_price,
                    pnl,
                    pnl_pct: pnl / pos.size,
                });
            }
        }

        // Remove closed positions (in reverse order)
        for idx in closed_positions.into_iter().rev() {
            positions.remove(idx);
        }

        // Generate new signals (every few days)
        if day % 3 == 0 {
            for series in &all_series {
                // Skip if already have position
                if positions.iter().any(|p| p.symbol == series.symbol) {
                    continue;
                }

                // Get subset of data up to current day
                let subset = PriceSeries::new(
                    series.symbol.clone(),
                    series.candles[..=day].to_vec(),
                );

                // Generate prediction
                let prediction = predictor.predict_from_series(&subset);
                let signal_result = signal_gen.generate(&prediction);

                // Open position if signal is actionable
                if signal_result.signal != Signal::Hold {
                    let current_exposure: f64 = positions.iter().map(|p| p.size).sum::<f64>() / portfolio_value;

                    if risk_mgr.can_open_position(current_exposure, signal_result.position_size) {
                        let position_size = risk_mgr.adjust_position_size(
                            signal_result.position_size * portfolio_value,
                            current_exposure,
                        );

                        if position_size > 100.0 { // Minimum position size
                            let entry_price = series.candles[day].close;

                            positions.push(Position {
                                symbol: series.symbol.clone(),
                                is_long: signal_result.signal == Signal::Buy,
                                entry_price,
                                current_price: entry_price,
                                size: position_size,
                                unrealized_pnl: 0.0,
                            });

                            cash -= position_size;
                        }
                    }
                }
            }
        }
    }

    // Close all remaining positions at the end
    for pos in &positions {
        let pnl = pos.unrealized_pnl;
        cash += pos.size + pnl;

        trade_history.push(Trade {
            symbol: pos.symbol.clone(),
            is_long: pos.is_long,
            entry_price: pos.entry_price,
            exit_price: pos.current_price,
            pnl,
            pnl_pct: pnl / pos.size,
        });
    }

    portfolio_value = cash;

    // Calculate metrics
    let total_return = (portfolio_value - initial_capital) / initial_capital;
    let avg_daily_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
    let volatility = calculate_volatility(&daily_returns);
    let sharpe = avg_daily_return / volatility * (252.0_f64).sqrt();

    let winning_trades: Vec<_> = trade_history.iter().filter(|t| t.pnl > 0.0).collect();
    let losing_trades: Vec<_> = trade_history.iter().filter(|t| t.pnl < 0.0).collect();
    let win_rate = winning_trades.len() as f64 / trade_history.len().max(1) as f64;

    // Display results
    println!("{:-<60}", "");
    println!(" Backtest Results");
    println!("{:-<60}", "");
    println!("\nPerformance Metrics:");
    println!("  Initial Capital:    ${:.2}", initial_capital);
    println!("  Final Value:        ${:.2}", portfolio_value);
    println!("  Total Return:       {:.2}%", total_return * 100.0);
    println!("  Annualized Vol:     {:.2}%", volatility * (252.0_f64).sqrt() * 100.0);
    println!("  Sharpe Ratio:       {:.2}", sharpe);
    println!("  Max Drawdown:       {:.2}%", max_drawdown * 100.0);

    println!("\nTrading Statistics:");
    println!("  Total Trades:       {}", trade_history.len());
    println!("  Winning Trades:     {}", winning_trades.len());
    println!("  Losing Trades:      {}", losing_trades.len());
    println!("  Win Rate:           {:.1}%", win_rate * 100.0);

    if !winning_trades.is_empty() {
        let avg_win: f64 = winning_trades.iter().map(|t| t.pnl_pct).sum::<f64>()
            / winning_trades.len() as f64;
        println!("  Avg Win:            {:.2}%", avg_win * 100.0);
    }

    if !losing_trades.is_empty() {
        let avg_loss: f64 = losing_trades.iter().map(|t| t.pnl_pct).sum::<f64>()
            / losing_trades.len() as f64;
        println!("  Avg Loss:           {:.2}%", avg_loss * 100.0);
    }

    println!("\n{:-<60}", "");
    println!(" Recent Trades");
    println!("{:-<60}", "");
    println!("{:<10} {:>6} {:>10} {:>10} {:>10}", "Symbol", "Side", "Entry", "Exit", "P&L %");

    for trade in trade_history.iter().rev().take(10) {
        println!(
            "{:<10} {:>6} {:>10.2} {:>10.2} {:>9.2}%",
            trade.symbol,
            if trade.is_long { "LONG" } else { "SHORT" },
            trade.entry_price,
            trade.exit_price,
            trade.pnl_pct * 100.0,
        );
    }

    println!("\n{:-<60}", "");
    println!("\nDisclaimer: Past performance does not guarantee future results.");
    println!("This is a simulation for educational purposes only.\n");

    Ok(())
}

#[derive(Debug)]
struct Position {
    symbol: String,
    is_long: bool,
    entry_price: f64,
    current_price: f64,
    size: f64,
    unrealized_pnl: f64,
}

#[derive(Debug)]
struct Trade {
    symbol: String,
    is_long: bool,
    entry_price: f64,
    exit_price: f64,
    pnl: f64,
    pnl_pct: f64,
}

fn calculate_volatility(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;

    variance.sqrt()
}
