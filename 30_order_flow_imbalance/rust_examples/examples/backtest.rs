//! # Backtest Example
//!
//! Demonstrates backtesting an OFI strategy on historical data.
//!
//! Run with: `cargo run --example backtest`

use anyhow::Result;
use chrono::Utc;
use order_flow_imbalance::backtest::engine::{BacktestConfig, BacktestEngine};
use order_flow_imbalance::backtest::report::BacktestReport;
use order_flow_imbalance::data::orderbook::{OrderBook, OrderBookLevel};
use order_flow_imbalance::data::trade::Trade;
use rand::Rng;

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              Backtest Engine Demo                          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 10000.0,
        position_size: 0.1,
        commission_rate: 0.0004,
        slippage_bps: 1.0,
        max_holding_time: 120,
        max_daily_loss: 500.0,
    };

    println!("Backtest Configuration:");
    println!("───────────────────────────────────────────────────────────");
    println!("  Initial Capital:   ${:.2}", config.initial_capital);
    println!("  Position Size:     {} BTC", config.position_size);
    println!("  Commission Rate:   {:.2} bps", config.commission_rate * 10000.0);
    println!("  Slippage:          {:.1} bps", config.slippage_bps);
    println!("  Max Holding Time:  {} seconds", config.max_holding_time);
    println!("  Max Daily Loss:    ${:.2}", config.max_daily_loss);
    println!();

    let mut engine = BacktestEngine::new(config.clone());

    // Generate synthetic historical data
    println!("Generating synthetic historical data...");
    let (orderbooks, trades) = generate_historical_data(5000);
    println!("  Order Book Snapshots: {}", orderbooks.len());
    println!("  Trades:               {}", trades.len());
    println!();

    // Run backtest
    println!("Running backtest...");
    println!();

    let mut trade_idx = 0;

    for (i, orderbook) in orderbooks.iter().enumerate() {
        // Process trades that occurred before this orderbook
        while trade_idx < trades.len() && trades[trade_idx].timestamp <= orderbook.timestamp {
            engine.process_trade(&trades[trade_idx]);
            trade_idx += 1;
        }

        // Process orderbook
        engine.process_orderbook(orderbook);

        // Progress update
        if (i + 1) % 1000 == 0 {
            println!("  Processed {} / {} snapshots...", i + 1, orderbooks.len());
        }
    }

    println!();
    println!("Backtest complete!");
    println!();

    // Generate report
    let report = BacktestReport::from_engine(&engine, config.initial_capital);

    // Print formatted report
    println!("{}", report.format());

    // Additional analysis
    println!("TRADE ANALYSIS");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let trades = engine.trade_log();
    if !trades.is_empty() {
        println!("Trade Log (last 10 trades):");
        println!("───────────────────────────────────────────────────────────");
        println!("  {:^23} │ {:^6} │ {:^12} │ {:^10} │ {:^8}",
            "Time", "Side", "Price", "P&L", "Reason"
        );
        println!("  ───────────────────────┼────────┼──────────────┼────────────┼──────────");

        for trade in trades.iter().rev().take(10) {
            println!("  {} │ {:^6} │ ${:>10.2} │ ${:>8.2} │ {}",
                trade.timestamp.format("%Y-%m-%d %H:%M:%S"),
                trade.side,
                trade.price,
                trade.pnl,
                trade.exit_reason
            );
        }
        println!();
    }

    // Equity curve analysis
    let equity_curve = engine.equity_curve();
    if !equity_curve.is_empty() {
        println!("Equity Curve Summary:");
        println!("───────────────────────────────────────────────────────────");

        let initial = config.initial_capital;
        let final_equity = equity_curve.last().map(|e| e.equity).unwrap_or(initial);
        let peak = equity_curve.iter().map(|e| e.equity).fold(initial, f64::max);
        let trough = equity_curve.iter().map(|e| e.equity).fold(initial, f64::min);
        let max_dd = equity_curve.iter().map(|e| e.drawdown).fold(0.0_f64, f64::max);

        println!("  Data Points:   {}", equity_curve.len());
        println!("  Initial:       ${:.2}", initial);
        println!("  Final:         ${:.2}", final_equity);
        println!("  Peak:          ${:.2}", peak);
        println!("  Trough:        ${:.2}", trough);
        println!("  Max Drawdown:  {:.2}%", max_dd * 100.0);
        println!();

        // Simple equity chart
        println!("Equity Chart (simplified):");
        println!("───────────────────────────────────────────────────────────");

        let step = equity_curve.len() / 20;
        let range = peak - trough;

        for i in (0..equity_curve.len()).step_by(step.max(1)) {
            if i < equity_curve.len() {
                let e = &equity_curve[i];
                let normalized = if range > 0.0 {
                    ((e.equity - trough) / range * 40.0) as usize
                } else {
                    20
                };
                let bar = "█".repeat(normalized);
                println!("  ${:>8.0} │{}", e.equity, bar);
            }
        }
        println!();
    }

    // Export report
    println!("Exporting results...");
    let json = report.to_json()?;
    println!("  Report JSON size: {} bytes", json.len());
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("Backtest simulation complete.");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

/// Generate synthetic historical data for backtesting
fn generate_historical_data(n_snapshots: usize) -> (Vec<OrderBook>, Vec<Trade>) {
    let mut rng = rand::thread_rng();
    let mut orderbooks = Vec::with_capacity(n_snapshots);
    let mut trades = Vec::new();

    let mut price = 50000.0;
    let mut timestamp = Utc::now() - chrono::Duration::hours(24);

    for i in 0..n_snapshots {
        // Random walk for price
        let change = rng.gen_range(-50.0..50.0);
        price += change;
        price = price.max(40000.0).min(60000.0);

        // Random spread
        let spread = rng.gen_range(5.0..30.0);
        let bid = price - spread / 2.0;
        let ask = price + spread / 2.0;

        // Generate order book levels
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        for level in 1..=10 {
            let bid_price = bid - (level as f64 - 1.0) * 5.0;
            let ask_price = ask + (level as f64 - 1.0) * 5.0;

            bids.push(OrderBookLevel::new(
                bid_price,
                rng.gen_range(0.5..5.0),
                level,
            ));
            asks.push(OrderBookLevel::new(
                ask_price,
                rng.gen_range(0.5..5.0),
                level,
            ));
        }

        let orderbook = OrderBook::new(
            "BTCUSDT".to_string(),
            timestamp,
            bids,
            asks,
        );
        orderbooks.push(orderbook);

        // Generate some trades
        let n_trades = rng.gen_range(0..5);
        for _ in 0..n_trades {
            let trade_price = if rng.gen_bool(0.5) {
                ask + rng.gen_range(-1.0..1.0)
            } else {
                bid + rng.gen_range(-1.0..1.0)
            };

            let trade = Trade::new(
                "BTCUSDT".to_string(),
                timestamp + chrono::Duration::milliseconds(rng.gen_range(0..100)),
                trade_price,
                rng.gen_range(0.01..0.5),
                rng.gen_bool(0.5),
                format!("trade_{}", i * 10 + trades.len()),
            );
            trades.push(trade);
        }

        // Advance timestamp
        timestamp += chrono::Duration::milliseconds(rng.gen_range(100..500));
    }

    (orderbooks, trades)
}
