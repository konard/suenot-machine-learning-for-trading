//! Full trading strategy example using DiD-based signals.
//!
//! Demonstrates backtesting a DiD-based trading strategy on cryptocurrency data.

use std::collections::HashMap;

use diff_in_diff_trading::backtest::{Backtester, Event, PerformanceMetrics};
use diff_in_diff_trading::data::generate_synthetic_candles;
use diff_in_diff_trading::model::{DiDEstimator, DiDResults, PanelData};
use diff_in_diff_trading::trading::TradingStrategy;

fn main() {
    println!("=== DiD Trading Strategy Backtest ===\n");

    // Generate price data for multiple assets
    let n_days = 200;
    let btc_candles = generate_synthetic_candles(n_days, 40000.0, "BTCUSDT");
    let eth_candles = generate_synthetic_candles(n_days, 2500.0, "ETHUSDT");
    let sol_candles = generate_synthetic_candles(n_days, 100.0, "SOLUSDT");
    let avax_candles = generate_synthetic_candles(n_days, 30.0, "AVAXUSDT");

    let mut price_data = HashMap::new();
    price_data.insert("BTCUSDT".to_string(), btc_candles.clone());
    price_data.insert("ETHUSDT".to_string(), eth_candles.clone());
    price_data.insert("SOLUSDT".to_string(), sol_candles.clone());
    price_data.insert("AVAXUSDT".to_string(), avax_candles.clone());

    println!("Price data generated:");
    println!("  - BTCUSDT: {} days", btc_candles.len());
    println!("  - ETHUSDT: {} days", eth_candles.len());
    println!("  - SOLUSDT: {} days", sol_candles.len());
    println!("  - AVAXUSDT: {} days", avax_candles.len());
    println!();

    // Simulate events throughout the backtest period
    let events = simulate_events(&btc_candles, &eth_candles, &sol_candles, &avax_candles);

    println!("Simulated {} trading events\n", events.len());

    // Configure trading strategy
    let strategy = TradingStrategy::new()
        .with_max_position(0.15)
        .with_stop_loss(0.08)
        .with_take_profit(0.12)
        .with_holding_period(10);

    println!("Strategy configuration:");
    println!("  - Max position size: 15%");
    println!("  - Stop-loss: 8%");
    println!("  - Take-profit: 12%");
    println!("  - Max holding period: 10 days");
    println!();

    // Run backtest
    println!("Running backtest...\n");

    let mut backtester = Backtester::new(100000.0, strategy);
    let metrics = backtester.run(&events, &price_data);

    // Display results
    println!("=== Backtest Results ===");
    println!("{}", metrics);
    println!();

    // Trade analysis
    let trades = backtester.get_trades();
    if !trades.is_empty() {
        println!("=== Trade Analysis ===");
        println!(
            "{:<12} {:<12} {:<10} {:<10} {:<12} {:<15}",
            "Symbol", "Direction", "Entry", "Exit", "PnL", "Reason"
        );
        println!("{:-<75}", "");

        for trade in trades.iter().take(10) {
            let direction = if trade.direction > 0 { "LONG" } else { "SHORT" };
            let pnl = trade.pnl.unwrap_or(0.0);
            let reason = trade.exit_reason.as_deref().unwrap_or("-");

            println!(
                "{:<12} {:<12} ${:<9.2} ${:<9.2} ${:<11.2} {:<15}",
                trade.symbol, direction, trade.entry_price, trade.exit_price.unwrap_or(0.0), pnl, reason
            );
        }

        if trades.len() > 10 {
            println!("... and {} more trades", trades.len() - 10);
        }
        println!();
    }

    // Equity curve summary
    let equity_curve = backtester.get_equity_curve();
    if !equity_curve.is_empty() {
        println!("=== Equity Curve Summary ===");
        println!("Starting equity: ${:.2}", equity_curve.first().map(|(_, v)| *v).unwrap_or(100000.0));
        println!("Ending equity:   ${:.2}", equity_curve.last().map(|(_, v)| *v).unwrap_or(100000.0));
        println!("Peak equity:     ${:.2}", equity_curve.iter().map(|(_, v)| *v).fold(0.0, f64::max));
        println!("Trough equity:   ${:.2}", equity_curve.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min));
        println!();
    }

    // Performance comparison
    println!("=== Performance Comparison ===");
    print_performance_summary(&metrics);

    println!("\n=== Backtest Complete ===");
}

/// Simulate trading events using DiD estimation.
fn simulate_events(
    btc: &[diff_in_diff_trading::data::Candle],
    eth: &[diff_in_diff_trading::data::Candle],
    sol: &[diff_in_diff_trading::data::Candle],
    avax: &[diff_in_diff_trading::data::Candle],
) -> Vec<Event> {
    let mut events = Vec::new();
    let estimator = DiDEstimator::new().with_bootstrap(100);

    // Event 1: SOL listing (day 50)
    if sol.len() > 60 {
        let event_date = sol[50].timestamp;
        let did_results = estimate_event_effect(sol, eth, 50, 10, 10, &estimator);
        events.push(Event {
            date: event_date,
            treated_symbol: "SOLUSDT".to_string(),
            control_symbol: "ETHUSDT".to_string(),
            did_results,
        });
    }

    // Event 2: AVAX partnership (day 90)
    if avax.len() > 100 {
        let event_date = avax[90].timestamp;
        let did_results = estimate_event_effect(avax, sol, 90, 10, 10, &estimator);
        events.push(Event {
            date: event_date,
            treated_symbol: "AVAXUSDT".to_string(),
            control_symbol: "SOLUSDT".to_string(),
            did_results,
        });
    }

    // Event 3: ETH upgrade (day 130)
    if eth.len() > 140 {
        let event_date = eth[130].timestamp;
        let did_results = estimate_event_effect(eth, btc, 130, 10, 10, &estimator);
        events.push(Event {
            date: event_date,
            treated_symbol: "ETHUSDT".to_string(),
            control_symbol: "BTCUSDT".to_string(),
            did_results,
        });
    }

    // Event 4: BTC halving anticipation (day 170)
    if btc.len() > 180 {
        let event_date = btc[170].timestamp;
        let did_results = estimate_event_effect(btc, eth, 170, 10, 10, &estimator);
        events.push(Event {
            date: event_date,
            treated_symbol: "BTCUSDT".to_string(),
            control_symbol: "ETHUSDT".to_string(),
            did_results,
        });
    }

    events
}

/// Estimate DiD effect for an event.
fn estimate_event_effect(
    treated: &[diff_in_diff_trading::data::Candle],
    control: &[diff_in_diff_trading::data::Candle],
    event_idx: usize,
    pre_window: usize,
    post_window: usize,
    estimator: &DiDEstimator,
) -> DiDResults {
    let mut panel_data = Vec::new();

    // Add treated data
    for i in (event_idx.saturating_sub(pre_window))..=(event_idx + post_window).min(treated.len() - 1) {
        let is_post = i >= event_idx;
        let event_time = i as i32 - event_idx as i32;

        let outcome = if i > 0 {
            (treated[i].close - treated[i - 1].close) / treated[i - 1].close
        } else {
            0.0
        };

        panel_data.push(PanelData {
            unit_id: "treated".to_string(),
            timestamp: treated[i].timestamp,
            outcome,
            treated: true,
            post_treatment: is_post,
            event_time: Some(event_time),
            covariates: HashMap::new(),
        });
    }

    // Add control data
    for i in (event_idx.saturating_sub(pre_window))..=(event_idx + post_window).min(control.len() - 1) {
        let is_post = i >= event_idx;
        let event_time = i as i32 - event_idx as i32;

        let outcome = if i > 0 {
            (control[i].close - control[i - 1].close) / control[i - 1].close
        } else {
            0.0
        };

        panel_data.push(PanelData {
            unit_id: "control".to_string(),
            timestamp: control[i].timestamp,
            outcome,
            treated: false,
            post_treatment: is_post,
            event_time: Some(event_time),
            covariates: HashMap::new(),
        });
    }

    estimator.fit(&panel_data)
}

/// Print performance summary with benchmarks.
fn print_performance_summary(metrics: &PerformanceMetrics) {
    let buy_hold_approx = 0.15; // Approximate buy-and-hold for comparison

    println!(
        "{:<25} {:<15} {:<15}",
        "Metric", "DiD Strategy", "Buy & Hold*"
    );
    println!("{:-<55}", "");
    println!(
        "{:<25} {:>13.2}% {:>13.2}%",
        "Total Return",
        metrics.total_return * 100.0,
        buy_hold_approx * 100.0
    );
    println!(
        "{:<25} {:>13.2}% {:>14}",
        "Annualized Return",
        metrics.annualized_return * 100.0,
        "N/A"
    );
    println!(
        "{:<25} {:>13.2}% {:>14}",
        "Volatility",
        metrics.volatility * 100.0,
        "~25%"
    );
    println!(
        "{:<25} {:>14.2} {:>14}",
        "Sharpe Ratio",
        metrics.sharpe_ratio,
        "~0.5"
    );
    println!(
        "{:<25} {:>13.2}% {:>14}",
        "Max Drawdown",
        metrics.max_drawdown * 100.0,
        "~30%"
    );
    println!(
        "{:<25} {:>14} {:>14}",
        "Number of Trades",
        metrics.num_trades,
        "1"
    );
    println!("{:-<55}", "");
    println!("* Buy & Hold figures are approximate benchmarks");
}
