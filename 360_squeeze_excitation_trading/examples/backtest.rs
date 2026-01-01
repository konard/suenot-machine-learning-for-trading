//! Backtesting Example with SE-Enhanced Strategy
//!
//! This example runs a backtest simulation using the SE Momentum Strategy
//! and compares it with a baseline approach.

use se_trading::prelude::*;
use se_trading::data::bybit::generate_sample_data;
use se_trading::strategies::se_momentum::{SEMomentumStrategy, SEMomentumConfig, StrategyAction};
use se_trading::utils::metrics::{PerformanceMetrics, Trade};
use se_trading::strategies::signals::Direction;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║      SE-Enhanced Strategy Backtest Simulation             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Configuration
    let initial_capital = 10000.0;
    let data_length = 1000;
    let starting_price = 50000.0;

    println!("Simulation parameters:");
    println!("  - Initial capital: ${:.2}", initial_capital);
    println!("  - Data points: {}", data_length);
    println!("  - Starting price: ${:.2}", starting_price);
    println!();

    // Generate sample data
    println!("Generating sample market data...");
    let klines = generate_sample_data(data_length, starting_price);

    // Run SE-enhanced backtest
    println!("Running SE-enhanced strategy backtest...\n");
    let se_results = run_se_backtest(&klines, initial_capital);

    // Run baseline backtest (simple momentum)
    println!("Running baseline strategy backtest...\n");
    let baseline_results = run_baseline_backtest(&klines, initial_capital);

    // Display results comparison
    println!("═══════════════════════════════════════════════════════════");
    println!("                   BACKTEST RESULTS                         ");
    println!("═══════════════════════════════════════════════════════════\n");

    print_comparison(&se_results, &baseline_results);

    // Print detailed SE strategy results
    println!("\n═══════════════════════════════════════════════════════════");
    println!("            SE-ENHANCED STRATEGY DETAILS                    ");
    println!("═══════════════════════════════════════════════════════════");
    println!("{}", se_results.summary().to_table());

    // Trade analysis
    println!("═══════════════════════════════════════════════════════════");
    println!("                   TRADE ANALYSIS                           ");
    println!("═══════════════════════════════════════════════════════════\n");

    analyze_trades(&se_results);

    // Equity curve visualization
    println!("═══════════════════════════════════════════════════════════");
    println!("                   EQUITY CURVE                             ");
    println!("═══════════════════════════════════════════════════════════\n");

    print_equity_curve(&se_results, &baseline_results, initial_capital);

    println!("\n✓ Backtest complete!");
}

/// Run backtest with SE-enhanced strategy
fn run_se_backtest(klines: &[se_trading::data::bybit::Kline], initial_capital: f64) -> PerformanceMetrics {
    let config = SEMomentumConfig {
        entry_threshold: 0.25,
        exit_threshold: 0.1,
        lookback_window: 50,
        max_position_size: 0.1, // 10% of capital
        stop_loss_pct: 3.0,
        take_profit_pct: 6.0,
        min_signal_strength: 0.2,
        signal_cooldown: 3,
        ..Default::default()
    };

    let mut strategy = SEMomentumStrategy::new(config);
    let mut metrics = PerformanceMetrics::new(initial_capital);

    let mut trade_count = 0;
    let mut current_entry: Option<(u64, f64, f64, bool)> = None; // (time, price, size, is_long)

    // Start after warmup period
    let warmup = 60;

    for i in warmup..klines.len() {
        let window = &klines[..=i];
        let current_price = klines[i].close;
        let current_time = klines[i].start_time;

        let action = strategy.on_bar(window, current_price);

        match action {
            StrategyAction::OpenLong { size, .. } => {
                if current_entry.is_none() {
                    let position_value = initial_capital * size;
                    let qty = position_value / current_price;
                    current_entry = Some((current_time, current_price, qty, true));
                    trade_count += 1;
                }
            }
            StrategyAction::OpenShort { size, .. } => {
                if current_entry.is_none() {
                    let position_value = initial_capital * size;
                    let qty = position_value / current_price;
                    current_entry = Some((current_time, current_price, qty, false));
                    trade_count += 1;
                }
            }
            StrategyAction::ClosePosition { .. } => {
                if let Some((entry_time, entry_price, size, is_long)) = current_entry.take() {
                    let trade = Trade::new(
                        entry_time,
                        current_time,
                        entry_price,
                        current_price,
                        size,
                        is_long,
                    );
                    metrics.record_trade(trade);
                }
            }
            StrategyAction::Reverse { from, to, size, .. } => {
                // Close existing position
                if let Some((entry_time, entry_price, old_size, is_long)) = current_entry.take() {
                    let trade = Trade::new(
                        entry_time,
                        current_time,
                        entry_price,
                        current_price,
                        old_size,
                        is_long,
                    );
                    metrics.record_trade(trade);
                }

                // Open new position
                let position_value = initial_capital * size;
                let qty = position_value / current_price;
                let new_is_long = to == Direction::Long;
                current_entry = Some((current_time, current_price, qty, new_is_long));
                trade_count += 1;
            }
            StrategyAction::Hold => {}
        }
    }

    // Close any remaining position
    if let Some((entry_time, entry_price, size, is_long)) = current_entry {
        let last_price = klines.last().unwrap().close;
        let last_time = klines.last().unwrap().start_time;
        let trade = Trade::new(entry_time, last_time, entry_price, last_price, size, is_long);
        metrics.record_trade(trade);
    }

    metrics
}

/// Run baseline backtest (simple momentum crossover)
fn run_baseline_backtest(klines: &[se_trading::data::bybit::Kline], initial_capital: f64) -> PerformanceMetrics {
    let mut metrics = PerformanceMetrics::new(initial_capital);
    let mut current_entry: Option<(u64, f64, f64, bool)> = None;

    let fast_period = 10;
    let slow_period = 30;
    let position_size = 0.1;

    for i in slow_period..klines.len() {
        let fast_ma: f64 = klines[(i - fast_period)..i].iter().map(|k| k.close).sum::<f64>()
            / fast_period as f64;
        let slow_ma: f64 = klines[(i - slow_period)..i].iter().map(|k| k.close).sum::<f64>()
            / slow_period as f64;

        let prev_fast_ma: f64 = klines[(i - fast_period - 1)..(i - 1)]
            .iter()
            .map(|k| k.close)
            .sum::<f64>()
            / fast_period as f64;
        let prev_slow_ma: f64 = klines[(i - slow_period - 1)..(i - 1)]
            .iter()
            .map(|k| k.close)
            .sum::<f64>()
            / slow_period as f64;

        let current_price = klines[i].close;
        let current_time = klines[i].start_time;

        // Golden cross - go long
        if fast_ma > slow_ma && prev_fast_ma <= prev_slow_ma {
            // Close short if any
            if let Some((entry_time, entry_price, size, is_long)) = current_entry.take() {
                if !is_long {
                    let trade = Trade::new(
                        entry_time,
                        current_time,
                        entry_price,
                        current_price,
                        size,
                        false,
                    );
                    metrics.record_trade(trade);
                }
            }

            // Open long
            if current_entry.is_none() {
                let position_value = initial_capital * position_size;
                let qty = position_value / current_price;
                current_entry = Some((current_time, current_price, qty, true));
            }
        }

        // Death cross - go short
        if fast_ma < slow_ma && prev_fast_ma >= prev_slow_ma {
            // Close long if any
            if let Some((entry_time, entry_price, size, is_long)) = current_entry.take() {
                if is_long {
                    let trade = Trade::new(
                        entry_time,
                        current_time,
                        entry_price,
                        current_price,
                        size,
                        true,
                    );
                    metrics.record_trade(trade);
                }
            }

            // Open short
            if current_entry.is_none() {
                let position_value = initial_capital * position_size;
                let qty = position_value / current_price;
                current_entry = Some((current_time, current_price, qty, false));
            }
        }
    }

    // Close any remaining position
    if let Some((entry_time, entry_price, size, is_long)) = current_entry {
        let last_price = klines.last().unwrap().close;
        let last_time = klines.last().unwrap().start_time;
        let trade = Trade::new(entry_time, last_time, entry_price, last_price, size, is_long);
        metrics.record_trade(trade);
    }

    metrics
}

/// Print comparison between strategies
fn print_comparison(se: &PerformanceMetrics, baseline: &PerformanceMetrics) {
    let se_summary = se.summary();
    let base_summary = baseline.summary();

    fn compare(se_val: f64, base_val: f64) -> &'static str {
        if se_val > base_val * 1.05 {
            "✓ SE wins"
        } else if base_val > se_val * 1.05 {
            "✗ Base wins"
        } else {
            "≈ Tie"
        }
    }

    println!("  {:20} {:>12} {:>12} {:>15}",
             "Metric", "SE Strategy", "Baseline", "Comparison");
    println!("  {}", "-".repeat(60));

    println!("  {:20} {:>12} {:>12} {:>15}",
             "Total Trades",
             se_summary.total_trades,
             base_summary.total_trades,
             "");

    println!("  {:20} {:>11.1}% {:>11.1}% {:>15}",
             "Win Rate",
             se_summary.win_rate,
             base_summary.win_rate,
             compare(se_summary.win_rate, base_summary.win_rate));

    println!("  {:20} {:>12.2} {:>12.2} {:>15}",
             "Total PnL",
             se_summary.total_pnl,
             base_summary.total_pnl,
             compare(se_summary.total_pnl, base_summary.total_pnl));

    println!("  {:20} {:>11.2}% {:>11.2}% {:>15}",
             "Total Return",
             se_summary.total_return_pct,
             base_summary.total_return_pct,
             compare(se_summary.total_return_pct, base_summary.total_return_pct));

    println!("  {:20} {:>12.2} {:>12.2} {:>15}",
             "Profit Factor",
             se_summary.profit_factor,
             base_summary.profit_factor,
             compare(se_summary.profit_factor, base_summary.profit_factor));

    println!("  {:20} {:>11.2}% {:>11.2}% {:>15}",
             "Max Drawdown",
             se_summary.max_drawdown,
             base_summary.max_drawdown,
             compare(-se_summary.max_drawdown, -base_summary.max_drawdown));

    println!("  {:20} {:>12.2} {:>12.2} {:>15}",
             "Sharpe Ratio",
             se_summary.sharpe_ratio,
             base_summary.sharpe_ratio,
             compare(se_summary.sharpe_ratio, base_summary.sharpe_ratio));
}

/// Analyze individual trades
fn analyze_trades(metrics: &PerformanceMetrics) {
    let trades = metrics.trades();

    if trades.is_empty() {
        println!("  No trades to analyze.");
        return;
    }

    // Best and worst trades
    let mut sorted_by_pnl: Vec<&Trade> = trades.iter().collect();
    sorted_by_pnl.sort_by(|a, b| b.pnl.partial_cmp(&a.pnl).unwrap());

    let best = sorted_by_pnl.first().unwrap();
    let worst = sorted_by_pnl.last().unwrap();

    println!("  Best trade:  {:+.2} ({:.2}% return)",
             best.pnl, best.return_pct);
    println!("  Worst trade: {:+.2} ({:.2}% return)",
             worst.pnl, worst.return_pct);

    // Win/loss streaks
    let mut max_win_streak = 0;
    let mut max_loss_streak = 0;
    let mut current_win_streak = 0;
    let mut current_loss_streak = 0;

    for trade in trades {
        if trade.is_winner() {
            current_win_streak += 1;
            current_loss_streak = 0;
            max_win_streak = max_win_streak.max(current_win_streak);
        } else {
            current_loss_streak += 1;
            current_win_streak = 0;
            max_loss_streak = max_loss_streak.max(current_loss_streak);
        }
    }

    println!("\n  Max winning streak: {} trades", max_win_streak);
    println!("  Max losing streak:  {} trades", max_loss_streak);

    // Long vs Short performance
    let long_trades: Vec<&Trade> = trades.iter().filter(|t| t.is_long).collect();
    let short_trades: Vec<&Trade> = trades.iter().filter(|t| !t.is_long).collect();

    let long_pnl: f64 = long_trades.iter().map(|t| t.pnl).sum();
    let short_pnl: f64 = short_trades.iter().map(|t| t.pnl).sum();

    println!("\n  Long trades:  {} (PnL: {:+.2})", long_trades.len(), long_pnl);
    println!("  Short trades: {} (PnL: {:+.2})", short_trades.len(), short_pnl);
}

/// Print ASCII equity curve
fn print_equity_curve(
    se: &PerformanceMetrics,
    baseline: &PerformanceMetrics,
    initial_capital: f64,
) {
    let se_curve = se.equity_curve();
    let base_curve = baseline.equity_curve();

    if se_curve.is_empty() && base_curve.is_empty() {
        println!("  No equity data to display.");
        return;
    }

    // Find min/max for scaling
    let all_values: Vec<f64> = se_curve.iter().chain(base_curve.iter()).cloned().collect();
    let min_val = all_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let height = 10;
    let width = 60;

    // Sample curves to fit width
    let se_sampled = sample_curve(se_curve, width);
    let base_sampled = sample_curve(base_curve, width);

    println!("  Legend: █ SE Strategy  ░ Baseline  ─ Initial Capital\n");

    for row in (0..height).rev() {
        let threshold = min_val + (max_val - min_val) * row as f64 / height as f64;
        let next_threshold = min_val + (max_val - min_val) * (row + 1) as f64 / height as f64;

        // Y-axis label
        if row == height - 1 {
            print!("  {:>8.0} │", max_val);
        } else if row == 0 {
            print!("  {:>8.0} │", min_val);
        } else if row == height / 2 {
            print!("  {:>8.0} │", initial_capital);
        } else {
            print!("           │");
        }

        // Plot points
        for i in 0..width {
            let se_val = se_sampled.get(i).cloned().unwrap_or(initial_capital);
            let base_val = base_sampled.get(i).cloned().unwrap_or(initial_capital);

            let se_in_range = se_val >= threshold && se_val < next_threshold;
            let base_in_range = base_val >= threshold && base_val < next_threshold;
            let init_in_range = initial_capital >= threshold && initial_capital < next_threshold;

            if se_in_range && base_in_range {
                print!("▓");
            } else if se_in_range {
                print!("█");
            } else if base_in_range {
                print!("░");
            } else if init_in_range {
                print!("─");
            } else {
                print!(" ");
            }
        }
        println!();
    }

    // X-axis
    print!("           └");
    print!("{}", "─".repeat(width));
    println!();
    println!("            Start{:>width$}End", "", width = width - 8);
}

/// Sample a curve to a specific number of points
fn sample_curve(curve: &[f64], target_len: usize) -> Vec<f64> {
    if curve.is_empty() {
        return vec![];
    }

    if curve.len() <= target_len {
        return curve.to_vec();
    }

    let step = curve.len() as f64 / target_len as f64;
    (0..target_len)
        .map(|i| {
            let idx = (i as f64 * step) as usize;
            curve[idx.min(curve.len() - 1)]
        })
        .collect()
}
