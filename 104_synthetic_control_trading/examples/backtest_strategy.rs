//! Backtest Strategy Example
//!
//! Demonstrates backtesting an SCM-based event trading strategy.

use synthetic_control_trading::backtest::engine::{
    generate_sample_events, print_report, BacktestEngine, StrategyConfig,
};

fn main() {
    println!("Synthetic Control Method - Backtest Strategy");
    println!("=============================================\n");

    // Generate sample events with known treatment effect
    println!("Generating 100 sample events with 3% treatment effect...");
    let events = generate_sample_events(
        100,   // Number of events
        60,    // Pre-treatment days
        20,    // Post-treatment days
        5,     // Number of donors per event
        0.03,  // Treatment effect (3%)
    );

    println!("  Generated {} events", events.len());
    println!("  Pre-treatment period: 60 days");
    println!("  Post-treatment period: 20 days");
    println!("  Donors per event: 5");

    // Configure strategy
    let strategy = StrategyConfig {
        entry_threshold: 0.02, // Enter if CAR > 2%
        exit_days: 5,          // Hold for 5 days
        stop_loss: -0.05,      // 5% stop loss
        take_profit: 0.10,     // 10% take profit
        direction: "long".to_string(),
    };

    println!("\nStrategy configuration:");
    println!("  Entry threshold: {:.1}%", strategy.entry_threshold * 100.0);
    println!("  Exit days: {}", strategy.exit_days);
    println!("  Stop loss: {:.1}%", strategy.stop_loss * 100.0);
    println!("  Take profit: {:.1}%", strategy.take_profit * 100.0);
    println!("  Direction: {}", strategy.direction);

    // Create backtest engine
    let engine = BacktestEngine::new(100_000.0, 0.001)
        .with_position_size(0.1)
        .with_max_positions(5);

    println!("\nBacktest configuration:");
    println!("  Initial capital: $100,000");
    println!("  Transaction cost: 0.1%");
    println!("  Position size: 10% of capital");

    // Run backtest
    println!("\nRunning backtest...");
    let results = engine.run(&events, &strategy).expect("Backtest failed");

    // Print results
    print_report(&results);

    // Additional analysis
    println!("\nTrade Analysis:");
    if !results.trades.is_empty() {
        let winning_trades: Vec<_> = results.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<_> = results.trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let avg_win = if !winning_trades.is_empty() {
            winning_trades.iter().map(|t| t.return_pct).sum::<f64>() / winning_trades.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing_trades.is_empty() {
            losing_trades.iter().map(|t| t.return_pct).sum::<f64>() / losing_trades.len() as f64
        } else {
            0.0
        };

        println!("  Winning trades: {}", winning_trades.len());
        println!("  Losing trades: {}", losing_trades.len());
        println!("  Average win: {:.2}%", avg_win * 100.0);
        println!("  Average loss: {:.2}%", avg_loss * 100.0);

        // Average holding period
        let avg_holding =
            results.trades.iter().map(|t| t.holding_days).sum::<usize>() as f64 / results.n_trades as f64;
        println!("  Average holding: {:.1} days", avg_holding);

        // CAR distribution
        let car_values: Vec<f64> = results.trades.iter().map(|t| t.car_at_entry).collect();
        let avg_entry_car = car_values.iter().sum::<f64>() / car_values.len() as f64;
        println!("  Average CAR at entry: {:.2}%", avg_entry_car * 100.0);
    }

    // Compare with different strategy configurations
    println!("\n{}", "=".repeat(50));
    println!("Strategy Comparison");
    println!("{}", "=".repeat(50));

    let strategies = vec![
        ("Long only, 2% threshold", StrategyConfig {
            entry_threshold: 0.02,
            exit_days: 5,
            stop_loss: -0.05,
            take_profit: 0.10,
            direction: "long".to_string(),
        }),
        ("Long only, 3% threshold", StrategyConfig {
            entry_threshold: 0.03,
            exit_days: 5,
            stop_loss: -0.05,
            take_profit: 0.10,
            direction: "long".to_string(),
        }),
        ("Both directions, 2% threshold", StrategyConfig {
            entry_threshold: 0.02,
            exit_days: 5,
            stop_loss: -0.05,
            take_profit: 0.10,
            direction: "both".to_string(),
        }),
        ("Long only, 10-day hold", StrategyConfig {
            entry_threshold: 0.02,
            exit_days: 10,
            stop_loss: -0.05,
            take_profit: 0.10,
            direction: "long".to_string(),
        }),
    ];

    println!("\n{:<30} {:>10} {:>10} {:>10}", "Strategy", "Trades", "Return", "Sharpe");
    println!("{}", "-".repeat(62));

    for (name, strat) in strategies {
        let res = engine.run(&events, &strat).unwrap();
        println!(
            "{:<30} {:>10} {:>9.1}% {:>10.3}",
            name,
            res.n_trades,
            res.total_return * 100.0,
            res.sharpe_ratio
        );
    }

    println!("\nBacktest completed successfully!");
}
