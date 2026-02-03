//! Example: Backtest with counterfactual risk management
//!
//! This example demonstrates how to use counterfactual explanations
//! for risk management in a trading strategy.
//!
//! Run with: cargo run --example backtest

use rand::Rng;
use rust_counterfactual::{
    counterfactual::CounterfactualOptimizer,
    data::get_sample_data,
    model::TradingClassifier,
    strategy::Backtester,
};

fn main() {
    println!("Backtesting with Counterfactual Risk Management");
    println!("================================================\n");

    // Load sample data and train model
    println!("1. Setting up...");
    let (features, labels, feature_names) = get_sample_data();

    let mut model = TradingClassifier::new(feature_names.len(), 64, 3);
    model.train(&features, &labels, 30, 0.01);
    println!("   Model trained.\n");

    // Generate synthetic prices
    println!("2. Generating synthetic price data...");
    let n = features.len();
    let mut rng = rand::thread_rng();
    let mut prices = vec![100.0; n];
    for i in 1..n {
        let ret = rng.gen::<f64>() * 0.04 - 0.02; // -2% to +2%
        prices[i] = prices[i - 1] * (1.0 + ret);
    }
    println!("   Generated {} price points.\n", n);

    // Generate predictions
    println!("3. Generating trading signals...");
    let signals: Vec<usize> = features.iter().map(|x| model.predict(x)).collect();
    let buy_count = signals.iter().filter(|&&s| s == 2).count();
    let sell_count = signals.iter().filter(|&&s| s == 0).count();
    let hold_count = signals.iter().filter(|&&s| s == 1).count();
    println!("   Signals: {} BUY, {} SELL, {} HOLD\n", buy_count, sell_count, hold_count);

    // Calculate counterfactual distances
    println!("4. Calculating counterfactual distances...");
    let optimizer = CounterfactualOptimizer::new(&model);
    let mut cf_distances = vec![0.0; n];

    for (i, x) in features.iter().enumerate() {
        let pred = signals[i];
        if pred != 1 { // Only for non-HOLD
            let target = if pred == 2 { 0 } else { 2 };
            let result = optimizer.generate(x, target, 30, 0.05);
            cf_distances[i] = result.l1_distance;
        }
    }

    let avg_dist: f64 = cf_distances.iter().sum::<f64>() / n as f64;
    println!("   Average CF distance: {:.4}\n", avg_dist);

    // Run backtests
    let backtester = Backtester::new(10000.0)
        .with_commission(0.001)
        .with_slippage(0.0005);

    // Backtest WITHOUT CF filtering
    println!("5. Backtest WITHOUT counterfactual filtering:");
    let result_no_cf = backtester.run(&prices, &signals, None, None);
    println!("{}", result_no_cf);

    // Backtest WITH CF filtering (threshold = 0.3)
    println!("\n6. Backtest WITH counterfactual filtering (threshold=0.3):");
    let result_with_cf = backtester.run(
        &prices,
        &signals,
        Some(&cf_distances),
        Some(0.3),
    );
    println!("{}", result_with_cf);

    // Compare results
    println!("\n7. Comparison:");
    println!("   {:>25} {:>15} {:>15}", "", "No CF Filter", "CF Filter");
    println!("   {:-<25} {:-<15} {:-<15}", "", "", "");
    println!("   {:>25} {:>14.2}% {:>14.2}%", "Total Return",
             result_no_cf.total_return * 100.0,
             result_with_cf.total_return * 100.0);
    println!("   {:>25} {:>15.2} {:>15.2}", "Sharpe Ratio",
             result_no_cf.sharpe_ratio,
             result_with_cf.sharpe_ratio);
    println!("   {:>25} {:>14.2}% {:>14.2}%", "Max Drawdown",
             result_no_cf.max_drawdown * 100.0,
             result_with_cf.max_drawdown * 100.0);
    println!("   {:>25} {:>14.2}% {:>14.2}%", "Win Rate",
             result_no_cf.win_rate * 100.0,
             result_with_cf.win_rate * 100.0);
    println!("   {:>25} {:>15} {:>15}", "Num Trades",
             result_no_cf.num_trades,
             result_with_cf.num_trades);

    // Analyze trades by CF distance
    if !result_no_cf.trades.is_empty() {
        println!("\n8. Trade analysis by CF distance:");

        let trades_with_cf: Vec<_> = result_no_cf.trades.iter()
            .filter(|t| t.counterfactual_distance.is_some())
            .collect();

        if trades_with_cf.len() > 1 {
            let distances: Vec<f64> = trades_with_cf.iter()
                .map(|t| t.counterfactual_distance.unwrap())
                .collect();
            let pnls: Vec<f64> = trades_with_cf.iter()
                .map(|t| t.pnl_pct)
                .collect();

            let median_dist = {
                let mut sorted = distances.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            };

            let high_conf_wins: Vec<_> = trades_with_cf.iter()
                .filter(|t| t.counterfactual_distance.unwrap() >= median_dist)
                .filter(|t| t.pnl_pct > 0.0)
                .collect();

            let low_conf_wins: Vec<_> = trades_with_cf.iter()
                .filter(|t| t.counterfactual_distance.unwrap() < median_dist)
                .filter(|t| t.pnl_pct > 0.0)
                .collect();

            let high_conf_total = trades_with_cf.iter()
                .filter(|t| t.counterfactual_distance.unwrap() >= median_dist)
                .count();
            let low_conf_total = trades_with_cf.iter()
                .filter(|t| t.counterfactual_distance.unwrap() < median_dist)
                .count();

            println!("   Median CF distance: {:.4}", median_dist);
            if high_conf_total > 0 {
                println!("   High confidence win rate: {:.1}% ({}/{})",
                         high_conf_wins.len() as f64 / high_conf_total as f64 * 100.0,
                         high_conf_wins.len(), high_conf_total);
            }
            if low_conf_total > 0 {
                println!("   Low confidence win rate: {:.1}% ({}/{})",
                         low_conf_wins.len() as f64 / low_conf_total as f64 * 100.0,
                         low_conf_wins.len(), low_conf_total);
            }
        }
    }

    println!("\nDone!");
}
