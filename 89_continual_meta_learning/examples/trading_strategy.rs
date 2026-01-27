//! Trading strategy example using Continual MAML.
//!
//! This example demonstrates:
//! - Setting up a trading strategy with continual meta-learning
//! - Pre-training on historical regimes
//! - Running the strategy on new market data
//! - Evaluating trading performance

use continual_meta_learning::{
    data::bybit::SimulatedDataGenerator,
    data::features::FeatureGenerator,
    continual::algorithm::{ContinualMAMLTrainer, TaskData},
    model::network::TradingModel,
    backtest::engine::{BacktestEngine, BacktestConfig},
};

fn main() {
    println!("=== Continual MAML Trading Strategy Example ===\n");

    let feature_gen = FeatureGenerator::default_window();

    // Phase 1: Pre-train on historical regimes
    println!("Phase 1: Pre-training on historical market regimes...");
    let model = TradingModel::new(11, 32, 1);
    let mut trainer = ContinualMAMLTrainer::new(
        model, 0.01, 0.001, 5, true, 100.0, 50,
    );

    let historical_regimes: Vec<(&str, f64, f64)> = vec![
        ("Bull", 0.015, 0.0003),
        ("Bear", 0.02, -0.0003),
        ("Sideways", 0.008, 0.0),
        ("Volatile", 0.03, 0.0),
    ];

    for (regime_id, (name, vol, trend)) in historical_regimes.iter().enumerate() {
        let klines = SimulatedDataGenerator::generate_trending_klines(
            300, 50000.0, *vol, *trend,
        );
        let features = feature_gen.compute_features(&klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        if features.len() < 40 {
            continue;
        }

        let split = features.len() / 2;
        let mut tasks = Vec::new();

        for _ in 0..3 {
            let sf = features[..split].to_vec();
            let sl = returns[..split].to_vec();
            let qf = features[split..].to_vec();
            let ql_len = qf.len().min(returns.len() - split);
            let ql = returns[split..split + ql_len].to_vec();

            if sf.len() == sl.len() && qf.len() == ql.len() {
                tasks.push(TaskData::new(sf, sl, qf, ql, regime_id));
            }
        }

        if !tasks.is_empty() {
            let losses = trainer.learn_regime(&tasks, regime_id, 10, 0);
            println!("  {} regime: loss {:.6} -> {:.6}",
                name,
                losses.first().unwrap_or(&0.0),
                losses.last().unwrap_or(&0.0)
            );
        }
    }

    println!("  Pre-training complete. {} regimes learned.\n",
        trainer.regimes_seen().len()
    );

    // Phase 2: Backtest on new regime-changing data
    println!("Phase 2: Backtesting on new regime-changing data...");
    let test_klines = SimulatedDataGenerator::generate_regime_changing_klines(500, 50000.0);
    println!("  Generated {} test candles", test_klines.len());

    let config = BacktestConfig {
        initial_capital: 10000.0,
        transaction_cost: 0.001,
        slippage: 0.0005,
        threshold: 0.001,
        adaptation_window: 20,
        adaptation_steps: 5,
    };

    let engine = BacktestEngine::new(config);
    let results = engine.run(&trainer, &test_klines);

    // Phase 3: Display results
    println!("\n=== Backtest Results ===");
    println!("Capital: ${:.2} -> ${:.2}", results.initial_capital, results.final_capital);
    println!("Total Return: {:.2}%", results.total_return * 100.0);
    println!("Sharpe Ratio: {:.3}", results.sharpe_ratio);
    println!("Sortino Ratio: {:.3}", results.sortino_ratio);
    println!("Max Drawdown: {:.2}%", results.max_drawdown * 100.0);
    println!("Trades: {}", results.num_trades);
    println!("Win Rate: {:.1}%", results.win_rate * 100.0);
    println!("Profit Factor: {:.2}", results.profit_factor);

    // Compare with buy-and-hold
    let first_price = test_klines.first().map(|k| k.close).unwrap_or(1.0);
    let last_price = test_klines.last().map(|k| k.close).unwrap_or(1.0);
    let buy_hold_return = (last_price / first_price) - 1.0;

    println!("\n=== Comparison ===");
    println!("Continual MAML: {:+.2}%", results.total_return * 100.0);
    println!("Buy & Hold:     {:+.2}%", buy_hold_return * 100.0);

    let outperformance = results.total_return - buy_hold_return;
    if outperformance > 0.0 {
        println!("Continual MAML outperformed by {:.2}%", outperformance * 100.0);
    } else {
        println!("Buy & Hold outperformed by {:.2}%", -outperformance * 100.0);
    }

    println!("\n=== Example Complete ===");
}
