//! Trading example: compare Behavioral Cloning vs MaxEnt IRL
//!
//! Fetches BTCUSDT data from Bybit, generates expert demonstrations,
//! trains both methods, and shows comparative results.

use anyhow::Result;
use imitation_learning_trading::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Imitation Learning for Trading ===\n");

    // ------------------------------------------------------------------
    // 1. Fetch market data from Bybit (fallback to synthetic if offline)
    // ------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT data from Bybit...");
    let states = match fetch_bybit_klines("BTCUSDT", "60", 200).await {
        Ok(candles) => {
            println!("    Fetched {} candles from Bybit", candles.len());
            let s = candles_to_states(&candles, 10);
            println!("    Derived {} trading states", s.len());
            s
        }
        Err(e) => {
            println!("    Bybit API unavailable ({}), using synthetic data", e);
            generate_synthetic_states(200, 42)
        }
    };

    // ------------------------------------------------------------------
    // 2. Generate expert demonstrations using different strategies
    // ------------------------------------------------------------------
    println!("\n[2] Generating expert demonstrations...");

    let momentum_expert = ExpertPolicy::new(ExpertType::Momentum, 0.05);
    let reversion_expert = ExpertPolicy::new(ExpertType::MeanReversion, 0.01);

    let momentum_demos = momentum_expert.generate_demonstrations(&states);
    let reversion_demos = reversion_expert.generate_demonstrations(&states);

    println!(
        "    Momentum expert: {} demos (Buy: {}, Hold: {}, Sell: {})",
        momentum_demos.len(),
        momentum_demos.iter().filter(|d| d.action == TradingAction::Buy).count(),
        momentum_demos.iter().filter(|d| d.action == TradingAction::Hold).count(),
        momentum_demos.iter().filter(|d| d.action == TradingAction::Sell).count(),
    );
    println!(
        "    MeanReversion expert: {} demos (Buy: {}, Hold: {}, Sell: {})",
        reversion_demos.len(),
        reversion_demos.iter().filter(|d| d.action == TradingAction::Buy).count(),
        reversion_demos.iter().filter(|d| d.action == TradingAction::Hold).count(),
        reversion_demos.iter().filter(|d| d.action == TradingAction::Sell).count(),
    );

    // ------------------------------------------------------------------
    // 3. Train and compare methods on Momentum expert
    // ------------------------------------------------------------------
    println!("\n[3] Training on Momentum expert demonstrations...");
    let split = momentum_demos.len() * 3 / 4;
    let (train, test) = momentum_demos.split_at(split);

    println!("    Train: {} samples, Test: {} samples", train.len(), test.len());

    let mut bc = BehavioralCloner::new(0.01, 100);
    bc.train(train)?;
    let bc_metrics = bc.evaluate(test);

    let mut irl = MaxEntropyIRL::new(0.05, 200);
    irl.train(train)?;
    let irl_metrics = irl.evaluate(test);

    println!("\n    --- Momentum Expert Results ---");
    println!("    {:<20} {:>10} {:>10}", "Metric", "BC", "IRL");
    println!("    {:-<20} {:->10} {:->10}", "", "", "");
    println!(
        "    {:<20} {:>10.4} {:>10.4}",
        "Accuracy", bc_metrics.accuracy, irl_metrics.accuracy
    );
    println!(
        "    {:<20} {:>10.4} {:>10.4}",
        "Total PnL", bc_metrics.total_pnl, irl_metrics.total_pnl
    );
    println!(
        "    {:<20} {:>10.4} {:>10.4}",
        "Sharpe Ratio", bc_metrics.sharpe_ratio, irl_metrics.sharpe_ratio
    );
    println!(
        "    {:<20} {:>10.4} {:>10.4}",
        "Max Drawdown", bc_metrics.max_drawdown, irl_metrics.max_drawdown
    );

    // ------------------------------------------------------------------
    // 4. Train and compare methods on MeanReversion expert
    // ------------------------------------------------------------------
    println!("\n[4] Training on MeanReversion expert demonstrations...");
    let split = reversion_demos.len() * 3 / 4;
    let (train_r, test_r) = reversion_demos.split_at(split);

    let mut bc_r = BehavioralCloner::new(0.01, 100);
    bc_r.train(train_r)?;
    let bc_r_metrics = bc_r.evaluate(test_r);

    let mut irl_r = MaxEntropyIRL::new(0.05, 200);
    irl_r.train(train_r)?;
    let irl_r_metrics = irl_r.evaluate(test_r);

    println!("\n    --- MeanReversion Expert Results ---");
    println!("    {:<20} {:>10} {:>10}", "Metric", "BC", "IRL");
    println!("    {:-<20} {:->10} {:->10}", "", "", "");
    println!(
        "    {:<20} {:>10.4} {:>10.4}",
        "Accuracy", bc_r_metrics.accuracy, irl_r_metrics.accuracy
    );
    println!(
        "    {:<20} {:>10.4} {:>10.4}",
        "Total PnL", bc_r_metrics.total_pnl, irl_r_metrics.total_pnl
    );
    println!(
        "    {:<20} {:>10.4} {:>10.4}",
        "Sharpe Ratio", bc_r_metrics.sharpe_ratio, irl_r_metrics.sharpe_ratio
    );
    println!(
        "    {:<20} {:>10.4} {:>10.4}",
        "Max Drawdown", bc_r_metrics.max_drawdown, irl_r_metrics.max_drawdown
    );

    // ------------------------------------------------------------------
    // 5. Show IRL reward interpretability
    // ------------------------------------------------------------------
    println!("\n[5] IRL Reward Interpretability...");
    let weights = irl_r.get_reward_weights();
    let feature_names = ["return", "volatility", "momentum", "volume"];
    let action_names = ["Buy", "Hold", "Sell"];

    println!("    Learned reward weights (MeanReversion expert):");
    for (a, aname) in action_names.iter().enumerate() {
        print!("    {:<6}", aname);
        for (f, fname) in feature_names.iter().enumerate() {
            let idx = a * TradingState::NUM_FEATURES + f;
            print!("  {}={:+.4}", fname, weights[idx]);
        }
        println!();
    }

    // ------------------------------------------------------------------
    // 6. When each method wins
    // ------------------------------------------------------------------
    println!("\n[6] When each method wins:");
    println!("    - BC excels when expert demonstrations are abundant and the");
    println!("      task horizon is short (fewer compounding errors).");
    println!("    - IRL excels when you need interpretable reward functions or");
    println!("      want to transfer the learned objective to new market conditions.");
    println!("    - For momentum-following experts, BC typically matches IRL");
    println!("      because the decision boundary is simple and linear.");
    println!("    - For mean-reversion experts, IRL often provides more robust");
    println!("      generalisation by capturing the underlying reward structure.");

    println!("\n=== Done ===");
    Ok(())
}
