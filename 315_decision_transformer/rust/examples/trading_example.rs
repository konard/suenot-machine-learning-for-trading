//! Decision Transformer Trading Example
//!
//! Demonstrates:
//! 1. Fetching BTCUSDT data from Bybit
//! 2. Building an offline dataset from OHLCV candles
//! 3. Training a Decision Transformer
//! 4. Generating trading actions conditioned on target return

use decision_transformer::{
    fetch_bybit_candles, generate_synthetic_candles, normalize_candles,
    DecisionTransformer, TrajectoryDataset, TradingAction,
};

fn main() {
    println!("=== Decision Transformer for Trading ===\n");

    // Step 1: Fetch data from Bybit (fall back to synthetic if unavailable)
    println!("Step 1: Fetching BTCUSDT data from Bybit...");
    let candles = match fetch_bybit_candles("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} candles from Bybit", c.len());
            if c.len() < 10 {
                println!("  Too few candles, using synthetic data");
                generate_synthetic_candles(200)
            } else {
                c
            }
        }
        Err(e) => {
            println!("  Bybit API unavailable ({}), using synthetic data", e);
            generate_synthetic_candles(200)
        }
    };

    println!(
        "  Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // Step 2: Build offline dataset
    println!("Step 2: Building offline dataset...");
    let episode_len = 20;
    let dataset = TrajectoryDataset::from_candles(&candles, episode_len);
    let stats = dataset.stats();
    println!("  Trajectories: {}", stats.n_trajectories);
    println!("  Total transitions: {}", stats.total_transitions);
    println!("  Average episode return: {:.6}", stats.avg_return);
    println!("  Max episode return: {:.6}", stats.max_return);
    println!();

    // Step 3: Create and train Decision Transformer
    println!("Step 3: Training Decision Transformer...");
    let state_dim = 5;   // OHLCV
    let n_actions = 3;   // sell, hold, buy
    let d_model = 32;
    let n_heads = 2;
    let n_layers = 2;
    let max_seq_len = episode_len;
    let lr = 0.001;
    let n_epochs = 20;

    let mut dt = DecisionTransformer::new(
        state_dim, n_actions, d_model, n_heads, n_layers, max_seq_len, lr,
    );

    for epoch in 1..=n_epochs {
        let loss = dt.train_epoch(&dataset);
        if epoch == 1 || epoch % 5 == 0 || epoch == n_epochs {
            println!("  Epoch {}/{}: loss = {:.4}", epoch, n_epochs, loss);
        }
    }
    println!();

    // Step 4: Generate trading actions with target return
    println!("Step 4: Generating trading actions...");

    let normalized = normalize_candles(&candles);
    let states: Vec<Vec<f64>> = normalized
        .iter()
        .map(|c| vec![c.open, c.high, c.low, c.close, c.volume])
        .collect();

    // Test with different target returns
    let target_returns = vec![0.01, 0.02, 0.05];

    for target in &target_returns {
        println!("\n  --- Target Return: {:.1}% ---", target * 100.0);

        let n_steps = 10;
        let start_idx = states.len().saturating_sub(n_steps + 5);
        let test_states = &states[start_idx..];

        let results = dt.generate_actions(&test_states[0], *target, test_states, n_steps);

        let mut position = 0i32;
        let mut cumulative_pnl = 0.0;

        for (step, (action, confidence)) in results.iter().enumerate() {
            let state_preview: Vec<String> = test_states[step]
                .iter()
                .take(4)
                .map(|v| format!("{:.3}", v))
                .collect();

            // Update position
            match action {
                TradingAction::Buy => position = 1,
                TradingAction::Sell => position = -1,
                TradingAction::Hold => {}
            }

            // Compute step PnL
            let step_pnl = if step + 1 < test_states.len() {
                let curr_close = test_states[step][3];
                let next_close = test_states[step + 1][3];
                if curr_close > 0.0 {
                    position as f64 * (next_close - curr_close) / curr_close
                } else {
                    0.0
                }
            } else {
                0.0
            };
            cumulative_pnl += step_pnl;

            println!(
                "  Step {:2}: State=[{}] Action={:4} Confidence={:.3} Position={:2} PnL={:+.4}%",
                step + 1,
                state_preview.join(", "),
                action.label(),
                confidence,
                position,
                step_pnl * 100.0
            );
        }
        println!(
            "  Cumulative PnL: {:+.4}% (Target: {:.1}%)",
            cumulative_pnl * 100.0,
            target * 100.0
        );
    }

    println!("\n=== Done ===");
}
