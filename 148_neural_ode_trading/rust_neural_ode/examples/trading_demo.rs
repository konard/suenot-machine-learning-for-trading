//! Trading Demo Example
//!
//! Demonstrates a complete Neural ODE trading pipeline:
//! 1. Generate/fetch market data
//! 2. Create and configure Neural ODE model
//! 3. Generate trading signals
//! 4. Backtest the strategy
//! 5. Print performance metrics
//!
//! Run: cargo run --example trading_demo

use neural_ode_trading::backtest::Backtester;
use neural_ode_trading::data::generate_synthetic_klines;
use neural_ode_trading::model::NeuralODE;
use neural_ode_trading::ode_solver::{ODESolver, RK4Solver};
use ndarray::Array1;

fn main() {
    println!("═══════════════════════════════════════════════════");
    println!("  Neural ODE Trading - Complete Demo");
    println!("═══════════════════════════════════════════════════\n");

    // ── Step 1: Generate Market Data ──
    println!("Step 1: Generating synthetic BTCUSDT data...");
    let btc_data = generate_synthetic_klines("BTCUSDT", 500);
    println!("  Generated {} candles", btc_data.len());

    let first_price = btc_data.first().unwrap().close;
    let last_price = btc_data.last().unwrap().close;
    println!(
        "  Price range: ${:.2} -> ${:.2} ({:+.2}%)",
        first_price,
        last_price,
        (last_price / first_price - 1.0) * 100.0
    );

    // ── Step 2: Create Neural ODE Model ──
    println!("\nStep 2: Creating Neural ODE model...");
    let window_size = 10;
    let input_dim = window_size * 3; // 3 features per timestep
    let hidden_dim = 32;
    let output_dim = 1;

    let model = NeuralODE::new(input_dim, hidden_dim, output_dim, 0.05);
    println!("  Input dim:  {}", input_dim);
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Parameters: {}", model.num_parameters());

    // ── Step 3: Demonstrate ODE Dynamics ──
    println!("\nStep 3: Neural ODE forward pass demonstration...");

    // Create a sample input
    let sample_input: Vec<f64> = (0..input_dim)
        .map(|i| (i as f64 * 0.1).sin() * 0.5)
        .collect();

    // Single-step prediction
    let prediction = model.forward(&sample_input);
    println!("  Single prediction: {:.6}", prediction[0]);

    // Multi-step trajectory prediction
    let t_points: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
    let trajectory = model.forward_trajectory(&sample_input, &t_points);
    println!("\n  Trajectory at different time points:");
    println!("  {:>6} {:>12}", "t", "Output");
    for (t, pred) in t_points.iter().zip(trajectory.iter()) {
        println!("  {:>6.1} {:>12.6}", t, pred[0]);
    }

    // ── Step 4: ODE Solver Internals ──
    println!("\nStep 4: ODE solver analysis...");

    let solver = RK4Solver;
    let h0 = Array1::from_vec(vec![0.5; hidden_dim]);

    // Create a standalone ODE function to demonstrate dynamics
    let simple_dynamics = |_t: f64, h: &Array1<f64>| {
        // Simple decay dynamics for demonstration
        h.mapv(|x| -0.5 * x + 0.1 * x.tanh())
    };

    let h_final = solver.solve(&simple_dynamics, &h0, 0.0, 1.0, 0.01);
    println!(
        "  Initial state norm: {:.4}",
        h0.mapv(|x| x * x).sum().sqrt()
    );
    println!(
        "  Final state norm:   {:.4}",
        h_final.mapv(|x| x * x).sum().sqrt()
    );
    println!("  State changed by:   {:.4} (L2 distance)",
        (&h_final - &h0).mapv(|x| x * x).sum().sqrt()
    );

    // ── Step 5: Backtesting ──
    println!("\nStep 5: Running backtest on BTCUSDT...");

    let backtester = Backtester::new(100_000.0, 0.001);
    let signals = backtester.generate_signals(&model, &btc_data, window_size);
    let result = backtester.run(&btc_data, &signals, window_size);
    result.print_summary();

    // ── Step 6: Multi-asset comparison ──
    println!("Step 6: Multi-asset comparison...\n");

    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    println!(
        "  {:<10} {:>12} {:>10} {:>12} {:>10}",
        "Symbol", "Return", "Sharpe", "MaxDD", "Trades"
    );
    println!("  {}", "-".repeat(58));

    for symbol in &symbols {
        let data = generate_synthetic_klines(symbol, 500);
        let model = NeuralODE::new(input_dim, hidden_dim, output_dim, 0.05);
        let bt = Backtester::new(100_000.0, 0.001);
        let sigs = bt.generate_signals(&model, &data, window_size);
        let res = bt.run(&data, &sigs, window_size);

        println!(
            "  {:<10} {:>11.2}% {:>10.3} {:>11.2}% {:>10}",
            symbol,
            res.total_return * 100.0,
            res.sharpe_ratio,
            res.max_drawdown * 100.0,
            res.n_trades,
        );
    }

    println!("\n═══════════════════════════════════════════════════");
    println!("  Demo complete. Key points:");
    println!("  - Neural ODEs model continuous-time market dynamics");
    println!("  - ODE solvers (RK4, Dormand-Prince) integrate the dynamics");
    println!("  - The model handles irregular timestamps natively");
    println!("  - Backtest framework evaluates trading performance");
    println!("═══════════════════════════════════════════════════");
}
