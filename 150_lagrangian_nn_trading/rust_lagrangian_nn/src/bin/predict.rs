//! Predict and backtest using a trained Lagrangian Neural Network.
//!
//! Usage:
//!   cargo run --release --bin predict -- --model output/lnn_model.json --symbol BTCUSDT

use clap::Parser;
use lagrangian_nn_trading::nn::LagrangianNN;
use lagrangian_nn_trading::data::{BybitClient, construct_config_space, normalize_config_space};
use lagrangian_nn_trading::trading::TradingStrategy;
use lagrangian_nn_trading::integrator::integrate_trajectory;
use lagrangian_nn_trading::utils;

#[derive(Parser, Debug)]
#[command(name = "predict", about = "Predict and backtest with Lagrangian NN")]
struct Args {
    /// Path to saved model JSON
    #[arg(short, long)]
    model: String,

    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval
    #[arg(short, long, default_value = "5")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value_t = 5000)]
    limit: usize,

    /// MA window
    #[arg(long, default_value_t = 20)]
    ma_window: usize,

    /// Initial capital for backtest
    #[arg(long, default_value_t = 100000.0)]
    initial_capital: f64,

    /// Commission per trade
    #[arg(long, default_value_t = 0.001)]
    commission: f64,

    /// Prediction horizon (integration steps)
    #[arg(long, default_value_t = 10)]
    prediction_horizon: usize,

    /// Entry threshold for signals
    #[arg(long, default_value_t = 0.5)]
    entry_threshold: f64,

    /// Output directory
    #[arg(short, long, default_value = "output")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Lagrangian NN Trading: Predict & Backtest ===");
    println!("Model:    {}", args.model);
    println!("Symbol:   {}", args.symbol);
    println!("Capital:  ${:.2}", args.initial_capital);
    println!();

    std::fs::create_dir_all(&args.output)?;

    // Load model
    let model_json = std::fs::read_to_string(&args.model)?;
    let model: LagrangianNN = serde_json::from_str(&model_json)?;
    println!("Loaded model ({} parameters)", model.num_parameters());

    // Fetch data
    let client = BybitClient::new();
    println!("Fetching data...");
    let candles = client.fetch_extended(&args.symbol, &args.interval, args.limit).await?;
    println!("Fetched {} candles", candles.len());

    // Construct config space
    let config_data = construct_config_space(&candles, args.ma_window);
    let (normalized, _stats) = normalize_config_space(&config_data);

    println!("Config space: {} samples", normalized.q.len());

    // Demonstrate trajectory prediction
    println!("\n--- Trajectory Prediction ---");
    let sample_idx = normalized.q.len() / 2;
    let q_sample = &normalized.q[sample_idx];
    let qdot_sample = &normalized.qdot[sample_idx];

    println!("Starting state: q={:.4}, q-dot={:.4}", q_sample[0], qdot_sample[0]);
    println!("Starting energy: {:.6}", model.energy(q_sample, qdot_sample));

    let (traj_q, traj_v) = integrate_trajectory(
        &model, q_sample, qdot_sample, 0.1, args.prediction_horizon,
    );

    println!("Predicted trajectory ({} steps):", args.prediction_horizon);
    for (i, (q, v)) in traj_q.iter().zip(traj_v.iter()).enumerate() {
        if i % 3 == 0 || i == traj_q.len() - 1 {
            let e = model.energy(q, v);
            println!("  t={}: q={:.4}, q-dot={:.4}, E={:.6}", i, q[0], v[0], e);
        }
    }

    let predicted_change = traj_q.last().unwrap()[0] - traj_q[0][0];
    println!("Predicted q change: {:.6}", predicted_change);
    println!("Signal: {}", if predicted_change > args.entry_threshold {
        "BUY"
    } else if predicted_change < -args.entry_threshold {
        "SELL"
    } else {
        "HOLD"
    });

    // Energy conservation check
    let energies = utils::energy_along_trajectory(&model, &traj_q, &traj_v);
    let e_drift = if let (Some(first), Some(last)) = (energies.first(), energies.last()) {
        (last - first) / (first.abs() + 1e-10) * 100.0
    } else {
        0.0
    };
    println!("Energy drift: {:.4}%", e_drift);

    // Run backtest
    println!("\n--- Backtest ---");
    let mut strategy = TradingStrategy::new(
        model, args.prediction_horizon, 0.1, args.entry_threshold,
    );

    let result = strategy.backtest(
        &normalized.prices,
        &normalized.q,
        &normalized.qdot,
        args.initial_capital,
        args.commission,
    );

    println!("==========================================");
    println!("BACKTEST RESULTS (Lagrangian NN Strategy)");
    println!("==========================================");
    println!("Initial Capital:   ${:.2}", result.initial_capital);
    println!("Final Capital:     ${:.2}", result.final_capital);
    println!("Total Return:      {:+.2}%", result.total_return * 100.0);
    println!("Max Drawdown:      {:.2}%", result.max_drawdown * 100.0);
    println!("Sharpe Ratio:      {:.3}", result.sharpe_ratio);
    println!("Win Rate:          {:.1}%", result.win_rate * 100.0);
    println!("Number of Trades:  {}", result.n_trades);
    println!("==========================================");

    // Save results
    let results_json = serde_json::to_string_pretty(&result)?;
    let results_path = format!("{}/backtest_results.json", args.output);
    std::fs::write(&results_path, results_json)?;
    println!("\nResults saved to {}", results_path);

    // Save equity curve
    let equity_data: Vec<Vec<f64>> = result.equity_curve.iter()
        .enumerate()
        .map(|(i, &e)| vec![i as f64, e])
        .collect();
    let equity_path = format!("{}/equity_curve.csv", args.output);
    utils::export_csv(&equity_path, &["step", "equity"], &equity_data)?;
    println!("Equity curve saved to {}", equity_path);

    println!("\nPrediction and backtesting complete!");
    Ok(())
}
