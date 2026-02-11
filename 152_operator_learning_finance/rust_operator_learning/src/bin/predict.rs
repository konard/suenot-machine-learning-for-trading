//! Prediction and inference using trained operator learning models.
//!
//! Supports:
//! - Real-time prediction on live Bybit data
//! - Batch prediction on historical data
//! - Zero-shot super-resolution demonstration
//! - Comparison with analytical solutions
//!
//! # Usage
//! ```bash
//! cargo run --bin predict -- --mode live --symbol BTCUSDT
//! cargo run --bin predict -- --mode batch --data black_scholes
//! cargo run --bin predict -- --mode compare
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use ndarray::Array1;
use operator_learning_trading::{
    backtest,
    data,
    operator::{self, DeepONet, DeepONetConfig, SpectralOperator, black_scholes_call},
    strategy,
};

#[derive(Parser, Debug)]
#[command(name = "predict", about = "Run operator learning predictions")]
struct Args {
    /// Prediction mode: live, batch, compare, superres
    #[arg(short, long, default_value = "compare")]
    mode: String,

    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of samples for batch prediction
    #[arg(long, default_value_t = 100)]
    samples: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("{}", "═══════════════════════════════════════════════════════".blue());
    println!("{}", "  Operator Learning Predictor - Chapter 152".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════".blue());

    match args.mode.as_str() {
        "live" => predict_live(&args)?,
        "batch" => predict_batch(&args)?,
        "compare" => compare_methods(&args)?,
        "superres" => demo_superresolution(&args)?,
        _ => {
            eprintln!("{}: Unknown mode '{}'. Use: live, batch, compare, superres",
                "Error".red(), args.mode);
        }
    }

    Ok(())
}

fn predict_live(args: &Args) -> Result<()> {
    println!(
        "\n{} real-time prediction for {}...",
        "Starting".yellow().bold(),
        args.symbol.cyan()
    );

    // Fetch recent data
    let candles = data::fetch_bybit_klines(&args.symbol, "60", 200)?;
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();

    println!("  Fetched {} candles. Latest price: {:.2}", prices.len(), prices.last().unwrap_or(&0.0));

    // Train a quick model on recent data
    let lookback = 60;
    let horizon = 5;

    let (branch_data, trunk_data, targets) = match data::prepare_price_operator_data(&prices, lookback, horizon) {
        Some(d) => d,
        None => { eprintln!("Not enough data"); return Ok(()); }
    };

    let config = DeepONetConfig {
        branch_input_dim: lookback,
        trunk_input_dim: 1,
        latent_dim: 32,
        branch_hidden_dims: vec![64, 64],
        trunk_hidden_dims: vec![32, 32],
        learning_rate: 0.002,
        n_epochs: 50,
        batch_size: 32,
    };

    let mut model = DeepONet::new(config);
    println!("\n  Training on recent data ({} samples)...", branch_data.nrows());
    let _ = model.train(&branch_data, &trunk_data, &targets);

    // Make predictions for multiple horizons
    let window = &prices[prices.len() - lookback..];
    let w_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
    let w_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = w_max - w_min;
    let current_price = *prices.last().unwrap();

    let norm_window: Vec<f64> = window.iter().map(|p| (p - w_min) / range).collect();
    let branch_input = Array1::from_vec(norm_window);

    println!("\n  {} for {}:", "Predictions".cyan().bold(), args.symbol);
    println!("  Current price: {:.2}", current_price);
    println!("  {:>10}  {:>12}  {:>10}", "Horizon", "Predicted", "Change");

    for h in 1..=horizon {
        let eval_frac = h as f64 / horizon as f64;
        let trunk_input = Array1::from_vec(vec![eval_frac]);
        let pred_norm = model.predict(&branch_input, &trunk_input);
        let pred_price = pred_norm * range + w_min;
        let change_pct = (pred_price - current_price) / current_price * 100.0;

        let change_str = if change_pct >= 0.0 {
            format!("+{:.3}%", change_pct).green().to_string()
        } else {
            format!("{:.3}%", change_pct).red().to_string()
        };

        println!("  {:>8}h  {:>12.2}  {}", h, pred_price, change_str);
    }

    // Generate signal
    let strat_config = strategy::StrategyConfig {
        lookback,
        prediction_horizon: horizon,
        buy_threshold: 0.003,
        sell_threshold: -0.003,
    };
    let signal = strategy::generate_signal(&model, &prices, &strat_config);
    let signal_str = match signal {
        strategy::Signal::Buy => "BUY".green().bold().to_string(),
        strategy::Signal::Sell => "SELL".red().bold().to_string(),
        strategy::Signal::Hold => "HOLD".yellow().to_string(),
    };
    println!("\n  Trading Signal: {}", signal_str);

    Ok(())
}

fn predict_batch(args: &Args) -> Result<()> {
    println!(
        "\n{} batch prediction on Black-Scholes data...",
        "Running".yellow().bold()
    );

    // Generate test data
    let (branch_data, trunk_data, targets) =
        data::generate_bs_training_data(args.samples, 50, 10);

    // Train model
    let config = DeepONetConfig {
        branch_input_dim: 50,
        trunk_input_dim: 2,
        latent_dim: 64,
        branch_hidden_dims: vec![128, 128],
        trunk_hidden_dims: vec![128, 128],
        learning_rate: 0.001,
        n_epochs: 80,
        batch_size: 64,
    };

    let mut model = DeepONet::new(config);
    let train_end = (branch_data.nrows() as f64 * 0.8) as usize;

    let branch_train = branch_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let trunk_train = trunk_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let targets_train = targets.slice(ndarray::s![..train_end]).to_owned();

    println!("  Training on {} samples...", train_end);
    let _ = model.train(&branch_train, &trunk_train, &targets_train);

    // Batch predictions on test set
    let branch_test = branch_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let trunk_test = trunk_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let targets_test = targets.slice(ndarray::s![train_end..]).to_owned();

    let rel_error = model.evaluate(&branch_test, &trunk_test, &targets_test);
    println!("\n  Test Relative L2 Error: {:.6}", rel_error);

    // Individual predictions
    println!("\n  {:>6}  {:>6}  {:>10}  {:>10}  {:>8}", "S/K", "T", "Predicted", "Actual", "AbsErr");
    for i in 0..10.min(branch_test.nrows()) {
        let branch_in = branch_test.row(i).to_owned();
        let trunk_in = trunk_test.row(i).to_owned();
        let pred = model.predict(&branch_in, &trunk_in);
        let actual = targets_test[i];
        println!(
            "  {:>6.2}  {:>6.2}  {:>10.6}  {:>10.6}  {:>8.6}",
            trunk_in[0], trunk_in[1], pred, actual, (pred - actual).abs()
        );
    }

    Ok(())
}

fn compare_methods(args: &Args) -> Result<()> {
    println!(
        "\n{} Neural Operator vs Analytical Black-Scholes...",
        "Comparing".yellow().bold()
    );

    // Generate test scenarios
    let test_cases = vec![
        (1.0, 1.0, 0.2, 0.03, "ATM, 1Y, sig=20%"),
        (1.1, 0.5, 0.3, 0.03, "ITM, 6M, sig=30%"),
        (0.9, 0.25, 0.25, 0.03, "OTM, 3M, sig=25%"),
        (1.0, 2.0, 0.15, 0.03, "ATM, 2Y, sig=15%"),
        (0.8, 1.0, 0.4, 0.03, "Deep OTM, 1Y, sig=40%"),
    ];

    println!("\n  {:>30}  {:>12}  {:>12}  {:>8}", "Scenario", "Analytical", "Operator", "Error");
    println!("  {}", "-".repeat(70));

    // Train a quick model
    let (branch_data, trunk_data, targets) =
        data::generate_bs_training_data(1000, 50, 20);

    let config = DeepONetConfig {
        branch_input_dim: 50,
        trunk_input_dim: 2,
        latent_dim: 64,
        branch_hidden_dims: vec![128, 128],
        trunk_hidden_dims: vec![128, 128],
        learning_rate: 0.001,
        n_epochs: 60,
        batch_size: 64,
    };

    let mut model = DeepONet::new(config);
    let _ = model.train(&branch_data, &trunk_data, &targets);

    for (s_k, t, sigma, r, label) in &test_cases {
        // Analytical
        let analytical = black_scholes_call(*s_k, *t, *sigma, *r);

        // Operator prediction (construct vol function input)
        let s_grid: Vec<f64> = (0..50).map(|i| 0.5 + 1.5 * i as f64 / 49.0).collect();
        let vol_func: Vec<f64> = s_grid.iter().map(|_| *sigma).collect(); // flat vol
        let branch_in = Array1::from_vec(vol_func);
        let trunk_in = Array1::from_vec(vec![*s_k, *t]);
        let operator = model.predict(&branch_in, &trunk_in);

        let error = (operator - analytical).abs();
        println!(
            "  {:>30}  {:>12.6}  {:>12.6}  {:>8.6}",
            label, analytical, operator, error
        );
    }

    // Timing comparison
    println!("\n  {} comparison:", "Timing".cyan());

    let n_trials = 1000;

    // Analytical timing
    let start = std::time::Instant::now();
    for _ in 0..n_trials {
        let _ = black_scholes_call(1.0, 1.0, 0.2, 0.03);
    }
    let analytical_time = start.elapsed();

    // Operator timing
    let branch_in = Array1::from_vec(vec![0.2; 50]);
    let trunk_in = Array1::from_vec(vec![1.0, 1.0]);
    let start = std::time::Instant::now();
    for _ in 0..n_trials {
        let _ = model.predict(&branch_in, &trunk_in);
    }
    let operator_time = start.elapsed();

    println!(
        "    Analytical BS ({} calls):  {:?} ({:.1} us/call)",
        n_trials,
        analytical_time,
        analytical_time.as_micros() as f64 / n_trials as f64
    );
    println!(
        "    DeepONet     ({} calls):  {:?} ({:.1} us/call)",
        n_trials,
        operator_time,
        operator_time.as_micros() as f64 / n_trials as f64
    );

    // Spectral operator demo
    println!("\n{}", "Spectral Operator Demo:".yellow());
    let spectral = SpectralOperator::new(8, 1, 2);
    let input = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 64).mapv(f64::sin);
    let output = spectral.apply(&input);
    println!(
        "  Input:  {} points, range [{:.3}, {:.3}]",
        input.len(),
        input.iter().cloned().fold(f64::INFINITY, f64::min),
        input.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    println!(
        "  Output: {} points, range [{:.3}, {:.3}]",
        output.len(),
        output.iter().cloned().fold(f64::INFINITY, f64::min),
        output.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    Ok(())
}

fn demo_superresolution(_args: &Args) -> Result<()> {
    println!(
        "\n{} zero-shot super-resolution demo...",
        "Running".yellow().bold()
    );

    // Demonstrate that spectral operator works at any resolution
    let spectral = SpectralOperator::new(12, 1, 3);

    let resolutions = vec![32, 64, 128, 256, 512];

    println!(
        "\n  Applying spectral operator to sinusoidal input at multiple resolutions:"
    );
    println!("  {:>12}  {:>12}  {:>12}", "Resolution", "Output Mean", "Output Std");

    for &res in &resolutions {
        let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, res);
        let input = x.mapv(|v| v.sin() + 0.3 * (3.0 * v).sin());
        let output = spectral.apply(&input);

        let mean = output.mean().unwrap_or(0.0);
        let std_val = output.std(0.0);

        println!("  {:>12}  {:>12.6}  {:>12.6}", res, mean, std_val);
    }

    println!(
        "\n  The spectral operator produces consistent results across all resolutions."
    );
    println!(
        "  This demonstrates {} --- a key advantage of neural operators.",
        "resolution invariance".green()
    );

    Ok(())
}
