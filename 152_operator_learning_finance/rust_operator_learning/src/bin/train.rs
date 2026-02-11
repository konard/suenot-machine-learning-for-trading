//! Train operator learning models for financial applications.
//!
//! Supports training DeepONet on:
//! - Synthetic Black-Scholes option pricing data
//! - Synthetic yield curve evolution data
//! - Real crypto price data from Bybit
//!
//! # Usage
//! ```bash
//! cargo run --bin train -- --data black_scholes --epochs 100
//! cargo run --bin train -- --data yield_curve --epochs 50
//! cargo run --bin train -- --data crypto --symbol BTCUSDT --epochs 80
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use ndarray::{Array1, Array2};
use operator_learning_trading::{
    data,
    operator::{DeepONet, DeepONetConfig, relative_l2_error},
};

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train operator learning models")]
struct Args {
    /// Data source: black_scholes, yield_curve, or crypto
    #[arg(short, long, default_value = "black_scholes")]
    data: String,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 100)]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f64,

    /// Batch size
    #[arg(short, long, default_value_t = 64)]
    batch_size: usize,

    /// Latent dimension (number of basis functions)
    #[arg(long, default_value_t = 64)]
    latent_dim: usize,

    /// Number of training samples
    #[arg(long, default_value_t = 2000)]
    samples: usize,

    /// Crypto symbol (for --data crypto)
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,

    /// Output model file
    #[arg(short, long, default_value = "model.json")]
    output: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("{}", "═══════════════════════════════════════════════════════".blue());
    println!("{}", "  Operator Learning Trainer - Chapter 152".blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════".blue());

    match args.data.as_str() {
        "black_scholes" => train_black_scholes(&args)?,
        "yield_curve" => train_yield_curve(&args)?,
        "crypto" => train_crypto(&args)?,
        _ => {
            eprintln!("{}: Unknown data source '{}'. Use: black_scholes, yield_curve, crypto",
                "Error".red(), args.data);
            std::process::exit(1);
        }
    }

    println!("\n{}", "Training complete!".green().bold());
    Ok(())
}

fn train_black_scholes(args: &Args) -> Result<()> {
    println!(
        "\n{} Black-Scholes option pricing operator...",
        "Training".yellow().bold()
    );
    println!("  Samples: {}, Epochs: {}, LR: {}", args.samples, args.epochs, args.lr);

    // Generate data
    let n_vol_points = 50;
    let n_eval = 20;
    let (branch_data, trunk_data, targets) =
        data::generate_bs_training_data(args.samples, n_vol_points, n_eval);

    let total = branch_data.nrows();
    let train_end = (total as f64 * 0.8) as usize;

    println!(
        "  Total data points: {} (train: {}, test: {})",
        total, train_end, total - train_end
    );

    // Split
    let branch_train = branch_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let trunk_train = trunk_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let targets_train = targets.slice(ndarray::s![..train_end]).to_owned();

    let branch_test = branch_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let trunk_test = trunk_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let targets_test = targets.slice(ndarray::s![train_end..]).to_owned();

    // Create model
    let config = DeepONetConfig {
        branch_input_dim: n_vol_points,
        trunk_input_dim: 2,
        latent_dim: args.latent_dim,
        branch_hidden_dims: vec![128, 128, 128],
        trunk_hidden_dims: vec![128, 128, 128],
        learning_rate: args.lr,
        n_epochs: args.epochs,
        batch_size: args.batch_size,
    };

    let mut model = DeepONet::new(config);

    // Train
    println!("\n{}", "Training:".yellow());
    let losses = model.train(&branch_train, &trunk_train, &targets_train);

    // Evaluate
    let test_error = model.evaluate(&branch_test, &trunk_test, &targets_test);
    println!("\n  {} Relative L2 Error: {:.6}", "Test".cyan(), test_error);

    // Show sample predictions
    println!("\n  {} predictions:", "Sample".cyan());
    for i in 0..5.min(branch_test.nrows()) {
        let branch_in = branch_test.row(i).to_owned();
        let trunk_in = trunk_test.row(i).to_owned();
        let pred = model.predict(&branch_in, &trunk_in);
        let actual = targets_test[i];
        println!(
            "    S/K={:.2}, T={:.2}: Pred={:.4}, Actual={:.4}, Error={:.4}",
            trunk_in[0], trunk_in[1], pred, actual, (pred - actual).abs()
        );
    }

    // Print training summary
    println!("\n{}", "Training Summary:".yellow());
    println!("  Initial loss: {:.6}", losses.first().unwrap_or(&0.0));
    println!("  Final loss:   {:.6}", losses.last().unwrap_or(&0.0));
    println!("  Test Rel L2:  {:.6}", test_error);

    Ok(())
}

fn train_yield_curve(args: &Args) -> Result<()> {
    println!(
        "\n{} yield curve evolution operator...",
        "Training".yellow().bold()
    );

    let n_maturities = 30;
    let (curves_today, curves_future) =
        data::generate_yield_curve_data(args.samples, n_maturities);

    // Prepare DeepONet data: branch=today's curve, trunk=maturity, target=future yield
    let n_samples = curves_today.nrows();
    let total = n_samples * n_maturities;

    let mut branch_data = Array2::zeros((total, n_maturities));
    let mut trunk_data = Array2::zeros((total, 1));
    let mut targets = Array1::zeros(total);

    let maturities: Vec<f64> = (0..n_maturities)
        .map(|i| 0.25 + 29.75 * i as f64 / (n_maturities - 1) as f64)
        .collect();

    let mut idx = 0;
    for i in 0..n_samples {
        for j in 0..n_maturities {
            for k in 0..n_maturities {
                branch_data[[idx, k]] = curves_today[[i, k]];
            }
            trunk_data[[idx, 0]] = maturities[j] / 30.0; // normalize to [0, 1]
            targets[idx] = curves_future[[i, j]];
            idx += 1;
        }
    }

    let train_end = (total as f64 * 0.8) as usize;
    println!("  Total data points: {} (train: {})", total, train_end);

    let branch_train = branch_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let trunk_train = trunk_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let targets_train = targets.slice(ndarray::s![..train_end]).to_owned();

    let branch_test = branch_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let trunk_test = trunk_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let targets_test = targets.slice(ndarray::s![train_end..]).to_owned();

    let config = DeepONetConfig {
        branch_input_dim: n_maturities,
        trunk_input_dim: 1,
        latent_dim: args.latent_dim,
        branch_hidden_dims: vec![128, 128],
        trunk_hidden_dims: vec![64, 64],
        learning_rate: args.lr,
        n_epochs: args.epochs,
        batch_size: args.batch_size,
    };

    let mut model = DeepONet::new(config);
    println!("\n{}", "Training:".yellow());
    let losses = model.train(&branch_train, &trunk_train, &targets_train);

    let test_error = model.evaluate(&branch_test, &trunk_test, &targets_test);
    println!("\n  {} Relative L2 Error: {:.6}", "Test".cyan(), test_error);

    // Show a sample yield curve prediction
    println!("\n  {} yield curve prediction:", "Sample".cyan());
    let sample_idx = 0;
    println!("    {:>8}  {:>10}  {:>10}  {:>8}", "Maturity", "Actual", "Predicted", "Error");
    for j in (0..n_maturities).step_by(5) {
        let data_idx = train_end + sample_idx * n_maturities + j;
        if data_idx >= total { break; }
        let branch_in = branch_test.row(data_idx - train_end).to_owned();
        let trunk_in = trunk_test.row(data_idx - train_end).to_owned();
        let pred = model.predict(&branch_in, &trunk_in);
        let actual = targets_test[data_idx - train_end];
        println!(
            "    {:>6.1}Y  {:>10.4}%  {:>10.4}%  {:>8.4}%",
            maturities[j],
            actual * 100.0,
            pred * 100.0,
            (pred - actual).abs() * 100.0
        );
    }

    Ok(())
}

fn train_crypto(args: &Args) -> Result<()> {
    println!(
        "\n{} crypto price operator on {}...",
        "Training".yellow().bold(),
        args.symbol.cyan()
    );

    // Fetch data from Bybit
    println!("  Fetching data from Bybit...");
    let candles = data::fetch_bybit_klines(&args.symbol, "60", 1000)?;
    println!("  Received {} candles", candles.len());

    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let lookback = 60;
    let horizon = 5;

    let (branch_data, trunk_data, targets) = match data::prepare_price_operator_data(&prices, lookback, horizon) {
        Some(data) => data,
        None => {
            eprintln!("{}: Not enough data points", "Error".red());
            std::process::exit(1);
        }
    };

    let total = branch_data.nrows();
    let train_end = (total as f64 * 0.8) as usize;
    println!("  Operator data points: {} (train: {})", total, train_end);

    let branch_train = branch_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let trunk_train = trunk_data.slice(ndarray::s![..train_end, ..]).to_owned();
    let targets_train = targets.slice(ndarray::s![..train_end]).to_owned();

    let branch_test = branch_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let trunk_test = trunk_data.slice(ndarray::s![train_end.., ..]).to_owned();
    let targets_test = targets.slice(ndarray::s![train_end..]).to_owned();

    let config = DeepONetConfig {
        branch_input_dim: lookback,
        trunk_input_dim: 1,
        latent_dim: args.latent_dim,
        branch_hidden_dims: vec![128, 128],
        trunk_hidden_dims: vec![64, 64],
        learning_rate: args.lr,
        n_epochs: args.epochs,
        batch_size: args.batch_size,
    };

    let mut model = DeepONet::new(config);
    println!("\n{}", "Training:".yellow());
    let losses = model.train(&branch_train, &trunk_train, &targets_train);

    let test_error = model.evaluate(&branch_test, &trunk_test, &targets_test);
    println!("\n  {} Relative L2 Error: {:.6}", "Test".cyan(), test_error);

    // Generate trading signals on test set
    println!("\n  {} trading signals on test data...", "Generating".yellow());
    let test_start = candles.len() - (total - train_end);
    let test_prices = &prices[test_start..];

    let strategy_config = operator_learning_trading::strategy::StrategyConfig {
        lookback,
        prediction_horizon: horizon,
        ..Default::default()
    };

    let signals = operator_learning_trading::strategy::generate_signal_series(
        &model,
        test_prices,
        &strategy_config,
    );

    let buy_count = signals.iter().filter(|&&s| s == operator_learning_trading::strategy::Signal::Buy).count();
    let sell_count = signals.iter().filter(|&&s| s == operator_learning_trading::strategy::Signal::Sell).count();
    let hold_count = signals.iter().filter(|&&s| s == operator_learning_trading::strategy::Signal::Hold).count();
    println!("  Buy: {}, Sell: {}, Hold: {}", buy_count, sell_count, hold_count);

    // Quick backtest
    let bt_config = operator_learning_trading::backtest::BacktestConfig::default();
    let result = operator_learning_trading::backtest::run_backtest(test_prices, &signals, &bt_config);
    operator_learning_trading::backtest::print_results("DeepONet Crypto Operator", &result);

    Ok(())
}
