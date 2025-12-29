//! Backtest a trading strategy
//!
//! Usage: cargo run --bin backtest -- --data BTCUSDT_60.csv --model model.json

use anyhow::Result;
use rust_nn_crypto::{
    backtest::{Backtester, BacktestConfig},
    data::OHLCVSeries,
    nn::NeuralNetwork,
    strategy::StrategyConfig,
};
use std::env;

fn main() -> Result<()> {
    env_logger::init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let mut data_path = "BTCUSDT_60.csv".to_string();
    let mut model_path = "model.json".to_string();
    let mut initial_capital = 10000.0f64;
    let mut stop_loss_pct = 0.02f64;
    let mut take_profit_pct = 0.04f64;
    let mut allow_short = true;
    let mut walk_forward = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                data_path = args.get(i + 1).cloned().unwrap_or(data_path);
                i += 2;
            }
            "--model" | "-m" => {
                model_path = args.get(i + 1).cloned().unwrap_or(model_path);
                i += 2;
            }
            "--capital" | "-c" => {
                initial_capital = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(initial_capital);
                i += 2;
            }
            "--stop-loss" | "-sl" => {
                stop_loss_pct = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(stop_loss_pct);
                i += 2;
            }
            "--take-profit" | "-tp" => {
                take_profit_pct = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(take_profit_pct);
                i += 2;
            }
            "--no-short" => {
                allow_short = false;
                i += 1;
            }
            "--walk-forward" | "-wf" => {
                walk_forward = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("            Cryptocurrency Trading Backtest");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Load data
    println!("Loading data from {}...", data_path);
    let series = OHLCVSeries::load_csv(&data_path, "UNKNOWN".to_string(), "60".to_string())?;
    println!("Loaded {} candles", series.len());

    // Load model
    println!("Loading model from {}...", model_path);
    let mut model = NeuralNetwork::load(&model_path)?;
    model.summary();

    // Configure strategy
    let mut strategy_config = StrategyConfig::default();
    strategy_config.stop_loss_pct = Some(stop_loss_pct);
    strategy_config.take_profit_pct = Some(take_profit_pct);
    strategy_config.allow_short = allow_short;

    println!("\nStrategy Configuration:");
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Stop Loss: {:.2}%", stop_loss_pct * 100.0);
    println!("  Take Profit: {:.2}%", take_profit_pct * 100.0);
    println!("  Allow Short: {}", allow_short);
    println!("  Walk Forward: {}", walk_forward);

    // Configure backtest
    let mut backtest_config = BacktestConfig::default();
    backtest_config.initial_capital = initial_capital;
    backtest_config.walk_forward = walk_forward;

    // Run backtest
    println!("\nRunning backtest...");
    println!("─────────────────────────────────────────────────────────────────");

    let mut backtester = Backtester::new(backtest_config);
    let result = backtester.run(&mut model, &series, strategy_config);

    // Print results
    result.metrics.print_report();

    // Signal distribution
    use rust_nn_crypto::strategy::Signal;
    let mut signal_counts = [0usize; 5];
    for signal in &result.signals {
        match signal {
            Signal::StrongBuy => signal_counts[0] += 1,
            Signal::Buy => signal_counts[1] += 1,
            Signal::Hold => signal_counts[2] += 1,
            Signal::Sell => signal_counts[3] += 1,
            Signal::StrongSell => signal_counts[4] += 1,
        }
    }

    println!();
    println!("Signal Distribution:");
    println!("  Strong Buy:  {} ({:.1}%)", signal_counts[0], signal_counts[0] as f64 / result.signals.len() as f64 * 100.0);
    println!("  Buy:         {} ({:.1}%)", signal_counts[1], signal_counts[1] as f64 / result.signals.len() as f64 * 100.0);
    println!("  Hold:        {} ({:.1}%)", signal_counts[2], signal_counts[2] as f64 / result.signals.len() as f64 * 100.0);
    println!("  Sell:        {} ({:.1}%)", signal_counts[3], signal_counts[3] as f64 / result.signals.len() as f64 * 100.0);
    println!("  Strong Sell: {} ({:.1}%)", signal_counts[4], signal_counts[4] as f64 / result.signals.len() as f64 * 100.0);

    // Prediction statistics
    let mean_pred: f64 = result.predictions.iter().sum::<f64>() / result.predictions.len() as f64;
    let max_pred = result.predictions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_pred = result.predictions.iter().cloned().fold(f64::INFINITY, f64::min);

    println!();
    println!("Prediction Statistics:");
    println!("  Mean: {:.6}", mean_pred);
    println!("  Min:  {:.6}", min_pred);
    println!("  Max:  {:.6}", max_pred);

    // Save equity curve
    let equity_path = format!("{}_equity.csv", data_path.replace(".csv", ""));
    save_equity_curve(&result.equity_curve, &equity_path)?;
    println!();
    println!("Equity curve saved to: {}", equity_path);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    Backtest Complete!");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn save_equity_curve(equity: &[f64], path: &str) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(&["index", "equity"])?;

    for (i, &eq) in equity.iter().enumerate() {
        writer.write_record(&[i.to_string(), eq.to_string()])?;
    }

    writer.flush()?;
    Ok(())
}

fn print_help() {
    println!("Backtest a trading strategy using a trained neural network");
    println!();
    println!("USAGE:");
    println!("    cargo run --bin backtest -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -d, --data <PATH>          Input CSV data file");
    println!("    -m, --model <PATH>         Trained model file (default: model.json)");
    println!("    -c, --capital <AMOUNT>     Initial capital (default: 10000)");
    println!("    -sl, --stop-loss <PCT>     Stop loss percentage (default: 0.02)");
    println!("    -tp, --take-profit <PCT>   Take profit percentage (default: 0.04)");
    println!("        --no-short             Disable short selling");
    println!("    -wf, --walk-forward        Use walk-forward optimization");
    println!("    -h, --help                 Print help information");
    println!();
    println!("EXAMPLES:");
    println!("    cargo run --bin backtest -- --data BTCUSDT_60.csv --model model.json");
    println!("    cargo run --bin backtest -- -d data.csv -m btc_model.json -c 50000 --no-short");
}
