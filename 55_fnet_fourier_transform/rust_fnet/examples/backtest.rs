//! Example: Run backtest with FNet predictions on historical data.
//!
//! Usage:
//!   cargo run --example backtest -- --symbol BTCUSDT --interval 60

use anyhow::Result;
use clap::Parser;

use fnet_trading::{
    calculate_features, Backtester, BacktesterConfig, BybitClient, FeatureConfig, FNet,
    SignalGeneratorConfig, TradingDataset,
};

#[derive(Parser, Debug)]
#[command(name = "backtest")]
#[command(about = "Run backtest with FNet trading strategy")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval
    #[arg(short, long, default_value = "60")]
    interval: String,

    /// Number of candles
    #[arg(short, long, default_value = "2000")]
    limit: usize,

    /// Sequence length for model
    #[arg(long, default_value = "168")]
    seq_len: usize,

    /// Prediction horizon
    #[arg(long, default_value = "24")]
    horizon: usize,

    /// Initial capital
    #[arg(long, default_value = "100000")]
    capital: f64,

    /// Use synthetic data (for testing without API)
    #[arg(long)]
    synthetic: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("FNet Trading - Backtester");
    println!("=========================");
    println!();

    // Get data
    let (features, prices, timestamps) = if args.synthetic {
        println!("Using synthetic data for demonstration...");
        generate_synthetic_data(2000)
    } else {
        println!("Fetching data from Bybit...");
        fetch_real_data(&args.symbol, &args.interval, args.limit)?
    };

    println!("Data points: {}", features.len());
    println!("Sequence length: {}", args.seq_len);
    println!("Prediction horizon: {}", args.horizon);
    println!();

    // Create dataset
    println!("Creating training sequences...");
    let feature_matrix = features.to_matrix();
    let dataset = TradingDataset::from_features(
        &feature_matrix,
        &prices,
        &timestamps,
        args.seq_len,
        args.horizon,
        0, // log_return column
    );

    println!("Total sequences: {}", dataset.len());

    // Split data
    let (train, _val, test) = dataset.split(0.7, 0.15);
    println!("Train: {}, Test: {}", train.len(), test.len());
    println!();

    // Create and "train" model (in real use, you would actually train it)
    println!("Creating FNet model...");
    let model = FNet::new(
        8,    // n_features
        64,   // d_model (smaller for demo)
        2,    // n_layers
        128,  // d_ff
        args.seq_len,
    );
    println!("Model parameters: {}", model.num_parameters());
    println!();

    // Generate predictions on test set
    println!("Generating predictions...");
    let predictions = generate_predictions(&model, &test);
    println!("Generated {} predictions", predictions.len());

    // Run backtest
    println!("\nRunning backtest...");
    let backtester = Backtester::new(BacktesterConfig {
        initial_capital: args.capital,
        transaction_cost: 0.001,
        slippage: 0.0005,
    });

    let signal_config = SignalGeneratorConfig {
        threshold: 0.001,
        confidence_threshold: 0.4,
        position_size: 1.0,
        stop_loss: 0.02,
        take_profit: 0.04,
        max_holding_period: args.horizon,
    };

    let result = backtester.run(&predictions, &test.prices, &test.timestamps, signal_config);

    // Print results
    println!("\n{}", "=".repeat(50));
    println!("BACKTEST RESULTS");
    println!("{}", "=".repeat(50));
    println!();
    println!("{}", result.metrics.summary());

    // Calculate buy-and-hold for comparison
    if !test.prices.is_empty() {
        let buy_hold_return = (test.prices.last().unwrap() / test.prices.first().unwrap()) - 1.0;
        println!();
        println!("Comparison:");
        println!("  Strategy Return: {:>8.2}%", result.metrics.total_return * 100.0);
        println!("  Buy & Hold:      {:>8.2}%", buy_hold_return * 100.0);
        println!(
            "  Outperformance:  {:>8.2}%",
            (result.metrics.total_return - buy_hold_return) * 100.0
        );
    }

    // Show trade distribution
    println!("\nTrade Analysis:");
    println!("  Total trades: {}", result.trades.len());

    let take_profits = result
        .trades
        .iter()
        .filter(|t| matches!(t.exit_reason, fnet_trading::strategy::backtest::ExitReason::TakeProfit))
        .count();
    let stop_losses = result
        .trades
        .iter()
        .filter(|t| matches!(t.exit_reason, fnet_trading::strategy::backtest::ExitReason::StopLoss))
        .count();
    let max_holding = result
        .trades
        .iter()
        .filter(|t| matches!(t.exit_reason, fnet_trading::strategy::backtest::ExitReason::MaxHoldingPeriod))
        .count();
    let reversals = result
        .trades
        .iter()
        .filter(|t| matches!(t.exit_reason, fnet_trading::strategy::backtest::ExitReason::SignalReverse))
        .count();

    println!("  Take Profit: {}", take_profits);
    println!("  Stop Loss: {}", stop_losses);
    println!("  Max Holding: {}", max_holding);
    println!("  Reversals: {}", reversals);

    println!("\nBacktest complete!");
    Ok(())
}

fn fetch_real_data(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<(fnet_trading::TradingFeatures, Vec<f64>, Vec<u64>)> {
    let client = BybitClient::new();
    let klines = client.fetch_klines(symbol, interval, limit)?;

    if klines.is_empty() {
        anyhow::bail!("No data fetched");
    }

    let config = FeatureConfig::default();
    let features = calculate_features(&klines, &config);

    Ok((
        features.clone(),
        features.close_prices.clone(),
        features.timestamps.clone(),
    ))
}

fn generate_synthetic_data(n: usize) -> (fnet_trading::TradingFeatures, Vec<f64>, Vec<u64>) {
    use std::f64::consts::PI;

    // Generate synthetic price series with trend and cycles
    let mut prices = Vec::with_capacity(n);
    let mut price = 50000.0;

    for i in 0..n {
        let trend = 0.0001 * (i as f64 / n as f64); // Slight uptrend
        let cycle1 = 0.002 * (2.0 * PI * i as f64 / 24.0).sin(); // Daily cycle
        let cycle2 = 0.001 * (2.0 * PI * i as f64 / 168.0).sin(); // Weekly cycle
        let noise = 0.0005 * (i as f64 * 0.1).sin(); // Random-ish noise

        price *= 1.0 + trend + cycle1 + cycle2 + noise;
        prices.push(price);
    }

    // Create synthetic klines
    let klines: Vec<fnet_trading::Kline> = prices
        .iter()
        .enumerate()
        .map(|(i, &p)| fnet_trading::Kline {
            timestamp: 1700000000 + i as u64 * 3600,
            open: p * 0.999,
            high: p * 1.01,
            low: p * 0.99,
            close: p,
            volume: 1000.0 * (1.0 + 0.5 * (i as f64 * 0.2).cos()),
            turnover: p * 1000.0,
        })
        .collect();

    let config = FeatureConfig::default();
    let features = calculate_features(&klines, &config);

    (
        features.clone(),
        features.close_prices.clone(),
        features.timestamps.clone(),
    )
}

fn generate_predictions(model: &FNet, dataset: &TradingDataset) -> Vec<f64> {
    let mut predictions = Vec::with_capacity(dataset.len());

    for i in 0..dataset.len() {
        let input = dataset.features.slice(ndarray::s![i..i + 1, .., ..]).to_owned();
        let output = model.forward(&input);
        predictions.push(output[[0, 0]]);
    }

    predictions
}
