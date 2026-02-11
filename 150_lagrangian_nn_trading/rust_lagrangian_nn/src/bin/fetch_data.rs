//! Fetch market data from Bybit and construct configuration space for LNN.
//!
//! Usage:
//!   cargo run --bin fetch_data -- --symbol BTCUSDT --interval 5 --limit 5000

use clap::Parser;
use lagrangian_nn_trading::data::{BybitClient, construct_config_space, normalize_config_space};
use lagrangian_nn_trading::utils;

#[derive(Parser, Debug)]
#[command(name = "fetch_data", about = "Fetch Bybit data for Lagrangian NN trading")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval in minutes (1, 3, 5, 15, 30, 60, 240, D, W)
    #[arg(short, long, default_value = "5")]
    interval: String,

    /// Number of candles to fetch
    #[arg(short, long, default_value_t = 5000)]
    limit: usize,

    /// Moving average window for configuration space
    #[arg(short, long, default_value_t = 20)]
    ma_window: usize,

    /// Output directory
    #[arg(short, long, default_value = "data")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== Lagrangian NN Trading: Data Fetcher ===");
    println!("Symbol:    {}", args.symbol);
    println!("Interval:  {} min", args.interval);
    println!("Limit:     {} candles", args.limit);
    println!("MA Window: {}", args.ma_window);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Fetch data
    let client = BybitClient::new();
    println!("Fetching data from Bybit...");
    let candles = client.fetch_extended(&args.symbol, &args.interval, args.limit).await?;
    println!("Fetched {} candles", candles.len());

    if candles.len() < 100 {
        anyhow::bail!("Not enough data: need at least 100 candles, got {}", candles.len());
    }

    let first_ts = candles.first().unwrap().timestamp;
    let last_ts = candles.last().unwrap().timestamp;
    let price_min = candles.iter().map(|c| c.close).fold(f64::INFINITY, f64::min);
    let price_max = candles.iter().map(|c| c.close).fold(f64::NEG_INFINITY, f64::max);

    println!("Time range: {} to {}", first_ts, last_ts);
    println!("Price range: {:.2} to {:.2}", price_min, price_max);

    // Save raw data
    let raw_path = format!("{}/{}_raw.csv", args.output, args.symbol);
    let raw_data: Vec<Vec<f64>> = candles.iter()
        .map(|c| vec![c.timestamp as f64, c.open, c.high, c.low, c.close, c.volume])
        .collect();
    utils::export_csv(
        &raw_path,
        &["timestamp", "open", "high", "low", "close", "volume"],
        &raw_data,
    )?;
    println!("Saved raw data to {}", raw_path);

    // Construct configuration space
    println!("\nConstructing configuration space (q, q-dot, q-ddot)...");
    let config_data = construct_config_space(&candles, args.ma_window);
    println!("Config space samples: {}", config_data.q.len());

    if config_data.q.is_empty() {
        anyhow::bail!("Configuration space is empty after filtering");
    }

    let q_min = config_data.q.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min);
    let q_max = config_data.q.iter().map(|v| v[0]).fold(f64::NEG_INFINITY, f64::max);
    let qdot_min = config_data.qdot.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min);
    let qdot_max = config_data.qdot.iter().map(|v| v[0]).fold(f64::NEG_INFINITY, f64::max);
    let qddot_min = config_data.qddot.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min);
    let qddot_max = config_data.qddot.iter().map(|v| v[0]).fold(f64::NEG_INFINITY, f64::max);

    println!("q range:     [{:.6}, {:.6}]", q_min, q_max);
    println!("q-dot range: [{:.6}, {:.6}]", qdot_min, qdot_max);
    println!("q-ddot range:[{:.6}, {:.6}]", qddot_min, qddot_max);

    // Normalize
    let (normalized, stats) = normalize_config_space(&config_data);

    // Save configuration space
    let config_path = format!("{}/{}_config_space.csv", args.output, args.symbol);
    let config_rows: Vec<Vec<f64>> = (0..normalized.q.len())
        .map(|i| {
            vec![
                normalized.q[i][0],
                normalized.qdot[i][0],
                normalized.qddot[i][0],
                normalized.prices[i],
            ]
        })
        .collect();
    utils::export_csv(
        &config_path,
        &["q", "qdot", "qddot", "price"],
        &config_rows,
    )?;
    println!("Saved config space to {}", config_path);

    // Save normalization stats
    let stats_json = serde_json::to_string_pretty(&stats)?;
    let stats_path = format!("{}/{}_norm_stats.json", args.output, args.symbol);
    std::fs::write(&stats_path, stats_json)?;
    println!("Saved normalization stats to {}", stats_path);

    // Train/test split info
    let n = normalized.q.len();
    let split = (n as f64 * 0.8) as usize;
    println!("\nTrain samples: {}", split);
    println!("Test samples:  {}", n - split);

    println!("\nData fetching complete!");
    Ok(())
}
