//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates fetching OHLCV data from the Bybit API
//! and processing it for use with the ELECTRA trading pipeline.

use electra_finance::api::bybit::{BybitClient, Interval};
use electra_finance::data::processor::DataProcessor;
use electra_finance::data::features::FeatureEngineering;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== ELECTRA Finance: Fetching Bybit Data ===\n");

    // Create Bybit API client
    let client = BybitClient::new();

    // Fetch daily klines for BTCUSDT
    let symbol = "BTCUSDT";
    println!("Fetching {} kline data...", symbol);

    let klines = client.get_klines(symbol, Interval::Day1, 100, None, None)?;
    println!("Fetched {} klines for {}", klines.len(), symbol);

    if let Some(first) = klines.first() {
        println!("First kline: {} Open={:.2} Close={:.2}",
            first.datetime().format("%Y-%m-%d"),
            first.open,
            first.close
        );
    }
    if let Some(last) = klines.last() {
        println!("Last kline:  {} Open={:.2} Close={:.2}",
            last.datetime().format("%Y-%m-%d"),
            last.open,
            last.close
        );
    }

    // Process data
    let processor = DataProcessor::new(20);
    let processed = processor.process_klines(&klines);
    println!("\nProcessed {} data points", processed.len());

    // Compute features
    let features = FeatureEngineering::new();
    let feature_matrix = features.compute_features(&klines);
    println!("Feature matrix shape: {} x {}", feature_matrix.nrows(), feature_matrix.ncols());

    // Normalize features
    let (normalized, means, stds) = features.normalize(&feature_matrix);
    println!("\nFeature statistics:");
    let feature_names = ["Returns", "LogReturns", "Volatility", "RSI", "MACD", "ATR", "VolumeRatio"];
    for (i, name) in feature_names.iter().enumerate() {
        println!("  {}: mean={:.6}, std={:.6}", name, means[i], stds[i]);
    }

    // Show recent data
    println!("\nRecent processed data:");
    for data in processed.iter().rev().take(5) {
        let direction_str = match data.direction {
            1 => "UP  ",
            -1 => "DOWN",
            _ => "FLAT",
        };
        println!("  {} Return={:+.4}% Vol={:.4} Dir={}",
            chrono::Utc.timestamp_millis_opt(data.timestamp).unwrap().format("%Y-%m-%d"),
            data.returns * 100.0,
            data.volatility,
            direction_str
        );
    }

    // Also fetch ETHUSDT for multi-asset analysis
    let eth_symbol = "ETHUSDT";
    println!("\nFetching {} kline data...", eth_symbol);
    let eth_klines = client.get_klines(eth_symbol, Interval::Day1, 100, None, None)?;
    println!("Fetched {} klines for {}", eth_klines.len(), eth_symbol);

    // Export to CSV
    let csv_path = "btcusdt_data.csv";
    let mut writer = csv::Writer::from_path(csv_path)?;
    writer.write_record(["timestamp", "open", "high", "low", "close", "volume", "returns", "direction"])?;

    for data in &processed {
        writer.write_record(&[
            data.timestamp.to_string(),
            format!("{:.2}", data.open),
            format!("{:.2}", data.high),
            format!("{:.2}", data.low),
            format!("{:.2}", data.close),
            format!("{:.2}", data.volume),
            format!("{:.6}", data.returns),
            data.direction.to_string(),
        ])?;
    }
    writer.flush()?;
    println!("\nData exported to {}", csv_path);

    println!("\n=== Data fetching complete ===");

    Ok(())
}

use chrono::TimeZone;
