//! Generate live trading signals using TCN
//!
//! Usage:
//!     cargo run --bin live_signals -- --symbol BTCUSDT

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use rust_tcn_trading::api::{BybitClient, TimeFrame};
use rust_tcn_trading::features::{Normalizer, TechnicalIndicators};
use rust_tcn_trading::tcn::{TCN, TCNConfig};
use rust_tcn_trading::trading::{RiskConfig, RiskManager, SignalGenerator, PortfolioState};
use std::time::Duration;

/// Generate live trading signals
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Trading pair symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Time interval
    #[arg(short, long, default_value = "15m")]
    interval: String,

    /// Confidence threshold for trading
    #[arg(long, default_value_t = 0.65)]
    threshold: f64,

    /// Refresh interval in seconds
    #[arg(long, default_value_t = 60)]
    refresh: u64,

    /// Run continuously
    #[arg(long, default_value_t = false)]
    continuous: bool,

    /// Portfolio capital for position sizing
    #[arg(long, default_value_t = 100000.0)]
    capital: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("=== TCN Live Signal Generator ===\n");
    println!("Symbol:    {}", args.symbol);
    println!("Interval:  {}", args.interval);
    println!("Threshold: {:.0}%", args.threshold * 100.0);
    println!("Capital:   ${:.2}\n", args.capital);

    // Parse timeframe
    let timeframe = TimeFrame::from_str(&args.interval)
        .ok_or_else(|| anyhow::anyhow!("Invalid interval: {}", args.interval))?;

    // Create client
    let client = BybitClient::new();

    // Create TCN model
    let config = TCNConfig {
        input_size: 20, // Will be updated based on features
        output_size: 3,
        num_channels: vec![32, 32, 32],
        kernel_size: 3,
        dropout: 0.1,
    };

    // Create risk manager
    let risk_config = RiskConfig::default();
    let risk_manager = RiskManager::new(risk_config.clone());
    let mut portfolio_state = PortfolioState::new(args.capital);

    loop {
        println!("\n{}", "=".repeat(60));
        println!("Timestamp: {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        println!("{}", "=".repeat(60));

        // Fetch latest data
        println!("\nFetching market data...");
        let data = client
            .get_klines(&args.symbol, timeframe, Some(200), None, None)
            .await?;

        if data.len() < 100 {
            println!("Warning: Only {} candles available", data.len());
            if !args.continuous {
                break;
            }
            tokio::time::sleep(Duration::from_secs(args.refresh)).await;
            continue;
        }

        let latest_candle = data.candles.last().unwrap();
        println!("Latest price: ${:.2}", latest_candle.close);
        println!("24h range:    ${:.2} - ${:.2}",
                 latest_candle.low, latest_candle.high);

        // Calculate features
        let features = TechnicalIndicators::calculate_all(&data.candles);

        // Normalize
        let mut normalizer = Normalizer::zscore();
        let normalized = normalizer.fit_transform(&features.data);

        // Get last N bars for prediction
        let seq_len = 50;
        let window = normalized.slice(ndarray::s![.., (normalized.ncols() - seq_len)..]).to_owned();

        // Create model with correct input size
        let config = TCNConfig {
            input_size: features.num_features,
            output_size: 3,
            num_channels: vec![32, 32, 32],
            kernel_size: 3,
            dropout: 0.1,
        };
        let tcn = TCN::new(config);

        // Create signal generator
        let signal_gen = SignalGenerator::new(tcn, args.threshold, args.threshold);

        // Generate signal
        let feature_matrix = rust_tcn_trading::features::FeatureMatrix {
            feature_names: features.feature_names.clone(),
            data: window,
            num_features: features.num_features,
            seq_len,
        };

        let signal = signal_gen.generate_signal(&feature_matrix);

        // Print prediction
        println!("\n--- TCN Prediction ---");
        if let Some(probs) = &signal.probabilities {
            println!("  Down probability:    {:.1}%", probs[0] * 100.0);
            println!("  Neutral probability: {:.1}%", probs[1] * 100.0);
            println!("  Up probability:      {:.1}%", probs[2] * 100.0);
        }

        // Print signal
        println!("\n--- Trading Signal ---");
        let signal_str = match signal.signal_type {
            rust_tcn_trading::trading::SignalType::Long => "LONG (BUY)",
            rust_tcn_trading::trading::SignalType::Short => "SHORT (SELL)",
            rust_tcn_trading::trading::SignalType::Neutral => "NEUTRAL (HOLD)",
        };

        println!("  Signal:     {}", signal_str);
        println!("  Confidence: {:.1}%", signal.confidence * 100.0);

        // Validate signal with risk manager
        let validated = risk_manager.validate_signal(&signal, &portfolio_state, Some(&args.symbol));

        println!("\n--- Risk Management ---");
        match &validated {
            rust_tcn_trading::trading::ValidatedSignal::Approved(s) => {
                let position_value = s.position_size * args.capital;
                println!("  Status:         APPROVED");
                println!("  Position size:  {:.1}% (${:.2})", s.position_size * 100.0, position_value);

                if signal.signal_type != rust_tcn_trading::trading::SignalType::Neutral {
                    let is_long = signal.signal_type == rust_tcn_trading::trading::SignalType::Long;
                    let stop_loss = risk_manager.calculate_stop_loss(latest_candle.close, is_long);
                    let take_profit = risk_manager.calculate_take_profit(latest_candle.close, is_long);

                    println!("  Entry price:    ${:.2}", latest_candle.close);
                    println!("  Stop loss:      ${:.2} ({:.1}%)",
                             stop_loss, (stop_loss / latest_candle.close - 1.0).abs() * 100.0);
                    println!("  Take profit:    ${:.2} ({:.1}%)",
                             take_profit, (take_profit / latest_candle.close - 1.0).abs() * 100.0);
                }
            }
            rust_tcn_trading::trading::ValidatedSignal::Reduced(s, reason) => {
                println!("  Status:         REDUCED");
                println!("  Position size:  {:.1}% (${:.2})",
                         s.position_size * 100.0, s.position_size * args.capital);
                println!("  Reason:         {}", reason);
            }
            rust_tcn_trading::trading::ValidatedSignal::Blocked(reason) => {
                println!("  Status:         BLOCKED");
                println!("  Reason:         {}", reason);
            }
        }

        // Print some technical indicators
        println!("\n--- Technical Indicators ---");
        if let Some(rsi) = features.get_feature("rsi_14") {
            if let Some(&val) = rsi.last() {
                let rsi_signal = if val < 30.0 {
                    "Oversold"
                } else if val > 70.0 {
                    "Overbought"
                } else {
                    "Neutral"
                };
                println!("  RSI(14):      {:.1} ({})", val, rsi_signal);
            }
        }

        if let Some(macd) = features.get_feature("macd") {
            if let Some(&macd_val) = macd.last() {
                if let Some(signal_line) = features.get_feature("macd_signal") {
                    if let Some(&sig_val) = signal_line.last() {
                        let macd_signal = if macd_val > sig_val {
                            "Bullish"
                        } else {
                            "Bearish"
                        };
                        println!("  MACD:         {:.4} ({})", macd_val - sig_val, macd_signal);
                    }
                }
            }
        }

        if let Some(vol) = features.get_feature("volatility_20") {
            if let Some(&val) = vol.last() {
                println!("  Volatility:   {:.2}%", val * 100.0);
            }
        }

        // Fetch order book for additional context
        let orderbook = client.get_orderbook(&args.symbol, Some(5)).await?;
        println!("\n--- Order Book ---");
        if let (Some(bid), Some(ask)) = (orderbook.best_bid(), orderbook.best_ask()) {
            println!("  Best bid:     ${:.2}", bid);
            println!("  Best ask:     ${:.2}", ask);
            println!("  Spread:       {:.2} bps", orderbook.spread_bps().unwrap_or(0.0));
            println!("  Imbalance:    {:.1}%", orderbook.imbalance(5) * 100.0);
        }

        if !args.continuous {
            break;
        }

        println!("\nRefreshing in {} seconds...", args.refresh);
        tokio::time::sleep(Duration::from_secs(args.refresh)).await;
    }

    println!("\n=== Signal Generator Stopped ===");

    // Disclaimer
    println!("\n{}", "=".repeat(60));
    println!("DISCLAIMER: This is for educational purposes only.");
    println!("Do NOT use for actual trading without thorough testing");
    println!("and risk management. Past performance does not guarantee");
    println!("future results. Trading cryptocurrencies carries high risk.");
    println!("{}", "=".repeat(60));

    Ok(())
}
