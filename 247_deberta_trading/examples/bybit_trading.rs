//! Bybit crypto trading example.
//!
//! Fetches real-time kline data from Bybit and generates
//! trading signals based on sentiment analysis.

use deberta_trading::{BybitClient, SentimentModel};
use deberta_trading::data::bybit::extract_returns;
use deberta_trading::trading::signals::{generate_signal, SignalDirection};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== DeBERTa Crypto Trading with Bybit Data ===\n");

    // Fetch BTC price data
    let client = BybitClient::new();
    let klines = client.fetch_klines("BTCUSDT", "D", 30).await?;
    println!("Fetched {} daily candles for BTCUSDT", klines.len());

    if let Some(last) = klines.last() {
        println!(
            "Latest: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}",
            last.open, last.high, last.low, last.close
        );
    }

    // Calculate returns
    let returns = extract_returns(&klines);
    if !returns.is_empty() {
        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = {
            let var = returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            var.sqrt()
        };
        println!(
            "Average daily return: {:.4}%, Volatility: {:.4}%",
            avg_return * 100.0,
            volatility * 100.0
        );
    }

    // Generate sentiment signals
    let model = SentimentModel::new();
    let sample_headlines = vec![
        ("Bitcoin ETF sees record inflows", "positive event"),
        ("Regulatory concerns grow for crypto", "negative event"),
        ("Bitcoin hashrate reaches all-time high", "positive event"),
    ];

    println!("\n=== Sentiment Signals ===\n");

    if let Some(latest_kline) = klines.last() {
        for (headline, event_type) in &sample_headlines {
            let signal = generate_signal(&model, headline, latest_kline, 0.5);
            println!(
                "[{}] {} → {} (strength: {:.4})",
                event_type, headline, signal.direction, signal.strength
            );

            match signal.direction {
                SignalDirection::Buy => {
                    println!("  → Consider LONG BTCUSDT @ {:.2}", signal.price);
                }
                SignalDirection::Sell => {
                    println!("  → Consider SHORT BTCUSDT @ {:.2}", signal.price);
                }
                SignalDirection::Hold => {
                    println!("  → No action recommended");
                }
            }
        }
    }

    Ok(())
}
