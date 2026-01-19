//! # Fetch News Example
//!
//! Demonstrates fetching market data from Bybit and generating mock news.

use llm_sentiment::{BybitClient, Interval, NewsCollector};
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLM Sentiment Analysis - Data Fetching Demo ===\n");

    // Initialize Bybit client
    let client = BybitClient::new();

    // Fetch current ticker for BTC
    println!("Fetching BTCUSDT ticker...");
    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!("Symbol: {}", ticker.symbol);
            println!("Last Price: ${:.2}", ticker.last_price);
            println!("24h Change: {:.2}%", ticker.price_change_percent_24h);
            println!("24h Volume: {:.2}", ticker.volume_24h);
            println!("24h High: ${:.2}", ticker.high_24h);
            println!("24h Low: ${:.2}", ticker.low_24h);
        }
        Err(e) => {
            println!("Error fetching ticker: {}", e);
            println!("(This may happen if there's no network connection)");
        }
    }

    println!();

    // Fetch recent candlestick data
    println!("Fetching recent 1-hour candles...");
    let end_time = Utc::now();
    let start_time = end_time - Duration::hours(24);

    match client.get_klines("BTCUSDT", Interval::OneHour, start_time, end_time).await {
        Ok(candles) => {
            println!("Fetched {} candles", candles.len());
            if let Some(last) = candles.last() {
                println!("Latest candle:");
                println!("  Open: ${:.2}", last.open);
                println!("  High: ${:.2}", last.high);
                println!("  Low: ${:.2}", last.low);
                println!("  Close: ${:.2}", last.close);
                println!("  Volume: {:.2}", last.volume);
            }
        }
        Err(e) => {
            println!("Error fetching candles: {}", e);
        }
    }

    println!();

    // Generate mock news data
    println!("=== Mock News Data Demo ===\n");
    let news_collector = NewsCollector::new();

    println!("Supported symbols: {:?}\n", news_collector.supported_symbols());

    let btc_news = news_collector.get_mock_crypto_news("BTC");
    println!("Generated {} mock news articles for BTC:\n", btc_news.len());

    for (i, article) in btc_news.iter().enumerate() {
        println!("{}. [{}] {}", i + 1, article.source.name(), article.title);
        if let Some(engagement) = article.engagement {
            println!("   Engagement: {}", engagement);
        }
        println!();
    }

    println!("=== Data Fetching Complete ===");

    Ok(())
}
