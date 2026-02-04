//! Example: Fetch market data from Bybit

use conformal_trading::api::client::BybitClient;
use conformal_trading::DEFAULT_SYMBOLS;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("Conformal Prediction Trading - Data Fetcher");
    println!("{}", "=".repeat(50));

    let client = BybitClient::new();

    // Fetch ticker
    println!("\nFetching BTC/USDT ticker...");
    let ticker = client.get_ticker("BTCUSDT").await?;
    println!("  Symbol: {}", ticker.symbol);
    println!("  Last Price: ${}", ticker.last_price);
    println!("  24h Change: {}%", ticker.price_24h_pcnt);

    // Fetch recent klines
    println!("\nFetching recent 1-hour klines...");
    let klines = client.get_klines("BTCUSDT", "60", 24).await?;
    println!("  Fetched {} klines", klines.len());

    if let Some(latest) = klines.last() {
        println!("  Latest candle:");
        println!("    Time: {}", latest.datetime());
        println!("    Open: ${:.2}", latest.open);
        println!("    High: ${:.2}", latest.high);
        println!("    Low: ${:.2}", latest.low);
        println!("    Close: ${:.2}", latest.close);
        println!("    Volume: {:.2}", latest.volume);
    }

    // Fetch historical data
    println!("\nFetching 7 days of historical data...");
    let historical = client.get_historical_klines("BTCUSDT", "60", 7).await?;
    println!("  Fetched {} candles", historical.len());

    if !historical.is_empty() {
        println!("  Date range: {} to {}",
            historical.first().unwrap().datetime(),
            historical.last().unwrap().datetime()
        );

        // Calculate some basic stats
        let closes: Vec<f64> = historical.iter().map(|k| k.close).collect();
        let returns: Vec<f64> = closes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt();

        println!("\n  Statistics:");
        println!("    Mean hourly return: {:.4}%", mean_return * 100.0);
        println!("    Hourly volatility: {:.4}%", volatility * 100.0);
        println!("    Annualized volatility: {:.2}%", volatility * (252.0 * 24.0_f64).sqrt() * 100.0);
    }

    // Fetch multiple symbols
    println!("\nFetching tickers for default symbols...");
    for symbol in DEFAULT_SYMBOLS.iter().take(5) {
        match client.get_ticker(symbol).await {
            Ok(t) => println!("  {}: ${}", t.symbol, t.last_price),
            Err(e) => println!("  {}: Error - {}", symbol, e),
        }
    }

    println!("\nData fetch complete!");
    Ok(())
}
