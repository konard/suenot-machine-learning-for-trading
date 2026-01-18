//! Example: Fetching cryptocurrency data from Bybit
//!
//! Demonstrates how to use the Bybit client to fetch OHLCV data.

use anyhow::Result;
use hot_crypto::data::BybitClient;
use hot_crypto::get_crypto_universe;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("HOT Crypto - Data Fetch Example");
    println!("================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch data for a few symbols
    let symbols = &get_crypto_universe()[..3]; // BTC, ETH, SOL

    for symbol in symbols {
        println!("Fetching {} data...", symbol);

        // Fetch 60 daily candles
        let series = client.get_klines(symbol, "D", None, None, Some(60)).await?;

        println!("  Fetched {} candles", series.len());

        if let Some(latest) = series.latest() {
            println!("  Latest candle:");
            println!("    Time:   {}", latest.timestamp);
            println!("    Open:   ${:.2}", latest.open);
            println!("    High:   ${:.2}", latest.high);
            println!("    Low:    ${:.2}", latest.low);
            println!("    Close:  ${:.2}", latest.close);
            println!("    Volume: {:.2}", latest.volume);
        }

        // Calculate some statistics
        let returns = series.returns();
        let volatility = series.rolling_volatility(20);

        // Get latest values
        let last_return = returns[returns.len() - 1];
        let last_vol = volatility[volatility.len() - 1];

        println!("  Statistics:");
        println!("    Latest return:    {:.2}%", last_return * 100.0);
        println!("    Annualized vol:   {:.2}%", last_vol * 100.0);
        println!();

        // Small delay to avoid rate limiting
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    // Fetch current tickers
    println!("\nCurrent Prices:");
    println!("{:-<50}", "");

    for symbol in symbols {
        match client.get_ticker(symbol).await {
            Ok((price, volume)) => {
                println!("{:<10} ${:>12.2}  Vol: {:>15.0}", symbol, price, volume);
            }
            Err(e) => {
                println!("{:<10} Error: {}", symbol, e);
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    println!("\nDone!");

    Ok(())
}
