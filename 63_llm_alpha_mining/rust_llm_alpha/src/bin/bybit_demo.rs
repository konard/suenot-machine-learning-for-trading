//! Bybit Data Loading Demo
//!
//! Demonstrates how to load cryptocurrency data from Bybit.

use llm_alpha_mining::data::{BybitLoader, generate_synthetic_data, calculate_features};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("====================================================");
    println!("LLM Alpha Mining - Bybit Data Loading Demo (Rust)");
    println!("====================================================");

    // 1. Synthetic data (no API calls)
    println!("\n1. SYNTHETIC DATA");
    println!("{}", "-".repeat(40));

    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    for symbol in &symbols {
        let data = generate_synthetic_data(symbol, 180, 42);
        let closes = data.close_prices();

        let start_price = closes.first().unwrap_or(&0.0);
        let end_price = closes.last().unwrap_or(&0.0);
        let ret = (end_price / start_price - 1.0) * 100.0;

        println!("\n{} ({}):", symbol, data.source);
        println!("  Records: {}", data.len());
        println!("  Price range: ${:.2} - ${:.2}",
                 closes.iter().cloned().fold(f64::INFINITY, f64::min),
                 closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        println!("  Total return: {:+.2}%", ret);
    }

    // 2. Calculate features
    println!("\n2. TECHNICAL FEATURES");
    println!("{}", "-".repeat(40));

    let btc_data = generate_synthetic_data("BTCUSDT", 180, 42);
    let features = calculate_features(&btc_data);

    println!("\nCalculated {} features for BTCUSDT:", features.len());
    for (name, values) in &features {
        let valid: Vec<_> = values.iter().filter(|v| !v.is_nan()).collect();
        if !valid.is_empty() {
            let mean: f64 = valid.iter().copied().sum::<f64>() / valid.len() as f64;
            println!("  {}: {} valid values, mean={:.4}", name, valid.len(), mean);
        }
    }

    // 3. Returns and volatility
    println!("\n3. RETURNS ANALYSIS");
    println!("{}", "-".repeat(40));

    let returns = btc_data.returns();
    let _log_returns = btc_data.log_returns();

    if !returns.is_empty() {
        let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std_ret = variance.sqrt();

        println!("\nDaily Returns:");
        println!("  Count: {}", returns.len());
        println!("  Mean: {:.4}%", mean_ret * 100.0);
        println!("  Std Dev: {:.4}%", std_ret * 100.0);
        println!("  Annualized Vol: {:.2}%", std_ret * 252.0_f64.sqrt() * 100.0);
    }

    // 4. Live data example (code shown, not executed to avoid API calls)
    println!("\n4. LIVE DATA EXAMPLE");
    println!("{}", "-".repeat(40));

    println!(r#"
To fetch live Bybit data:

    let loader = BybitLoader::new();
    let btc = loader.load("BTCUSDT", "60", 30).await?;
    println!("Loaded {{}} hourly candles", btc.len());

    // Get funding rates
    let rates = loader.load_funding_rate("BTCUSDT", 7).await?;
    for (timestamp, rate) in rates {{
        println!("  {{}}: {{:.4}}%", timestamp, rate * 100.0);
    }}
"#);

    // 5. Optional: Actually fetch live data
    println!("\n5. ATTEMPTING LIVE DATA FETCH");
    println!("{}", "-".repeat(40));

    // Uncomment to test live API
    let loader = BybitLoader::new();
    match loader.load("BTCUSDT", "60", 1).await {
        Ok(data) => {
            println!("Successfully loaded {} candles from Bybit API", data.len());
            if let Some(last) = data.candles.last() {
                println!("Latest candle:");
                println!("  Time: {}", last.timestamp);
                println!("  Close: ${:.2}", last.close);
                println!("  Volume: {:.2}", last.volume);
            }
        }
        Err(e) => {
            println!("Note: Live fetch skipped or failed: {}", e);
            println!("(This is expected if running without network access)");
        }
    }

    println!("\n====================================================");
    println!("Demo complete!");

    Ok(())
}
