//! Cryptocurrency Trading Example
//!
//! Demonstrates how to use the Bybit data loader and trading signal generator
//! for cryptocurrency analysis.

use llm_finance::{BybitDataLoader, TradingSignalGenerator, SignalType, Sentiment, TradingSignal};

fn main() {
    println!("=== Cryptocurrency Trading Analysis ===\n");

    // Create Bybit data loader (using testnet)
    println!("--- Bybit Data Loader ---");
    let loader = BybitDataLoader::new(true); // Use testnet
    println!("Created Bybit data loader (testnet mode)");
    println!("Available symbols: BTCUSDT, ETHUSDT, SOLUSDT, etc.");

    // Get kline data
    println!("\n--- Fetching Kline Data ---");
    match loader.get_klines("BTCUSDT", "60", 100) {
        Ok(series) => {
            println!("Fetched {} candles for {}", series.data.len(), series.symbol);
            if let (Some(first), Some(last)) = (series.data.first(), series.data.last()) {
                println!("  - Price range: ${:.2} - ${:.2}", first.close, last.close);
                println!("  - Time range: {} to {}", first.timestamp, last.timestamp);
            }

            // Calculate indicators
            let sma = series.sma(20);
            let rsi = series.rsi(14);
            let volatility = series.volatility(20);

            if let Some(last_sma) = sma.last().and_then(|v| *v) {
                println!("  - Current SMA(20): ${:.2}", last_sma);
            }
            if let Some(last_rsi) = rsi.last().and_then(|v| *v) {
                println!("  - Current RSI(14): {:.2}", last_rsi);
            }
            if let Some(last_vol) = volatility.last().and_then(|v| *v) {
                println!("  - Current volatility: {:.4}", last_vol);
            }
        }
        Err(e) => println!("Error fetching klines: {}", e),
    }

    // Get ticker information
    println!("\n--- Ticker Information ---");
    match loader.get_ticker("BTCUSDT") {
        Ok(ticker) => {
            println!("Symbol: {}", ticker.symbol);
            println!("  - Last price: ${:.2}", ticker.last_price);
            println!("  - 24h change: ${:.2}", ticker.price_change_24h);
            println!("  - 24h high/low: ${:.2} / ${:.2}", ticker.high_24h, ticker.low_24h);
            println!("  - 24h volume: ${:.2}M", ticker.volume_24h / 1_000_000.0);
        }
        Err(e) => println!("Error fetching ticker: {}", e),
    }

    // Get orderbook
    println!("\n--- Orderbook ---");
    match loader.get_orderbook("BTCUSDT", 5) {
        Ok(orderbook) => {
            println!("Symbol: {}", orderbook.symbol);
            if let (Some(bid), Some(ask)) = (orderbook.best_bid(), orderbook.best_ask()) {
                println!("  - Best bid: ${:.2}", bid);
                println!("  - Best ask: ${:.2}", ask);
            }
            if let Some(spread) = orderbook.spread() {
                println!("  - Spread: ${:.2}", spread);
            }
            if let Some(mid) = orderbook.mid_price() {
                println!("  - Mid price: ${:.2}", mid);
            }
        }
        Err(e) => println!("Error fetching orderbook: {}", e),
    }

    // Get funding rate
    println!("\n--- Funding Rate ---");
    match loader.get_funding_rate("BTCUSDT") {
        Ok(rate) => {
            println!("Current funding rate: {:.4}%", rate * 100.0);
            if rate > 0.0 {
                println!("  (Longs pay shorts)");
            } else if rate < 0.0 {
                println!("  (Shorts pay longs)");
            }
        }
        Err(e) => println!("Error fetching funding rate: {}", e),
    }

    // Create trading signal generator
    println!("\n--- Trading Signal Generator ---");
    let generator = TradingSignalGenerator::new();
    println!("Signal generator configuration (default):");
    println!("  - Bullish threshold: 0.6");
    println!("  - Bearish threshold: 0.6");
    println!("  - Confidence threshold: 0.5");

    // Generate signals based on sentiment
    println!("\n--- Generating Trading Signals ---");
    let sentiments = [
        (Sentiment::Bullish, 0.8),
        (Sentiment::Bearish, 0.7),
        (Sentiment::Neutral, 0.9),
        (Sentiment::Bullish, 0.4), // Low confidence
    ];

    for (sentiment, confidence) in sentiments.iter() {
        let signal = generator.generate_signal(*sentiment, *confidence);
        let signal_str = match signal.signal {
            SignalType::Buy => "BUY ",
            SignalType::Sell => "SELL",
            SignalType::Hold => "HOLD",
        };
        println!(
            "  {:?} with confidence {:.1} -> {} (final confidence: {:.2})",
            sentiment, confidence, signal_str, signal.confidence
        );
    }

    // Demonstrate signal aggregation
    println!("\n--- Signal Aggregation ---");
    let signals = vec![
        TradingSignal::new(SignalType::Buy, 0.8, Sentiment::Bullish),
        TradingSignal::new(SignalType::Buy, 0.7, Sentiment::Bullish),
        TradingSignal::new(SignalType::Sell, 0.6, Sentiment::Bearish),
    ];

    let majority = generator.aggregate_signals(&signals, "majority");
    let weighted = generator.aggregate_signals(&signals, "weighted");
    let unanimous = generator.aggregate_signals(&signals, "unanimous");

    println!("Signals: 2x BUY, 1x SELL");
    println!("  - Majority vote: {:?} (confidence: {:.2})", majority.signal, majority.confidence);
    println!("  - Weighted: {:?} (confidence: {:.2})", weighted.signal, weighted.confidence);
    println!("  - Unanimous: {:?} (confidence: {:.2})", unanimous.signal, unanimous.confidence);

    println!("\nCryptocurrency trading example completed successfully!");
}
