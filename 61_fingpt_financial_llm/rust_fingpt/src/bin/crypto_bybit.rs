//! FinGPT Cryptocurrency Analysis with Bybit Demo
//!
//! This example demonstrates how to combine FinGPT sentiment analysis
//! with Bybit cryptocurrency market data.

use fingpt_trading::{SentimentAnalyzer, TradingSignalGenerator};
use fingpt_trading::signals::NewsItem;
use fingpt_trading::api::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("{}", "=".repeat(70));
    println!("FinGPT Cryptocurrency Analysis with Bybit Data");
    println!("{}", "=".repeat(70));

    // Initialize components (mock mode for demo)
    let analyzer = SentimentAnalyzer::new_mock();
    let generator = TradingSignalGenerator::new(analyzer);
    let bybit = BybitClient::new_mock();

    // Crypto pairs to analyze
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    // Simulated crypto news feed
    let crypto_news: Vec<(&str, Vec<NewsItem>)> = vec![
        (
            "BTCUSDT",
            vec![
                NewsItem::new(
                    "Bitcoin spot ETF sees record $500M inflows as institutional adoption accelerates.",
                    "bloomberg",
                ),
                NewsItem::new(
                    "Major mining company reports 40% increase in hash rate.",
                    "reuters",
                ),
            ],
        ),
        (
            "ETHUSDT",
            vec![
                NewsItem::new(
                    "Ethereum staking rewards increase after network upgrade.",
                    "reuters",
                ),
                NewsItem::new(
                    "DeFi total value locked on Ethereum reaches new all-time high.",
                    "bloomberg",
                ),
            ],
        ),
        (
            "SOLUSDT",
            vec![
                NewsItem::new(
                    "Solana network processes 65,000 TPS in stress test, outperforming competitors.",
                    "reuters",
                ),
                NewsItem::new(
                    "Major NFT marketplace announces Solana integration.",
                    "twitter",
                ),
            ],
        ),
    ];

    // Analysis results
    struct AnalysisResult {
        symbol: String,
        price: f64,
        change_24h: f64,
        technical_bias: String,
        sentiment_bias: String,
        signal_type: String,
        strength: f64,
    }

    let mut results = Vec::new();

    for symbol in &symbols {
        println!("\n{}", "=".repeat(70));
        println!("Analyzing: {}", symbol);
        println!("{}", "=".repeat(70));

        // Fetch current market data from Bybit
        let ticker = bybit.get_ticker(symbol).await?;
        let klines = bybit.get_klines(symbol, "1h", 24).await?;

        // Display market data
        let last_price: f64 = ticker.last_price.parse().unwrap_or(0.0);
        let change_24h: f64 = ticker.price_24h_pcnt.parse().unwrap_or(0.0);
        let volume_24h: f64 = ticker.volume_24h.parse().unwrap_or(0.0);
        let bid: f64 = ticker.bid_price.parse().unwrap_or(0.0);
        let ask: f64 = ticker.ask_price.parse().unwrap_or(0.0);

        println!("\nMarket Data (Bybit):");
        println!("  Last Price: ${:.2}", last_price);
        println!("  24h Change: {:+.2}%", change_24h * 100.0);
        println!("  24h Volume: ${:.0}", volume_24h);
        println!("  Bid: ${:.2}", bid);
        println!("  Ask: ${:.2}", ask);

        // Calculate technical indicators from klines
        if !klines.is_empty() {
            let closes: Vec<f64> = klines
                .iter()
                .filter_map(|k| k.close.parse().ok())
                .collect();
            let highs: Vec<f64> = klines
                .iter()
                .filter_map(|k| k.high.parse().ok())
                .collect();
            let lows: Vec<f64> = klines
                .iter()
                .filter_map(|k| k.low.parse().ok())
                .collect();

            if closes.len() >= 12 {
                let sma_12: f64 = closes.iter().rev().take(12).sum::<f64>() / 12.0;
                let sma_24: f64 = closes.iter().sum::<f64>() / closes.len() as f64;
                let high_24h = highs.iter().cloned().fold(f64::MIN, f64::max);
                let low_24h = lows.iter().cloned().fold(f64::MAX, f64::min);
                let price_position = (last_price - low_24h) / (high_24h - low_24h) * 100.0;

                println!("\nTechnical Indicators (24h):");
                println!("  SMA-12: ${:.2}", sma_12);
                println!("  SMA-24: ${:.2}", sma_24);
                println!("  24h High: ${:.2}", high_24h);
                println!("  24h Low: ${:.2}", low_24h);
                println!("  Price Position: {:.1}% of range", price_position);
            }
        }

        // Analyze news sentiment
        println!("\nNews Sentiment Analysis:");

        let news_items: Vec<NewsItem> = crypto_news
            .iter()
            .filter(|(s, _)| s == symbol)
            .flat_map(|(_, items)| items.clone())
            .collect();

        let analyzer_ref = SentimentAnalyzer::new_mock();
        for (i, item) in news_items.iter().enumerate() {
            let result = analyzer_ref.analyze(&item.text).await?;
            let truncated = if item.text.len() > 50 {
                format!("{}...", &item.text[..50])
            } else {
                item.text.clone()
            };
            println!(
                "  [{}] {} ({:+.2}): {}",
                i + 1,
                result.sentiment.to_string().to_uppercase(),
                result.score,
                truncated
            );
        }

        // Generate trading signal
        let signal = generator.generate_signal(symbol, &news_items).await?;

        // Combine technical and sentiment analysis
        let technical_bias = if change_24h > 0.0 { "bullish" } else { "bearish" };
        let sentiment_bias = if signal.strength > 0.0 {
            "bullish"
        } else if signal.strength < 0.0 {
            "bearish"
        } else {
            "neutral"
        };

        println!("\nCombined Analysis:");
        println!("  Technical Bias: {}", technical_bias.to_uppercase());
        println!("  Sentiment Bias: {}", sentiment_bias.to_uppercase());
        println!(
            "  Signal: {} (strength: {:+.2})",
            signal.signal_type.to_string().to_uppercase(),
            signal.strength
        );

        results.push(AnalysisResult {
            symbol: symbol.to_string(),
            price: last_price,
            change_24h,
            technical_bias: technical_bias.to_string(),
            sentiment_bias: sentiment_bias.to_string(),
            signal_type: signal.signal_type.to_string(),
            strength: signal.strength,
        });
    }

    // Summary dashboard
    println!("\n{}", "=".repeat(70));
    println!("CRYPTO TRADING DASHBOARD");
    println!("{}", "=".repeat(70));

    println!(
        "\n{:<12} {:<12} {:<10} {:<12} {:<10} {:<10}",
        "Symbol", "Price", "24h Chg", "Technical", "Sentiment", "Signal"
    );
    println!("{}", "-".repeat(70));

    for result in &results {
        println!(
            "{:<12} ${:<10.2} {:+.1}%      {:<12} {:<10} {:<10}",
            result.symbol,
            result.price,
            result.change_24h * 100.0,
            result.technical_bias,
            result.sentiment_bias,
            result.signal_type.to_uppercase()
        );
    }

    // Trading recommendations
    println!("\n{}", "=".repeat(70));
    println!("TRADING RECOMMENDATIONS");
    println!("{}", "=".repeat(70));

    for result in &results {
        if result.technical_bias == "bullish" && result.sentiment_bias == "bullish" {
            println!("\n{}: STRONG BUY", result.symbol);
            println!("  Both technical and sentiment indicators are bullish");
        } else if result.technical_bias == "bearish" && result.sentiment_bias == "bearish" {
            println!("\n{}: STRONG SELL", result.symbol);
            println!("  Both technical and sentiment indicators are bearish");
        } else if result.technical_bias != result.sentiment_bias {
            println!("\n{}: HOLD/WAIT", result.symbol);
            println!(
                "  Mixed signals - technical is {}, sentiment is {}",
                result.technical_bias, result.sentiment_bias
            );
            println!("  Wait for clearer direction");
        } else {
            println!("\n{}: NEUTRAL", result.symbol);
            println!("  No clear trading opportunity");
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("Demo completed successfully!");
    println!("{}", "=".repeat(70));

    Ok(())
}
