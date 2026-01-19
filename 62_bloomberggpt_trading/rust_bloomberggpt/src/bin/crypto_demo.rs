//! Cryptocurrency Demo with Bybit API
//!
//! Demonstrates using the Bybit API for cryptocurrency data and analysis.
//!
//! Run with: cargo run --bin crypto_demo

use bloomberggpt_trading::{
    SentimentAnalyzer,
    TradingSignalGenerator,
    data::{MarketDataLoader, Interval},
    api::BybitClient,
    signals::SignalSource,
};
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=".repeat(60));
    println!("BloombergGPT Trading - Cryptocurrency Demo");
    println!("=".repeat(60));

    // Cryptocurrency symbols to analyze
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    println!("\nSymbols to analyze: {}", symbols.join(", "));

    // Initialize Bybit client
    let bybit = BybitClient::new();

    // Get server time to verify connection
    println!("\nConnecting to Bybit API...");
    match bybit.server_time().await {
        Ok(time) => println!("  Server time: {}", time),
        Err(e) => {
            println!("  Warning: Could not connect to Bybit API: {}", e);
            println!("  Continuing with mock data...");
        }
    }

    // Fetch ticker data
    println!("\nFetching ticker data...");
    println!("{}", "-".repeat(50));

    for symbol in &symbols {
        match bybit.get_ticker(symbol).await {
            Ok(ticker) => {
                let change_emoji = if ticker.price_change_24h >= 0.0 { "[+]" } else { "[-]" };
                println!("\n{}:", symbol);
                println!("  Last Price: ${:.2}", ticker.last_price);
                println!("  24h Change: {} {:.2}%", change_emoji, ticker.price_change_24h * 100.0);
                println!("  24h Volume: {:.2}", ticker.volume_24h);
                println!("  Bid/Ask: ${:.2} / ${:.2}", ticker.bid_price, ticker.ask_price);
            }
            Err(e) => {
                println!("\n{}: Could not fetch ticker: {}", symbol, e);
            }
        }
    }

    // Fetch historical data
    println!("\n{}", "=".repeat(60));
    println!("HISTORICAL DATA");
    println!("{}", "-".repeat(50));

    let loader = MarketDataLoader::bybit();
    let end = Utc::now();
    let start = end - Duration::days(30);

    for symbol in &symbols {
        println!("\nFetching {} historical data...", symbol);

        match loader.fetch_ohlcv(symbol, Interval::Day1, start, end).await {
            Ok(bars) => {
                if bars.is_empty() {
                    println!("  No data returned");
                    continue;
                }

                println!("  Fetched {} daily bars", bars.len());

                // Calculate statistics
                let returns: Vec<f64> = bars.windows(2)
                    .map(|w| (w[1].close - w[0].close) / w[0].close)
                    .collect();

                let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns.iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                let volatility = variance.sqrt() * (365.0_f64).sqrt();

                let total_return = (bars.last().unwrap().close - bars.first().unwrap().close)
                    / bars.first().unwrap().close;

                println!("  Statistics (30-day):");
                println!("    Total Return: {:.2}%", total_return * 100.0);
                println!("    Annualized Volatility: {:.2}%", volatility * 100.0);
                println!("    Price Range: ${:.2} - ${:.2}",
                    bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min),
                    bars.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max)
                );
            }
            Err(e) => {
                println!("  Could not fetch data: {}", e);
            }
        }
    }

    // Sentiment analysis on crypto news
    println!("\n{}", "=".repeat(60));
    println!("CRYPTO SENTIMENT ANALYSIS");
    println!("{}", "-".repeat(50));

    let analyzer = SentimentAnalyzer::new_mock();
    let generator = TradingSignalGenerator::new();

    let crypto_news = vec![
        ("BTC", "Bitcoin ETF sees record inflows as institutional adoption accelerates"),
        ("BTC", "Major bank announces Bitcoin custody services for clients"),
        ("ETH", "Ethereum staking yields rise after network upgrade"),
        ("ETH", "DeFi protocol exploit leads to $10M in losses"),
        ("SOL", "Solana network experiences brief outage during high traffic"),
        ("SOL", "New gaming partnerships drive Solana NFT volume surge"),
    ];

    println!("\nAnalyzing crypto news...\n");

    for (crypto, news) in &crypto_news {
        let result = analyzer.analyze(news).await?;

        let sentiment_indicator = if result.score > 0.2 {
            "[+]"
        } else if result.score < -0.2 {
            "[-]"
        } else {
            "[ ]"
        };

        println!("{} {} | {} (score: {:.2})",
            sentiment_indicator,
            crypto,
            result.sentiment,
            result.score
        );
        println!("   \"{}\"", truncate_text(news, 55));

        // Generate trading signal
        let symbol = format!("{}USDT", crypto);
        if let Some(signal) = generator.generate_signal(&result, &symbol, SignalSource::News)? {
            println!("   -> Signal: {} (strength: {:.2})", signal.signal_type, signal.strength);
        }
        println!();
    }

    // Recent trades (if API available)
    println!("{}", "=".repeat(60));
    println!("RECENT MARKET ACTIVITY");
    println!("{}", "-".repeat(50));

    for symbol in &["BTCUSDT", "ETHUSDT"] {
        match bybit.get_recent_trades(symbol, 5).await {
            Ok(trades) => {
                println!("\n{} Recent Trades:", symbol);
                for trade in trades {
                    let side = if trade.side == "Buy" { "[B]" } else { "[S]" };
                    println!("  {} ${:.2} x {:.4}", side, trade.price, trade.size);
                }
            }
            Err(e) => {
                println!("\n{}: Could not fetch trades: {}", symbol, e);
            }
        }
    }

    // Orderbook snapshot
    println!("\n{}", "-".repeat(50));
    println!("ORDERBOOK SNAPSHOT");

    for symbol in &["BTCUSDT"] {
        match bybit.get_orderbook(symbol, 5).await {
            Ok(book) => {
                println!("\n{} Orderbook:", symbol);
                println!("  Asks (Sell Orders):");
                for level in book.asks.iter().take(3).rev() {
                    println!("    ${:.2} | {:.4}", level.price, level.quantity);
                }
                println!("  -------");
                println!("  Bids (Buy Orders):");
                for level in book.bids.iter().take(3) {
                    println!("    ${:.2} | {:.4}", level.price, level.quantity);
                }

                // Calculate spread
                if let (Some(best_bid), Some(best_ask)) = (book.bids.first(), book.asks.first()) {
                    let spread = (best_ask.price - best_bid.price) / best_bid.price * 100.0;
                    println!("\n  Spread: {:.4}%", spread);
                }
            }
            Err(e) => {
                println!("\n{}: Could not fetch orderbook: {}", symbol, e);
            }
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Demo complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}

fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}
