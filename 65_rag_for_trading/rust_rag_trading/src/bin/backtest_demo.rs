//! Backtesting demonstration.
//!
//! This example shows how to:
//! - Backtest a RAG-based trading strategy
//! - Configure backtest parameters
//! - Analyze performance metrics

use chrono::{TimeZone, Utc};
use rag_trading::{
    backtest::{BacktestConfig, PriceBar, RAGBacktester},
    Document, DocumentRetriever, RAGSignalGenerator,
};

fn main() {
    println!("=== RAG Trading: Backtest Demo ===\n");

    // Create document retriever with sample news
    let mut retriever = DocumentRetriever::new();

    // Add documents that will influence signals
    let documents = vec![
        Document::new(
            "AAPL reported strong quarterly earnings with beats on both revenue \
             and EPS. iPhone sales exceeded expectations. Services growth \
             continues momentum. Analysts upgrade to buy.",
            Some("AAPL".to_string()),
            "Reuters",
            Utc::now(),
        ),
        Document::new(
            "Apple announces breakthrough AI features for iOS. Strong \
             momentum in wearables. Bullish outlook from management.",
            Some("AAPL".to_string()),
            "Bloomberg",
            Utc::now(),
        ),
        Document::new(
            "Apple faces supply chain concerns in China. iPhone demand \
             slightly weak in some markets. Competition increasing.",
            Some("AAPL".to_string()),
            "Financial Times",
            Utc::now(),
        ),
    ];

    retriever.add_documents(documents);
    println!("Indexed {} documents for AAPL\n", retriever.document_count());

    // Create signal generator
    let generator = RAGSignalGenerator::new(retriever);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.1, // 10% per trade
        min_confidence: 0.5,
        hold_period: 5, // 5 bars
        commission: 0.001,
    };

    println!("Backtest Configuration:");
    println!("  Initial Capital: ${:.2}", config.initial_capital);
    println!("  Position Size: {:.0}%", config.position_size * 100.0);
    println!("  Min Confidence: {:.0}%", config.min_confidence * 100.0);
    println!("  Hold Period: {} bars", config.hold_period);
    println!("  Commission: {:.2}%\n", config.commission * 100.0);

    // Create backtester
    let backtester = RAGBacktester::new(generator, config);

    // Generate sample price data (simulating 50 trading days)
    let prices = generate_sample_prices(50);
    println!("Generated {} price bars for testing\n", prices.len());
    println!(
        "Price range: ${:.2} - ${:.2}",
        prices.iter().map(|p| p.low).fold(f64::MAX, f64::min),
        prices.iter().map(|p| p.high).fold(f64::MIN, f64::max)
    );
    println!();

    // Run backtest
    println!("Running backtest...\n");
    let result = backtester.run("AAPL", &prices);

    // Display results
    println!("=== Backtest Results ===\n");
    println!("{}", result);

    // Display trade details
    if !result.trades.is_empty() {
        println!("\n--- Trade Details ---\n");

        for (i, trade) in result.trades.iter().enumerate() {
            let pnl_str = trade
                .pnl
                .map(|p| format!("{:+.2}", p))
                .unwrap_or_else(|| "Open".to_string());

            let return_str = trade
                .return_pct()
                .map(|r| format!("{:+.2}%", r * 100.0))
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "Trade {}: {} {} @ ${:.2} -> {} | PnL: ${} ({})",
                i + 1,
                trade.direction,
                trade.ticker,
                trade.entry_price,
                trade
                    .exit_price
                    .map(|p| format!("${:.2}", p))
                    .unwrap_or_else(|| "Open".to_string()),
                pnl_str,
                return_str
            );
        }
    }

    // Performance analysis
    println!("\n--- Performance Analysis ---\n");

    if result.num_trades > 0 {
        let avg_return = result.trades
            .iter()
            .filter_map(|t| t.return_pct())
            .sum::<f64>()
            / result.num_trades as f64;

        println!("Average Return per Trade: {:.2}%", avg_return * 100.0);
        println!(
            "Profit Factor: {:.2}",
            if result.losing_trades > 0 {
                result.winning_trades as f64 / result.losing_trades as f64
            } else {
                f64::INFINITY
            }
        );

        // Calculate expectancy
        let expectancy = result.avg_profit;
        println!("Expectancy per Trade: ${:.2}", expectancy);
    } else {
        println!("No trades executed during backtest period.");
        println!("This could mean:");
        println!("  - Confidence threshold too high");
        println!("  - Insufficient document data for signals");
        println!("  - Signal generator returning neutral signals");
    }

    println!("\n=== Demo Complete ===");
}

/// Generate sample price data for testing.
fn generate_sample_prices(num_bars: usize) -> Vec<PriceBar> {
    let base_price = 175.0; // Starting price for AAPL
    let mut prices = Vec::with_capacity(num_bars);
    let mut price = base_price;

    for i in 0..num_bars {
        // Create timestamp
        let ts = Utc
            .with_ymd_and_hms(2024, 1, 2 + (i as u32 % 28), 9, 30, 0)
            .unwrap();

        // Simulate price movement with some volatility
        let trend = 0.001 * (i as f64 - num_bars as f64 / 2.0); // Slight uptrend
        let noise = 0.02 * ((i * 17 + 13) % 100) as f64 / 100.0 - 0.01;

        price *= 1.0 + trend + noise;

        // Generate OHLCV
        let volatility = price * 0.015;
        let open = price;
        let movement = ((i * 7 + 3) % 20) as f64 / 20.0 - 0.5;
        let close = price * (1.0 + movement * 0.02);
        let high = open.max(close) + volatility * ((i * 11) % 10) as f64 / 10.0;
        let low = open.min(close) - volatility * ((i * 13) % 10) as f64 / 10.0;
        let volume = 50_000_000.0 * (1.0 + 0.3 * ((i * 5) % 10) as f64 / 10.0);

        prices.push(PriceBar::new(ts, open, high, low, close, volume));

        price = close;
    }

    prices
}
