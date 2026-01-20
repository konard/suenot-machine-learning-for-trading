//! Bybit exchange integration example.
//!
//! This example demonstrates how to:
//! 1. Connect to Bybit API (testnet)
//! 2. Fetch market data
//! 3. Analyze order book for execution
//!
//! Note: For actual trading, you need valid API credentials.

use llm_trade_execution::{
    BybitClient, BybitConfig, TimeFrame, MarketImpactEstimator,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Bybit Integration Example ===\n");

    // Create a Bybit client for public endpoints (no auth needed)
    // For testnet:
    // let client = BybitClient::public_testnet();
    // For mainnet:
    let client = BybitClient::public();

    let symbol = "BTCUSDT";

    // Example 1: Fetch current ticker
    println!("--- Fetching Ticker ---");
    match client.get_ticker(symbol).await {
        Ok(ticker) => {
            println!("Symbol: {}", ticker.symbol);
            println!("Last Price: {:.2}", ticker.last_price);
            println!("Bid: {:.2} (qty: {:.4})", ticker.bid_price, ticker.bid_qty);
            println!("Ask: {:.2} (qty: {:.4})", ticker.ask_price, ticker.ask_qty);
            println!("Spread: {:.2} ({:.2} bps)", ticker.spread(), ticker.spread_bps());
            println!("24h High: {:.2}", ticker.high_24h);
            println!("24h Low: {:.2}", ticker.low_24h);
            println!("24h Volume: {:.2}", ticker.volume_24h);

            if let Some(oi) = ticker.open_interest {
                println!("Open Interest: {:.2}", oi);
            }
            if let Some(fr) = ticker.funding_rate {
                println!("Funding Rate: {:.4}%", fr * 100.0);
            }
        }
        Err(e) => {
            println!("Failed to fetch ticker: {}", e);
            println!("(This is expected if running without network access)");
        }
    }
    println!();

    // Example 2: Fetch order book and analyze
    println!("--- Fetching Order Book ---");
    match client.get_orderbook(symbol, Some(25)).await {
        Ok(book) => {
            println!("Order Book for {}", symbol);
            println!("Best Bid: {:.2}", book.best_bid().unwrap_or(0.0));
            println!("Best Ask: {:.2}", book.best_ask().unwrap_or(0.0));
            println!("Mid Price: {:.2}", book.mid_price().unwrap_or(0.0));
            println!("Spread: {:.2} bps", book.spread_bps().unwrap_or(0.0));
            println!("Bid Depth (10 levels): {:.4}", book.bid_depth(10));
            println!("Ask Depth (10 levels): {:.4}", book.ask_depth(10));
            println!("Imbalance: {:.2}", book.imbalance(10));

            // Estimate impact for different order sizes
            println!("\n--- Impact Estimation ---");
            let estimator = MarketImpactEstimator::crypto();

            for qty in [0.1, 0.5, 1.0, 5.0, 10.0] {
                let estimate = estimator.estimate(qty, 1.0, Some(&book));
                println!(
                    "  {:.1} BTC: {:.2} bps total ({:.2} permanent + {:.2} temporary)",
                    qty,
                    estimate.impact.as_bps(),
                    estimate.impact.permanent * 10000.0,
                    estimate.impact.temporary * 10000.0
                );
            }

            // Direct order book impact
            println!("\n--- Direct Book Impact ---");
            for qty in [0.1, 0.5, 1.0] {
                if let Some((avg_price, impact)) = book.buy_impact(qty) {
                    println!(
                        "  Buy {:.1} BTC: avg price {:.2}, impact {:.4}%",
                        qty, avg_price, impact * 100.0
                    );
                }
            }
        }
        Err(e) => {
            println!("Failed to fetch order book: {}", e);
        }
    }
    println!();

    // Example 3: Fetch historical klines
    println!("--- Fetching Historical Data ---");
    match client.get_klines(symbol, TimeFrame::H1, Some(24), None, None).await {
        Ok(bars) => {
            println!("Got {} hourly bars", bars.len());

            if !bars.is_empty() {
                // Calculate some basic stats
                let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
                let avg_volume: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;

                let ranges: Vec<f64> = bars.iter().map(|b| b.range()).collect();
                let avg_range: f64 = ranges.iter().sum::<f64>() / ranges.len() as f64;

                println!("Average hourly volume: {:.2}", avg_volume);
                println!("Average hourly range: {:.2}", avg_range);

                // Show latest bar
                if let Some(latest) = bars.last() {
                    println!("\nLatest bar:");
                    println!("  Time: {}", latest.timestamp);
                    println!("  Open: {:.2}", latest.open);
                    println!("  High: {:.2}", latest.high);
                    println!("  Low: {:.2}", latest.low);
                    println!("  Close: {:.2}", latest.close);
                    println!("  Volume: {:.2}", latest.volume);
                }
            }
        }
        Err(e) => {
            println!("Failed to fetch klines: {}", e);
        }
    }
    println!();

    // Example 4: Fetch recent trades
    println!("--- Fetching Recent Trades ---");
    match client.get_trades(symbol, Some(20)).await {
        Ok(trades) => {
            println!("Got {} recent trades", trades.len());

            if !trades.is_empty() {
                // Calculate trade statistics
                let total_volume: f64 = trades.iter().map(|t| t.quantity).sum();
                let total_value: f64 = trades.iter().map(|t| t.value()).sum();
                let vwap = total_value / total_volume;

                let buy_volume: f64 = trades
                    .iter()
                    .filter(|t| t.direction == llm_trade_execution::TradeDirection::Buy)
                    .map(|t| t.quantity)
                    .sum();

                println!("Total Volume: {:.4}", total_volume);
                println!("Buy Volume: {:.4} ({:.1}%)", buy_volume, buy_volume / total_volume * 100.0);
                println!("VWAP: {:.2}", vwap);

                // Show a few trades
                println!("\nRecent trades:");
                for trade in trades.iter().take(5) {
                    println!(
                        "  {} {:.4} @ {:.2} ({})",
                        if trade.direction == llm_trade_execution::TradeDirection::Buy { "BUY " } else { "SELL" },
                        trade.quantity,
                        trade.price,
                        trade.timestamp.format("%H:%M:%S")
                    );
                }
            }
        }
        Err(e) => {
            println!("Failed to fetch trades: {}", e);
        }
    }

    println!("\n=== Example Complete ===");
    println!("Note: This example uses public endpoints only.");
    println!("For actual trading, configure API credentials in environment:");
    println!("  BYBIT_API_KEY=your_key");
    println!("  BYBIT_API_SECRET=your_secret");
    println!("  BYBIT_TESTNET=true  # for testnet");

    Ok(())
}
