//! # Fetch Trades Example
//!
//! Demonstrates fetching recent trades from Bybit exchange.
//!
//! Run with: `cargo run --example fetch_trades`

use anyhow::Result;
use order_flow_imbalance::BybitClient;
use order_flow_imbalance::data::trade::TradeStats;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let symbol = env::args().nth(1).unwrap_or_else(|| "BTCUSDT".to_string());

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║            Trade Fetcher - Bybit Exchange                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    let client = BybitClient::new();

    println!("Fetching recent trades for {}...", symbol);
    println!();

    // Fetch last 100 trades
    let trades = client.get_trades(&symbol, 100).await?;

    println!("═══════════════════════════════════════════════════════════");
    println!("                   RECENT TRADES: {}                        ", symbol);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  Time                  │ Side │     Price     │    Size    ");
    println!("  ──────────────────────┼──────┼───────────────┼────────────");

    for trade in trades.iter().take(20) {
        let side = if trade.is_buy() { "BUY " } else { "SELL" };
        let side_color = if trade.is_buy() { "+" } else { "-" };

        println!(
            "  {} │ {} {} │ ${:>12.2} │ {:>10.4}",
            trade.timestamp.format("%H:%M:%S%.3f"),
            side_color,
            side,
            trade.price,
            trade.size
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");

    // Calculate statistics
    let stats = TradeStats::from_trades(&trades);

    println!();
    println!("TRADE STATISTICS (Last {} trades)", trades.len());
    println!("───────────────────────────────────────────────────────────");
    println!("  Total Volume:      {:>12.4}", stats.volume);
    println!("  Buy Volume:        {:>12.4} ({:.1}%)",
        stats.buy_volume,
        stats.buy_volume / stats.volume * 100.0
    );
    println!("  Sell Volume:       {:>12.4} ({:.1}%)",
        stats.sell_volume,
        stats.sell_volume / stats.volume * 100.0
    );
    println!("  Trade Imbalance:   {:>12.4}", stats.trade_imbalance());
    println!();
    println!("  VWAP:              ${:>12.2}", stats.vwap);
    println!("  High:              ${:>12.2}", stats.high);
    println!("  Low:               ${:>12.2}", stats.low);
    println!("  Range:             {:>12.2}%", stats.range_pct());
    println!();
    println!("  Trade Count:       {:>12}", stats.count);
    println!("  Buy Count:         {:>12}", stats.buy_count);
    println!("  Sell Count:        {:>12}", stats.sell_count);
    println!("  Avg Trade Size:    {:>12.4}", stats.avg_size);
    println!("  Max Trade Size:    {:>12.4}", stats.max_size);
    println!();

    // Identify large trades
    let large_threshold = stats.avg_size * 2.0;
    let large_trades: Vec<_> = trades.iter().filter(|t| t.size > large_threshold).collect();

    if !large_trades.is_empty() {
        println!("LARGE TRADES (> 2x average)");
        println!("───────────────────────────────────────────────────────────");
        for trade in large_trades.iter().take(5) {
            let side = if trade.is_buy() { "BUY " } else { "SELL" };
            println!(
                "  {} {} {:>10.4} @ ${:.2}",
                trade.timestamp.format("%H:%M:%S"),
                side,
                trade.size,
                trade.price
            );
        }
        println!();
    }

    Ok(())
}
