//! # Real-time Analysis Example
//!
//! Demonstrates real-time order flow analysis with live data.
//!
//! Run with: `cargo run --example realtime_analysis`

use anyhow::Result;
use order_flow_imbalance::BybitClient;
use order_flow_imbalance::features::engine::FeatureEngine;
use order_flow_imbalance::orderflow::ofi::OrderFlowCalculator;
use order_flow_imbalance::orderflow::vpin::VpinCalculator;
use order_flow_imbalance::orderflow::toxicity::ToxicityIndicator;
use order_flow_imbalance::data::trade::TradeStats;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║          Real-Time Market Microstructure Analysis          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Analyzing BTCUSDT in real-time...");
    println!("Press Ctrl+C to stop");
    println!();

    let client = BybitClient::new();

    let mut ofi_calc = OrderFlowCalculator::new();
    let mut vpin_calc = VpinCalculator::new(5.0, 20); // Smaller buckets for demo
    let mut feature_engine = FeatureEngine::new();
    let mut toxicity = ToxicityIndicator::new();

    // Track spreads for z-score
    let mut spread_history: Vec<f64> = Vec::new();

    let mut iteration = 0;

    loop {
        iteration += 1;

        // Fetch data
        let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
        let trades = client.get_trades("BTCUSDT", 100).await?;

        // Update calculators
        ofi_calc.update(&orderbook);
        for trade in &trades {
            vpin_calc.add_trade(trade);
        }
        feature_engine.update_orderbook(&orderbook);
        for trade in &trades {
            feature_engine.update_trade(trade);
        }

        // Calculate stats
        let mid = orderbook.mid_price().unwrap_or(0.0);
        let spread_bps = orderbook.spread_bps().unwrap_or(0.0);
        spread_history.push(spread_bps);
        if spread_history.len() > 100 {
            spread_history.remove(0);
        }

        let avg_spread = spread_history.iter().sum::<f64>() / spread_history.len() as f64;
        let std_spread = {
            let variance = spread_history.iter().map(|s| (s - avg_spread).powi(2)).sum::<f64>()
                / spread_history.len() as f64;
            variance.sqrt()
        };

        let trade_stats = TradeStats::from_trades(&trades);

        // Update toxicity
        toxicity.update(
            vpin_calc.current_vpin(),
            ofi_calc.z_score(50),
            &orderbook,
            &trade_stats,
            avg_spread,
            std_spread,
        );

        // Clear screen and display dashboard
        print!("\x1B[2J\x1B[1;1H"); // Clear screen

        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║       BTCUSDT Real-Time Microstructure Dashboard          ║");
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║  Time: {} | Iteration: {:>5}  ║",
            orderbook.timestamp.format("%H:%M:%S%.3f"), iteration
        );
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();

        // Price section
        println!("  PRICE");
        println!("  ─────────────────────────────────────────────────────────");
        println!("    Mid Price:        ${:>12.2}", mid);
        println!("    Best Bid:         ${:>12.2}  ({:>8.4} BTC)",
            orderbook.best_bid().map(|l| l.price).unwrap_or(0.0),
            orderbook.best_bid().map(|l| l.size).unwrap_or(0.0)
        );
        println!("    Best Ask:         ${:>12.2}  ({:>8.4} BTC)",
            orderbook.best_ask().map(|l| l.price).unwrap_or(0.0),
            orderbook.best_ask().map(|l| l.size).unwrap_or(0.0)
        );
        println!("    Spread:           {:>12.2} bps", spread_bps);
        println!();

        // Order Flow section
        println!("  ORDER FLOW IMBALANCE");
        println!("  ─────────────────────────────────────────────────────────");

        let ofi_1min = ofi_calc.ofi_1min();
        let ofi_zscore = ofi_calc.z_score(50).unwrap_or(0.0);

        // Visual indicator for OFI
        let ofi_bar = create_bar(ofi_zscore, 3.0, 20);
        println!("    OFI 1-min:        {:>+12.2}", ofi_1min);
        println!("    OFI Z-Score:      {:>+12.2}  [{}]", ofi_zscore, ofi_bar);
        println!("    Cumulative OFI:   {:>+12.2}", ofi_calc.cumulative());

        // Signal interpretation
        let ofi_signal = if ofi_zscore > 2.0 {
            "▲▲ STRONG BUY PRESSURE"
        } else if ofi_zscore > 1.0 {
            "▲  Buy pressure"
        } else if ofi_zscore < -2.0 {
            "▼▼ STRONG SELL PRESSURE"
        } else if ofi_zscore < -1.0 {
            "▼  Sell pressure"
        } else {
            "◆  Neutral"
        };
        println!("    Signal:           {}", ofi_signal);
        println!();

        // VPIN section
        println!("  VPIN (Flow Toxicity)");
        println!("  ─────────────────────────────────────────────────────────");
        if let Some(vpin) = vpin_calc.current_vpin() {
            let vpin_bar = create_bar(vpin, 1.0, 20);
            println!("    VPIN:             {:>12.4}  [{}]", vpin, vpin_bar);

            let stats = vpin_calc.statistics();
            let level_str = match stats.toxicity_level() {
                order_flow_imbalance::orderflow::vpin::ToxicityLevel::Low => "🟢 LOW",
                order_flow_imbalance::orderflow::vpin::ToxicityLevel::Medium => "🟡 MEDIUM",
                order_flow_imbalance::orderflow::vpin::ToxicityLevel::High => "🟠 HIGH",
                order_flow_imbalance::orderflow::vpin::ToxicityLevel::VeryHigh => "🔴 VERY HIGH",
            };
            println!("    Toxicity Level:   {}", level_str);
        } else {
            println!("    VPIN:             Collecting data... ({}/20 buckets)",
                vpin_calc.bucket_count()
            );
        }
        println!();

        // Depth section
        println!("  ORDER BOOK DEPTH");
        println!("  ─────────────────────────────────────────────────────────");
        let depth_imb = orderbook.depth_imbalance(5);
        let imb_bar = create_bar(depth_imb, 1.0, 20);
        println!("    Bid Depth (L5):   {:>12.4} BTC", orderbook.bid_depth(5));
        println!("    Ask Depth (L5):   {:>12.4} BTC", orderbook.ask_depth(5));
        println!("    Imbalance:        {:>+12.4}  [{}]", depth_imb, imb_bar);
        println!();

        // Trade Flow section
        println!("  TRADE FLOW (Last 100 trades)");
        println!("  ─────────────────────────────────────────────────────────");
        println!("    Buy Volume:       {:>12.4} BTC", trade_stats.buy_volume);
        println!("    Sell Volume:      {:>12.4} BTC", trade_stats.sell_volume);
        println!("    Trade Imbalance:  {:>+12.4}", trade_stats.trade_imbalance());
        println!("    VWAP:             ${:>12.2}", trade_stats.vwap);
        println!();

        // Composite Analysis
        println!("  COMPOSITE ANALYSIS");
        println!("  ─────────────────────────────────────────────────────────");
        let composite_bar = create_bar(toxicity.composite_score, 1.0, 20);
        println!("    Toxicity Score:   {:>12.4}  [{}]", toxicity.composite_score, composite_bar);
        println!("    Level:            {}", toxicity.level());
        println!("    Recommendation:   {}", toxicity.recommendation());
        println!();

        // Market Assessment
        println!("═══════════════════════════════════════════════════════════");
        let assessment = if ofi_zscore > 1.5 && depth_imb > 0.2 {
            "📈 BULLISH: Strong buy pressure detected"
        } else if ofi_zscore < -1.5 && depth_imb < -0.2 {
            "📉 BEARISH: Strong sell pressure detected"
        } else if toxicity.composite_score > 0.6 {
            "⚠️  CAUTION: High toxicity - informed traders active"
        } else if spread_bps > avg_spread + 2.0 * std_spread {
            "⚠️  CAUTION: Spread widening detected"
        } else {
            "➡️  NEUTRAL: Normal market conditions"
        };
        println!("  {}", assessment);
        println!("═══════════════════════════════════════════════════════════");

        sleep(Duration::from_millis(500)).await;

        if iteration >= 200 {
            println!();
            println!("Demo complete after {} iterations.", iteration);
            break;
        }
    }

    Ok(())
}

fn create_bar(value: f64, max: f64, width: usize) -> String {
    let normalized = ((value / max + 1.0) / 2.0).clamp(0.0, 1.0);
    let pos = (normalized * width as f64) as usize;

    let mut bar = String::with_capacity(width);
    for i in 0..width {
        if i == width / 2 {
            bar.push('│');
        } else if i == pos {
            bar.push('●');
        } else if (i < width / 2 && i >= pos) || (i > width / 2 && i <= pos) {
            bar.push('─');
        } else {
            bar.push(' ');
        }
    }
    bar
}
