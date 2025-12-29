//! # Trading Strategy Example
//!
//! Demonstrates a complete OFI-based trading strategy.
//!
//! Run with: `cargo run --example trading_strategy`

use anyhow::Result;
use order_flow_imbalance::BybitClient;
use order_flow_imbalance::features::engine::FeatureEngine;
use order_flow_imbalance::strategy::signal::{Signal, SignalConfig, SignalGenerator};
use order_flow_imbalance::strategy::position::{Position, PositionManager, PositionSide, ExitReason};
use order_flow_imbalance::metrics::trading::TradingMetrics;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           OFI Trading Strategy Simulator                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("⚠️  DISCLAIMER: This is a SIMULATION for educational purposes.");
    println!("    No real trades are being executed.");
    println!();

    let client = BybitClient::new();
    let mut feature_engine = FeatureEngine::new();

    // Configure signal generator
    let signal_config = SignalConfig {
        prob_threshold_long: 0.55,
        prob_threshold_short: 0.45,
        ofi_threshold: 1.0,  // Lower threshold for demo
        max_spread_bps: 20.0,
        min_confidence: 0.05,
        stop_loss_pct: 0.1,
        take_profit_pct: 0.15,
    };

    let mut signal_generator = SignalGenerator::new(signal_config);
    signal_generator.set_cooldown(5000); // 5 second cooldown

    let mut position_manager = PositionManager::new(
        0.1,    // 0.1 BTC max position
        120,    // 2 minute max hold
        100.0,  // $100 max daily loss
    );

    let mut metrics = TradingMetrics::new();

    println!("Strategy Configuration:");
    println!("  OFI Threshold:     1.0 Z-score");
    println!("  Max Spread:        20 bps");
    println!("  Position Size:     0.1 BTC");
    println!("  Max Hold Time:     2 minutes");
    println!("  Stop Loss:         0.1%");
    println!("  Take Profit:       0.15%");
    println!();

    // Warm up
    println!("Warming up feature engine...");
    for _ in 0..15 {
        let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
        let trades = client.get_trades("BTCUSDT", 50).await?;

        feature_engine.update_orderbook(&orderbook);
        for trade in &trades {
            feature_engine.update_trade(trade);
        }
        sleep(Duration::from_millis(200)).await;
    }
    println!("Done!");
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("                  LIVE SIMULATION                           ");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let mut iteration = 0;

    loop {
        iteration += 1;

        // Fetch data
        let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
        let trades = client.get_trades("BTCUSDT", 50).await?;

        // Update feature engine
        feature_engine.update_orderbook(&orderbook);
        for trade in &trades {
            feature_engine.update_trade(trade);
        }

        let mid_price = orderbook.mid_price().unwrap_or(0.0);
        let spread_bps = orderbook.spread_bps().unwrap_or(0.0);

        // Check for exit
        if let Some(exit_reason) = position_manager.update(mid_price) {
            if let Some(closed) = position_manager.close_position(mid_price, exit_reason) {
                let notional = closed.size * mid_price;
                let commission = notional * 0.0004 * 2.0;
                let net_pnl = closed.pnl - commission;

                metrics.record_trade(closed.pnl, commission);

                println!();
                println!("╔═══════════════════════════════════════════════════════╗");
                println!("║                    TRADE CLOSED                       ║");
                println!("╠═══════════════════════════════════════════════════════╣");
                println!("║  Side:        {:?}", closed.side);
                println!("║  Entry:       ${:.2}", closed.entry_price);
                println!("║  Exit:        ${:.2}", closed.exit_price);
                println!("║  Size:        {:.4} BTC", closed.size);
                println!("║  Gross P&L:   ${:.2}", closed.pnl);
                println!("║  Commission:  ${:.2}", commission);
                println!("║  Net P&L:     ${:.2}", net_pnl);
                println!("║  Reason:      {:?}", closed.exit_reason);
                println!("╚═══════════════════════════════════════════════════════╝");
                println!();
            }
        }

        // Extract features and generate signal
        let features = feature_engine.extract_features(&orderbook);
        let signal = signal_generator.generate(&features, mid_price, spread_bps);

        // Display status
        let ofi_1min = features.get("ofi_1min").unwrap_or(0.0);
        let ofi_zscore = features.get("ofi_zscore").unwrap_or(0.0);
        let depth_imb = features.get("depth_imbalance_l5").unwrap_or(0.0);

        println!("─────────────────────────────────────────────────────────────");
        println!("  #{} | {} | ${:.2} | Spread: {:.1} bps",
            iteration,
            orderbook.timestamp.format("%H:%M:%S"),
            mid_price,
            spread_bps
        );
        println!("  OFI 1min: {:+.2} | Z: {:+.2} | Depth Imb: {:+.2}",
            ofi_1min, ofi_zscore, depth_imb
        );

        // Show position status
        if let Some(pos) = position_manager.position() {
            let mut pos_display = pos.clone();
            pos_display.update_pnl(mid_price);
            println!("  Position: {:?} {:.4} BTC @ ${:.2} | P&L: ${:.2}",
                pos.side, pos.size, pos.entry_price, pos_display.unrealized_pnl
            );
        } else {
            println!("  Position: FLAT");
        }

        // Handle signal
        if position_manager.is_flat() && signal.signal != Signal::Hold {
            println!();
            println!("  🔔 SIGNAL: {} (confidence: {:.1}%)",
                signal.signal, signal.confidence * 100.0
            );

            let side = match signal.signal {
                Signal::Long => PositionSide::Long,
                Signal::Short => PositionSide::Short,
                _ => PositionSide::Flat,
            };

            if side != PositionSide::Flat {
                let position = Position::new(
                    "BTCUSDT".to_string(),
                    side,
                    0.1,
                    mid_price,
                    signal.stop_loss,
                    signal.take_profit,
                );

                if position_manager.open_position(position) {
                    println!("  ✅ Opened {:?} position at ${:.2}", side, mid_price);
                    if let Some(sl) = signal.stop_loss {
                        println!("     Stop Loss: ${:.2}", sl);
                    }
                    if let Some(tp) = signal.take_profit {
                        println!("     Take Profit: ${:.2}", tp);
                    }
                }
            }
        }

        println!();

        // Show running stats every 20 iterations
        if iteration % 20 == 0 && metrics.total_trades > 0 {
            println!("═══════════════════════════════════════════════════════════");
            println!("                   SESSION STATISTICS                       ");
            println!("───────────────────────────────────────────────────────────");
            println!("  Trades: {} | Win Rate: {:.1}% | Net P&L: ${:.2}",
                metrics.total_trades,
                metrics.win_rate() * 100.0,
                metrics.net_pnl
            );
            println!("  Profit Factor: {:.2} | Sharpe: {:.2}",
                metrics.profit_factor(),
                metrics.sharpe_ratio()
            );
            println!("═══════════════════════════════════════════════════════════");
            println!();
        }

        sleep(Duration::from_millis(500)).await;

        // Stop after 200 iterations
        if iteration >= 200 {
            break;
        }
    }

    // Final report
    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                   FINAL REPORT                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("{}", metrics.summary());

    Ok(())
}
