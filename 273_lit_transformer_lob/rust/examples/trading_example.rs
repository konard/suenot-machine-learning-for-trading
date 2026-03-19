use anyhow::Result;
use lit_transformer_lob::{
    build_lob_features, build_trade_features, BybitClient, LitModel,
};

fn main() -> Result<()> {
    let d_model = 16;
    let max_levels = 100; // up to 50 bid + 50 ask
    let d_hidden = 32;

    println!("=== LIT Transformer for Limit Order Book ===\n");

    // Initialize model
    let model = LitModel::new(d_model, max_levels, d_hidden);
    println!("[1/4] Model initialized (d_model={}, d_hidden={})", d_model, d_hidden);

    // Fetch live data from Bybit
    let client = BybitClient::new();

    println!("[2/4] Fetching BTCUSDT orderbook from Bybit...");
    let (bids, asks) = match client.fetch_orderbook("BTCUSDT", 25) {
        Ok(data) => {
            println!(
                "       Got {} bid levels, {} ask levels",
                data.0.len(),
                data.1.len()
            );
            if let (Some(best_bid), Some(best_ask)) = (data.0.first(), data.1.first()) {
                println!(
                    "       Best bid: {:.2} ({:.4}), Best ask: {:.2} ({:.4})",
                    best_bid.0, best_bid.1, best_ask.0, best_ask.1
                );
                let mid = (best_bid.0 + best_ask.0) / 2.0;
                let spread_bps = (best_ask.0 - best_bid.0) / mid * 10000.0;
                println!("       Mid price: {:.2}, Spread: {:.2} bps", mid, spread_bps);
            }
            data
        }
        Err(e) => {
            println!("       Failed to fetch orderbook: {}. Using synthetic data.", e);
            let bids: Vec<(f64, f64)> = (0..25)
                .map(|i| (50000.0 - i as f64 * 0.5, 0.1 + i as f64 * 0.05))
                .collect();
            let asks: Vec<(f64, f64)> = (0..25)
                .map(|i| (50000.5 + i as f64 * 0.5, 0.1 + i as f64 * 0.05))
                .collect();
            (bids, asks)
        }
    };

    println!("[3/4] Fetching BTCUSDT recent trades from Bybit...");
    let trades = match client.fetch_recent_trades("BTCUSDT", 50) {
        Ok(data) => {
            println!("       Got {} recent trades", data.len());
            if let Some(last) = data.first() {
                println!(
                    "       Last trade: price={:.2}, qty={:.4}, side={}",
                    last.price,
                    last.qty,
                    if last.is_buyer_maker { "sell" } else { "buy" }
                );
            }
            data
        }
        Err(e) => {
            println!("       Failed to fetch trades: {}. Using synthetic data.", e);
            (0..50)
                .map(|i| lit_transformer_lob::TradeRecord {
                    price: 50000.0 + (i as f64 * 0.1).sin() * 5.0,
                    qty: 0.01 + (i as f64 * 0.05),
                    timestamp: 1700000000000 + i * 500,
                    is_buyer_maker: i % 3 == 0,
                })
                .collect()
        }
    };

    // Compute mid-price for feature engineering
    let mid_price = if !bids.is_empty() && !asks.is_empty() {
        (bids[0].0 + asks[0].0) / 2.0
    } else {
        50000.0
    };

    // Build feature matrices
    let lob_features = build_lob_features(&bids, &asks, d_model);
    let trade_features = build_trade_features(&trades, mid_price, d_model);

    println!(
        "       LOB features shape: {:?}, Trade features shape: {:?}",
        lob_features.shape(),
        trade_features.shape()
    );

    // Run inference
    println!("\n[4/4] Running LIT inference...");
    let probs = model.forward(&lob_features, &trade_features);

    println!("\n=== Prediction Results ===");
    println!("  P(price UP)   = {:.4}", probs[0]);
    println!("  P(price DOWN) = {:.4}", probs[1]);
    println!("  P(price STAY) = {:.4}", probs[2]);

    let prediction = if probs[0] > probs[1] && probs[0] > probs[2] {
        "UP"
    } else if probs[1] > probs[0] && probs[1] > probs[2] {
        "DOWN"
    } else {
        "STAY"
    };
    println!("\n  => Model predicts: {}", prediction);

    println!("\nNote: This model uses random weights for demonstration.");
    println!("In production, weights would be trained on historical LOB + trade data.");

    Ok(())
}
