//! Trading Example: Compute microstructure features from Bybit BTCUSDT data.
//!
//! Usage: cargo run --example trading_example

use microstructure_features::bybit;
use microstructure_features::{MicrostructureCalculator, Trade};

fn main() {
    println!("=== Microstructure Features for BTCUSDT (Bybit) ===\n");

    // ─── Fetch live data from Bybit ────────────────────────────────────────
    println!("Fetching orderbook...");
    let orderbook = match bybit::fetch_orderbook("BTCUSDT", 25) {
        Ok(book) => {
            println!(
                "  Best bid: {:.2} ({:.6}), Best ask: {:.2} ({:.6})",
                book.bids[0].price, book.bids[0].quantity,
                book.asks[0].price, book.asks[0].quantity,
            );
            book
        }
        Err(e) => {
            eprintln!("Failed to fetch orderbook: {}. Using synthetic data.", e);
            create_synthetic_orderbook()
        }
    };

    println!("Fetching recent trades...");
    let trades = match bybit::fetch_recent_trades("BTCUSDT", 500) {
        Ok(t) => {
            println!("  Fetched {} trades", t.len());
            t
        }
        Err(e) => {
            eprintln!("Failed to fetch trades: {}. Using synthetic data.", e);
            create_synthetic_trades()
        }
    };

    // ─── Compute all features ──────────────────────────────────────────────
    let calc = MicrostructureCalculator::new()
        .with_depth_levels(10)
        .with_vpin_params(0.1, 50);

    let interval_ms = 1000; // 1-second intervals for Kyle's lambda and Amihud
    let features = calc.compute_all(&orderbook, &trades, interval_ms);

    println!("\n--- Computed Microstructure Features ---");
    println!("  Midprice:             {:.2}", features.midprice);
    println!("  Absolute Spread:      {:.4}", features.absolute_spread);
    println!("  Relative Spread:      {:.6} ({:.4} bps)", features.relative_spread, features.relative_spread * 10000.0);
    println!("  Effective Spread:     {:.4}", features.effective_spread);
    println!("  Order Imbalance:      {:.4}", features.order_imbalance);
    println!("  Trade Imbalance:      {:.4}", features.trade_imbalance);
    println!("  Kyle's Lambda:        {:.8}", features.kyles_lambda);
    println!("  Amihud Illiquidity:   {:.8}", features.amihud_illiquidity);
    println!("  VPIN:                 {:.4}", features.vpin);

    // ─── Feature time series ───────────────────────────────────────────────
    println!("\n--- Feature Time Series (rolling over trade windows) ---");
    let window_size = 50;
    if trades.len() >= window_size * 3 {
        println!("{:>8} {:>12} {:>12} {:>12}", "Window", "TradeImbal", "VPIN", "Amihud");
        for i in 0..((trades.len() - window_size) / window_size).min(10) {
            let start = i * window_size;
            let end = start + window_size;
            let window = &trades[start..end];

            let ti = calc.trade_imbalance(window);
            let vpin = MicrostructureCalculator::new()
                .with_vpin_params(0.01, 20)
                .vpin(window);
            let amihud = calc.amihud_from_trades(window, interval_ms);

            println!("{:>8} {:>12.4} {:>12.4} {:>12.8}", i, ti, vpin, amihud);
        }
    } else {
        println!("  Not enough trades for time series (need {}+, got {})", window_size * 3, trades.len());
    }

    // ─── Correlation with future returns ───────────────────────────────────
    println!("\n--- Order Imbalance vs Future Returns ---");
    let lookback = 20;
    let lookahead = 10;
    if trades.len() >= lookback + lookahead + 10 {
        let mut imbalances = Vec::new();
        let mut future_returns = Vec::new();

        for i in lookback..(trades.len() - lookahead) {
            let window = &trades[(i - lookback)..i];
            let ti = calc.trade_imbalance(window);
            let current_price = trades[i].price;
            let future_price = trades[i + lookahead].price;
            let ret = if current_price > 0.0 {
                (future_price - current_price) / current_price
            } else {
                0.0
            };
            imbalances.push(ti);
            future_returns.push(ret);
        }

        let corr = pearson_correlation(&imbalances, &future_returns);
        println!("  Pearson correlation (trade imbalance, {}t-ahead return): {:.4}", lookahead, corr);
        println!("  Number of observations: {}", imbalances.len());

        // Show a few sample points
        println!("\n  Sample (imbalance -> future return):");
        for i in (0..imbalances.len()).step_by(imbalances.len() / 5) {
            println!(
                "    Imbalance: {:>8.4}  -> Return: {:>10.6}",
                imbalances[i], future_returns[i]
            );
        }
    } else {
        println!("  Not enough trades for correlation analysis");
    }

    println!("\n=== Done ===");
}

/// Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mean_x = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y = y[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    cov / denom
}

/// Create synthetic orderbook for offline testing.
fn create_synthetic_orderbook() -> microstructure_features::OrderBook {
    use microstructure_features::OrderBookLevel;

    let base_price = 50000.0;
    let mut bids = Vec::new();
    let mut asks = Vec::new();

    for i in 0..25 {
        let offset = (i as f64 + 1.0) * 0.5;
        bids.push(OrderBookLevel {
            price: base_price - offset,
            quantity: 0.1 + (i as f64) * 0.05,
        });
        asks.push(OrderBookLevel {
            price: base_price + offset,
            quantity: 0.1 + (i as f64) * 0.04,
        });
    }

    microstructure_features::OrderBook {
        bids,
        asks,
        timestamp: 0,
    }
}

/// Create synthetic trades for offline testing.
fn create_synthetic_trades() -> Vec<Trade> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut trades = Vec::new();
    let mut price = 50000.0;

    for i in 0..500 {
        let is_buy = rng.gen_bool(0.52); // slight buy bias
        let qty = 0.001 + rng.gen::<f64>() * 0.05;
        let impact = if is_buy { 0.5 } else { -0.5 };
        price += impact + rng.gen::<f64>() * 0.2 - 0.1;

        trades.push(Trade {
            price,
            quantity: qty,
            is_buyer_maker: !is_buy,
            timestamp: i * 200, // 200ms between trades
        });
    }
    trades
}
