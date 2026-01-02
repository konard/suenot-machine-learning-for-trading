//! # Backtesting Example
//!
//! Demonstrates backtesting a score matching trading strategy.
//!
//! Run with: cargo run --example backtest

use chrono::{Duration, TimeZone, Utc};
use score_matching_trading::{
    api::BybitClient,
    backtest::{BacktestConfig, Backtester},
    score::MultiScaleScoreNetwork,
    trading::{ScoreMatchingTrader, TraderConfig},
    utils::Candle,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Score Matching Backtest Demo ===\n");

    // 1. Generate or fetch historical data
    println!("1. Preparing Historical Data...");

    // Try to fetch from Bybit, fall back to synthetic data
    let candles = match fetch_bybit_data().await {
        Ok(data) if data.len() >= 500 => {
            println!("   Fetched {} candles from Bybit", data.len());
            data
        }
        _ => {
            println!("   Using synthetic data for demonstration");
            generate_synthetic_price_data(1000)
        }
    };

    println!("   Data range: {} to {}",
        candles.first().map(|c| c.timestamp.format("%Y-%m-%d").to_string()).unwrap_or_default(),
        candles.last().map(|c| c.timestamp.format("%Y-%m-%d").to_string()).unwrap_or_default()
    );
    println!("   Price range: ${:.2} to ${:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
    );

    // 2. Create score network
    println!("\n2. Creating Score Network...");
    let network = MultiScaleScoreNetwork::new(
        10,   // input_dim
        64,   // hidden_dim
        3,    // num_layers
        10,   // num_noise_levels
        0.01, // sigma_min
        1.0,  // sigma_max
    );
    println!("   Network created with {} parameters", network.num_parameters());

    // 3. Create trader with configuration
    println!("\n3. Configuring Trader...");
    let trader_config = TraderConfig {
        return_feature_idx: 0,
        confidence_threshold: 0.2,
        density_threshold: -10.0, // More permissive for demo
        denoise_steps: 10,
        lookback: 20,
    };
    let trader = ScoreMatchingTrader::with_config(network, trader_config);
    println!("   Confidence threshold: {}", trader.config().confidence_threshold);
    println!("   Lookback period: {}", trader.config().lookback);

    // 4. Configure backtest
    println!("\n4. Configuring Backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: 10000.0,
        max_position: 1.0,
        trading_cost: 0.001, // 0.1%
        warmup: 100,
        allow_short: true,
    };
    println!("   Initial capital: ${:.2}", backtest_config.initial_capital);
    println!("   Max position: {:.0}%", backtest_config.max_position * 100.0);
    println!("   Trading cost: {:.2}%", backtest_config.trading_cost * 100.0);
    println!("   Allow shorting: {}", backtest_config.allow_short);

    // 5. Run backtest
    println!("\n5. Running Backtest...");
    let backtester = Backtester::new(trader, backtest_config);
    let result = backtester.run(&candles);

    // 6. Display results
    println!("\n6. Backtest Results:");
    result.metrics.print_summary();

    // 7. Show sample daily results
    if !result.daily_results.is_empty() {
        println!("7. Sample Daily Results (last 10 days):");
        println!("   {:>12} {:>10} {:>8} {:>10} {:>8} {:>12}",
            "Date", "Price", "Signal", "Confidence", "Position", "Cum. PnL");
        println!("   {}", "-".repeat(70));

        for day in result.daily_results.iter().rev().take(10).rev() {
            println!(
                "   {:>12} {:>10.2} {:>8.4} {:>10.2}% {:>8.2}% {:>12.2}",
                day.timestamp.format("%Y-%m-%d"),
                day.price,
                day.signal,
                day.confidence * 100.0,
                day.position * 100.0,
                day.cumulative_pnl
            );
        }
    }

    // 8. Show sample trades
    if !result.trades.is_empty() {
        println!("\n8. Sample Trades (last 5):");
        println!("   {:>12} {:>12} {:>10} {:>10} {:>8} {:>10}",
            "Entry", "Exit", "Entry $", "Exit $", "Dir", "Return");
        println!("   {}", "-".repeat(72));

        for trade in result.trades.iter().rev().take(5).rev() {
            println!(
                "   {:>12} {:>12} {:>10.2} {:>10.2} {:>8} {:>10.2}%",
                trade.entry_time.format("%m-%d %H:%M"),
                trade.exit_time.format("%m-%d %H:%M"),
                trade.entry_price,
                trade.exit_price,
                if trade.position > 0.0 { "LONG" } else { "SHORT" },
                trade.return_pct
            );
        }
    }

    // 9. Performance analysis
    println!("\n9. Performance Analysis:");

    let total_days = result.daily_results.len();
    let trading_days = result.daily_results.iter()
        .filter(|d| d.position != 0.0)
        .count();
    let in_dist_days = result.daily_results.iter()
        .filter(|d| d.in_distribution)
        .count();

    println!("   Total days analyzed: {}", total_days);
    println!("   Days with positions: {} ({:.1}%)",
        trading_days,
        trading_days as f64 / total_days as f64 * 100.0
    );
    println!("   Days in distribution: {} ({:.1}%)",
        in_dist_days,
        in_dist_days as f64 / total_days as f64 * 100.0
    );

    // Calculate monthly returns
    if !result.daily_results.is_empty() {
        println!("\n   Monthly PnL:");
        let mut current_month = result.daily_results[0].timestamp.format("%Y-%m").to_string();
        let mut month_start_pnl = 0.0;

        for day in &result.daily_results {
            let month = day.timestamp.format("%Y-%m").to_string();
            if month != current_month {
                let month_pnl = day.cumulative_pnl - month_start_pnl;
                println!("     {}: {:+.2}", current_month, month_pnl);
                current_month = month;
                month_start_pnl = day.cumulative_pnl;
            }
        }
        // Print last month
        if let Some(last) = result.daily_results.last() {
            let month_pnl = last.cumulative_pnl - month_start_pnl;
            println!("     {}: {:+.2}", current_month, month_pnl);
        }
    }

    // 10. Risk metrics
    println!("\n10. Risk Analysis:");

    // Calculate drawdown series
    let mut peak = 0.0_f64;
    let mut drawdowns: Vec<f64> = Vec::new();
    for day in &result.daily_results {
        peak = peak.max(day.cumulative_pnl);
        let drawdown = peak - day.cumulative_pnl;
        drawdowns.push(drawdown);
    }

    let avg_drawdown = if !drawdowns.is_empty() {
        drawdowns.iter().sum::<f64>() / drawdowns.len() as f64
    } else {
        0.0
    };

    println!("   Maximum Drawdown: ${:.2}", result.metrics.max_drawdown);
    println!("   Average Drawdown: ${:.2}", avg_drawdown);

    // Calmar ratio (annual return / max drawdown)
    let calmar = if result.metrics.max_drawdown > 0.0 {
        result.metrics.annualized_return * 10000.0 / result.metrics.max_drawdown
    } else {
        0.0
    };
    println!("   Calmar Ratio: {:.2}", calmar);

    // 11. Recommendations
    println!("\n11. Strategy Recommendations:");

    if result.metrics.sharpe_ratio > 1.0 {
        println!("   [OK] Sharpe ratio > 1.0 suggests reasonable risk-adjusted returns");
    } else {
        println!("   [!] Sharpe ratio < 1.0 - consider adjusting strategy parameters");
    }

    if result.metrics.win_rate > 0.5 {
        println!("   [OK] Win rate > 50% indicates more winners than losers");
    } else {
        println!("   [!] Win rate < 50% - need high reward:risk to compensate");
    }

    if result.metrics.in_distribution_ratio > 0.8 {
        println!("   [OK] High in-distribution ratio suggests model fits market well");
    } else {
        println!("   [!] Low in-distribution ratio - model may need retraining");
    }

    if result.metrics.profit_factor > 1.5 {
        println!("   [OK] Profit factor > 1.5 indicates profitable strategy");
    } else if result.metrics.profit_factor > 1.0 {
        println!("   [~] Profit factor 1.0-1.5 - marginally profitable");
    } else {
        println!("   [X] Profit factor < 1.0 - strategy is not profitable");
    }

    println!("\n=== Backtest Complete ===");

    Ok(())
}

/// Fetch historical data from Bybit
async fn fetch_bybit_data() -> anyhow::Result<Vec<Candle>> {
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "60", 1000).await?;
    Ok(candles)
}

/// Generate synthetic price data for testing
fn generate_synthetic_price_data(n_candles: usize) -> Vec<Candle> {
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n_candles);

    let mut price = 50000.0; // Starting price
    let drift = 0.0001; // Slight upward drift
    let volatility = 0.02; // Daily volatility

    let normal = Normal::new(0.0, volatility).unwrap();

    for i in 0..n_candles {
        // Generate return with drift and noise
        let return_pct = drift + normal.sample(&mut rng);

        // Add some mean reversion
        let mean_price = 50000.0;
        let reversion = (mean_price - price) / mean_price * 0.01;

        price = price * (1.0 + return_pct + reversion);
        price = price.max(1000.0); // Floor

        // Generate OHLC from close
        let high = price * (1.0 + rng.gen::<f64>() * 0.01);
        let low = price * (1.0 - rng.gen::<f64>() * 0.01);
        let open = low + (high - low) * rng.gen::<f64>();

        // Volume with some randomness
        let base_volume = 100.0;
        let volume = base_volume * (0.5 + rng.gen::<f64>());

        let timestamp = Utc::now() - Duration::hours((n_candles - i) as i64);

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close: price,
            volume,
        });
    }

    candles
}
