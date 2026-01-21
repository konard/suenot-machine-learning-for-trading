//! Trading Strategy Example
//!
//! This example demonstrates using CML for a complete trading strategy:
//! - Creating synthetic market data
//! - Running a backtest
//! - Analyzing performance metrics
//!
//! Run with: cargo run --example trading_strategy

use continual_meta_learning::prelude::*;
use continual_meta_learning::backtest::engine::print_report;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== CML Trading Strategy Example ===\n");

    // Create synthetic market data
    println!("Generating synthetic market data...\n");
    let klines = generate_synthetic_klines(500);
    println!("Generated {} candles\n", klines.len());

    // Create trading features
    let feature_config = FeatureConfig {
        sma_window: 20,
        ema_short: 12,
        ema_long: 26,
        rsi_period: 14,
        vol_window: 20,
        momentum_period: 10,
        normalize: true,
    };

    let features = TradingFeatures::from_klines(&klines, feature_config);
    println!("Computed {} feature vectors", features.len());
    println!("Feature dimension: {}\n", features.feature_dim());

    // Display regime distribution
    print_regime_distribution(&features);

    // Create CML learner
    let cml_config = CMLConfig {
        input_size: features.feature_dim(),
        hidden_size: 32,
        output_size: 1,
        inner_lr: 0.01,
        outer_lr: 0.001,
        inner_steps: 5,
        memory_size: 200,
        ewc_lambda: 500.0,
    };

    let learner = ContinualMetaLearner::new(cml_config);

    // Create trading strategy
    let strategy_config = StrategyConfig {
        max_position: 1.0,
        min_position: 0.1,
        position_multiplier: 1.0,
        allow_short: true,
        warmup_samples: 30,
        adaptation_interval: 50,
        risk_per_trade: 0.02,
        stop_loss: 0.05,
        take_profit: 0.10,
        ..Default::default()
    };

    let mut strategy = CMLStrategy::new(learner, strategy_config);

    // Create backtester
    let backtest_config = BacktestConfig {
        initial_capital: 10000.0,
        trading_fee: 0.001,  // 0.1% fee
        slippage: 0.0005,    // 0.05% slippage
        reinvest_profits: true,
        risk_free_rate: 0.02,
        trading_days_per_year: 365.0, // Crypto trades 24/7
        log_trades: true,
    };

    let mut backtester = Backtester::new(backtest_config);

    // Run backtest
    println!("\nRunning backtest...\n");
    let result = backtester.run(&mut strategy, &klines, &features);

    // Print detailed report
    print_report(&result);

    // Additional analysis
    println!("=== Additional Analysis ===\n");

    // Strategy statistics
    let strategy_stats = strategy.stats();
    println!("Strategy Statistics:");
    println!("  Total trades: {}", strategy_stats.total_trades);
    println!("  Adaptations: {}", strategy_stats.adaptations);
    println!("  Final regime: {:?}", strategy_stats.current_regime);
    println!("  Buy signals: {}", strategy_stats.buy_signals);
    println!("  Sell signals: {}", strategy_stats.sell_signals);

    // Equity curve analysis
    println!("\nEquity Curve:");
    let equity = &result.equity_curve;
    if equity.len() >= 5 {
        println!("  Start: ${:.2}", equity[0]);
        println!("  25%:   ${:.2}", equity[equity.len() / 4]);
        println!("  50%:   ${:.2}", equity[equity.len() / 2]);
        println!("  75%:   ${:.2}", equity[3 * equity.len() / 4]);
        println!("  End:   ${:.2}", equity[equity.len() - 1]);
    }

    // Trade analysis
    if !result.trades.is_empty() {
        println!("\nTrade Analysis:");
        let winning: Vec<_> = result.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing: Vec<_> = result.trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let avg_win = if !winning.is_empty() {
            winning.iter().map(|t| t.pnl).sum::<f64>() / winning.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing.is_empty() {
            losing.iter().map(|t| t.pnl).sum::<f64>() / losing.len() as f64
        } else {
            0.0
        };

        println!("  Winning trades: {}", winning.len());
        println!("  Losing trades: {}", losing.len());
        println!("  Average win: ${:.2}", avg_win);
        println!("  Average loss: ${:.2}", avg_loss);

        if !losing.is_empty() && avg_loss != 0.0 {
            println!("  Risk/Reward ratio: {:.2}", avg_win / avg_loss.abs());
        }

        // Show sample trades
        println!("\nSample Trades (first 5):");
        for trade in result.trades.iter().take(5) {
            println!(
                "  #{}: {:?} | Entry: ${:.2} | Exit: ${:.2} | PnL: ${:.2} ({:.2}%)",
                trade.index,
                trade.regime,
                trade.entry_price,
                trade.exit_price.unwrap_or(0.0),
                trade.pnl,
                trade.return_pct * 100.0
            );
        }
    }

    // Compare with buy-and-hold
    println!("\n=== Comparison with Buy-and-Hold ===\n");
    let bh_return = (klines.last().unwrap().close - klines.first().unwrap().close)
        / klines.first().unwrap().close;

    println!("CML Strategy Return: {:.2}%", result.total_return * 100.0);
    println!("Buy-and-Hold Return: {:.2}%", bh_return * 100.0);
    println!(
        "Alpha (excess return): {:.2}%",
        (result.total_return - bh_return) * 100.0
    );

    println!("\nDone!");
}

/// Generate synthetic market data with different regimes.
fn generate_synthetic_klines(num_candles: usize) -> Vec<Kline> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut klines = Vec::with_capacity(num_candles);
    let mut price = 100.0;
    let mut timestamp = 1700000000000i64; // Starting timestamp

    // Define regime periods
    let regimes = [
        (MarketRegime::Bull, 0.0003, 0.01),         // Positive drift, low vol
        (MarketRegime::HighVolatility, 0.0, 0.03),  // No drift, high vol
        (MarketRegime::Bear, -0.0003, 0.015),       // Negative drift, moderate vol
        (MarketRegime::LowVolatility, 0.0001, 0.005), // Slight drift, very low vol
        (MarketRegime::Sideways, 0.0, 0.008),      // No drift, low vol
    ];

    let regime_duration = num_candles / regimes.len();

    for i in 0..num_candles {
        // Determine current regime
        let regime_idx = (i / regime_duration).min(regimes.len() - 1);
        let (_, drift, volatility) = regimes[regime_idx];

        // Generate price movement
        let return_val = drift + rng.gen_range(-volatility..volatility);
        let new_price = price * (1.0 + return_val);

        // Generate OHLC
        let open = price;
        let close = new_price;
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..volatility * 0.5));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..volatility * 0.5));

        // Generate volume (higher in volatile periods)
        let base_volume = 1000.0;
        let volume = base_volume * (1.0 + volatility * 50.0) * rng.gen_range(0.5..1.5);

        klines.push(Kline {
            start_time: timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: volume * (open + close) / 2.0,
        });

        price = new_price;
        timestamp += 3600000; // 1 hour
    }

    klines
}

/// Print the distribution of market regimes in the data.
fn print_regime_distribution(features: &TradingFeatures) {
    let mut counts: std::collections::HashMap<MarketRegime, usize> = std::collections::HashMap::new();

    for i in 0..features.len() {
        if let Some(regime) = features.get_regime(i) {
            *counts.entry(regime).or_insert(0) += 1;
        }
    }

    println!("Market Regime Distribution:");
    for (regime, count) in &counts {
        let pct = (*count as f64 / features.len() as f64) * 100.0;
        println!("  {:?}: {} ({:.1}%)", regime, count, pct);
    }
    println!();
}
