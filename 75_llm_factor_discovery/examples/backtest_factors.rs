//! Factor backtesting example
//!
//! Demonstrates how to backtest alpha factors using historical data.
//!
//! Run with: cargo run --example backtest_factors

use llm_factor_discovery::{
    data::PriceData,
    evaluation::{BacktestConfig, Backtester, IcCalculator, MetricsCalculator},
    parser::FactorExpr,
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== Factor Backtesting Example ===\n");

    // Generate simulated price data
    info!("Generating simulated price data...");
    let (prices, volumes) = generate_simulated_data(500);
    info!("  Generated {} data points\n", prices.len());

    // Define factors to test
    let factors = vec![
        ("momentum_5", "returns(close, 5)"),
        ("momentum_20", "returns(close, 20)"),
        ("volatility_10", "volatility(close, 10)"),
        ("volume_trend", "ts_mean(volume, 10)"),
        ("mean_reversion", "sub(ts_mean(close, 20), close)"),
    ];

    info!("Testing {} factors:\n", factors.len());

    // IC Calculator for evaluation
    let ic_calc = IcCalculator::new();
    let metrics_calc = MetricsCalculator::daily();

    // Calculate forward returns
    let forward_returns_1d = IcCalculator::calculate_forward_returns(&prices, 1);
    let forward_returns_5d = IcCalculator::calculate_forward_returns(&prices, 5);

    // Test each factor
    for (name, expression) in &factors {
        info!("Factor: {}", name);
        info!("  Expression: {}", expression);

        // Parse expression
        let expr = match FactorExpr::parse(expression) {
            Ok(e) => e,
            Err(err) => {
                info!("  Failed to parse: {}", err);
                continue;
            }
        };

        // Calculate factor values (simplified - just using returns for demo)
        let factor_values = calculate_simple_factor(&prices, &volumes, name);

        if factor_values.len() < 100 {
            info!("  Not enough data points for analysis");
            continue;
        }

        // Calculate IC with 1-day forward returns
        let min_len = factor_values.len().min(forward_returns_1d.len());
        let fv = &factor_values[..min_len];
        let fr = &forward_returns_1d[..min_len];

        if let Some(ic) = ic_calc.calculate_ic(fv, fr) {
            info!("  IC (1-day): {:.4}", ic);
        }

        if let Some(rank_ic) = ic_calc.calculate_rank_ic(fv, fr) {
            info!("  Rank IC (1-day): {:.4}", rank_ic);
        }

        // Calculate IC with 5-day forward returns
        let min_len_5d = factor_values.len().min(forward_returns_5d.len());
        if min_len_5d >= 50 {
            let fv_5d = &factor_values[..min_len_5d];
            let fr_5d = &forward_returns_5d[..min_len_5d];

            if let Some(ic_5d) = ic_calc.calculate_ic(fv_5d, fr_5d) {
                info!("  IC (5-day): {:.4}", ic_5d);
            }
        }

        // Calculate rolling IC statistics
        let rolling_ic = ic_calc.rolling_ic(fv, fr, 20);
        let ic_stats = ic_calc.ic_statistics(&rolling_ic);

        info!("  IC Mean: {:.4}", ic_stats.mean);
        info!("  IC Std: {:.4}", ic_stats.std);
        info!("  IC IR: {:.4}", ic_stats.ir);
        info!("  IC Hit Rate: {:.1}%", ic_stats.hit_rate * 100.0);
        info!("  T-stat: {:.2}", ic_stats.t_stat);

        // Check significance
        if ic_stats.is_significant() {
            info!("  Status: SIGNIFICANT (|t| > 2)");
        } else {
            info!("  Status: Not significant");
        }

        info!("");
    }

    // Example of using simulated backtest
    info!("Running full backtest simulation...\n");

    let config = BacktestConfig {
        universe: vec!["BTCUSDT".to_string()],
        forward_periods: vec![1, 5, 10],
        ..Default::default()
    };

    let backtester = Backtester::new(config);

    let expr = FactorExpr::parse("rank(ts_mean(returns(close, 5), 20))")?;
    let result = backtester.run(&expr).await?;

    info!("Backtest Results for: {}", expr);
    info!("  IC: {:.4}", result.ic);
    info!("  IC IR: {:.4}", result.ic_ir);
    info!("  Rank IC: {:.4}", result.rank_ic);
    info!("  Turnover: {:.2}", result.turnover);

    if let Some(returns) = result.strategy_returns {
        info!("  Strategy Returns: {:.2}%", returns * 100.0);
    }

    if let Some(sharpe) = result.sharpe_ratio {
        info!("  Sharpe Ratio: {:.2}", sharpe);
    }

    info!("\n  Quality Score: {:.4}", result.quality_score());

    if result.meets_threshold(0.02, 0.4) {
        info!("  PASSED quality threshold (IC >= 0.02, IR >= 0.4)");
    } else {
        info!("  Did not pass quality threshold");
    }

    info!("\n=== Backtest Complete ===");

    Ok(())
}

/// Generate simulated price and volume data with some structure
fn generate_simulated_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);

    let mut price = 100.0;

    for i in 0..n {
        // Trend component
        let trend = 0.0001 * (i as f64 / n as f64);

        // Mean reversion component
        let mean_reversion = (100.0 - price) * 0.01;

        // Random component with momentum
        let t = i as f64;
        let noise = 0.02 * ((t * 0.1).sin() + (t * 0.05).cos() * 0.5);

        let ret = trend + mean_reversion * 0.1 + noise;
        price *= 1.0 + ret;

        prices.push(price);

        // Volume with some correlation to absolute returns
        let base_volume = 1000.0;
        let volume = base_volume * (1.0 + ret.abs() * 10.0 + (t * 0.2).sin().abs());
        volumes.push(volume);
    }

    (prices, volumes)
}

/// Calculate simple factor values for demonstration
fn calculate_simple_factor(prices: &[f64], volumes: &[f64], factor_name: &str) -> Vec<f64> {
    match factor_name {
        "momentum_5" => {
            // 5-period returns
            if prices.len() < 6 {
                return vec![];
            }
            prices.windows(6)
                .map(|w| (w[5] - w[0]) / w[0])
                .collect()
        }
        "momentum_20" => {
            // 20-period returns
            if prices.len() < 21 {
                return vec![];
            }
            prices.windows(21)
                .map(|w| (w[20] - w[0]) / w[0])
                .collect()
        }
        "volatility_10" => {
            // 10-period volatility
            if prices.len() < 11 {
                return vec![];
            }

            let returns: Vec<f64> = prices.windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();

            returns.windows(10)
                .map(|w| {
                    let mean = w.iter().sum::<f64>() / 10.0;
                    let variance = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / 9.0;
                    variance.sqrt()
                })
                .collect()
        }
        "volume_trend" => {
            // 10-period moving average of volume
            if volumes.len() < 10 {
                return vec![];
            }
            volumes.windows(10)
                .map(|w| w.iter().sum::<f64>() / 10.0)
                .collect()
        }
        "mean_reversion" => {
            // Distance from 20-period moving average
            if prices.len() < 20 {
                return vec![];
            }
            let mut result = Vec::new();
            for i in 19..prices.len() {
                let ma = prices[i-19..=i].iter().sum::<f64>() / 20.0;
                result.push((ma - prices[i]) / prices[i]);
            }
            result
        }
        _ => vec![],
    }
}
