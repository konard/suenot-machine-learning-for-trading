//! IC calculation experiments
//!
//! Test Information Coefficient calculation with various scenarios.
//!
//! Run with: cargo run --example test_ic_calculation

use llm_factor_discovery::evaluation::{IcCalculator, MetricsCalculator};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== IC Calculation Experiments ===\n");

    let ic_calc = IcCalculator::new();

    // Experiment 1: Perfect correlation
    info!("Experiment 1: Perfect Positive Correlation");
    let factor1: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let returns1: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();

    if let Some(ic) = ic_calc.calculate_ic(&factor1, &returns1) {
        info!("  Pearson IC: {:.6} (expected: 1.0)", ic);
    }
    if let Some(rank_ic) = ic_calc.calculate_rank_ic(&factor1, &returns1) {
        info!("  Spearman Rank IC: {:.6} (expected: 1.0)", rank_ic);
    }
    info!("");

    // Experiment 2: Perfect negative correlation
    info!("Experiment 2: Perfect Negative Correlation");
    let factor2: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let returns2: Vec<f64> = (0..100).map(|i| (100 - i) as f64 * 0.01).collect();

    if let Some(ic) = ic_calc.calculate_ic(&factor2, &returns2) {
        info!("  Pearson IC: {:.6} (expected: -1.0)", ic);
    }
    if let Some(rank_ic) = ic_calc.calculate_rank_ic(&factor2, &returns2) {
        info!("  Spearman Rank IC: {:.6} (expected: -1.0)", rank_ic);
    }
    info!("");

    // Experiment 3: Weak signal with noise
    info!("Experiment 3: Weak Signal with Noise");
    let n = 1000;
    let signal_strength = 0.05; // 5% signal

    let factor3: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.1).sin())
        .collect();

    let returns3: Vec<f64> = factor3
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let noise = ((i as f64 * 1.7).sin() + (i as f64 * 2.3).cos()) * 0.5;
            f * signal_strength + noise * (1.0 - signal_strength)
        })
        .collect();

    if let Some(ic) = ic_calc.calculate_ic(&factor3, &returns3) {
        info!("  Pearson IC: {:.4} (signal strength: {:.0}%)", ic, signal_strength * 100.0);
    }
    if let Some(rank_ic) = ic_calc.calculate_rank_ic(&factor3, &returns3) {
        info!("  Rank IC: {:.4}", rank_ic);
    }
    info!("");

    // Experiment 4: Rolling IC stability
    info!("Experiment 4: Rolling IC Stability");

    let rolling_ic = ic_calc.rolling_ic(&factor3, &returns3, 50);
    let ic_stats = ic_calc.ic_statistics(&rolling_ic);

    info!("  Rolling IC Statistics (window=50):");
    info!("    Count: {}", ic_stats.count);
    info!("    Mean IC: {:.4}", ic_stats.mean);
    info!("    Std IC: {:.4}", ic_stats.std);
    info!("    IC IR: {:.4}", ic_stats.ir);
    info!("    Hit Rate: {:.1}%", ic_stats.hit_rate * 100.0);
    info!("    T-stat: {:.2}", ic_stats.t_stat);
    info!("    Significant: {}", ic_stats.is_significant());
    info!("");

    // Experiment 5: Forward return horizons
    info!("Experiment 5: IC Decay with Horizon");

    let prices: Vec<f64> = (0..500)
        .map(|i| 100.0 * (1.0 + 0.001 * (i as f64 * 0.05).sin()))
        .scan(100.0, |price, change| {
            *price *= change;
            Some(*price)
        })
        .collect();

    // Simple momentum factor
    let momentum_factor: Vec<f64> = prices
        .windows(6)
        .map(|w| (w[5] - w[0]) / w[0])
        .collect();

    for horizon in [1, 5, 10, 20, 40] {
        let forward_returns = IcCalculator::calculate_forward_returns(&prices, horizon);

        let min_len = momentum_factor.len().min(forward_returns.len());
        if min_len < 50 {
            continue;
        }

        let fv = &momentum_factor[..min_len];
        let fr = &forward_returns[..min_len];

        if let Some(ic) = ic_calc.calculate_ic(fv, fr) {
            info!("  {}-period horizon: IC = {:.4}", horizon, ic);
        }
    }
    info!("");

    // Experiment 6: Different IC metrics comparison
    info!("Experiment 6: Pearson vs Spearman IC");

    // Create data with outliers
    let mut factor_outliers: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mut returns_outliers: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();

    // Add outliers
    factor_outliers[50] = 1000.0;
    returns_outliers[50] = 10.0;

    let pearson_ic = ic_calc.calculate_ic(&factor_outliers, &returns_outliers);
    let spearman_ic = ic_calc.calculate_rank_ic(&factor_outliers, &returns_outliers);

    info!("  With outliers:");
    info!("    Pearson IC: {:.4}", pearson_ic.unwrap_or(0.0));
    info!("    Spearman IC: {:.4}", spearman_ic.unwrap_or(0.0));
    info!("  Spearman is more robust to outliers");
    info!("");

    // Experiment 7: Performance metrics
    info!("Experiment 7: Strategy Performance Metrics");

    let metrics_calc = MetricsCalculator::daily();

    // Simulated daily returns
    let daily_returns: Vec<f64> = (0..252)
        .map(|i| {
            let trend = 0.0003; // Small positive drift
            let vol = 0.02;    // 2% daily vol
            let noise = ((i as f64 * 0.7).sin() + (i as f64 * 1.1).cos()) * 0.5;
            trend + vol * noise
        })
        .collect();

    let perf = metrics_calc.calculate(&daily_returns);

    info!("  Performance Metrics (1 year simulation):");
    info!("    Total Return: {:.2}%", perf.total_return * 100.0);
    info!("    Annual Return: {:.2}%", perf.annual_return * 100.0);
    info!("    Annual Volatility: {:.2}%", perf.annual_volatility * 100.0);
    info!("    Sharpe Ratio: {:.2}", perf.sharpe_ratio);
    info!("    Sortino Ratio: {:.2}", perf.sortino_ratio);
    info!("    Max Drawdown: {:.2}%", perf.max_drawdown * 100.0);
    info!("    Calmar Ratio: {:.2}", perf.calmar_ratio);
    info!("    Win Rate: {:.1}%", perf.win_rate * 100.0);
    info!("    Profit Factor: {:.2}", perf.profit_factor);

    info!("\n=== IC Calculation Experiments Complete ===");

    Ok(())
}
