//! Example: Full Trading Pipeline with Conformal Prediction
//!
//! This comprehensive example demonstrates the complete workflow:
//! 1. Fetch data from Bybit
//! 2. Generate features
//! 3. Train conformal predictor
//! 4. Run trading strategy backtest
//! 5. Evaluate with multiple metrics
//!
//! Run with: cargo run --example full_pipeline

use conformal_prediction_trading::{
    api::bybit::{BybitClient, Interval},
    conformal::{model::LinearModel, split::SplitConformalPredictor, PredictionInterval},
    data::{features::FeatureEngineering, processor::DataProcessor},
    metrics::{coverage::CoverageMetrics, trading::TradingMetrics},
    strategy::{
        sizing::PositionSizer,
        trading::{Backtester, ConformalTradingStrategy, TradeResult},
    },
};
use ndarray::Array2;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     CONFORMAL PREDICTION TRADING PIPELINE                     ║");
    println!("║     Trade Only When Confident!                                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════
    // Step 1: Data Fetching
    // ═══════════════════════════════════════════════════════════════
    println!("Step 1: Fetching Data");
    println!("─────────────────────────────────────────────────────────────────");

    let client = BybitClient::new();

    // Fetch data for multiple symbols
    let symbols = vec!["BTCUSDT", "ETHUSDT"];
    let mut all_results = Vec::new();

    for symbol in &symbols {
        println!("\nFetching {} data...", symbol);
        let klines = client.get_klines(symbol, Interval::Hour4, Some(500), None, None)?;
        println!("  Received {} candles", klines.len());

        if klines.len() < 100 {
            println!("  Skipping {} - insufficient data", symbol);
            continue;
        }

        // Run pipeline for this symbol
        let result = run_pipeline_for_symbol(symbol, &klines)?;
        all_results.push((symbol.to_string(), result));
    }

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                       FINAL SUMMARY                           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Symbol", "Trades", "Win Rate", "Sharpe", "Coverage", "Total PnL"
    );
    println!("{}", "─".repeat(74));

    for (symbol, result) in &all_results {
        println!(
            "{:>10} {:>12} {:>11.1}% {:>12.2} {:>11.1}% {:>11.4}%",
            symbol,
            result.n_trades,
            result.win_rate * 100.0,
            result.sharpe,
            result.coverage * 100.0,
            result.total_pnl * 100.0
        );
    }

    println!("\n{}", "═".repeat(74));
    println!("Pipeline complete! See above for detailed results.");

    Ok(())
}

struct PipelineResult {
    n_trades: usize,
    win_rate: f64,
    sharpe: f64,
    coverage: f64,
    total_pnl: f64,
}

fn run_pipeline_for_symbol(
    symbol: &str,
    klines: &[conformal_prediction_trading::api::bybit::Kline],
) -> anyhow::Result<PipelineResult> {
    // ═══════════════════════════════════════════════════════════════
    // Step 2: Feature Engineering
    // ═══════════════════════════════════════════════════════════════
    println!("\nStep 2: Feature Engineering for {}", symbol);
    println!("─────────────────────────────────────────────────────────────────");

    let (features, feature_names) = FeatureEngineering::generate_features(klines);
    let targets = FeatureEngineering::create_returns(klines, 1);

    // Skip warm-up period for indicators
    let valid_start = 30;
    let features = features.slice(ndarray::s![valid_start.., ..]).to_owned();
    let targets: Vec<f64> = targets[valid_start..].to_vec();

    println!("  Features: {:?}", &feature_names[..5]);
    println!(
        "  Data shape: {} samples x {} features",
        features.nrows(),
        features.ncols()
    );

    // Clean data (remove any NaN values)
    let (features, targets) = DataProcessor::remove_invalid_rows(&features, &targets);
    println!("  After cleaning: {} samples", features.nrows());

    // ═══════════════════════════════════════════════════════════════
    // Step 3: Train/Calibration/Test Split
    // ═══════════════════════════════════════════════════════════════
    println!("\nStep 3: Data Splitting");
    println!("─────────────────────────────────────────────────────────────────");

    let ((x_train, y_train), (x_calib, y_calib), (x_test, y_test)) =
        DataProcessor::train_calib_test_split(&features, &targets, 0.6, 0.2);

    println!(
        "  Train: {}, Calib: {}, Test: {}",
        x_train.nrows(),
        x_calib.nrows(),
        x_test.nrows()
    );

    // Standardize features
    let (x_train_std, means, stds) = DataProcessor::standardize(&x_train);
    let x_calib_std = DataProcessor::apply_standardization(&x_calib, &means, &stds);
    let x_test_std = DataProcessor::apply_standardization(&x_test, &means, &stds);

    // ═══════════════════════════════════════════════════════════════
    // Step 4: Train Conformal Predictor
    // ═══════════════════════════════════════════════════════════════
    println!("\nStep 4: Training Conformal Predictor (90% coverage)");
    println!("─────────────────────────────────────────────────────────────────");

    let model = LinearModel::new(true);
    let mut cp = SplitConformalPredictor::new(model, 0.1);
    cp.fit(&x_train_std, &y_train, &x_calib_std, &y_calib);

    let q_hat = cp.quantile().unwrap_or(0.0);
    println!("  Calibration quantile (q_hat): {:.6}", q_hat);
    println!("  Interval width: {:.4}%", cp.interval_width() * 100.0);

    // Verify calibration coverage
    let calib_coverage = cp.coverage(&x_calib_std, &y_calib);
    println!("  Calibration coverage: {:.1}%", calib_coverage * 100.0);

    // ═══════════════════════════════════════════════════════════════
    // Step 5: Generate Predictions
    // ═══════════════════════════════════════════════════════════════
    println!("\nStep 5: Generating Predictions");
    println!("─────────────────────────────────────────────────────────────────");

    let intervals = cp.predict(&x_test_std);
    let coverage_metrics = CoverageMetrics::calculate(&intervals, &y_test, 0.1);

    println!("  Test coverage: {:.1}%", coverage_metrics.coverage * 100.0);
    println!("  Avg interval width: {:.4}%", coverage_metrics.avg_width * 100.0);
    println!("  Winkler score: {:.6}", coverage_metrics.winkler_score);

    // ═══════════════════════════════════════════════════════════════
    // Step 6: Run Trading Strategy
    // ═══════════════════════════════════════════════════════════════
    println!("\nStep 6: Running Trading Strategy");
    println!("─────────────────────────────────────────────────────────────────");

    // Create strategy with inverse position sizing
    let sizer = PositionSizer::inverse().with_baseline_width(0.02);
    let strategy = ConformalTradingStrategy::with_sizer(0.015, 0.002, sizer);

    let mut backtester = Backtester::new(strategy);
    backtester.run(&intervals, &y_test);

    let n_trades = backtester.n_trades();
    let trade_freq = backtester.trade_frequency();
    let win_rate = backtester.win_rate();
    let sharpe = backtester.sharpe_ratio();
    let total_pnl = backtester.total_pnl();
    let coverage_on_trades = backtester.coverage();

    println!("  Trades: {} ({:.1}% of periods)", n_trades, trade_freq * 100.0);
    println!("  Win rate: {:.1}%", win_rate * 100.0);
    println!("  Sharpe ratio: {:.2}", sharpe);
    println!("  Total PnL: {:.4}%", total_pnl * 100.0);
    println!("  Coverage on trades: {:.1}%", coverage_on_trades * 100.0);

    // ═══════════════════════════════════════════════════════════════
    // Step 7: Comparison with Baseline
    // ═══════════════════════════════════════════════════════════════
    println!("\nStep 7: Comparison with Baseline");
    println!("─────────────────────────────────────────────────────────────────");

    // Baseline 1: Always long
    let baseline_long_pnl: f64 = y_test.iter().sum();
    println!("  Always Long PnL: {:.4}%", baseline_long_pnl * 100.0);

    // Baseline 2: Trade every period based on point prediction
    let baseline_preds = intervals.iter().map(|i| i.prediction).collect::<Vec<_>>();
    let baseline_signal_pnl: f64 = baseline_preds
        .iter()
        .zip(y_test.iter())
        .map(|(&pred, &actual)| {
            let dir = if pred > 0.0 { 1.0 } else { -1.0 };
            dir * actual
        })
        .sum();
    println!("  Signal-based (no CP) PnL: {:.4}%", baseline_signal_pnl * 100.0);

    // Conformal strategy
    println!("  Conformal Strategy PnL: {:.4}%", total_pnl * 100.0);
    println!(
        "  Improvement vs Always Long: {:.4}%",
        (total_pnl - baseline_long_pnl) * 100.0
    );
    println!(
        "  Improvement vs Signal-based: {:.4}%",
        (total_pnl - baseline_signal_pnl) * 100.0
    );

    // ═══════════════════════════════════════════════════════════════
    // Step 8: Trade Analysis
    // ═══════════════════════════════════════════════════════════════
    println!("\nStep 8: Trade Analysis");
    println!("─────────────────────────────────────────────────────────────────");

    let results = backtester.results();
    let trades: Vec<&TradeResult> = results.iter().filter(|r| r.signal.trade).collect();

    if !trades.is_empty() {
        // Long vs Short performance
        let long_trades: Vec<_> = trades.iter().filter(|t| t.signal.direction > 0).collect();
        let short_trades: Vec<_> = trades.iter().filter(|t| t.signal.direction < 0).collect();

        let long_pnl: f64 = long_trades.iter().map(|t| t.pnl).sum();
        let short_pnl: f64 = short_trades.iter().map(|t| t.pnl).sum();

        println!(
            "  Long trades: {} (PnL: {:.4}%)",
            long_trades.len(),
            long_pnl * 100.0
        );
        println!(
            "  Short trades: {} (PnL: {:.4}%)",
            short_trades.len(),
            short_pnl * 100.0
        );

        // Best and worst trades
        let mut sorted_trades = trades.clone();
        sorted_trades.sort_by(|a, b| {
            b.pnl.partial_cmp(&a.pnl).unwrap_or(std::cmp::Ordering::Equal)
        });

        if sorted_trades.len() >= 3 {
            println!("\n  Best trades:");
            for t in sorted_trades.iter().take(3) {
                let dir = if t.signal.direction > 0 { "LONG" } else { "SHORT" };
                println!(
                    "    {} size={:.2} actual={:.4}% pnl={:.4}%",
                    dir,
                    t.signal.size,
                    t.actual_return * 100.0,
                    t.pnl * 100.0
                );
            }

            println!("\n  Worst trades:");
            for t in sorted_trades.iter().rev().take(3) {
                let dir = if t.signal.direction > 0 { "LONG" } else { "SHORT" };
                println!(
                    "    {} size={:.2} actual={:.4}% pnl={:.4}%",
                    dir,
                    t.signal.size,
                    t.actual_return * 100.0,
                    t.pnl * 100.0
                );
            }
        }
    }

    Ok(PipelineResult {
        n_trades,
        win_rate,
        sharpe,
        coverage: coverage_metrics.coverage,
        total_pnl,
    })
}
