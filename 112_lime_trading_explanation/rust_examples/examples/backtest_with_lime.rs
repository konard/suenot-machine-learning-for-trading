//! Example: Backtesting with LIME-based signal filtering
//!
//! This example demonstrates how to use LIME explanations to filter
//! trading signals and compare performance with and without filtering.

use lime_trading::{
    api::bybit::BybitClient,
    data::processor::DataProcessor,
    data::features::FeatureEngineering,
    explainer::lime::LimeExplainer,
    models::random_forest::RandomForestClassifier,
};
use ndarray::Array2;

/// Simple backtesting results
struct BacktestResult {
    total_return: f64,
    num_trades: usize,
    win_rate: f64,
    avg_return_per_trade: f64,
}

impl std::fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Total Return: {:.2}%, Trades: {}, Win Rate: {:.1}%, Avg Return/Trade: {:.4}%",
            self.total_return * 100.0,
            self.num_trades,
            self.win_rate * 100.0,
            self.avg_return_per_trade * 100.0
        )
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("Backtesting with LIME Filtering");
    println!("================================\n");

    // 1. Fetch data
    println!("1. Fetching market data...");
    let client = BybitClient::new();
    let candles = client.fetch_klines("BTCUSDT", "60", 1000)?;
    println!("   Fetched {} candles\n", candles.len());

    // 2. Prepare features
    println!("2. Preparing features...");
    let processor = DataProcessor::new();
    let (features, targets, valid_indices) = processor.prepare_features(&candles);

    if features.nrows() < 200 {
        println!("   Not enough data. Need at least 200 samples.");
        return Ok(());
    }

    println!("   Features: {} samples x {} features\n", features.nrows(), features.ncols());

    // 3. Split data
    let train_ratio = 0.7;
    let (x_train, x_test, y_train, y_test) =
        FeatureEngineering::train_test_split(&features, &targets, train_ratio);

    println!("3. Data split:");
    println!("   Training: {} samples", x_train.nrows());
    println!("   Testing: {} samples\n", x_test.nrows());

    // 4. Train model
    println!("4. Training Random Forest...");
    let mut model = RandomForestClassifier::new(50, 8, 10);
    model.fit(&x_train, &y_train);
    println!("   Model trained!\n");

    // Get prices for test period
    let train_size = (valid_indices.len() as f64 * train_ratio) as usize;
    let test_indices: Vec<usize> = valid_indices[train_size..].to_vec();
    let test_prices: Vec<f64> = test_indices.iter().map(|&i| candles[i].close).collect();

    // 5. Create LIME explainer
    println!("5. Creating LIME explainer...");
    let model_for_lime = {
        let mut m = RandomForestClassifier::new(50, 8, 10);
        m.fit(&x_train, &y_train);
        m
    };
    let predict_fn = move |x: &Array2<f64>| model_for_lime.predict_proba(x);
    let explainer = LimeExplainer::new(
        predict_fn,
        processor.feature_names.clone(),
        Some(&x_train),
        Some(500),
    );
    println!("   LIME explainer ready!\n");

    // 6. Backtest without LIME filtering
    println!("6. Running backtest WITHOUT LIME filtering...");
    let result_without = run_backtest(
        &model,
        &x_test,
        &y_test,
        &test_prices,
        None, // No explainer = no filtering
        0.0,
    );
    println!("   {}\n", result_without);

    // 7. Backtest with LIME filtering
    println!("7. Running backtest WITH LIME filtering (R² > 0.5)...");

    // Train another model for the explainer in backtest
    let model_for_backtest = {
        let mut m = RandomForestClassifier::new(50, 8, 10);
        m.fit(&x_train, &y_train);
        m
    };
    let predict_fn_backtest = move |x: &Array2<f64>| model_for_backtest.predict_proba(x);
    let explainer_backtest = LimeExplainer::new(
        predict_fn_backtest,
        processor.feature_names.clone(),
        Some(&x_train),
        Some(500),
    );

    let result_with = run_backtest(
        &model,
        &x_test,
        &y_test,
        &test_prices,
        Some(&explainer_backtest),
        0.5,  // Minimum R² threshold
    );
    println!("   {}\n", result_with);

    // 8. Compare results
    println!("8. Comparison:");
    println!("   {:30} {:>20} {:>20}", "Metric", "Without LIME", "With LIME");
    println!("   {}", "-".repeat(70));
    println!(
        "   {:30} {:>19.2}% {:>19.2}%",
        "Total Return",
        result_without.total_return * 100.0,
        result_with.total_return * 100.0
    );
    println!(
        "   {:30} {:>20} {:>20}",
        "Number of Trades",
        result_without.num_trades,
        result_with.num_trades
    );
    println!(
        "   {:30} {:>19.1}% {:>19.1}%",
        "Win Rate",
        result_without.win_rate * 100.0,
        result_with.win_rate * 100.0
    );
    println!(
        "   {:30} {:>19.4}% {:>19.4}%",
        "Avg Return per Trade",
        result_without.avg_return_per_trade * 100.0,
        result_with.avg_return_per_trade * 100.0
    );

    println!("\nDone!");
    Ok(())
}

fn run_backtest(
    model: &RandomForestClassifier,
    x_test: &ndarray::Array2<f64>,
    y_test: &ndarray::Array1<f64>,
    prices: &[f64],
    explainer: Option<&LimeExplainer>,
    min_r2: f64,
) -> BacktestResult {
    let mut returns: Vec<f64> = Vec::new();
    let mut trades = 0;
    let mut wins = 0;

    let n_samples = x_test.nrows();
    let min_prob = 0.55;

    for i in 0..(n_samples - 1) {
        let instance = x_test.row(i).to_owned();
        let proba = model.predict_proba(&instance.clone().insert_axis(ndarray::Axis(0)));
        let prob_up = proba[[0, 1]];

        // Check if we should trade
        let mut should_trade = prob_up > min_prob || prob_up < (1.0 - min_prob);

        // Apply LIME filtering if explainer is provided
        if should_trade {
            if let Some(exp) = explainer {
                let explanation = exp.explain(&instance, 1);
                if explanation.local_model_score < min_r2 {
                    should_trade = false;
                }
            }
        }

        if should_trade {
            trades += 1;

            // Determine direction
            let direction: f64 = if prob_up > 0.5 { 1.0 } else { -1.0 };

            // Calculate return
            let actual_return = (prices[i + 1] / prices[i]) - 1.0;
            let trade_return = direction * actual_return;

            returns.push(trade_return);

            if trade_return > 0.0 {
                wins += 1;
            }
        }
    }

    let total_return = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;
    let win_rate = if trades > 0 {
        wins as f64 / trades as f64
    } else {
        0.0
    };
    let avg_return = if trades > 0 {
        returns.iter().sum::<f64>() / trades as f64
    } else {
        0.0
    };

    BacktestResult {
        total_return,
        num_trades: trades,
        win_rate,
        avg_return_per_trade: avg_return,
    }
}
