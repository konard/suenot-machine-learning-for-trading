//! Example: LIME explanations for trading predictions
//!
//! This example demonstrates how to use LIME to explain predictions
//! from a trading model trained on cryptocurrency data.

use lime_trading::{
    api::bybit::BybitClient,
    data::processor::DataProcessor,
    data::features::FeatureEngineering,
    explainer::lime::{LimeExplainer, LimeAnalyzer},
    models::random_forest::RandomForestClassifier,
};
use ndarray::Array2;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    println!("LIME for Trading Model Explanation");
    println!("===================================\n");

    // 1. Fetch data from Bybit
    println!("1. Fetching market data from Bybit...");
    let client = BybitClient::new();
    let candles = client.fetch_klines("BTCUSDT", "60", 500)?;
    println!("   Fetched {} candles\n", candles.len());

    // 2. Prepare features
    println!("2. Computing features...");
    let processor = DataProcessor::new();
    let (features, targets, _valid_indices) = processor.prepare_features(&candles);

    if features.nrows() < 100 {
        println!("   Not enough data for training. Need at least 100 samples.");
        return Ok(());
    }

    println!("   Features shape: {} x {}", features.nrows(), features.ncols());
    println!("   Feature names: {:?}\n", processor.feature_names);

    // 3. Split data
    println!("3. Splitting data into train/test sets...");
    let (x_train, x_test, y_train, y_test) =
        FeatureEngineering::train_test_split(&features, &targets, 0.8);
    println!("   Training samples: {}", x_train.nrows());
    println!("   Test samples: {}\n", x_test.nrows());

    // 4. Train Random Forest model
    println!("4. Training Random Forest classifier...");
    let mut model = RandomForestClassifier::new(50, 8, 10);
    model.fit(&x_train, &y_train);
    println!("   Model trained!\n");

    // 5. Evaluate model
    println!("5. Evaluating model on test set...");
    let predictions = model.predict(&x_test);
    let mut correct = 0;
    for i in 0..predictions.len() {
        if (predictions[i] == 1) == (y_test[i] > 0.5) {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / predictions.len() as f64;
    println!("   Accuracy: {:.2}%\n", accuracy * 100.0);

    // 6. Create LIME explainer
    println!("6. Creating LIME explainer...");

    // Clone model for closure (needed since Rust closure ownership)
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
        Some(1000),
    );
    println!("   LIME explainer created!\n");

    // 7. Explain a single prediction
    println!("7. Explaining a single prediction...");
    let test_instance = x_test.row(0).to_owned();
    let explanation = explainer.explain(&test_instance, 1);

    println!("\n{}\n", explanation);

    // 8. Get top contributing features
    println!("8. Top 5 contributing features:");
    for (name, contrib) in explanation.get_top_features(5) {
        let sign = if contrib >= 0.0 { "+" } else { "" };
        println!("   {}{:.4}  {}", sign, contrib, name);
    }
    println!();

    // 9. Batch explanations
    println!("9. Generating batch explanations...");
    let n_explain = x_test.nrows().min(20);
    let batch_instances = x_test.slice(ndarray::s![..n_explain, ..]).to_owned();
    let explanations = explainer.explain_batch(&batch_instances, 1);
    println!("   Generated {} explanations\n", explanations.len());

    // 10. Analyze explanations
    println!("10. Analyzing explanations...");

    // Aggregate feature importance
    let aggregated = LimeAnalyzer::aggregate_explanations(&explanations);
    println!("    Average absolute feature contributions:");
    let mut sorted: Vec<_> = aggregated.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (name, value) in sorted.iter().take(5) {
        println!("      {:.4}  {}", value, name);
    }
    println!();

    // Explanation consistency
    let consistency = LimeAnalyzer::explanation_consistency(&explanations, 5);
    println!("    Explanation consistency (top 5): {:.2}%", consistency * 100.0);
    println!();

    // Local model quality distribution
    let r2_scores: Vec<f64> = explanations.iter().map(|e| e.local_model_score).collect();
    let avg_r2 = r2_scores.iter().sum::<f64>() / r2_scores.len() as f64;
    let high_quality = r2_scores.iter().filter(|&&r| r > 0.5).count();
    println!("    Average local model R²: {:.4}", avg_r2);
    println!("    High-quality explanations (R² > 0.5): {}/{}\n", high_quality, r2_scores.len());

    // 11. Filter by explanation quality
    println!("11. Filtering by explanation quality...");
    let filtered = LimeAnalyzer::filter_by_confidence(explanations, 0.5);
    println!("    Retained {} out of {} explanations\n", filtered.len(), n_explain);

    println!("Done!");
    Ok(())
}
