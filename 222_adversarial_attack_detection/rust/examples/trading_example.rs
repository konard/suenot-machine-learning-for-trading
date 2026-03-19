//! Trading example: Adversarial attack detection on Bybit BTCUSDT data.
//!
//! This example:
//! 1. Fetches BTCUSDT kline data from Bybit
//! 2. Generates clean and adversarial market data
//! 3. Runs all detectors
//! 4. Compares detection rates and false positive rates

use adversarial_attack_detection::*;
use ndarray::Array1;

fn main() {
    println!("=== Adversarial Attack Detection - Trading Example ===\n");

    // ─── Step 1: Fetch real market data from Bybit ──────────────────────
    println!("[1] Fetching BTCUSDT data from Bybit...");

    let features = match BybitClient::new().fetch_klines("BTCUSDT", "5", 200) {
        Ok(klines) => {
            println!("    Fetched {} klines from Bybit", klines.len());
            let feats = BybitClient::klines_to_features(&klines);
            println!("    Extracted {} feature vectors (dim={})", feats.len(), feats.first().map_or(0, |f| f.len()));
            feats
        }
        Err(e) => {
            println!("    Warning: Could not fetch from Bybit: {}", e);
            println!("    Using synthetic data instead.\n");
            generate_clean_data(200, 4)
        }
    };

    if features.is_empty() {
        println!("    No data available. Exiting.");
        return;
    }

    // Split into train and test
    let split = features.len() * 3 / 4;
    let train_data = &features[..split];
    let test_clean = &features[split..];
    println!("    Train: {} samples, Test (clean): {} samples\n", train_data.len(), test_clean.len());

    // ─── Step 2: Generate adversarial data ──────────────────────────────
    println!("[2] Generating adversarial data...");
    let epsilon = 0.05;
    let test_adversarial = generate_adversarial_data(test_clean, epsilon);
    println!("    Generated {} adversarial samples (epsilon={})\n", test_adversarial.len(), epsilon);

    // Also generate PGD adversarial samples
    let weights = Array1::from_vec(vec![1.0, -0.5, 0.3, -0.8]);
    let pgd_adversarial: Vec<Array1<f64>> = test_clean
        .iter()
        .map(|x| pgd_attack(x, &weights, epsilon, 0.01, 10))
        .collect();
    println!("    Generated {} PGD adversarial samples\n", pgd_adversarial.len());

    // ─── Step 3: Fit detectors ──────────────────────────────────────────
    println!("[3] Fitting detectors on training data...");

    // Combined detector
    let mut combined = AdversarialDetector::new(0.1, 5, 2);
    combined.fit(train_data);
    println!("    Combined detector fitted.\n");

    // Individual detectors for comparison
    let mut stat_detector = StatisticalDetector::new(0.1);
    stat_detector.fit(train_data, 5.0);

    let mut lid_estimator = LIDEstimator::new(5);
    lid_estimator.fit(train_data, 95.0);

    let fs_detector = FeatureSqueezingDetector::new(4, 3, 0.02);

    // ─── Step 4: Run detection ──────────────────────────────────────────
    println!("[4] Running detection on test data...\n");

    // Collect predictions and labels
    let mut all_predictions: Vec<bool> = Vec::new();
    let mut all_labels: Vec<bool> = Vec::new();
    let mut all_scores: Vec<f64> = Vec::new();

    // --- Individual detector results ---
    let mut fs_preds = Vec::new();
    let mut stat_preds = Vec::new();
    let mut lid_preds = Vec::new();

    // Test on clean data
    for x in test_clean {
        let result = combined.detect(x);
        all_predictions.push(result.is_adversarial);
        all_labels.push(false); // clean
        all_scores.push(result.score);

        fs_preds.push(fs_detector.detect(x).0);
        stat_preds.push(stat_detector.detect(x).0);
        lid_preds.push(lid_estimator.detect(x).0);
    }

    // Test on FGSM adversarial data
    for x in &test_adversarial {
        let result = combined.detect(x);
        all_predictions.push(result.is_adversarial);
        all_labels.push(true); // adversarial
        all_scores.push(result.score);

        fs_preds.push(fs_detector.detect(x).0);
        stat_preds.push(stat_detector.detect(x).0);
        lid_preds.push(lid_estimator.detect(x).0);
    }

    let n_clean = test_clean.len();
    let n_adv = test_adversarial.len();
    let clean_labels: Vec<bool> = vec![false; n_clean];
    let adv_labels: Vec<bool> = vec![true; n_adv];
    let all_labels_ind: Vec<bool> = [clean_labels.as_slice(), adv_labels.as_slice()].concat();

    // ─── Step 5: Compute and display metrics ────────────────────────────
    println!("=== Detection Results ===\n");

    // Combined detector
    let tpr = DetectionMetrics::true_positive_rate(&all_predictions, &all_labels);
    let fpr = DetectionMetrics::false_positive_rate(&all_predictions, &all_labels);
    let auc = DetectionMetrics::auc_roc(&all_scores, &all_labels);
    println!("Combined Detector (majority vote):");
    println!("  True Positive Rate:  {:.4}", tpr);
    println!("  False Positive Rate: {:.4}", fpr);
    println!("  AUC-ROC:             {:.4}", auc);
    println!();

    // Feature Squeezing
    let fs_tpr = DetectionMetrics::true_positive_rate(&fs_preds, &all_labels_ind);
    let fs_fpr = DetectionMetrics::false_positive_rate(&fs_preds, &all_labels_ind);
    println!("Feature Squeezing Detector:");
    println!("  True Positive Rate:  {:.4}", fs_tpr);
    println!("  False Positive Rate: {:.4}", fs_fpr);
    println!();

    // Statistical (KDE)
    let stat_tpr = DetectionMetrics::true_positive_rate(&stat_preds, &all_labels_ind);
    let stat_fpr = DetectionMetrics::false_positive_rate(&stat_preds, &all_labels_ind);
    println!("Statistical (KDE) Detector:");
    println!("  True Positive Rate:  {:.4}", stat_tpr);
    println!("  False Positive Rate: {:.4}", stat_fpr);
    println!();

    // LID
    let lid_tpr = DetectionMetrics::true_positive_rate(&lid_preds, &all_labels_ind);
    let lid_fpr = DetectionMetrics::false_positive_rate(&lid_preds, &all_labels_ind);
    println!("LID Detector:");
    println!("  True Positive Rate:  {:.4}", lid_tpr);
    println!("  False Positive Rate: {:.4}", lid_fpr);
    println!();

    // ─── Step 6: PGD attack comparison ──────────────────────────────────
    println!("=== PGD Attack Detection ===\n");
    let mut pgd_combined_preds = Vec::new();
    for x in &pgd_adversarial {
        let result = combined.detect(x);
        pgd_combined_preds.push(result.is_adversarial);
    }
    let pgd_labels: Vec<bool> = vec![true; pgd_adversarial.len()];
    let pgd_tpr = DetectionMetrics::true_positive_rate(&pgd_combined_preds, &pgd_labels);
    println!("Combined Detector vs PGD attack:");
    println!("  True Positive Rate:  {:.4}", pgd_tpr);
    println!();

    // ─── Step 7: Example individual detections ──────────────────────────
    println!("=== Sample Detections ===\n");
    if let Some(clean_sample) = test_clean.first() {
        let result = combined.detect(clean_sample);
        println!("Clean sample:  adversarial={}, score={:.6}", result.is_adversarial, result.score);
        println!("  feature_squeeze={:.6}, statistical={:.6}, lid={:.6}, reconstruction={:.6}",
            result.feature_squeeze_score, result.statistical_score,
            result.lid_score, result.reconstruction_score);
    }
    if let Some(adv_sample) = test_adversarial.first() {
        let result = combined.detect(adv_sample);
        println!("FGSM sample:   adversarial={}, score={:.6}", result.is_adversarial, result.score);
        println!("  feature_squeeze={:.6}, statistical={:.6}, lid={:.6}, reconstruction={:.6}",
            result.feature_squeeze_score, result.statistical_score,
            result.lid_score, result.reconstruction_score);
    }
    if let Some(pgd_sample) = pgd_adversarial.first() {
        let result = combined.detect(pgd_sample);
        println!("PGD sample:    adversarial={}, score={:.6}", result.is_adversarial, result.score);
        println!("  feature_squeeze={:.6}, statistical={:.6}, lid={:.6}, reconstruction={:.6}",
            result.feature_squeeze_score, result.statistical_score,
            result.lid_score, result.reconstruction_score);
    }

    println!("\n=== Done ===");
}
