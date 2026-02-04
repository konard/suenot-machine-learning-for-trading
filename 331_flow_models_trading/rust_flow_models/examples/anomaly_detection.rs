//! Example: Anomaly detection using flow model
//!
//! This example demonstrates how to use normalizing flows for
//! detecting unusual market conditions via likelihood estimation.

use flow_models_trading::prelude::*;
use ndarray::Array2;
use rand_distr::{Distribution, Normal};

fn main() {
    // Initialize logging
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("          Flow Models Trading - Anomaly Detection");
    println!("═══════════════════════════════════════════════════════════════");

    // Generate normal market data
    println!("\n1. Generating normal market data...");
    let normal_data = generate_normal_data(500, 8);
    println!("   Normal samples: {}", normal_data.nrows());

    // Create and initialize flow model
    println!("\n2. Creating flow model...");
    let config = FlowConfig::default()
        .with_input_dim(8)
        .with_num_layers(4)
        .with_hidden_dim(64);

    let mut model = NormalizingFlow::new(config);

    // "Train" by running forward pass to initialize
    println!("\n3. Initializing model on normal data...");
    let _ = model.forward(&normal_data);

    // Create anomaly detector
    println!("\n4. Fitting anomaly detector...");
    let mut detector = AnomalyDetector::new(5.0); // 5th percentile threshold
    detector.fit(&mut model, &normal_data);

    println!("   Threshold (5th percentile): {:.4}", detector.get_threshold().unwrap_or(0.0));

    if let Some((mean, std, min, max)) = detector.reference_stats() {
        println!("   Reference log-prob stats:");
        println!("     Mean: {:.4}", mean);
        println!("     Std:  {:.4}", std);
        println!("     Min:  {:.4}", min);
        println!("     Max:  {:.4}", max);
    }

    // Generate test data with some anomalies
    println!("\n5. Generating test data with anomalies...");
    let (test_data, anomaly_labels) = generate_test_data_with_anomalies(100, 8);
    let true_anomaly_count = anomaly_labels.iter().filter(|&&a| a).count();
    println!("   Test samples: {}", test_data.nrows());
    println!("   True anomalies: {}", true_anomaly_count);

    // Detect anomalies
    println!("\n6. Running anomaly detection...");
    let (is_anomaly, log_probs) = detector.detect(&mut model, &test_data);

    let detected_count = is_anomaly.iter().filter(|&&a| a).count();
    println!("   Detected anomalies: {}", detected_count);

    // Compute detection metrics
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut true_negatives = 0;
    let mut false_negatives = 0;

    for i in 0..test_data.nrows() {
        match (anomaly_labels[i], is_anomaly[i]) {
            (true, true) => true_positives += 1,
            (true, false) => false_negatives += 1,
            (false, true) => false_positives += 1,
            (false, false) => true_negatives += 1,
        }
    }

    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };

    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!("\n7. Detection Results:");
    println!("   ─────────────────────────────────────");
    println!("   True Positives:  {:3}", true_positives);
    println!("   False Positives: {:3}", false_positives);
    println!("   True Negatives:  {:3}", true_negatives);
    println!("   False Negatives: {:3}", false_negatives);
    println!("   ─────────────────────────────────────");
    println!("   Precision: {:.2}%", precision * 100.0);
    println!("   Recall:    {:.2}%", recall * 100.0);
    println!("   F1 Score:  {:.2}%", f1 * 100.0);

    // Show some examples
    println!("\n8. Sample detections:");
    println!("   {:>5} {:>12} {:>10} {:>10}", "Index", "Log-Prob", "Detected", "True Label");
    println!("   {:-<45}", "");

    for i in 0..test_data.nrows().min(10) {
        let detected_str = if is_anomaly[i] { "ANOMALY" } else { "normal" };
        let true_str = if anomaly_labels[i] { "ANOMALY" } else { "normal" };
        println!(
            "   {:>5} {:>12.4} {:>10} {:>10}",
            i, log_probs[i], detected_str, true_str
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                  Anomaly detection complete!");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Generate normal market data
fn generate_normal_data(n_samples: usize, dim: usize) -> Array2<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    Array2::from_shape_fn((n_samples, dim), |_| {
        normal.sample(&mut rng) * 0.02  // Normal market volatility
    })
}

/// Generate test data with some anomalies
fn generate_test_data_with_anomalies(n_samples: usize, dim: usize) -> (Array2<f64>, Vec<bool>) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    let anomaly_rate = 0.15; // 15% anomalies
    let n_anomalies = (n_samples as f64 * anomaly_rate) as usize;

    let mut data = Array2::zeros((n_samples, dim));
    let mut labels = vec![false; n_samples];

    for i in 0..n_samples {
        if i < n_anomalies {
            // Anomaly: extreme values
            for j in 0..dim {
                data[[i, j]] = normal.sample(&mut rng) * 0.1; // Much higher volatility
            }
            labels[i] = true;
        } else {
            // Normal data
            for j in 0..dim {
                data[[i, j]] = normal.sample(&mut rng) * 0.02;
            }
        }
    }

    // Shuffle
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let mut shuffled_data = Array2::zeros((n_samples, dim));
    let mut shuffled_labels = vec![false; n_samples];

    for (new_i, &old_i) in indices.iter().enumerate() {
        shuffled_data.row_mut(new_i).assign(&data.row(old_i));
        shuffled_labels[new_i] = labels[old_i];
    }

    (shuffled_data, shuffled_labels)
}
