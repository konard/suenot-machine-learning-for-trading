//! Universal Encoder Analysis Example
//!
//! Demonstrates the shared encoder's representation learning capabilities
//! and how different market conditions map to the representation space.
//!
//! Run with: cargo run --example universal_encoder

use task_agnostic_trading::model::{
    UniversalEncoder, EncoderConfig,
    TaskHead, TaskHeadConfig, TaskType,
};
use ndarray::{Array2, Axis};

fn main() {
    println!("=== Universal Encoder Analysis ===\n");

    // Create encoder
    let encoder_config = EncoderConfig {
        input_dim: 20,
        hidden_dims: vec![64, 32],
        repr_dim: 16,
        dropout_rate: 0.1,
        use_batch_norm: true,
        use_residual: true,
    };
    let encoder = UniversalEncoder::new(encoder_config);
    println!("Encoder: 20 -> [64, 32] -> 16 (with batch norm + residual)\n");

    // Generate different market conditions
    let uptrend = generate_condition_features(20, "uptrend");
    let downtrend = generate_condition_features(20, "downtrend");
    let volatile = generate_condition_features(20, "volatile");
    let calm = generate_condition_features(20, "calm");

    // Encode each condition
    let repr_up = encoder.forward(&uptrend);
    let repr_down = encoder.forward(&downtrend);
    let repr_vol = encoder.forward(&volatile);
    let repr_calm = encoder.forward(&calm);

    println!("--- Representation Statistics ---");
    for (name, repr) in &[
        ("Uptrend", &repr_up),
        ("Downtrend", &repr_down),
        ("Volatile", &repr_vol),
        ("Calm", &repr_calm),
    ] {
        let mean = repr.mean().unwrap_or(0.0);
        let std = repr.std(0.0);
        let norm: f64 = repr.rows().into_iter().map(|r| {
            r.iter().map(|x| x * x).sum::<f64>().sqrt()
        }).sum::<f64>() / repr.nrows() as f64;
        println!(
            "  {:<12} mean={:+.4}, std={:.4}, avg_norm={:.4}",
            name, mean, std, norm
        );
    }
    println!();

    // Compute pairwise distances between condition centroids
    println!("--- Pairwise Centroid Distances ---");
    let centroids: Vec<(&str, Vec<f64>)> = vec![
        ("Uptrend", compute_centroid(&repr_up)),
        ("Downtrend", compute_centroid(&repr_down)),
        ("Volatile", compute_centroid(&repr_vol)),
        ("Calm", compute_centroid(&repr_calm)),
    ];

    print!("{:<12}", "");
    for (name, _) in &centroids {
        print!("{:>12}", name);
    }
    println!();

    for (name_i, c_i) in &centroids {
        print!("{:<12}", name_i);
        for (_, c_j) in &centroids {
            let dist = euclidean_distance(c_i, c_j);
            print!("{:>12.4}", dist);
        }
        println!();
    }
    println!();

    // Test each task head on the encoded representations
    println!("--- Task Head Predictions per Condition ---\n");

    let task_configs = vec![
        TaskHeadConfig::new(TaskType::TrendPrediction, 3).with_input_dim(16),
        TaskHeadConfig::new(TaskType::VolatilityForecast, 1).with_input_dim(16),
        TaskHeadConfig::new(TaskType::RegimeDetection, 4).with_input_dim(16),
        TaskHeadConfig::new(TaskType::RiskAssessment, 3).with_input_dim(16),
    ];

    for task_config in &task_configs {
        let head = TaskHead::new(task_config.clone());
        println!("Task: {}", task_config.task_type.name());
        let labels = task_config.task_type.class_labels();

        for (name, repr) in &[
            ("Uptrend", &repr_up),
            ("Downtrend", &repr_down),
            ("Volatile", &repr_vol),
            ("Calm", &repr_calm),
        ] {
            let preds = head.forward(repr);
            let avg_pred = preds.mean_axis(Axis(0)).unwrap();

            if task_config.task_type.is_classification() {
                let best_idx = avg_pred
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                print!("  {:<12} -> {}", name, labels[best_idx]);
                print!(" [");
                for (i, &p) in avg_pred.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    print!("{}:{:.2}", labels[i], p);
                }
                println!("]");
            } else {
                println!("  {:<12} -> {:.4}", name, avg_pred[0]);
            }
        }
        println!();
    }

    // Representation quality analysis
    println!("--- Representation Quality ---");
    let inter_class = compute_inter_class_distance(&centroids);
    let intra_up = compute_intra_class_distance(&repr_up);
    let intra_down = compute_intra_class_distance(&repr_down);
    let intra_vol = compute_intra_class_distance(&repr_vol);
    let intra_calm = compute_intra_class_distance(&repr_calm);
    let avg_intra = (intra_up + intra_down + intra_vol + intra_calm) / 4.0;

    println!("  Avg inter-class distance: {:.4}", inter_class);
    println!("  Avg intra-class distance: {:.4}", avg_intra);
    println!(
        "  Separation ratio (higher is better): {:.4}",
        if avg_intra > 1e-10 { inter_class / avg_intra } else { 0.0 }
    );

    println!("\n=== Encoder Analysis Complete ===");
}

fn generate_condition_features(n_samples: usize, condition: &str) -> Array2<f64> {
    Array2::from_shape_fn((n_samples, 20), |(i, j)| {
        let base = match condition {
            "uptrend" => match j {
                0..=4 => 0.05 + (i as f64 * 0.001),   // Positive returns
                5..=9 => 0.7 + (j as f64 * 0.02),      // High momentum
                10..=14 => 0.15,                         // Low-moderate vol
                _ => 0.6,
            },
            "downtrend" => match j {
                0..=4 => -0.05 - (i as f64 * 0.001),   // Negative returns
                5..=9 => 0.3 - (j as f64 * 0.02),      // Low momentum
                10..=14 => 0.2,                          // Moderate vol
                _ => 0.4,
            },
            "volatile" => match j {
                0..=4 => ((i as f64 * 0.5).sin()) * 0.1,  // Oscillating returns
                5..=9 => 0.5,                               // Neutral momentum
                10..=14 => 0.8,                             // High volatility
                _ => 0.3,
            },
            "calm" => match j {
                0..=4 => 0.001,                            // Near-zero returns
                5..=9 => 0.5,                              // Neutral momentum
                10..=14 => 0.05,                           // Very low vol
                _ => 0.7,
            },
            _ => 0.5,
        };

        // Add small noise
        let noise_seed = (i * 71 + j * 13) as f64;
        let noise = (noise_seed * 0.37).sin() * 0.02;
        base + noise
    })
}

fn compute_centroid(data: &Array2<f64>) -> Vec<f64> {
    let mean = data.mean_axis(Axis(0)).unwrap();
    mean.to_vec()
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn compute_inter_class_distance(centroids: &[(&str, Vec<f64>)]) -> f64 {
    let n = centroids.len();
    let mut total = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            total += euclidean_distance(&centroids[i].1, &centroids[j].1);
            count += 1;
        }
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn compute_intra_class_distance(data: &Array2<f64>) -> f64 {
    let centroid = compute_centroid(data);
    let mut total = 0.0;
    for row in data.rows() {
        let point: Vec<f64> = row.to_vec();
        total += euclidean_distance(&point, &centroid);
    }
    total / data.nrows() as f64
}
