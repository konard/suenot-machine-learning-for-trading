//! # Knowledge Distillation Trading Example
//!
//! This example demonstrates:
//! 1. Fetching BTCUSDT kline data from Bybit
//! 2. Training a large teacher model on price direction prediction
//! 3. Distilling the teacher's knowledge into a compact student model
//! 4. Comparing teacher vs student accuracy and inference speed

use knowledge_distillation_trading::*;
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    println!("=== Knowledge Distillation Trading Example ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -----------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT kline data from Bybit...");
    let client = BybitClient::new();
    let candles = client.fetch_klines("BTCUSDT", "15", 200)?;
    println!("    Fetched {} candles", candles.len());

    if candles.len() < 30 {
        anyhow::bail!("Not enough candle data for training (need at least 30, got {})", candles.len());
    }

    println!(
        "    Price range: {:.2} - {:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max),
    );

    // -----------------------------------------------------------------------
    // Step 2: Prepare features and labels
    // -----------------------------------------------------------------------
    println!("\n[2] Preparing features and labels...");
    let threshold = 0.001; // 0.1% threshold for buy/sell classification
    let (features, labels) = prepare_features_and_labels(&candles, 20, threshold);
    println!("    Generated {} samples with 7 features each", features.len());

    if features.is_empty() {
        anyhow::bail!("No training samples generated. Check data quality.");
    }

    // Count class distribution
    let mut class_counts = [0usize; 3];
    for label in &labels {
        let cls = label
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        class_counts[cls] += 1;
    }
    println!(
        "    Class distribution: buy={}, hold={}, sell={}",
        class_counts[0], class_counts[1], class_counts[2]
    );

    // Train/test split (80/20)
    let split_idx = (features.len() as f64 * 0.8) as usize;
    let train_features = &features[..split_idx];
    let train_labels = &labels[..split_idx];
    let test_features = &features[split_idx..];
    let test_labels = &labels[split_idx..];
    println!(
        "    Train: {} samples, Test: {} samples",
        train_features.len(),
        test_features.len()
    );

    // -----------------------------------------------------------------------
    // Step 3: Train teacher model
    // -----------------------------------------------------------------------
    println!("\n[3] Training teacher model (128->64->32->3)...");
    let mut teacher = TeacherModel::new(7, 3);
    println!("    Teacher parameters: {}", teacher.num_parameters());

    let teacher_epochs = 100;
    let teacher_lr = 0.005;
    for epoch in 0..teacher_epochs {
        let mut total_loss = 0.0;
        for (feat, label) in train_features.iter().zip(train_labels.iter()) {
            total_loss += teacher.train_step(feat, label, teacher_lr);
        }
        if (epoch + 1) % 20 == 0 {
            let avg_loss = total_loss / train_features.len() as f64;
            println!("    Epoch {}/{}: avg loss = {:.6}", epoch + 1, teacher_epochs, avg_loss);
        }
    }

    // Evaluate teacher
    let teacher_preds: Vec<Array1<f64>> = test_features
        .iter()
        .map(|f| teacher.forward(f, 1.0))
        .collect();
    let (tc, tt, teacher_accuracy) = evaluate_accuracy(&teacher_preds, test_labels);
    println!("    Teacher accuracy: {}/{} = {:.2}%", tc, tt, teacher_accuracy * 100.0);

    // -----------------------------------------------------------------------
    // Step 4: Train student WITHOUT distillation (baseline)
    // -----------------------------------------------------------------------
    println!("\n[4] Training student model WITHOUT distillation (baseline)...");
    let mut student_baseline = StudentModel::new(7, 3);
    println!("    Student parameters: {}", student_baseline.num_parameters());

    let student_epochs = 100;
    let student_lr = 0.005;
    for epoch in 0..student_epochs {
        let mut total_loss = 0.0;
        for (feat, label) in train_features.iter().zip(train_labels.iter()) {
            total_loss += student_baseline.train_step(feat, label, student_lr);
        }
        if (epoch + 1) % 20 == 0 {
            let avg_loss = total_loss / train_features.len() as f64;
            println!("    Epoch {}/{}: avg loss = {:.6}", epoch + 1, student_epochs, avg_loss);
        }
    }

    let baseline_preds: Vec<Array1<f64>> = test_features
        .iter()
        .map(|f| student_baseline.forward(f, 1.0))
        .collect();
    let (bc, bt, baseline_accuracy) = evaluate_accuracy(&baseline_preds, test_labels);
    println!(
        "    Baseline student accuracy: {}/{} = {:.2}%",
        bc, bt, baseline_accuracy * 100.0
    );

    // -----------------------------------------------------------------------
    // Step 5: Train student WITH distillation
    // -----------------------------------------------------------------------
    println!("\n[5] Training student model WITH knowledge distillation...");
    let mut student_distilled = StudentModel::new(7, 3);

    let config = DistillationConfig {
        temperature: 5.0,
        alpha: 0.3,
        learning_rate: 0.005,
        epochs: 100,
    };

    println!("    Distillation config: T={}, alpha={}", config.temperature, config.alpha);
    let _losses = distillation_train(
        &teacher,
        &mut student_distilled,
        train_features,
        train_labels,
        &config,
    );

    let distilled_preds: Vec<Array1<f64>> = test_features
        .iter()
        .map(|f| student_distilled.forward(f, 1.0))
        .collect();
    let (dc, dt, distilled_accuracy) = evaluate_accuracy(&distilled_preds, test_labels);
    println!(
        "    Distilled student accuracy: {}/{} = {:.2}%",
        dc, dt, distilled_accuracy * 100.0
    );

    // -----------------------------------------------------------------------
    // Step 6: Benchmark inference speed
    // -----------------------------------------------------------------------
    println!("\n[6] Benchmarking inference speed...");
    let bench_input = Array1::from(vec![0.001, -0.005, 0.01, 0.02, 0.015, 0.3, 0.008]);
    let iterations = 10_000;

    let teacher_bench = benchmark_teacher(&teacher, &bench_input, iterations);
    let student_bench = benchmark_student(&student_distilled, &bench_input, iterations);

    println!("    {}", teacher_bench);
    println!("    {}", student_bench);

    let speedup = teacher_bench.mean_us / student_bench.mean_us.max(0.001);
    let compression = teacher.num_parameters() as f64 / student_distilled.num_parameters() as f64;

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("  Model             | Params | Accuracy | Mean Latency");
    println!("  ------------------|--------|----------|-------------");
    println!(
        "  Teacher           | {:>6} | {:>6.2}%  | {:.2} us",
        teacher.num_parameters(),
        teacher_accuracy * 100.0,
        teacher_bench.mean_us
    );
    println!(
        "  Student (baseline) | {:>6} | {:>6.2}%  | N/A",
        student_baseline.num_parameters(),
        baseline_accuracy * 100.0,
    );
    println!(
        "  Student (distill) | {:>6} | {:>6.2}%  | {:.2} us",
        student_distilled.num_parameters(),
        distilled_accuracy * 100.0,
        student_bench.mean_us
    );
    println!();
    println!("  Compression ratio: {:.1}x fewer parameters", compression);
    println!("  Speedup: {:.1}x faster inference", speedup);
    println!(
        "  Distillation improvement over baseline: {:.2}%",
        (distilled_accuracy - baseline_accuracy) * 100.0
    );

    Ok(())
}
