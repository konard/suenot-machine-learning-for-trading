use anyhow::Result;
use ndarray::Array1;
use teacher_student_trading::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Teacher-Student Trading Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Fetch data from Bybit
    // -----------------------------------------------------------------------
    println!("Fetching BTCUSDT data from Bybit...");
    let candles = fetch_bybit_klines("BTCUSDT", "15", 200).await?;
    println!("Fetched {} candles", candles.len());

    let window_size = 5;
    let input_size = window_size * 5; // 5 features per candle
    let (features, labels) = prepare_data(&candles, window_size);
    println!("Prepared {} samples with {} features each\n", features.len(), input_size);

    if features.len() < 20 {
        println!("Not enough data for meaningful training. Need at least 20 samples.");
        return Ok(());
    }

    // Split into train/test
    let split = features.len() * 80 / 100;
    let (train_features, test_features) = features.split_at(split);
    let (train_labels, test_labels) = labels.split_at(split);

    println!("Train: {} samples, Test: {} samples\n", train_features.len(), test_features.len());

    // -----------------------------------------------------------------------
    // 2. Train 3 specialist teachers for different market regimes
    // -----------------------------------------------------------------------
    println!("--- Training Specialist Teachers ---\n");

    let teacher_epochs = 30;
    let teacher_lr = 0.01;

    // Teacher 1: Bull market specialist
    let mut bull_teacher = TeacherModel::new(&[input_size, 128, 64, 32, 1], &[0, 1, 2]);
    println!("Training Bull Market Teacher...");
    for epoch in 0..teacher_epochs {
        let mut epoch_loss = 0.0;
        for (feat, &label) in train_features.iter().zip(train_labels.iter()) {
            let target = Array1::from_vec(vec![label]);
            epoch_loss += bull_teacher.train_step(feat, &target, teacher_lr);
        }
        if epoch % 10 == 0 || epoch == teacher_epochs - 1 {
            println!("  Epoch {}: loss = {:.6}", epoch, epoch_loss / train_features.len() as f64);
        }
    }
    let bull_acc = evaluate_teacher(&bull_teacher, test_features, test_labels);
    println!("  Bull Teacher Test Accuracy: {:.2}%\n", bull_acc * 100.0);

    // Teacher 2: Bear market specialist
    let mut bear_teacher = TeacherModel::new(&[input_size, 128, 64, 32, 1], &[0, 1, 2]);
    println!("Training Bear Market Teacher...");
    for epoch in 0..teacher_epochs {
        let mut epoch_loss = 0.0;
        for (feat, &label) in train_features.iter().zip(train_labels.iter()) {
            let target = Array1::from_vec(vec![label]);
            epoch_loss += bear_teacher.train_step(feat, &target, teacher_lr);
        }
        if epoch % 10 == 0 || epoch == teacher_epochs - 1 {
            println!("  Epoch {}: loss = {:.6}", epoch, epoch_loss / train_features.len() as f64);
        }
    }
    let bear_acc = evaluate_teacher(&bear_teacher, test_features, test_labels);
    println!("  Bear Teacher Test Accuracy: {:.2}%\n", bear_acc * 100.0);

    // Teacher 3: Sideways market specialist
    let mut sideways_teacher = TeacherModel::new(&[input_size, 128, 64, 32, 1], &[0, 1, 2]);
    println!("Training Sideways Market Teacher...");
    for epoch in 0..teacher_epochs {
        let mut epoch_loss = 0.0;
        for (feat, &label) in train_features.iter().zip(train_labels.iter()) {
            let target = Array1::from_vec(vec![label]);
            epoch_loss += sideways_teacher.train_step(feat, &target, teacher_lr);
        }
        if epoch % 10 == 0 || epoch == teacher_epochs - 1 {
            println!("  Epoch {}: loss = {:.6}", epoch, epoch_loss / train_features.len() as f64);
        }
    }
    let sideways_acc = evaluate_teacher(&sideways_teacher, test_features, test_labels);
    println!("  Sideways Teacher Test Accuracy: {:.2}%\n", sideways_acc * 100.0);

    // -----------------------------------------------------------------------
    // 3. Create multi-teacher ensemble
    // -----------------------------------------------------------------------
    println!("--- Multi-Teacher Ensemble ---\n");

    // Detect current regime to weight teachers
    let regime = detect_regime(&candles, 20);
    println!("Current detected regime: {:?}", regime);

    let weights = match regime {
        MarketRegime::Bull => vec![2.0, 0.5, 1.0],
        MarketRegime::Bear => vec![0.5, 2.0, 1.0],
        MarketRegime::Sideways => vec![1.0, 1.0, 2.0],
    };
    println!("Teacher weights (bull, bear, sideways): {:?}", weights);

    let aggregator = MultiTeacherAggregator::new(
        vec![bull_teacher.clone(), bear_teacher.clone(), sideways_teacher.clone()],
        weights,
    );
    let ensemble_acc = evaluate_ensemble(&aggregator, test_features, test_labels);
    println!("Ensemble Test Accuracy: {:.2}%\n", ensemble_acc * 100.0);

    // -----------------------------------------------------------------------
    // 4. Train student from multi-teacher ensemble
    // -----------------------------------------------------------------------
    println!("--- Training Student Model ---\n");

    let student = StudentModel::new(
        &[input_size, 32, 16, 1],
        &[0],                // hint at first hidden layer
        &[(32, 128)],        // project student 32-dim to teacher 128-dim
    );

    let mut trainer = TeacherStudentTrainer::new(student, 0.005)
        .with_loss_weights(1.0, 0.5, 0.3, 0.1);

    let student_epochs = 20;
    for epoch in 0..student_epochs {
        let loss = trainer.train_epoch_multi_teacher(&aggregator, train_features, train_labels);
        if epoch % 5 == 0 || epoch == student_epochs - 1 {
            let acc = trainer.evaluate(test_features, test_labels);
            println!("  Epoch {}: loss = {:.6}, accuracy = {:.2}%", epoch, loss, acc * 100.0);
        }
    }

    let student_acc = trainer.evaluate(test_features, test_labels);

    // -----------------------------------------------------------------------
    // 5. Also train a student from scratch (no teacher) for comparison
    // -----------------------------------------------------------------------
    println!("\n--- Training Baseline Student (No Teacher) ---\n");

    let mut baseline = TeacherModel::new(&[input_size, 32, 16, 1], &[]);
    for epoch in 0..student_epochs {
        let mut epoch_loss = 0.0;
        for (feat, &label) in train_features.iter().zip(train_labels.iter()) {
            let target = Array1::from_vec(vec![label]);
            epoch_loss += baseline.train_step(feat, &target, 0.005);
        }
        if epoch % 5 == 0 || epoch == student_epochs - 1 {
            let acc = evaluate_teacher(&baseline, test_features, test_labels);
            println!("  Epoch {}: loss = {:.6}, accuracy = {:.2}%",
                     epoch, epoch_loss / train_features.len() as f64, acc * 100.0);
        }
    }
    let baseline_acc = evaluate_teacher(&baseline, test_features, test_labels);

    // -----------------------------------------------------------------------
    // 6. Final comparison
    // -----------------------------------------------------------------------
    println!("\n========================================");
    println!("         FINAL RESULTS COMPARISON       ");
    println!("========================================\n");
    println!("Bull Teacher:          {:.2}%", bull_acc * 100.0);
    println!("Bear Teacher:          {:.2}%", bear_acc * 100.0);
    println!("Sideways Teacher:      {:.2}%", sideways_acc * 100.0);
    println!("Ensemble (3 teachers): {:.2}%", ensemble_acc * 100.0);
    println!("Student (distilled):   {:.2}%", student_acc * 100.0);
    println!("Baseline (no teacher): {:.2}%", baseline_acc * 100.0);
    println!();

    // Model size comparison
    let teacher_params = count_teacher_params(&bull_teacher);
    let student_params = count_student_params(&trainer.student);
    println!("Teacher parameters:    {}", teacher_params);
    println!("Student parameters:    {}", student_params);
    println!("Compression ratio:     {:.1}x", teacher_params as f64 / student_params as f64);

    Ok(())
}

fn count_teacher_params(model: &TeacherModel) -> usize {
    model.layers.iter().map(|l| {
        let (r, c) = l.weights.dim();
        r * c + l.biases.len()
    }).sum()
}

fn count_student_params(model: &StudentModel) -> usize {
    let layer_params: usize = model.layers.iter().map(|l| {
        let (r, c) = l.weights.dim();
        r * c + l.biases.len()
    }).sum();
    let regressor_params: usize = model.regressors.iter().map(|l| {
        let (r, c) = l.weights.dim();
        r * c + l.biases.len()
    }).sum();
    layer_params + regressor_params
}
