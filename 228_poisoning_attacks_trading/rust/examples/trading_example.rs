//! # Trading Example: Poisoning Attacks and Defenses
//!
//! Fetches BTCUSDT from Bybit, creates clean and poisoned datasets at various rates,
//! trains models on each, evaluates accuracy degradation, applies data sanitization
//! defense, and compares clean vs poisoned vs defended model performance.

use poisoning_attacks_trading::*;

fn main() {
    println!("=== Poisoning Attacks in Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch data from Bybit (fall back to synthetic data if unavailable)
    // -----------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT kline data from Bybit...");
    let klines = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(k) if k.len() >= 20 => {
            println!("    Fetched {} klines from Bybit API", k.len());
            k
        }
        Ok(_) | Err(_) => {
            println!("    Bybit API unavailable, using synthetic data");
            generate_synthetic_klines(200, 42)
        }
    };

    // Convert to dataset
    let full_dataset = klines_to_dataset(&klines);
    println!("    Dataset size: {} samples", full_dataset.len());
    println!(
        "    Features per sample: {}",
        full_dataset.samples[0].features.len()
    );

    // Split into train (80%) and test (20%)
    let split = (full_dataset.len() as f64 * 0.8) as usize;
    let train = Dataset::new(full_dataset.samples[..split].to_vec());
    let test = Dataset::new(full_dataset.samples[split..].to_vec());
    println!(
        "    Train: {} samples, Test: {} samples\n",
        train.len(),
        test.len()
    );

    let n_features = train.samples[0].features.len();

    // -----------------------------------------------------------------------
    // Step 2: Baseline clean model
    // -----------------------------------------------------------------------
    println!("[2] Training clean baseline model...");
    let mut clean_model = LinearClassifier::new(n_features, 0.1, 500);
    clean_model.train(&train);
    let clean_acc = clean_model.accuracy(&test);
    println!("    Clean model accuracy: {:.2}%\n", clean_acc * 100.0);

    // -----------------------------------------------------------------------
    // Step 3: Label flipping attack at various rates
    // -----------------------------------------------------------------------
    println!("[3] Label Flipping Attack - Accuracy Degradation:");
    println!("    {:<20} {:<15}", "Poison Rate", "Accuracy");
    println!("    {}", "-".repeat(35));

    let rates = vec![0.05, 0.10, 0.20];
    for &rate in &rates {
        let attack = LabelFlipAttack::new(rate, 42);
        let poisoned = attack.poison(&train);
        let mut model = LinearClassifier::new(n_features, 0.1, 500);
        model.train(&poisoned);
        let acc = model.accuracy(&test);
        let degradation = (clean_acc - acc) * 100.0;
        println!(
            "    {:<20} {:<15} (degradation: {:.1}%)",
            format!("{:.0}%", rate * 100.0),
            format!("{:.2}%", acc * 100.0),
            degradation
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 4: Feature poisoning attack
    // -----------------------------------------------------------------------
    println!("[4] Feature Poisoning Attack:");
    for &rate in &rates {
        let attack = FeaturePoisonAttack::new(rate, 1.5, 42);
        let poisoned = attack.poison(&train);
        let mut model = LinearClassifier::new(n_features, 0.1, 500);
        model.train(&poisoned);
        let acc = model.accuracy(&test);
        println!(
            "    Rate: {:.0}% -> Accuracy: {:.2}% (samples: {} -> {})",
            rate * 100.0,
            acc * 100.0,
            train.len(),
            poisoned.len()
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 5: Backdoor attack
    // -----------------------------------------------------------------------
    println!("[5] Backdoor Attack:");
    let trigger = vec![5.0, 5.0, 5.0, 5.0]; // extreme trigger pattern
    let backdoor = BackdoorAttack::new(trigger.clone(), 1, 0.1, 42);
    let backdoored_train = backdoor.poison(&train);
    let mut backdoor_model = LinearClassifier::new(n_features, 0.1, 500);
    backdoor_model.train(&backdoored_train);

    let normal_acc = backdoor_model.accuracy(&test);
    println!("    Normal accuracy (no trigger): {:.2}%", normal_acc * 100.0);

    // Test with trigger-embedded samples
    let triggered_samples: Vec<Sample> = test.samples.iter().map(|s| {
        let mut new_features = s.features.clone();
        let offset = new_features.len().saturating_sub(trigger.len());
        for (j, &tv) in trigger.iter().enumerate() {
            if offset + j < new_features.len() {
                new_features[offset + j] = tv;
            }
        }
        Sample {
            features: new_features,
            label: s.label,
        }
    }).collect();
    let triggered_test = Dataset::new(triggered_samples);

    // Check how many are classified as target label when trigger is present
    let preds = backdoor_model.predict(&triggered_test);
    let target_count = preds.iter().filter(|&&p| p == 1).count();
    println!(
        "    With trigger: {:.1}% classified as target label (attack success rate)\n",
        target_count as f64 / preds.len() as f64 * 100.0
    );

    // -----------------------------------------------------------------------
    // Step 6: Data sanitization defense
    // -----------------------------------------------------------------------
    println!("[6] Data Sanitization Defense:");
    println!("    {:<20} {:<15} {:<15} {:<15}", "Poison Rate", "Poisoned", "Defended", "Recovery");
    println!("    {}", "-".repeat(65));

    for &rate in &rates {
        // Poison with feature injection
        let attack = FeaturePoisonAttack::new(rate, 1.5, 42);
        let poisoned = attack.poison(&train);

        // Train on poisoned data
        let mut poison_model = LinearClassifier::new(n_features, 0.1, 500);
        poison_model.train(&poisoned);
        let poison_acc = poison_model.accuracy(&test);

        // Sanitize and retrain
        let sanitizer = DataSanitizer::new(2.5);
        let sanitized = sanitizer.sanitize(&poisoned);
        let mut defended_model = LinearClassifier::new(n_features, 0.1, 500);
        defended_model.train(&sanitized);
        let defended_acc = defended_model.accuracy(&test);

        let recovery = defended_acc - poison_acc;
        println!(
            "    {:<20} {:<15} {:<15} {:<15}",
            format!("{:.0}%", rate * 100.0),
            format!("{:.2}%", poison_acc * 100.0),
            format!("{:.2}%", defended_acc * 100.0),
            format!("{:+.2}%", recovery * 100.0)
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 7: Loss-based poison detection
    // -----------------------------------------------------------------------
    println!("[7] Loss-Based Poison Detection:");
    let attack = LabelFlipAttack::new(0.15, 42);
    let poisoned = attack.poison(&train);

    let mut detect_model = LinearClassifier::new(n_features, 0.1, 500);
    detect_model.train(&poisoned);

    let detector = PoisonDetector::new(0.85);
    let (cleaned, flagged) = detector.detect_and_remove(&poisoned, &detect_model.weights);
    println!("    Original size: {}", poisoned.len());
    println!("    Flagged as suspicious: {}", flagged.len());
    println!("    Cleaned size: {}", cleaned.len());

    let mut cleaned_model = LinearClassifier::new(n_features, 0.1, 500);
    cleaned_model.train(&cleaned);
    let cleaned_acc = cleaned_model.accuracy(&test);
    let poisoned_acc = detect_model.accuracy(&test);
    println!("    Poisoned model accuracy: {:.2}%", poisoned_acc * 100.0);
    println!("    Cleaned model accuracy:  {:.2}%", cleaned_acc * 100.0);
    println!();

    // -----------------------------------------------------------------------
    // Step 8: Influence-based analysis
    // -----------------------------------------------------------------------
    println!("[8] Influence Function Analysis:");
    let (train_x, train_y) = train.to_matrix();
    let test_sample = ndarray::Array1::from_vec(test.samples[0].features.clone());
    let test_label = test.samples[0].label as f64;

    let candidates = targeted_poisoning_candidates(
        &clean_model.weights,
        &train_x,
        &train_y,
        &test_sample,
        test_label,
        5,
    );
    println!("    Top 5 most influential training samples for first test point:");
    for (rank, &idx) in candidates.iter().enumerate() {
        println!(
            "      #{}: sample {} (label={})",
            rank + 1,
            idx,
            train.samples[idx].label
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("=== Summary ===");
    println!("  Clean baseline accuracy:    {:.2}%", clean_acc * 100.0);
    println!("  Worst poisoned accuracy:    {:.2}% (20% label flip)", {
        let attack = LabelFlipAttack::new(0.20, 42);
        let poisoned = attack.poison(&train);
        let mut m = LinearClassifier::new(n_features, 0.1, 500);
        m.train(&poisoned);
        m.accuracy(&test) * 100.0
    });
    println!("  Defense can recover significant accuracy via data sanitization");
    println!("  Multi-layer defense (sanitization + detection) recommended for production");
}
