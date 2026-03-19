use adversarial_examples_finance::*;


fn main() -> anyhow::Result<()> {
    println!("=== Adversarial Examples in Finance: Trading Example ===\n");

    // ---- Step 1: Fetch data from Bybit ----
    println!("[1] Fetching BTCUSDT klines from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("    Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("    Warning: Could not fetch from Bybit ({}), using synthetic data", e);
            generate_synthetic_candles(200)
        }
    };

    if candles.len() < 10 {
        println!("    Not enough candles, using synthetic data");
        let candles = generate_synthetic_candles(200);
        return run_analysis(&candles);
    }

    run_analysis(&candles)
}

fn run_analysis(candles: &[OhlcvCandle]) -> anyhow::Result<()> {
    let lookback = 5;

    // ---- Step 2: Extract features and labels ----
    println!("\n[2] Extracting features (lookback={})...", lookback);
    let features = extract_features(candles, lookback);
    let labels = generate_labels(candles, lookback);

    let n = features.len().min(labels.len());
    let features = &features[..n];
    let labels = &labels[..n];
    println!("    {} samples, {} features each", n, features[0].len());

    // Normalize
    let (norm_features, _mean, _std) = normalize_features(features);

    // Split train/test
    let split = (n as f64 * 0.8) as usize;
    let train_x = &norm_features[..split];
    let train_y = &labels[..split];
    let test_x = &norm_features[split..];
    let test_y = &labels[split..];

    println!("    Train: {}, Test: {}", train_x.len(), test_x.len());

    // ---- Step 3: Train trading model ----
    println!("\n[3] Training neural network...");
    let input_dim = features[0].len();
    let mut network = NeuralNetwork::new(input_dim, 32, 16, 1);
    network.train(train_x, train_y, 50, 0.01);

    // Evaluate clean accuracy
    let clean_correct: usize = test_x
        .iter()
        .zip(test_y.iter())
        .filter(|(x, &y)| network.predict(x) == y)
        .count();
    let clean_accuracy = clean_correct as f64 / test_x.len() as f64;
    println!("    Clean test accuracy: {:.1}%", clean_accuracy * 100.0);

    // ---- Step 4: Generate adversarial examples ----
    println!("\n[4] Generating adversarial examples...\n");

    let epsilons = [0.01, 0.05, 0.1, 0.2, 0.5];

    // FGSM attacks
    println!("--- FGSM Attack ---");
    println!("{:<12} {:<20} {:<20}", "Epsilon", "Success Rate", "Avg L2 Perturbation");
    for &eps in &epsilons {
        let eval = evaluate_attack_success_rate(&network, test_x, test_y, eps, AttackType::FGSM);
        println!(
            "{:<12.3} {:<20.1}% {:<20.6}",
            eps,
            eval.success_rate * 100.0,
            eval.avg_l2_perturbation
        );
    }

    // PGD attacks
    println!("\n--- PGD Attack (20 steps) ---");
    println!("{:<12} {:<20} {:<20}", "Epsilon", "Success Rate", "Avg L2 Perturbation");
    for &eps in &epsilons {
        let eval = evaluate_attack_success_rate(&network, test_x, test_y, eps, AttackType::PGD);
        println!(
            "{:<12.3} {:<20.1}% {:<20.6}",
            eps,
            eval.success_rate * 100.0,
            eval.avg_l2_perturbation
        );
    }

    // DeepFool attacks
    println!("\n--- DeepFool Attack ---");
    let eval = evaluate_attack_success_rate(&network, test_x, test_y, 0.0, AttackType::DeepFool);
    println!(
        "Success rate: {:.1}%, Avg L2 perturbation: {:.6}",
        eval.success_rate * 100.0,
        eval.avg_l2_perturbation
    );

    // ---- Step 5: Financial perturbation budget ----
    println!("\n[5] Financial perturbation budget (crypto-calibrated)...");
    let budget = FinancialPerturbationBudget::default_crypto();
    let budget_epsilons = budget.to_epsilon_vector(lookback);

    let mut fin_success = 0;
    let mut fin_total = 0;
    for (x, &y) in test_x.iter().zip(test_y.iter()) {
        if network.predict(x) == y {
            fin_total += 1;
            let result = fgsm_financial_attack(&network, x, y, &budget_epsilons);
            if result.attack_successful {
                fin_success += 1;
            }
        }
    }
    if fin_total > 0 {
        println!(
            "    Financial FGSM success rate: {:.1}% ({}/{})",
            fin_success as f64 / fin_total as f64 * 100.0,
            fin_success,
            fin_total
        );
    }

    // ---- Step 6: Feature vulnerability analysis ----
    println!("\n[6] Feature vulnerability analysis...");
    let vuln = analyze_feature_vulnerability(&network, test_x, test_y);

    let feature_names = build_feature_names(lookback);
    println!("\n    Top 5 most vulnerable features:");
    for (rank, &idx) in vuln.feature_sensitivity_ranking.iter().take(5).enumerate() {
        let name = feature_names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("feature_{}", idx));
        println!(
            "    {}. {} (gradient magnitude: {:.6})",
            rank + 1,
            name,
            vuln.mean_gradient_magnitude[idx]
        );
    }

    // ---- Step 7: Detailed single-sample analysis ----
    println!("\n[7] Detailed adversarial example analysis (first test sample)...");
    if let (Some(sample), Some(&label)) = (test_x.first(), test_y.first()) {
        println!("    Original prediction: {}", network.predict(sample));
        println!("    True label: {}", label);

        let fgsm_result = fgsm_attack(&network, sample, label, 0.1);
        println!("\n    FGSM (eps=0.1):");
        println!("      Adversarial prediction: {}", fgsm_result.adversarial_prediction);
        println!("      L_inf perturbation: {:.6}", fgsm_result.l_inf_norm);
        println!("      L_2 perturbation: {:.6}", fgsm_result.l2_norm);
        println!("      Attack successful: {}", fgsm_result.attack_successful);

        let pgd_result = pgd_attack(&network, sample, label, 0.1, 0.025, 20);
        println!("\n    PGD (eps=0.1, steps=20):");
        println!("      Adversarial prediction: {}", pgd_result.adversarial_prediction);
        println!("      L_inf perturbation: {:.6}", pgd_result.l_inf_norm);
        println!("      L_2 perturbation: {:.6}", pgd_result.l2_norm);
        println!("      Attack successful: {}", pgd_result.attack_successful);

        let df_result = deepfool_attack(&network, sample, label, 50, 0.02);
        println!("\n    DeepFool:");
        println!("      Adversarial prediction: {}", df_result.adversarial_prediction);
        println!("      L_inf perturbation: {:.6}", df_result.l_inf_norm);
        println!("      L_2 perturbation: {:.6}", df_result.l2_norm);
        println!("      Attack successful: {}", df_result.attack_successful);
    }

    println!("\n=== Analysis Complete ===");
    Ok(())
}

fn build_feature_names(lookback: usize) -> Vec<String> {
    let mut names = Vec::new();
    for i in 0..lookback {
        names.push(format!("return_t-{}", lookback - i));
    }
    for i in 0..lookback {
        names.push(format!("vol_change_t-{}", lookback - i));
    }
    for i in 0..lookback {
        names.push(format!("hl_range_t-{}", lookback - i));
    }
    names
}

fn generate_synthetic_candles(n: usize) -> Vec<OhlcvCandle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64;

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(OhlcvCandle {
            timestamp: i as u64 * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }
    candles
}
