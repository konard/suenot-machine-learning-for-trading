//! Trading example: Adversarial Training for Robust Price Prediction
//!
//! This example:
//! 1. Fetches BTCUSDT kline data from Bybit
//! 2. Trains a standard model and an adversarially trained model
//! 3. Attacks both with FGSM and PGD at various epsilon values
//! 4. Compares robustness: standard vs adversarial training

use adversarial_training_trading::*;

#[tokio::main]
async fn main() {
    println!("=== Adversarial Training for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch data from Bybit
    // -----------------------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT data from Bybit...");
    let (x_data, y_data) = match fetch_bybit_klines("BTCUSDT", "60", 200).await {
        Ok(candles) => {
            println!("  Fetched {} candles from Bybit", candles.len());
            let (features, labels) = candles_to_features(&candles);
            println!(
                "  Features shape: ({}, {}), Labels: {}",
                features.nrows(),
                features.ncols(),
                labels.len()
            );
            (features, labels)
        }
        Err(e) => {
            println!("  Failed to fetch from Bybit: {}. Using synthetic data.", e);
            let (features, labels) = generate_synthetic_data(200, 4);
            println!(
                "  Synthetic data shape: ({}, {})",
                features.nrows(),
                features.ncols()
            );
            (features, labels)
        }
    };

    let positive_ratio = y_data.iter().filter(|&&v| v == 1.0).count() as f64 / y_data.len() as f64;
    println!("  Label distribution: {:.1}% positive\n", positive_ratio * 100.0);

    // -----------------------------------------------------------------------
    // Step 2: Train standard model
    // -----------------------------------------------------------------------
    println!("Step 2: Training standard model...");
    let mut standard_model = NeuralNetwork::new(x_data.ncols(), 16);
    standard_model.train(&x_data, &y_data, 300, 0.1);
    let std_acc = standard_model.accuracy(&x_data, &y_data);
    println!("  Standard model clean accuracy: {:.4}\n", std_acc);

    // -----------------------------------------------------------------------
    // Step 3: Train adversarially robust model (PGD-AT)
    // -----------------------------------------------------------------------
    println!("Step 3: Training adversarially robust model (PGD-AT)...");
    let mut robust_model = NeuralNetwork::new(x_data.ncols(), 16);
    adversarial_train(&mut robust_model, &x_data, &y_data, 300, 0.1, 0.05);
    let rob_acc = robust_model.accuracy(&x_data, &y_data);
    println!("  Robust model clean accuracy: {:.4}\n", rob_acc);

    // -----------------------------------------------------------------------
    // Step 4: Train TRADES model
    // -----------------------------------------------------------------------
    println!("Step 4: Training TRADES model (beta=1.0)...");
    let mut trades_model = NeuralNetwork::new(x_data.ncols(), 16);
    trades_train(&mut trades_model, &x_data, &y_data, 300, 0.1, 1.0, 0.05);
    let trades_acc = trades_model.accuracy(&x_data, &y_data);
    println!("  TRADES model clean accuracy: {:.4}\n", trades_acc);

    // -----------------------------------------------------------------------
    // Step 5: Evaluate robustness under FGSM attack
    // -----------------------------------------------------------------------
    let epsilons = vec![0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5];

    println!("Step 5: Robustness evaluation under FGSM attack");
    println!("{:-<70}", "");
    println!(
        "{:<10} {:>15} {:>15} {:>15}",
        "Epsilon", "Standard", "PGD-AT", "TRADES"
    );
    println!("{:-<70}", "");

    let std_fgsm = evaluate_robustness_fgsm(&standard_model, &x_data, &y_data, &epsilons);
    let rob_fgsm = evaluate_robustness_fgsm(&robust_model, &x_data, &y_data, &epsilons);
    let trd_fgsm = evaluate_robustness_fgsm(&trades_model, &x_data, &y_data, &epsilons);

    for i in 0..epsilons.len() {
        println!(
            "{:<10.3} {:>15.4} {:>15.4} {:>15.4}",
            std_fgsm[i].0, std_fgsm[i].1, rob_fgsm[i].1, trd_fgsm[i].1
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 6: Evaluate robustness under PGD attack
    // -----------------------------------------------------------------------
    println!("Step 6: Robustness evaluation under PGD attack (20 steps)");
    println!("{:-<70}", "");
    println!(
        "{:<10} {:>15} {:>15} {:>15}",
        "Epsilon", "Standard", "PGD-AT", "TRADES"
    );
    println!("{:-<70}", "");

    let std_pgd = evaluate_robustness_pgd(&standard_model, &x_data, &y_data, &epsilons, 20);
    let rob_pgd = evaluate_robustness_pgd(&robust_model, &x_data, &y_data, &epsilons, 20);
    let trd_pgd = evaluate_robustness_pgd(&trades_model, &x_data, &y_data, &epsilons, 20);

    for i in 0..epsilons.len() {
        println!(
            "{:<10.3} {:>15.4} {:>15.4} {:>15.4}",
            std_pgd[i].0, std_pgd[i].1, rob_pgd[i].1, trd_pgd[i].1
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 7: Summary
    // -----------------------------------------------------------------------
    println!("=== Summary ===");
    println!("Clean accuracy:");
    println!("  Standard: {:.4}", std_acc);
    println!("  PGD-AT:   {:.4}", rob_acc);
    println!("  TRADES:   {:.4}", trades_acc);
    println!();

    let eps_test = 0.1;
    let std_under_attack = evaluate_robustness_pgd(
        &standard_model, &x_data, &y_data, &[eps_test], 20
    )[0].1;
    let rob_under_attack = evaluate_robustness_pgd(
        &robust_model, &x_data, &y_data, &[eps_test], 20
    )[0].1;
    let trd_under_attack = evaluate_robustness_pgd(
        &trades_model, &x_data, &y_data, &[eps_test], 20
    )[0].1;

    println!("Accuracy under PGD attack (eps={}):", eps_test);
    println!("  Standard: {:.4}", std_under_attack);
    println!("  PGD-AT:   {:.4}", rob_under_attack);
    println!("  TRADES:   {:.4}", trd_under_attack);
    println!();
    println!("Adversarially trained models should show higher accuracy under attack,");
    println!("demonstrating improved robustness for trading applications.");
}
