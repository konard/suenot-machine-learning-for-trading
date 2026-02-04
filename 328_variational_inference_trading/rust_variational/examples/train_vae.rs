//! Example: Train VAE model
//!
//! This example demonstrates how to create and use a VAE model
//! for market prediction.

use variational_inference_trading::prelude::*;
use ndarray::Array1;
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Variational Autoencoder Training Demo ===\n");

    // Configuration
    let config = VAEConfig::new(13, 16, vec![128, 64])
        .with_beta(4.0)
        .with_sequence_length(60);

    println!("Model Configuration:");
    println!("  Input dim:     {}", config.input_dim);
    println!("  Latent dim:    {}", config.latent_dim);
    println!("  Hidden dims:   {:?}", config.hidden_dims);
    println!("  Beta:          {}", config.beta);
    println!("  Sequence len:  {}", config.sequence_length);

    // Create VAE
    println!("\nCreating VAE model...");
    let vae = VAE::new(config.clone());

    // Generate synthetic data
    println!("\nGenerating synthetic market data...");
    let mut rng = rand::thread_rng();
    let input_size = config.input_dim * config.sequence_length;

    let samples: Vec<Array1<f64>> = (0..100)
        .map(|_| {
            Array1::from_vec(
                (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect()
            )
        })
        .collect();

    // Forward pass
    println!("\nRunning forward pass on sample...");
    let sample = &samples[0];
    let output = vae.forward(sample);

    println!("  Reconstruction shape: {} elements", output.x_recon.len());
    println!("  Latent mu shape:      {} elements", output.latent_mu.len());
    println!("  Predicted return:     {:.4}", output.return_mu);
    println!("  Return std:           {:.4}", (0.5 * output.return_logvar).exp());

    // Compute loss
    let target_return = 0.01;
    let loss = vae.compute_loss(sample, &output, Some(target_return));

    println!("\nLoss computation:");
    println!("  Total loss:       {:.4}", loss.total_loss);
    println!("  Reconstruction:   {:.4}", loss.recon_loss);
    println!("  KL divergence:    {:.4}", loss.kl_loss);
    println!("  Return loss:      {:.4}", loss.return_loss);

    // Generate prediction with uncertainty
    println!("\nGenerating prediction with uncertainty (100 samples)...");
    let prediction = vae.predict(sample, 100);

    println!("  Expected return:      {:.4}", prediction.expected_return);
    println!("  Return uncertainty:   {:.4}", prediction.return_std);
    println!("  Prob positive:        {:.2}%", prediction.prob_positive * 100.0);
    println!("  Predicted regime:     {} ({})", prediction.regime, prediction.regime_name());
    println!("  Regime probabilities: {:?}",
             prediction.regime_probs.iter()
                 .map(|p| format!("{:.2}", p))
                 .collect::<Vec<_>>());
    println!("  Confidence:           {:.2}%", prediction.confidence() * 100.0);

    if let Some(recon_error) = prediction.reconstruction_error {
        println!("  Reconstruction error: {:.4}", recon_error);
    }

    // Test multiple predictions
    println!("\nGenerating predictions for multiple samples...");
    let predictions: Vec<Prediction> = samples.iter()
        .take(10)
        .map(|s| vae.predict(s, 50))
        .collect();

    println!("\nPrediction summary:");
    println!("{:<5} {:>12} {:>12} {:>10} {:>8}",
             "Idx", "E[return]", "Uncertainty", "P(r>0)", "Regime");
    println!("{:-<55}", "");

    for (i, pred) in predictions.iter().enumerate() {
        println!("{:<5} {:>12.4} {:>12.4} {:>9.2}% {:>8}",
                 i,
                 pred.expected_return,
                 pred.return_std,
                 pred.prob_positive * 100.0,
                 pred.regime_name());
    }

    println!("\n=== VAE Demo Complete ===");

    Ok(())
}
