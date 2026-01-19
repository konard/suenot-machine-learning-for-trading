//! Sentiment Analysis Example
//!
//! Demonstrates how to use the FinancialSentimentModel with LoRA
//! for financial text sentiment classification.

use llm_finance::{FinancialSentimentModel, LoraLayer, TrainingConfig};
use ndarray::Array1;

fn main() {
    println!("=== Financial Sentiment Analysis with LoRA ===\n");

    // Model parameters
    let base_dim = 768; // BERT-like hidden dimension
    let num_heads = 12;
    let lora_rank = 8;
    let lora_alpha = 16.0;
    let num_classes = 3; // bullish, neutral, bearish

    // Create the model
    println!("Creating FinancialSentimentModel...");
    let model = FinancialSentimentModel::new(base_dim, num_heads, lora_rank, lora_alpha, num_classes);

    println!("Model configuration:");
    println!("  - Base dimension: {}", base_dim);
    println!("  - Attention heads: {}", num_heads);
    println!("  - LoRA rank: {}", lora_rank);
    println!("  - LoRA alpha: {}", lora_alpha);
    println!("  - Output classes: {}", num_classes);
    println!("  - Trainable parameters: {}", model.num_trainable_params());
    println!("  - Total parameters: {}", model.num_total_params());

    // Create a sample input (simulating BERT embeddings)
    println!("\n--- Running inference ---");
    let input = Array1::from_vec(vec![0.1; base_dim]);

    // Get prediction
    let logits = model.forward(&input);
    let proba = model.predict_proba(&input);
    let prediction = model.predict(&input);

    let class_names = ["Bearish", "Neutral", "Bullish"];

    println!("Logits: {:?}", logits.iter().take(3).collect::<Vec<_>>());
    println!("Probabilities:");
    for (i, &p) in proba.iter().enumerate() {
        println!("  - {}: {:.4}", class_names[i], p);
    }
    println!("Predicted class: {} ({})", prediction, class_names[prediction]);

    // Demonstrate LoRA layer
    println!("\n--- LoRA Layer Demo ---");
    let lora = LoraLayer::new(768, 768, 8, 16.0);
    println!("LoRA layer: {}x{} with rank {}", lora.in_features, lora.out_features, lora.rank);
    println!("Trainable params: {}", lora.num_trainable_params());

    let lora_input = Array1::from_vec(vec![0.5; 768]);
    let lora_output = lora.forward(&lora_input);
    println!("Output shape: {}", lora_output.len());

    // Training configuration
    println!("\n--- Training Configuration ---");
    let config = TrainingConfig::default()
        .with_learning_rate(2e-5)
        .with_epochs(3)
        .with_batch_size(16)
        .with_patience(5);

    println!("Training config:");
    println!("  - Learning rate: {}", config.learning_rate);
    println!("  - Epochs: {}", config.num_epochs);
    println!("  - Batch size: {}", config.batch_size);
    println!("  - Early stopping patience: {}", config.early_stopping_patience);

    println!("\nSentiment analysis example completed successfully!");
}
