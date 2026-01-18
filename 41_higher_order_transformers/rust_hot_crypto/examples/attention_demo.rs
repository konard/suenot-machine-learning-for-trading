//! Example: Higher Order Attention Demonstration
//!
//! Shows how the different attention mechanisms work and compares them.

use ndarray::Array2;
use hot_crypto::attention::{StandardAttention, HigherOrderAttention, KernelAttention};
use hot_crypto::tensor::{Tensor3D, CPDecomposition};

fn main() {
    println!("HOT Crypto - Attention Mechanism Demo");
    println!("======================================\n");

    // Model parameters
    let d_model = 64;
    let n_heads = 4;
    let seq_len = 10;
    let rank = 8;

    // Create random input (simulating embedded features)
    println!("Creating random input: seq_len={}, d_model={}", seq_len, d_model);
    let input = Array2::from_shape_fn((seq_len, d_model), |_| rand::random::<f64>() - 0.5);

    // 1. Standard Attention
    println!("\n--- Standard Attention ---");
    let std_attn = StandardAttention::new(d_model, n_heads);
    let std_output = std_attn.forward(&input);
    let std_weights = std_attn.get_attention_weights(&input);

    println!("Output shape: {:?}", std_output.shape());
    println!("Attention weights shape: {:?}", std_weights.shape());
    print_attention_matrix(&std_weights, "Standard Attention Weights (first 5x5)");

    // 2. Higher Order Attention
    println!("\n--- Higher Order Attention (rank={}) ---", rank);
    let ho_attn = HigherOrderAttention::new(d_model, n_heads, rank);
    let ho_output = ho_attn.forward(&input);
    let ho_weights = ho_attn.get_attention_weights(&input);

    println!("Output shape: {:?}", ho_output.shape());
    println!("Attention weights shape: {:?}", ho_weights.shape());
    print_attention_matrix(&ho_weights, "Higher Order Attention Weights (first 5x5)");

    // 3. Kernel Attention
    println!("\n--- Kernel Attention (linear complexity) ---");
    let k_attn = KernelAttention::new(d_model, n_heads, 32);
    let k_output = k_attn.forward(&input);
    let k_weights = k_attn.estimate_attention_weights(&input);

    println!("Output shape: {:?}", k_output.shape());
    println!("Attention weights shape: {:?}", k_weights.shape());
    print_attention_matrix(&k_weights, "Kernel Attention Weights (first 5x5)");

    // 4. Third-Order Tensor Visualization
    println!("\n--- Third-Order Attention Tensor ---");
    println!("(For small sequences only - O(n^3) complexity)\n");

    let small_input = Array2::from_shape_fn((5, d_model), |_| rand::random::<f64>() - 0.5);
    let tensor = ho_attn.compute_full_tensor(&small_input);

    println!("Tensor shape: {:?}", tensor.shape());
    println!("\nSlice T[0, :, :] (how position 0 attends to pairs):");
    for j in 0..5 {
        for k in 0..5 {
            print!("{:6.3} ", tensor[[0, j, k]]);
        }
        println!();
    }

    // 5. CP Decomposition Demo
    println!("\n--- CP Decomposition Demo ---");
    let demo_tensor = Tensor3D::random((5, 5, 5));
    let original_norm = demo_tensor.frobenius_norm();

    println!("Original tensor shape: {:?}", demo_tensor.shape());
    println!("Original Frobenius norm: {:.4}", original_norm);

    let cp = CPDecomposition::new(3); // rank-3 approximation
    let result = cp.decompose(&demo_tensor);

    println!("\nCP Decomposition (rank=3):");
    println!("  Iterations: {}", result.iterations);
    println!("  Final error: {:.4}", result.final_error);
    println!("  Relative error: {:.2}%", result.final_error / original_norm * 100.0);

    // Compare outputs
    println!("\n--- Output Comparison ---");
    compare_outputs(&std_output, &ho_output, "Standard vs Higher Order");
    compare_outputs(&std_output, &k_output, "Standard vs Kernel");
    compare_outputs(&ho_output, &k_output, "Higher Order vs Kernel");

    println!("\nDemo complete!");
}

fn print_attention_matrix(weights: &Array2<f64>, title: &str) {
    println!("\n{}:", title);
    let n = weights.nrows().min(5);
    let m = weights.ncols().min(5);

    print!("      ");
    for j in 0..m {
        print!(" pos{} ", j);
    }
    println!();

    for i in 0..n {
        print!("pos{} ", i);
        for j in 0..m {
            print!("{:5.3} ", weights[[i, j]]);
        }
        println!();
    }
}

fn compare_outputs(out1: &Array2<f64>, out2: &Array2<f64>, name: &str) {
    // Calculate cosine similarity of outputs
    let flat1: Vec<f64> = out1.iter().cloned().collect();
    let flat2: Vec<f64> = out2.iter().cloned().collect();

    let dot: f64 = flat1.iter().zip(&flat2).map(|(a, b)| a * b).sum();
    let norm1: f64 = flat1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = flat2.iter().map(|x| x * x).sum::<f64>().sqrt();

    let cosine_sim = dot / (norm1 * norm2);

    println!("  {}: cosine similarity = {:.4}", name, cosine_sim);
}
