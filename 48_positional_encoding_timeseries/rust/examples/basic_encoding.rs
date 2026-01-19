//! Example: Basic Positional Encoding
//!
//! This example demonstrates the fundamental positional encoding types:
//! sinusoidal, learned, relative, and rotary (RoPE).

use positional_encoding::{
    SinusoidalEncoding, LearnedEncoding, RelativeEncoding, RotaryEncoding,
    PositionalEncoding,
};
use ndarray::Array1;

fn main() {
    println!("Basic Positional Encoding Example");
    println!("==================================\n");

    // Sinusoidal Encoding
    println!("1. Sinusoidal Encoding");
    println!("-----------------------");

    let sinusoidal = SinusoidalEncoding::new(8, 100);
    let positions = vec![0, 1, 2, 5, 10];
    let encoded = sinusoidal.encode(&positions);

    println!("Dimension: {}", sinusoidal.dim());
    println!("Positions: {:?}", positions);
    for (i, pos) in positions.iter().enumerate() {
        let row: Vec<f64> = encoded.row(i).iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
        println!("  pos {}: {:?}", pos, row);
    }

    // Key insight: Position 0 has sin(0)=0, cos(0)=1 pattern
    println!("\nNote: At position 0, even indices are 0 (sin), odd indices are 1 (cos)");

    // Learned Encoding
    println!("\n2. Learned Encoding");
    println!("-------------------");

    let learned = LearnedEncoding::new(8, 100);
    let learned_encoded = learned.encode(&[0, 1, 2]);

    println!("Learned embeddings (randomly initialized):");
    for i in 0..3 {
        let row: Vec<f64> = learned_encoded.row(i).iter().map(|x| (x * 100.0).round() / 100.0).collect();
        println!("  pos {}: {:?}", i, row);
    }
    println!("\nNote: These would be trained during model training");

    // Relative Encoding
    println!("\n3. Relative Encoding");
    println!("--------------------");

    let relative = RelativeEncoding::new(8, 50);

    // Show relative encodings between different positions
    let pairs = vec![(0, 5), (5, 0), (10, 15), (15, 10)];
    for (from, to) in pairs {
        let enc = relative.encode_relative(from, to);
        let row: Vec<f64> = enc.iter().map(|x| (x * 1000.0).round() / 1000.0).collect();
        println!("  {} -> {} (dist {}): {:?}", from, to, (to as i32 - from as i32).abs(), row);
    }
    println!("\nNote: Same distance = same encoding magnitude, direction differs by sign");

    // Rotary Encoding (RoPE)
    println!("\n4. Rotary Encoding (RoPE)");
    println!("-------------------------");

    let rope = RotaryEncoding::new(8, 100);

    // Apply rotation to a sample vector
    let x = Array1::from_vec(vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5]);
    println!("Original vector: {:?}", x.as_slice().unwrap());

    for pos in [0, 1, 5, 10] {
        let rotated = rope.apply_rotation(&x, pos);
        let row: Vec<f64> = rotated.iter().map(|v| (v * 1000.0).round() / 1000.0).collect();
        println!("  Rotated (pos {}): {:?}", pos, row);
    }

    // Verify norm preservation
    let orig_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rot_norm: f64 = rope.apply_rotation(&x, 50).iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("\nNorm preservation: original={:.4}, rotated={:.4}", orig_norm, rot_norm);
    println!("Note: RoPE preserves vector magnitude (rotation property)");

    // Comparison
    println!("\n5. Encoding Comparison");
    println!("----------------------");
    println!("| Encoding   | Learnable | Relative | Extrapolation |");
    println!("|------------|-----------|----------|---------------|");
    println!("| Sinusoidal | No        | No       | Good          |");
    println!("| Learned    | Yes       | No       | Poor          |");
    println!("| Relative   | No        | Yes      | Good          |");
    println!("| RoPE       | No        | Yes      | Excellent     |");

    println!("\nRecommendations:");
    println!("- Short fixed sequences: Sinusoidal or Learned");
    println!("- Variable length: Relative");
    println!("- Long sequences: RoPE");
    println!("- Trading/Time series: RoPE + Calendar encoding");
}
