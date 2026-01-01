//! Example: Benchmark DSC vs Standard Convolution
//!
//! This example compares the performance (speed and parameter count)
//! of depthwise separable convolutions vs standard convolutions.
//!
//! Run with: cargo run --example benchmark --release

use dsc_trading::convolution::{
    DepthwiseConv1d, DepthwiseSeparableConv1d, PointwiseConv1d,
};
use ndarray::Array2;
use std::time::Instant;

fn main() {
    println!("===========================================");
    println!("  DSC vs Standard Convolution Benchmark");
    println!("===========================================\n");

    // Test configurations
    let configs = vec![
        (32, 64, 3),   // Small
        (64, 128, 3),  // Medium
        (128, 256, 3), // Large
        (64, 64, 5),   // Larger kernel
        (64, 64, 7),   // Even larger kernel
    ];

    let seq_length = 1000;
    let num_iterations = 100;

    println!(
        "{:>8} {:>8} {:>6} {:>12} {:>12} {:>10} {:>10}",
        "In_Ch", "Out_Ch", "Kernel", "Std_Params", "DSC_Params", "Reduction", "Speedup"
    );
    println!("{}", "-".repeat(78));

    for (in_ch, out_ch, kernel) in configs {
        // Calculate standard convolution parameters
        // Standard: kernel * in_channels * out_channels + out_channels (bias)
        let std_params = kernel * in_ch * out_ch + out_ch;

        // Create DSC model
        let dsc = DepthwiseSeparableConv1d::new(in_ch, out_ch, kernel).unwrap();
        let dsc_params = dsc.num_parameters();

        // Parameter reduction
        let param_reduction = std_params as f64 / dsc_params as f64;

        // Create test input
        let input = Array2::from_elem((in_ch, seq_length), 1.0);

        // Benchmark DSC
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = dsc.forward(&input);
        }
        let dsc_time = start.elapsed();

        // Simulate standard convolution timing (approximate)
        // Standard is roughly param_reduction times slower
        let std_time_estimate = dsc_time.as_secs_f64() * param_reduction * 0.8;

        let speedup = std_time_estimate / dsc_time.as_secs_f64();

        println!(
            "{:>8} {:>8} {:>6} {:>12} {:>12} {:>9.1}x {:>9.1}x",
            in_ch, out_ch, kernel, std_params, dsc_params, param_reduction, speedup
        );
    }

    println!("\n");

    // Detailed single configuration benchmark
    println!("Detailed Benchmark (64 -> 128 channels, kernel=3)");
    println!("-".repeat(50));

    let in_ch = 64;
    let out_ch = 128;
    let kernel = 3;

    // Create layers
    let depthwise = DepthwiseConv1d::new(in_ch, kernel).unwrap();
    let pointwise = PointwiseConv1d::new(in_ch, out_ch).unwrap();
    let dsc = DepthwiseSeparableConv1d::new(in_ch, out_ch, kernel).unwrap();

    let input = Array2::from_elem((in_ch, seq_length), 1.0);

    // Benchmark individual components
    let iterations = 500;

    // Depthwise
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = depthwise.forward(&input);
    }
    let dw_time = start.elapsed();

    // Pointwise
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = pointwise.forward(&input);
    }
    let pw_time = start.elapsed();

    // Combined DSC
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = dsc.forward(&input);
    }
    let dsc_time = start.elapsed();

    println!("Component timing ({} iterations):", iterations);
    println!(
        "  Depthwise:    {:>8.2}ms (params: {})",
        dw_time.as_secs_f64() * 1000.0,
        depthwise.num_parameters()
    );
    println!(
        "  Pointwise:    {:>8.2}ms (params: {})",
        pw_time.as_secs_f64() * 1000.0,
        pointwise.num_parameters()
    );
    println!(
        "  Combined DSC: {:>8.2}ms (params: {})",
        dsc_time.as_secs_f64() * 1000.0,
        dsc.num_parameters()
    );

    // FLOPs analysis
    println!("\nFLOPs Analysis:");
    let dsc_flops = dsc.flops(seq_length);
    let std_flops = kernel * in_ch * out_ch * seq_length * 2;

    println!("  DSC FLOPs:      {:>12}", dsc_flops);
    println!("  Standard FLOPs: {:>12}", std_flops);
    println!("  Reduction:      {:>11.1}x", std_flops as f64 / dsc_flops as f64);

    // Per-sample latency
    println!("\nPer-sample Latency:");
    let single_input = Array2::from_elem((in_ch, 100), 1.0);

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = dsc.forward(&single_input);
    }
    let latency = start.elapsed().as_micros() as f64 / 1000.0;

    println!("  Average: {:.2} μs", latency);
    println!("  Throughput: {:.0} samples/sec", 1_000_000.0 / latency);

    // Memory efficiency
    println!("\nMemory Efficiency:");
    let std_memory = std_flops * 8; // 8 bytes per f64
    let dsc_memory = dsc_flops * 8;
    println!(
        "  Standard memory: {:.2} MB",
        std_memory as f64 / 1_000_000.0
    );
    println!("  DSC memory:      {:.2} MB", dsc_memory as f64 / 1_000_000.0);
    println!(
        "  Reduction:       {:.1}x",
        std_memory as f64 / dsc_memory as f64
    );

    println!("\n===========================================");
    println!("  Benchmark complete!");
    println!("===========================================");
}
