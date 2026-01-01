//! Example: Dilated Convolution Demonstration
//!
//! This example demonstrates:
//! 1. How dilated convolutions work
//! 2. Receptive field calculations
//! 3. Multi-scale feature extraction

use dilated_conv_trading::conv::{
    calculate_receptive_field, DilatedConv1D, DilatedConvStack, MultiScaleDilatedConv,
};
use ndarray::Array2;

fn main() {
    println!("=== Dilated Convolution Demo ===\n");

    // 1. Demonstrate receptive field growth
    println!("1. Receptive Field Growth\n");
    println!("   Kernel size = 3, varying dilation rates:\n");

    for dilation in [1, 2, 4, 8, 16, 32] {
        let rf = calculate_receptive_field(3, &[dilation]);
        println!("   Dilation = {:2} → Receptive Field = {:3} timesteps", dilation, rf);
    }

    println!("\n   Stacking layers with exponential dilation:\n");
    let stack_configs = [
        vec![1, 2],
        vec![1, 2, 4],
        vec![1, 2, 4, 8],
        vec![1, 2, 4, 8, 16],
        vec![1, 2, 4, 8, 16, 32],
        vec![1, 2, 4, 8, 16, 32, 64],
        vec![1, 2, 4, 8, 16, 32, 64, 128],
    ];

    for config in &stack_configs {
        let rf = calculate_receptive_field(3, config);
        println!(
            "   {} layers → Receptive Field = {:4} timesteps",
            config.len(),
            rf
        );
    }

    // 2. Single dilated convolution
    println!("\n2. Single Dilated Convolution\n");

    let conv = DilatedConv1D::new(5, 32, 3, 4);
    println!("   Input channels: 5");
    println!("   Output channels: 32");
    println!("   Kernel size: 3");
    println!("   Dilation: 4");
    println!("   Receptive field: {} timesteps", conv.receptive_field());

    // Create sample input
    let input = Array2::from_shape_fn((5, 100), |(i, j)| {
        ((i as f64 * 0.1) + (j as f64 * 0.01)).sin()
    });

    let output = conv.forward(&input);
    println!("\n   Input shape: {:?}", input.dim());
    println!("   Output shape: {:?}", output.dim());

    // 3. Multi-scale convolution
    println!("\n3. Multi-Scale Dilated Convolution\n");

    let multi_scale = MultiScaleDilatedConv::new(5, 16, 3, &[1, 2, 4, 8]);
    println!("   Dilation rates: [1, 2, 4, 8]");
    println!("   Channels per scale: 16");
    println!("   Total output channels: {}", multi_scale.out_channels());

    let multi_output = multi_scale.forward(&input);
    println!("   Output shape: {:?}", multi_output.dim());

    // 4. Full WaveNet-style stack
    println!("\n4. WaveNet-Style Stack\n");

    let wavenet = DilatedConvStack::new(5, 32, &[1, 2, 4, 8, 16, 32]);
    println!("   Input channels: 5");
    println!("   Residual channels: 32");
    println!("   Number of blocks: {}", wavenet.num_blocks());
    println!("   Total receptive field: {} timesteps", wavenet.receptive_field());

    let wavenet_output = wavenet.forward(&input);
    println!("\n   Input shape: {:?}", input.dim());
    println!("   Output shape: {:?}", wavenet_output.dim());

    // Get prediction for last timestep
    let prediction = wavenet.predict_last(&input);
    println!("\n   Last timestep prediction: [{:.4}, {:.4}, {:.4}]",
        prediction[0], prediction[1], prediction[2]);
    println!("   (direction, magnitude, volatility)");

    // 5. Feature extraction
    println!("\n5. Multi-Scale Feature Extraction\n");

    let features = wavenet.extract_features(&input);
    println!("   Extracted {} feature maps from {} blocks:", features.len(), wavenet.num_blocks());

    for (i, feat) in features.iter().enumerate() {
        let dilation = 2_usize.pow(i as u32);
        println!("   Block {} (d={}): shape {:?}", i + 1, dilation, feat.dim());
    }

    // 6. Demonstrate causality
    println!("\n6. Causality Check\n");
    println!("   Verifying that output only depends on past inputs...\n");

    // Create two inputs that differ only in the future
    let mut input1 = Array2::zeros((5, 50));
    let mut input2 = Array2::zeros((5, 50));

    // Same past (t < 40)
    for t in 0..40 {
        for c in 0..5 {
            let val = (t as f64 * 0.1).sin();
            input1[[c, t]] = val;
            input2[[c, t]] = val;
        }
    }

    // Different future (t >= 40)
    for t in 40..50 {
        for c in 0..5 {
            input1[[c, t]] = 1.0;
            input2[[c, t]] = -1.0;
        }
    }

    let small_conv = DilatedConv1D::new(5, 8, 3, 1);
    let out1 = small_conv.forward(&input1);
    let out2 = small_conv.forward(&input2);

    // Check that outputs at t < 40 are the same
    let mut same = true;
    for t in 0..40 {
        for c in 0..8 {
            if (out1[[c, t]] - out2[[c, t]]).abs() > 1e-10 {
                same = false;
                break;
            }
        }
    }

    if same {
        println!("   ✓ Output at t < 40 is identical for both inputs");
        println!("   ✓ The convolution is causal (no future information leakage)");
    } else {
        println!("   ✗ Warning: Outputs differ - causality violation detected!");
    }

    println!("\n=== Demo Complete ===");
}
