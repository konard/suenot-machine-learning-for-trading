use fourier_neural_operator::spectral_engine;

fn main() {
    println!("--- Fourier Neural Operator (FNO) Spectral Engine ---");
    
    // Simulating very noisy Order Book ticks
    let raw_market_data = vec![
        1.0, 5.0, -2.0, 8.0, 1.0, -1.0, 9.0, 2.0
    ];
    
    println!("Raw HFT LOB Sequence: {:?}", raw_market_data);
    
    // We only preserve the lowest 2 Macro frequencies.
    // Equivalent to PyTorch's modes=2 configuration.
    let denoised_trend = spectral_engine::perform_fno_1d_pass(&raw_market_data, 2);
    
    println!("Fourier PDE Filtered Map: {:?}", denoised_trend);
    println!("Successfully stripped unpredictable high-freq noise mapped in O(N log N)!");
}
