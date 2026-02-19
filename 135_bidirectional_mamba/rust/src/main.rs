use ndarray::array;
use bidirectional_mamba::trading_mamba;

fn main() {
    println!("--- Vision Mamba / Bidirectional Rust Sequence Modeling Inference ---");
    
    // Simulate a multi-variate continuous limit order book or bar window
    // with N sequence length and 2 features
    let historical_window = array![
        [10.0, 15.0], // Time t-2
        [12.0, 16.0], // Time t-1
        [14.0, 18.0]  // Time t (Present)
    ];
    
    // Fusing all historical contextual information strictly symmetrically
    let predicted_score = trading_mamba::bidirectional_contextual_sweep(&historical_window);
    
    println!("Aggregated Market Representation Embedding Value: {}", predicted_score);
    println!("Completed strictly safe historical context fusion in O(N).");
}
