use neural_implicit_finance::calibration;
use ndarray::array;

fn main() {
    println!("--- Neural Implicit Finance (SIREN) Production Inference ---");
    
    // Load the calibrated model (mapping Moneyness/Maturity to Volatility)
    let model = calibration::get_mock_model();
    
    // Coordinate to query: Moneyness = 0.0 (At-the-money), Maturity = 1.0 (1 Year)
    let query_point = array![[0.0, 1.0]];
    
    println!("Querying IV Surface at point [M=0, T=1]...");
    
    let result = model.forward(&query_point);
    
    println!("Resulting Implied Volatility: {:.6}", result[[0, 0]]);
    println!("Inference successfully optimized via Rust static math.");
}
