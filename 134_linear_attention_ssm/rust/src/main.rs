mod lib;

fn main() {
    println!("--- Linear Attention SSM Rust Inference Tool ---");
    // Test stock inference
    let stock_prices = vec![150.0, 152.0, 154.0];
    let avg = lib::stock_market::process_stock_data("AAPL", &stock_prices);
    println!("AAPL Output Score: {}", avg);

    // Test crypto inference
    let crypto_prices = vec![50000.0, 52000.0, 54000.0];
    let btc_avg = lib::crypto_bybit::process_crypto_data("BTCUSDT", &crypto_prices);
    println!("BTCUSDT Output Score: {}", btc_avg);
}
