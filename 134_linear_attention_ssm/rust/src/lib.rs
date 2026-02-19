pub mod stock_market {
    pub fn process_stock_data(symbol: &str, prices: &[f64]) -> f64 {
        println!("Processing stock data for {}", symbol);
        // Simplified Linear Attention logic for SSM
        let mut moving_average = 0.0;
        for price in prices {
            moving_average += price;
        }
        if !prices.is_empty() {
            moving_average /= prices.len() as f64;
        }
        moving_average
    }
}

pub mod crypto_bybit {
    pub fn process_crypto_data(symbol: &str, prices: &[f64]) -> f64 {
        println!("Processing Bybit crypto data for {}", symbol);
        // Linear Attention step approximation
        let mut moving_average = 0.0;
        for price in prices {
            moving_average += price;
        }
        if !prices.is_empty() {
            moving_average /= prices.len() as f64;
        }
        moving_average
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stock_data() {
        let avg = stock_market::process_stock_data("AAPL", &[150.0, 152.0, 154.0]);
        assert_eq!(avg, 152.0);
    }

    #[test]
    fn test_crypto_data() {
        let avg = crypto_bybit::process_crypto_data("BTCUSDT", &[50000.0, 52000.0, 54000.0]);
        assert_eq!(avg, 52000.0);
    }
}
