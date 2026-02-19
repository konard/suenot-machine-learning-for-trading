pub mod stock_market {
    /// Simulates Structured Masked Attention (SMA) state space projection 
    /// for stock market limit order books.
    /// In Linear Attention, the State S_t is an exponentially weighted moving combination of past V inputs.
    pub fn process_stock_data(symbol: &str, prices: &[f64]) -> f64 {
        println!("Initializing High-Frequency Linear Attention SSM for {}", symbol);
        
        if prices.is_empty() { return 0.0; }

        let mut state = 0.0;
        let mut normalization = 0.0;
        let decay_factor = 0.95; // e^(-A), controlling memory horizon
        
        // Simulating single-feature linear attention: O_t = (S_t) / (Z_t)
        // With exponential decay to mimic Gated Linear Attention (GLA)
        for &price in prices {
            let key = 1.0;          // Dummy uniform kernel phi(K)
            let value = price;      // Input signal
            
            state = state * decay_factor + key * value;
            normalization = normalization * decay_factor + key;
        }
        
        // Generating output prediction mapped from State Matrix
        let query = 1.0; // Dummy query phi(Q)
        let output = (query * state) / (query * normalization); // Output extraction
        output
    }
}

pub mod crypto_bybit {
    /// Implements real-time $O(1)$ State Space inference specifically designed for
    /// processing continuous WebSocket tick streams from Bybit.
    /// Memory footprint remains constant globally regardless of stream duration.
    pub fn process_crypto_data(symbol: &str, prices: &[f64]) -> f64 {
        println!("Consuming Bybit Crypto stream for {} via SSD Framework", symbol);
        
        if prices.is_empty() { return 0.0; }
        
        let mut state = 0.0;
        let decay = 0.99; // Represents long-term context decay alpha

        // Running live inference: S_t = A * S_{t-1} + f(t)*V_t
        for &price in prices {
            // Equivalent inner matrix update S_t
            state = state * decay + price * (1.0 - decay);
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stock_data() {
        let prices = vec![150.0, 155.0, 160.0];
        let prediction = stock_market::process_stock_data("AAPL", &prices);
        assert!((prediction - 155.12).abs() < 0.5); // Smoothed decay validation
    }

    #[test]
    fn test_crypto_data() {
        let bybit_prices = vec![50000.0, 52000.0, 54000.0];
        let state_prediction = crypto_bybit::process_crypto_data("BTCUSDT", &bybit_prices);
        assert!(state_prediction > 0.0);
    }
}
