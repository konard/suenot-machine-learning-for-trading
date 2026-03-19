//! Stock market data provider with simulated fallback.
//!
//! Provides stock market data for traditional equities. Uses simulated
//! data with realistic sector-based volatility parameters for offline
//! testing and environments without market data API access.

use tracing::info;

use super::bybit::Kline;

/// Stock market parameters for realistic simulation.
struct StockParams {
    base_price: f64,
    annual_volatility: f64,
}

/// Get stock simulation parameters by ticker.
fn get_stock_params(ticker: &str) -> StockParams {
    match ticker {
        "AAPL" => StockParams { base_price: 180.0, annual_volatility: 0.25 },
        "MSFT" => StockParams { base_price: 420.0, annual_volatility: 0.23 },
        "GOOGL" => StockParams { base_price: 175.0, annual_volatility: 0.27 },
        "AMZN" => StockParams { base_price: 185.0, annual_volatility: 0.30 },
        "META" => StockParams { base_price: 500.0, annual_volatility: 0.35 },
        "NVDA" => StockParams { base_price: 880.0, annual_volatility: 0.45 },
        "TSLA" => StockParams { base_price: 250.0, annual_volatility: 0.50 },
        "JPM" => StockParams { base_price: 200.0, annual_volatility: 0.22 },
        "V" => StockParams { base_price: 280.0, annual_volatility: 0.20 },
        "JNJ" => StockParams { base_price: 155.0, annual_volatility: 0.15 },
        "WMT" => StockParams { base_price: 170.0, annual_volatility: 0.18 },
        "XOM" => StockParams { base_price: 105.0, annual_volatility: 0.25 },
        _ => StockParams { base_price: 100.0, annual_volatility: 0.25 },
    }
}

/// Generate simulated stock market kline data.
///
/// Uses geometric Brownian motion with sector-appropriate volatility
/// and daily timestamps (252 trading days per year).
///
/// # Arguments
/// * `ticker` - Stock ticker symbol (e.g., "AAPL")
/// * `n_points` - Number of daily candles to generate
pub fn generate_simulated_stock_klines(ticker: &str, n_points: usize) -> Vec<Kline> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal, Uniform};

    let seed = ticker.bytes().fold(42u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let params = get_stock_params(ticker);
    let daily_vol = params.annual_volatility / (252.0_f64).sqrt();
    let daily_drift = 0.0002; // small positive drift

    let returns_dist = Normal::new(daily_drift, daily_vol).unwrap();
    let noise_dist = Uniform::new(0.998, 1.002);
    let high_dist = Uniform::new(1.0, 1.015);
    let low_dist = Uniform::new(0.985, 1.0);

    let mut close = params.base_price;
    let mut klines = Vec::with_capacity(n_points);

    // Start from a fixed date (2023-01-03, first trading day of 2023)
    let base_timestamp: i64 = 1672704000;

    for t in 0..n_points {
        let ret: f64 = returns_dist.sample(&mut rng);
        close *= 1.0 + ret;

        let noise: f64 = noise_dist.sample(&mut rng);
        let open = close * noise;
        let high = open.max(close) * high_dist.sample(&mut rng);
        let low = open.min(close) * low_dist.sample(&mut rng);
        let volume = params.base_price * 50000.0 * (1.0 + ret.abs() * 5.0);

        klines.push(Kline {
            timestamp: base_timestamp + (t as i64) * 86400, // daily intervals
            open,
            high,
            low,
            close,
            volume,
        });
    }

    info!(
        "Generated {} simulated daily klines for {} (base_price={:.2}, vol={:.2}%)",
        n_points, ticker, params.base_price, params.annual_volatility * 100.0
    );

    klines
}

/// Load stock data for multiple tickers.
///
/// Returns a vector of (ticker, klines) pairs.
///
/// # Arguments
/// * `tickers` - Stock ticker symbols
/// * `n_points` - Number of daily data points per ticker
pub fn load_stock_data(tickers: &[&str], n_points: usize) -> Vec<(String, Vec<Kline>)> {
    tickers
        .iter()
        .map(|ticker| {
            let klines = generate_simulated_stock_klines(ticker, n_points);
            (ticker.to_string(), klines)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_stock_klines() {
        let klines = generate_simulated_stock_klines("AAPL", 252);
        assert_eq!(klines.len(), 252);
        assert!(klines[0].close > 0.0);
        assert!(klines[0].high >= klines[0].low);
        // Daily timestamps (86400 seconds apart)
        assert_eq!(klines[1].timestamp - klines[0].timestamp, 86400);
    }

    #[test]
    fn test_different_tickers_different_prices() {
        let aapl = generate_simulated_stock_klines("AAPL", 10);
        let msft = generate_simulated_stock_klines("MSFT", 10);
        // Different base prices should produce different series
        assert!((aapl[0].close - msft[0].close).abs() > 10.0);
    }

    #[test]
    fn test_load_stock_data() {
        let data = load_stock_data(&["AAPL", "MSFT", "GOOGL"], 100);
        assert_eq!(data.len(), 3);
        assert_eq!(data[0].0, "AAPL");
        assert_eq!(data[0].1.len(), 100);
    }
}
