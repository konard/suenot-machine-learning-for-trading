//! Bybit API client for cryptocurrency market data.

use serde::Deserialize;
use tracing::{info, warn};

/// Kline (candlestick) data for a single time period.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures.
#[derive(Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: Option<BybitResult>,
}

#[derive(Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Client for the Bybit public API.
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit API client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch kline data for a symbol.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (e.g., "60" for 1 hour)
    /// * `limit` - Number of candles (max 200)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        info!("Fetching klines: symbol={}, interval={}, limit={}", symbol, interval, limit);

        let resp = self
            .client
            .get(&url)
            .query(&[
                ("category", "linear"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.min(200).to_string()),
            ])
            .send()
            .await;

        match resp {
            Ok(r) => {
                let data: BybitResponse = r.json().await?;

                if data.ret_code != 0 {
                    warn!("Bybit API error: {}", data.ret_msg);
                    return Ok(generate_simulated_klines(symbol, limit));
                }

                let list = match data.result {
                    Some(r) => r.list,
                    None => {
                        warn!("Empty result from Bybit for {}", symbol);
                        return Ok(generate_simulated_klines(symbol, limit));
                    }
                };

                // Bybit returns newest first, reverse for chronological order
                let mut klines: Vec<Kline> = list
                    .iter()
                    .rev()
                    .filter_map(|r| {
                        if r.len() >= 6 {
                            Some(Kline {
                                timestamp: r[0].parse().unwrap_or(0),
                                open: r[1].parse().unwrap_or(0.0),
                                high: r[2].parse().unwrap_or(0.0),
                                low: r[3].parse().unwrap_or(0.0),
                                close: r[4].parse().unwrap_or(0.0),
                                volume: r[5].parse().unwrap_or(0.0),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                // Deduplicate by timestamp
                klines.dedup_by_key(|k| k.timestamp);

                Ok(klines)
            }
            Err(e) => {
                warn!("Failed to fetch Bybit data for {}: {}. Using simulated data.", symbol, e);
                Ok(generate_simulated_klines(symbol, limit))
            }
        }
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate simulated kline data for offline testing.
pub fn generate_simulated_klines(symbol: &str, n_points: usize) -> Vec<Kline> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal, Uniform};

    let seed = symbol.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let base_price = match symbol {
        "BTCUSDT" => 60000.0,
        "ETHUSDT" => 3000.0,
        "SOLUSDT" => 150.0,
        "AVAXUSDT" => 35.0,
        "MATICUSDT" => 0.8,
        _ => 100.0,
    };

    let returns_dist = Normal::new(0.0001, 0.02).unwrap();
    let noise_dist = Uniform::new(0.995, 1.005);
    let high_dist = Uniform::new(1.0, 1.02);
    let low_dist = Uniform::new(0.98, 1.0);

    let mut close = base_price;
    let mut klines = Vec::with_capacity(n_points);

    for t in 0..n_points {
        let ret: f64 = returns_dist.sample(&mut rng);
        close *= 1.0 + ret;

        let noise: f64 = noise_dist.sample(&mut rng);
        let open = close * noise;
        let high = open.max(close) * high_dist.sample(&mut rng);
        let low = open.min(close) * low_dist.sample(&mut rng);
        let volume = base_price * 10.0 * (1.0 + ret.abs() * 10.0);

        klines.push(Kline {
            timestamp: (1700000000 + t * 3600) as i64,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    klines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_klines() {
        let klines = generate_simulated_klines("BTCUSDT", 100);
        assert_eq!(klines.len(), 100);
        assert!(klines[0].close > 0.0);
        assert!(klines[0].high >= klines[0].low);
    }
}
