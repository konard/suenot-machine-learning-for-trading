//! Bybit exchange data client and simulated data generator.

use rand_distr::{Distribution, Normal};

/// Single candlestick data point
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
}

/// Bybit API client
pub struct BybitClient {
    base_url: String,
    client: reqwest::Client,
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Fetch historical klines from Bybit
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!("{}/v5/market/kline", self.base_url);

        let response = self.client
            .get(&url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;

        let list = data["result"]["list"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid API response"))?;

        let mut klines: Vec<Kline> = list.iter().filter_map(|item| {
            let arr = item.as_array()?;
            Some(Kline {
                timestamp: arr[0].as_str()?.parse().ok()?,
                open: arr[1].as_str()?.parse().ok()?,
                high: arr[2].as_str()?.parse().ok()?,
                low: arr[3].as_str()?.parse().ok()?,
                close: arr[4].as_str()?.parse().ok()?,
                volume: arr[5].as_str()?.parse().ok()?,
                turnover: arr[6].as_str()?.parse().ok()?,
            })
        }).collect();

        klines.reverse(); // Bybit returns descending, reverse to ascending
        Ok(klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated data generator for testing
pub struct SimulatedDataGenerator;

impl SimulatedDataGenerator {
    /// Generate random walk klines
    pub fn generate_klines(num: usize, base_price: f64, volatility: f64) -> Vec<Kline> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, volatility).unwrap();
        let mut price = base_price;
        let mut klines = Vec::with_capacity(num);

        for i in 0..num {
            let ret = normal.sample(&mut rng);
            let open = price;
            let close = price * (1.0 + ret);
            let high = open.max(close) * (1.0 + rand::random::<f64>() * 0.01);
            let low = open.min(close) * (1.0 - rand::random::<f64>() * 0.01);
            let volume = rand::random::<f64>() * 1_000_000.0;

            klines.push(Kline {
                timestamp: (i as i64) * 3600000,
                open,
                high,
                low,
                close,
                volume,
                turnover: volume * close,
            });
            price = close;
        }

        klines
    }

    /// Generate trending klines
    pub fn generate_trending_klines(
        num: usize,
        base_price: f64,
        volatility: f64,
        trend: f64,
    ) -> Vec<Kline> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, volatility).unwrap();
        let mut price = base_price;
        let mut klines = Vec::with_capacity(num);

        for i in 0..num {
            let ret = normal.sample(&mut rng) + trend;
            let open = price;
            let close = price * (1.0 + ret);
            let high = open.max(close) * (1.0 + rand::random::<f64>() * 0.01);
            let low = open.min(close) * (1.0 - rand::random::<f64>() * 0.01);
            let volume = rand::random::<f64>() * 1_000_000.0;

            klines.push(Kline {
                timestamp: (i as i64) * 3600000,
                open, high, low, close, volume,
                turnover: volume * close,
            });
            price = close;
        }

        klines
    }

    /// Generate regime-changing klines
    pub fn generate_regime_changing_klines(num: usize, base_price: f64) -> Vec<Kline> {
        let regimes = [
            (0.015, 0.0002, 0.2),
            (0.025, -0.0003, 0.15),
            (0.01, 0.0, 0.15),
            (0.02, 0.00015, 0.2),
            (0.03, -0.0001, 0.15),
            (0.012, 0.0001, 0.15),
        ];

        let mut rng = rand::thread_rng();
        let mut price = base_price;
        let mut klines = Vec::with_capacity(num);
        let mut idx = 0;

        for &(vol, trend, frac) in &regimes {
            let normal = Normal::new(0.0, vol).unwrap();
            let regime_len = (num as f64 * frac) as usize;

            for _ in 0..regime_len {
                if idx >= num { break; }

                let ret = normal.sample(&mut rng) + trend;
                let open = price;
                let close = price * (1.0 + ret);
                let high = open.max(close) * (1.0 + rand::random::<f64>() * 0.01);
                let low = open.min(close) * (1.0 - rand::random::<f64>() * 0.01);
                let volume = rand::random::<f64>() * 1_000_000.0;

                klines.push(Kline {
                    timestamp: (idx as i64) * 3600000,
                    open, high, low, close, volume,
                    turnover: volume * close,
                });
                price = close;
                idx += 1;
            }
            if idx >= num { break; }
        }

        // Fill remaining
        let normal = Normal::new(0.0, 0.015).unwrap();
        while klines.len() < num {
            let ret = normal.sample(&mut rng);
            let open = price;
            let close = price * (1.0 + ret);
            let high = open.max(close) * (1.0 + rand::random::<f64>() * 0.01);
            let low = open.min(close) * (1.0 - rand::random::<f64>() * 0.01);
            let volume = rand::random::<f64>() * 1_000_000.0;

            klines.push(Kline {
                timestamp: (klines.len() as i64) * 3600000,
                open, high, low, close, volume,
                turnover: volume * close,
            });
            price = close;
        }

        klines
    }
}
