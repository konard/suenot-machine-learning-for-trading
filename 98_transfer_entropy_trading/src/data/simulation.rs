//! Simulated data generation for testing TE strategies.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::data::bybit::Kline;

/// Generates simulated kline and return data with lead-lag structure.
pub struct SimulatedDataGenerator {
    rng: StdRng,
}

impl SimulatedDataGenerator {
    /// Create a new generator with a given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate simulated klines for a single asset.
    ///
    /// # Arguments
    /// * `n` - Number of klines.
    /// * `start_price` - Initial price.
    /// * `volatility` - Daily volatility (std dev of returns).
    /// * `drift` - Mean return per period.
    pub fn generate_klines(
        &mut self,
        n: usize,
        start_price: f64,
        volatility: f64,
        drift: f64,
    ) -> Vec<Kline> {
        let normal = Normal::new(drift, volatility).unwrap();
        let mut price = start_price;
        let mut klines = Vec::with_capacity(n);
        let base_ts = 1_700_000_000_000i64; // Approximate millisecond timestamp

        for i in 0..n {
            let ret = normal.sample(&mut self.rng);
            let open = price;
            let close = price * (1.0 + ret);
            let high = open.max(close) * (1.0 + self.rng.gen::<f64>() * volatility);
            let low = open.min(close) * (1.0 - self.rng.gen::<f64>() * volatility);
            let volume = 100.0 + self.rng.gen::<f64>() * 900.0;

            klines.push(Kline {
                timestamp: base_ts + (i as i64) * 3_600_000,
                open,
                high,
                low: low.max(0.01),
                close: close.max(0.01),
                volume,
                turnover: volume * close,
            });

            price = close.max(0.01);
        }
        klines
    }

    /// Generate return series for multiple assets with lead-lag structure.
    ///
    /// The first asset is the leader; subsequent assets follow with
    /// increasing lag and decreasing strength.
    ///
    /// # Arguments
    /// * `n_assets` - Number of assets.
    /// * `n_periods` - Number of time periods.
    /// * `lead_lag_strength` - Base coupling strength (0 to 1).
    /// * `volatility` - Return volatility.
    pub fn generate_lead_lag_returns(
        &mut self,
        n_assets: usize,
        n_periods: usize,
        lead_lag_strength: f64,
        volatility: f64,
    ) -> Vec<Vec<f64>> {
        let normal = Normal::new(0.0, volatility).unwrap();

        // Generate leader returns
        let leader: Vec<f64> = (0..n_periods)
            .map(|_| normal.sample(&mut self.rng))
            .collect();

        let mut all_returns = vec![vec![0.0; n_periods]; n_assets];
        all_returns[0] = leader.clone();

        // Generate follower returns
        for i in 1..n_assets {
            let lag = i;
            let strength = lead_lag_strength / i as f64;
            for t in 0..n_periods {
                let noise = normal.sample(&mut self.rng);
                if t >= lag {
                    all_returns[i][t] = strength * leader[t - lag] + (1.0 - strength) * noise;
                } else {
                    all_returns[i][t] = noise;
                }
            }
        }

        all_returns
    }

    /// Generate crypto-like return series (BTC, ETH, SOL, AVAX).
    ///
    /// BTC leads all others; ETH has moderate following; SOL and AVAX
    /// are strongest followers.
    pub fn generate_crypto_returns(&mut self, n_periods: usize) -> (Vec<String>, Vec<Vec<f64>>) {
        let symbols = vec![
            "BTC".to_string(),
            "ETH".to_string(),
            "SOL".to_string(),
            "AVAX".to_string(),
        ];

        let btc_dist = Normal::new(0.0005, 0.025).unwrap();
        let btc: Vec<f64> = (0..n_periods)
            .map(|_| btc_dist.sample(&mut self.rng))
            .collect();

        let eth_dist = Normal::new(0.0003, 0.03).unwrap();
        let mut eth = vec![0.0; n_periods];
        for t in 1..n_periods {
            eth[t] = 0.25 * btc[t - 1] + 0.75 * eth_dist.sample(&mut self.rng);
        }

        let sol_dist = Normal::new(0.0002, 0.04).unwrap();
        let mut sol = vec![0.0; n_periods];
        for t in 2..n_periods {
            sol[t] = 0.15 * btc[t - 2] + 0.10 * eth[t - 1] + 0.75 * sol_dist.sample(&mut self.rng);
        }

        let avax_dist = Normal::new(0.0001, 0.045).unwrap();
        let mut avax = vec![0.0; n_periods];
        for t in 3..n_periods {
            avax[t] = 0.12 * btc[t - 3] + 0.08 * sol[t - 1] + 0.80 * avax_dist.sample(&mut self.rng);
        }

        (symbols, vec![btc, eth, sol, avax])
    }
}
