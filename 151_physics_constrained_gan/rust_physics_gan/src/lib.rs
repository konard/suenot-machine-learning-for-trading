//! # Physics-Constrained GAN for Trading
//!
//! This library implements the core components for a physics-constrained
//! generative adversarial network applied to financial time series.
//!
//! ## Components
//!
//! - **Data structures**: OHLCV candles, return windows, trading signals
//! - **Bybit API client**: Fetching cryptocurrency market data
//! - **Physics constraints**: Financial stylized facts as differentiable penalties
//! - **Statistical utilities**: ACF, kurtosis, leverage effect computation
//! - **Simplified GAN**: Matrix-based generator/discriminator (demonstration)
//! - **Heston model**: Synthetic data generation with realistic properties
//!
//! ## Note
//!
//! The Rust implementation focuses on data fetching, preprocessing, statistical
//! validation, and backtesting. For GPU-accelerated GAN training, the Python
//! implementation with PyTorch is recommended.

pub mod bybit;
pub mod physics;
pub mod statistics;
pub mod heston;
pub mod trading;
pub mod gan;

use serde::{Deserialize, Serialize};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════════
// Core Data Structures
// ═══════════════════════════════════════════════════════════════════════════════

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// A window of log returns for GAN training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnWindow {
    pub returns: Vec<f64>,
    pub regime: MarketRegime,
    pub vol_level: VolatilityLevel,
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull = 0,
    Bear = 1,
    Sideways = 2,
    Crisis = 3,
}

impl fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketRegime::Bull => write!(f, "Bull"),
            MarketRegime::Bear => write!(f, "Bear"),
            MarketRegime::Sideways => write!(f, "Sideways"),
            MarketRegime::Crisis => write!(f, "Crisis"),
        }
    }
}

/// Volatility level classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VolatilityLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Extreme = 3,
}

impl fmt::Display for VolatilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VolatilityLevel::Low => write!(f, "Low"),
            VolatilityLevel::Medium => write!(f, "Medium"),
            VolatilityLevel::High => write!(f, "High"),
            VolatilityLevel::Extreme => write!(f, "Extreme"),
        }
    }
}

/// Physics constraint evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsReport {
    pub martingale_loss: f64,
    pub volatility_clustering_loss: f64,
    pub kurtosis_loss: f64,
    pub leverage_effect_loss: f64,
    pub autocorrelation_loss: f64,
    pub total_physics_loss: f64,
    pub statistics: ReturnStatistics,
}

impl fmt::Display for PhysicsReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Physics Constraint Report:")?;
        writeln!(f, "  Martingale loss:      {:.6}", self.martingale_loss)?;
        writeln!(f, "  Vol clustering loss:  {:.6}", self.volatility_clustering_loss)?;
        writeln!(f, "  Kurtosis loss:        {:.6}", self.kurtosis_loss)?;
        writeln!(f, "  Leverage effect loss: {:.6}", self.leverage_effect_loss)?;
        writeln!(f, "  Autocorrelation loss: {:.6}", self.autocorrelation_loss)?;
        writeln!(f, "  Total physics loss:   {:.6}", self.total_physics_loss)?;
        writeln!(f, "  Statistics:")?;
        write!(f, "{}", self.statistics)
    }
}

/// Summary statistics for return distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnStatistics {
    pub mean: f64,
    pub std: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub min: f64,
    pub max: f64,
    pub acf_abs_lag1: f64,
    pub acf_abs_lag5: f64,
    pub acf_sq_lag1: f64,
    pub leverage_corr: f64,
}

impl fmt::Display for ReturnStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    Mean:             {:.8}", self.mean)?;
        writeln!(f, "    Std:              {:.6}", self.std)?;
        writeln!(f, "    Skewness:         {:.4}", self.skewness)?;
        writeln!(f, "    Excess Kurtosis:  {:.4}", self.kurtosis)?;
        writeln!(f, "    Min:              {:.6}", self.min)?;
        writeln!(f, "    Max:              {:.6}", self.max)?;
        writeln!(f, "    ACF(|r|, lag=1):  {:.4}", self.acf_abs_lag1)?;
        writeln!(f, "    ACF(|r|, lag=5):  {:.4}", self.acf_abs_lag5)?;
        writeln!(f, "    ACF(r^2, lag=1):  {:.4}", self.acf_sq_lag1)?;
        write!(f, "    Leverage Corr:    {:.4}", self.leverage_corr)
    }
}

/// Trading signal from GAN-based strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub direction: SignalDirection,
    pub confidence: f64,
    pub position_size: f64,
    pub expected_return: f64,
    pub expected_risk: f64,
    pub regime: MarketRegime,
    pub vol_level: VolatilityLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SignalDirection {
    Long,
    Short,
    Neutral,
}

impl fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalDirection::Long => write!(f, "LONG"),
            SignalDirection::Short => write!(f, "SHORT"),
            SignalDirection::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// Backtest result summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub n_trades: usize,
    pub equity_curve: Vec<f64>,
}

impl fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Backtest Results:")?;
        writeln!(f, "  Total Return:    {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "  Annual Return:   {:.2}%", self.annualized_return * 100.0)?;
        writeln!(f, "  Sharpe Ratio:    {:.3}", self.sharpe_ratio)?;
        writeln!(f, "  Max Drawdown:    {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Win Rate:        {:.2}%", self.win_rate * 100.0)?;
        write!(f, "  Num Trades:      {}", self.n_trades)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute log returns from price series
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }
    prices.windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Convert returns to price paths
pub fn returns_to_prices(returns: &[f64], initial_price: f64) -> Vec<f64> {
    let mut prices = Vec::with_capacity(returns.len() + 1);
    prices.push(initial_price);
    let mut price = initial_price;
    for &r in returns {
        price *= (1.0 + r);
        prices.push(price);
    }
    prices
}

/// Segment a time series into overlapping windows
pub fn create_windows(data: &[f64], window_len: usize, stride: usize) -> Vec<Vec<f64>> {
    let mut windows = Vec::new();
    let n = data.len();
    let mut start = 0;
    while start + window_len <= n {
        windows.push(data[start..start + window_len].to_vec());
        start += stride;
    }
    windows
}

/// Detect market regime from recent returns
pub fn detect_regime(returns: &[f64]) -> (MarketRegime, VolatilityLevel) {
    if returns.is_empty() {
        return (MarketRegime::Sideways, VolatilityLevel::Medium);
    }

    let cum_return: f64 = returns.iter().sum();
    let volatility = statistics::std_dev(returns);

    let regime = if volatility > 0.05 {
        MarketRegime::Crisis
    } else if cum_return > 0.05 {
        MarketRegime::Bull
    } else if cum_return < -0.05 {
        MarketRegime::Bear
    } else {
        MarketRegime::Sideways
    };

    let vol_level = if volatility < 0.01 {
        VolatilityLevel::Low
    } else if volatility < 0.02 {
        VolatilityLevel::Medium
    } else if volatility < 0.04 {
        VolatilityLevel::High
    } else {
        VolatilityLevel::Extreme
    };

    (regime, vol_level)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bybit API Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod bybit {
    use super::*;
    use reqwest::blocking::Client;

    /// Bybit REST API client for fetching OHLCV data
    pub struct BybitClient {
        client: Client,
        base_url: String,
    }

    #[derive(Deserialize)]
    struct BybitResponse {
        #[serde(rename = "retCode")]
        ret_code: i32,
        #[serde(rename = "retMsg")]
        ret_msg: String,
        result: BybitResult,
    }

    #[derive(Deserialize)]
    struct BybitResult {
        list: Vec<Vec<serde_json::Value>>,
    }

    impl BybitClient {
        pub fn new() -> Self {
            Self {
                client: Client::builder()
                    .timeout(std::time::Duration::from_secs(30))
                    .build()
                    .expect("Failed to create HTTP client"),
                base_url: "https://api.bybit.com".to_string(),
            }
        }

        /// Fetch kline (candlestick) data
        pub fn fetch_klines(
            &self,
            symbol: &str,
            interval: &str,
            limit: usize,
        ) -> Result<Vec<Candle>, anyhow::Error> {
            let url = format!("{}/v5/market/kline", self.base_url);
            let resp: BybitResponse = self.client
                .get(&url)
                .query(&[
                    ("category", "spot"),
                    ("symbol", symbol),
                    ("interval", interval),
                    ("limit", &limit.min(1000).to_string()),
                ])
                .send()?
                .json()?;

            if resp.ret_code != 0 {
                anyhow::bail!("Bybit API error: {}", resp.ret_msg);
            }

            let mut candles: Vec<Candle> = resp.result.list.iter()
                .filter_map(|row| {
                    if row.len() >= 6 {
                        Some(Candle {
                            timestamp: row[0].as_str()?.parse::<i64>().ok()?,
                            open: row[1].as_str()?.parse::<f64>().ok()?,
                            high: row[2].as_str()?.parse::<f64>().ok()?,
                            low: row[3].as_str()?.parse::<f64>().ok()?,
                            close: row[4].as_str()?.parse::<f64>().ok()?,
                            volume: row[5].as_str()?.parse::<f64>().ok()?,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            // Bybit returns newest first; reverse to chronological order
            candles.reverse();
            Ok(candles)
        }

        /// Fetch extended data by paginating
        pub fn fetch_extended(
            &self,
            symbol: &str,
            interval: &str,
            n_candles: usize,
        ) -> Result<Vec<Candle>, anyhow::Error> {
            let mut all_candles = Vec::new();
            let mut end_time: Option<i64> = None;
            let mut remaining = n_candles;

            while remaining > 0 {
                let batch_size = remaining.min(1000);
                let url = format!("{}/v5/market/kline", self.base_url);

                let mut params = vec![
                    ("category".to_string(), "spot".to_string()),
                    ("symbol".to_string(), symbol.to_string()),
                    ("interval".to_string(), interval.to_string()),
                    ("limit".to_string(), batch_size.to_string()),
                ];

                if let Some(end) = end_time {
                    params.push(("end".to_string(), end.to_string()));
                }

                let resp: BybitResponse = self.client
                    .get(&url)
                    .query(&params)
                    .send()?
                    .json()?;

                if resp.ret_code != 0 {
                    anyhow::bail!("Bybit API error: {}", resp.ret_msg);
                }

                let mut candles: Vec<Candle> = resp.result.list.iter()
                    .filter_map(|row| {
                        if row.len() >= 6 {
                            Some(Candle {
                                timestamp: row[0].as_str()?.parse::<i64>().ok()?,
                                open: row[1].as_str()?.parse::<f64>().ok()?,
                                high: row[2].as_str()?.parse::<f64>().ok()?,
                                low: row[3].as_str()?.parse::<f64>().ok()?,
                                close: row[4].as_str()?.parse::<f64>().ok()?,
                                volume: row[5].as_str()?.parse::<f64>().ok()?,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                if candles.is_empty() {
                    break;
                }

                // Bybit returns newest first
                candles.reverse();
                remaining -= candles.len();

                if let Some(first) = candles.first() {
                    end_time = Some(first.timestamp - 1);
                }

                all_candles.splice(0..0, candles);

                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            Ok(all_candles)
        }
    }

    impl Default for BybitClient {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Statistics Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod statistics {
    /// Compute mean of a slice
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Compute variance of a slice
    pub fn variance(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let m = mean(data);
        data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    }

    /// Compute standard deviation
    pub fn std_dev(data: &[f64]) -> f64 {
        variance(data).sqrt()
    }

    /// Compute skewness
    pub fn skewness(data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }
        let m = mean(data);
        let s = std_dev(data);
        if s < 1e-12 {
            return 0.0;
        }
        let n = data.len() as f64;
        let sum_cubed: f64 = data.iter().map(|x| ((x - m) / s).powi(3)).sum();
        sum_cubed / n
    }

    /// Compute excess kurtosis
    pub fn excess_kurtosis(data: &[f64]) -> f64 {
        if data.len() < 4 {
            return 0.0;
        }
        let m = mean(data);
        let s = std_dev(data);
        if s < 1e-12 {
            return 0.0;
        }
        let n = data.len() as f64;
        let sum_fourth: f64 = data.iter().map(|x| ((x - m) / s).powi(4)).sum();
        sum_fourth / n - 3.0
    }

    /// Compute autocorrelation at given lag
    pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
        if lag >= data.len() || data.len() < 2 {
            return 0.0;
        }
        let m = mean(data);
        let var: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
        if var < 1e-12 {
            return 0.0;
        }
        let cov: f64 = data[..data.len() - lag].iter()
            .zip(data[lag..].iter())
            .map(|(a, b)| (a - m) * (b - m))
            .sum::<f64>() / (data.len() - lag) as f64;
        cov / var
    }

    /// Compute autocorrelation function up to max_lag
    pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
        (1..=max_lag).map(|k| autocorrelation(data, k)).collect()
    }

    /// Compute correlation between two slices
    pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 2 {
            return 0.0;
        }
        let mx = mean(&x[..n]);
        let my = mean(&y[..n]);
        let sx = std_dev(&x[..n]);
        let sy = std_dev(&y[..n]);
        if sx < 1e-12 || sy < 1e-12 {
            return 0.0;
        }
        let cov: f64 = x[..n].iter().zip(y[..n].iter())
            .map(|(a, b)| (a - mx) * (b - my))
            .sum::<f64>() / n as f64;
        cov / (sx * sy)
    }

    /// Compute leverage effect correlation: Corr(r_t, |r_{t+1}|)
    pub fn leverage_correlation(returns: &[f64]) -> f64 {
        if returns.len() < 3 {
            return 0.0;
        }
        let r_t: Vec<f64> = returns[..returns.len() - 1].to_vec();
        let abs_r_next: Vec<f64> = returns[1..].iter().map(|x| x.abs()).collect();
        correlation(&r_t, &abs_r_next)
    }

    /// Compute comprehensive return statistics
    pub fn compute_statistics(returns: &[f64]) -> super::ReturnStatistics {
        let abs_returns: Vec<f64> = returns.iter().map(|x| x.abs()).collect();
        let sq_returns: Vec<f64> = returns.iter().map(|x| x * x).collect();

        super::ReturnStatistics {
            mean: mean(returns),
            std: std_dev(returns),
            skewness: skewness(returns),
            kurtosis: excess_kurtosis(returns),
            min: returns.iter().cloned().fold(f64::INFINITY, f64::min),
            max: returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            acf_abs_lag1: autocorrelation(&abs_returns, 1),
            acf_abs_lag5: autocorrelation(&abs_returns, 5),
            acf_sq_lag1: autocorrelation(&sq_returns, 1),
            leverage_corr: leverage_correlation(returns),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Physics Constraints Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod physics {
    use super::statistics;
    use super::PhysicsReport;

    /// Configuration for physics constraint evaluation
    pub struct PhysicsConfig {
        pub target_kurtosis: f64,
        pub target_leverage_corr: f64,
        pub lambda_martingale: f64,
        pub lambda_volatility: f64,
        pub lambda_kurtosis: f64,
        pub lambda_leverage: f64,
        pub lambda_autocorr: f64,
        pub max_acf_lag: usize,
        pub target_vol_acf: Vec<f64>,
        pub target_sq_acf: Vec<f64>,
    }

    impl Default for PhysicsConfig {
        fn default() -> Self {
            Self {
                target_kurtosis: 8.0,
                target_leverage_corr: -0.25,
                lambda_martingale: 1.0,
                lambda_volatility: 0.5,
                lambda_kurtosis: 0.3,
                lambda_leverage: 0.2,
                lambda_autocorr: 0.3,
                max_acf_lag: 20,
                target_vol_acf: vec![0.25, 0.22, 0.20, 0.18, 0.16,
                                     0.14, 0.13, 0.12, 0.11, 0.10,
                                     0.09, 0.09, 0.08, 0.08, 0.07,
                                     0.07, 0.06, 0.06, 0.05, 0.05],
                target_sq_acf: vec![0.20, 0.17, 0.15, 0.13, 0.12,
                                    0.10, 0.09, 0.08, 0.07, 0.07],
            }
        }
    }

    impl PhysicsConfig {
        /// Set targets from real data
        pub fn set_targets_from_data(&mut self, returns: &[f64]) {
            let abs_returns: Vec<f64> = returns.iter().map(|x| x.abs()).collect();
            let sq_returns: Vec<f64> = returns.iter().map(|x| x * x).collect();

            self.target_kurtosis = statistics::excess_kurtosis(returns);
            self.target_leverage_corr = statistics::leverage_correlation(returns);

            self.target_vol_acf = statistics::acf(&abs_returns, self.max_acf_lag);
            self.target_sq_acf = statistics::acf(&sq_returns, 10);
        }
    }

    /// Compute martingale loss: E[r] should be approximately zero
    pub fn martingale_loss(returns: &[f64]) -> f64 {
        let m = statistics::mean(returns);
        m * m
    }

    /// Compute volatility clustering loss: ACF of |r| should match target
    pub fn volatility_clustering_loss(returns: &[f64], target_acf: &[f64]) -> f64 {
        let abs_returns: Vec<f64> = returns.iter().map(|x| x.abs()).collect();
        let gen_acf = statistics::acf(&abs_returns, target_acf.len());

        gen_acf.iter().zip(target_acf.iter())
            .map(|(g, t)| (g - t).powi(2))
            .sum::<f64>() / target_acf.len() as f64
    }

    /// Compute kurtosis loss: excess kurtosis should match target
    pub fn kurtosis_loss(returns: &[f64], target: f64) -> f64 {
        let k = statistics::excess_kurtosis(returns);
        (k - target).powi(2)
    }

    /// Compute leverage effect loss: Corr(r_t, |r_{t+1}|) should match target
    pub fn leverage_effect_loss(returns: &[f64], target_corr: f64) -> f64 {
        let corr = statistics::leverage_correlation(returns);
        (corr - target_corr).powi(2)
    }

    /// Compute autocorrelation structure loss
    pub fn autocorrelation_loss(returns: &[f64], target_sq_acf: &[f64]) -> f64 {
        // Raw returns should have near-zero ACF
        let return_acf = statistics::acf(returns, 5);
        let return_penalty: f64 = return_acf.iter().map(|a| a.powi(2)).sum();

        // Squared returns should match target ACF
        let sq_returns: Vec<f64> = returns.iter().map(|x| x * x).collect();
        let sq_acf = statistics::acf(&sq_returns, target_sq_acf.len());
        let sq_penalty: f64 = sq_acf.iter().zip(target_sq_acf.iter())
            .map(|(g, t)| (g - t).powi(2))
            .sum();

        (return_penalty + sq_penalty) / (5.0 + target_sq_acf.len() as f64)
    }

    /// Evaluate all physics constraints for a set of generated returns
    pub fn evaluate_physics(returns: &[f64], config: &PhysicsConfig) -> PhysicsReport {
        let l_mart = martingale_loss(returns);
        let l_vol = volatility_clustering_loss(returns, &config.target_vol_acf);
        let l_kurt = kurtosis_loss(returns, config.target_kurtosis);
        let l_lev = leverage_effect_loss(returns, config.target_leverage_corr);
        let l_acf = autocorrelation_loss(returns, &config.target_sq_acf);

        let total = config.lambda_martingale * l_mart
            + config.lambda_volatility * l_vol
            + config.lambda_kurtosis * l_kurt
            + config.lambda_leverage * l_lev
            + config.lambda_autocorr * l_acf;

        PhysicsReport {
            martingale_loss: l_mart,
            volatility_clustering_loss: l_vol,
            kurtosis_loss: l_kurt,
            leverage_effect_loss: l_lev,
            autocorrelation_loss: l_acf,
            total_physics_loss: total,
            statistics: statistics::compute_statistics(returns),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Heston Stochastic Volatility Model
// ═══════════════════════════════════════════════════════════════════════════════

pub mod heston {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    /// Heston model parameters
    pub struct HestonParams {
        pub mu: f64,          // Drift
        pub kappa: f64,       // Mean reversion speed
        pub theta: f64,       // Long-run variance
        pub xi: f64,          // Vol of vol
        pub rho: f64,         // Correlation (leverage effect)
        pub v0: f64,          // Initial variance
        pub dt: f64,          // Time step
    }

    impl Default for HestonParams {
        fn default() -> Self {
            Self {
                mu: 0.05,
                kappa: 2.0,
                theta: 0.04,
                xi: 0.3,
                rho: -0.7,
                v0: 0.04,
                dt: 1.0 / 252.0,
            }
        }
    }

    /// Generate synthetic returns using the Heston model
    pub fn generate_heston_path(params: &HestonParams, n_steps: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut v = params.v0;
        let mut returns = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let z1: f64 = normal.sample(&mut rng);
            let z2: f64 = params.rho * z1
                + (1.0 - params.rho * params.rho).sqrt() * normal.sample(&mut rng);

            let v_pos = v.max(1e-8);
            let sigma = v_pos.sqrt();

            // Log return
            let ret = (params.mu - 0.5 * v_pos) * params.dt
                + sigma * params.dt.sqrt() * z1;
            returns.push(ret);

            // CIR variance process
            v += params.kappa * (params.theta - v) * params.dt
                + params.xi * (v_pos * params.dt).sqrt() * z2;
            v = v.max(0.0);
        }

        returns
    }

    /// Generate multiple Heston paths
    pub fn generate_multiple_paths(
        params: &HestonParams,
        n_paths: usize,
        n_steps: usize,
    ) -> Vec<Vec<f64>> {
        (0..n_paths)
            .map(|_| generate_heston_path(params, n_steps))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Simplified GAN Module (demonstration, not GPU-accelerated)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod gan {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    /// Simple linear layer for demonstration
    #[derive(Clone)]
    pub struct LinearLayer {
        pub weights: Vec<Vec<f64>>,
        pub bias: Vec<f64>,
    }

    impl LinearLayer {
        /// Create with Xavier initialization
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let mut rng = rand::thread_rng();
            let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
            let normal = Normal::new(0.0, scale).unwrap();

            let weights = (0..output_dim)
                .map(|_| (0..input_dim).map(|_| normal.sample(&mut rng)).collect())
                .collect();
            let bias = vec![0.0; output_dim];

            Self { weights, bias }
        }

        /// Forward pass
        pub fn forward(&self, input: &[f64]) -> Vec<f64> {
            self.weights.iter().zip(self.bias.iter())
                .map(|(w, &b)| {
                    w.iter().zip(input.iter())
                        .map(|(&wi, &xi)| wi * xi)
                        .sum::<f64>() + b
                })
                .collect()
        }

        /// Update weights (SGD step)
        pub fn update(&mut self, grad_w: &[Vec<f64>], grad_b: &[f64], lr: f64) {
            for (row, grad_row) in self.weights.iter_mut().zip(grad_w.iter()) {
                for (w, g) in row.iter_mut().zip(grad_row.iter()) {
                    *w -= lr * g;
                }
            }
            for (b, g) in self.bias.iter_mut().zip(grad_b.iter()) {
                *b -= lr * g;
            }
        }
    }

    /// Simple generator for demonstration
    pub struct SimpleGenerator {
        pub layer1: LinearLayer,
        pub layer2: LinearLayer,
        pub layer3: LinearLayer,
        pub latent_dim: usize,
        pub output_dim: usize,
    }

    impl SimpleGenerator {
        pub fn new(latent_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
            Self {
                layer1: LinearLayer::new(latent_dim, hidden_dim),
                layer2: LinearLayer::new(hidden_dim, hidden_dim),
                layer3: LinearLayer::new(hidden_dim, output_dim),
                latent_dim,
                output_dim,
            }
        }

        /// Generate synthetic returns from noise
        pub fn forward(&self, z: &[f64]) -> Vec<f64> {
            let h1 = leaky_relu(&self.layer1.forward(z));
            let h2 = leaky_relu(&self.layer2.forward(&h1));
            let out = self.layer3.forward(&h2);
            // Scale output to reasonable return range
            out.iter().map(|x| x.tanh() * 0.05).collect()
        }

        /// Generate a batch of synthetic return paths
        pub fn generate_batch(&self, n_paths: usize) -> Vec<Vec<f64>> {
            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, 1.0).unwrap();

            (0..n_paths)
                .map(|_| {
                    let z: Vec<f64> = (0..self.latent_dim)
                        .map(|_| normal.sample(&mut rng))
                        .collect();
                    self.forward(&z)
                })
                .collect()
        }
    }

    fn leaky_relu(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| if v > 0.0 { v } else { 0.01 * v }).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Trading Module
// ═══════════════════════════════════════════════════════════════════════════════

pub mod trading {
    use super::*;

    /// Run a simple backtest on return data
    pub fn run_backtest(
        returns: &[f64],
        signals: &[f64],  // position sizes: +1 = long, -1 = short, 0 = flat
        transaction_cost: f64,
        initial_capital: f64,
    ) -> BacktestResult {
        let n = returns.len().min(signals.len());
        let mut equity = initial_capital;
        let mut peak = equity;
        let mut max_dd = 0.0_f64;
        let mut equity_curve = vec![equity];
        let mut trade_returns = Vec::new();
        let mut prev_signal = 0.0;

        for i in 0..n {
            let position = signals[i];
            let cost = (position - prev_signal).abs() * transaction_cost;
            let pnl = position * returns[i] - cost;

            equity *= 1.0 + pnl;
            equity_curve.push(equity);

            if position.abs() > 0.01 {
                trade_returns.push(pnl);
            }

            peak = peak.max(equity);
            let dd = (peak - equity) / peak;
            max_dd = max_dd.max(dd);

            prev_signal = position;
        }

        let total_return = (equity / initial_capital) - 1.0;
        let n_years = n as f64 / 252.0;
        let annualized = if n_years > 0.0 {
            (1.0 + total_return).powf(1.0 / n_years) - 1.0
        } else {
            0.0
        };

        let ret_slice: Vec<f64> = equity_curve.windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect();
        let sharpe = if ret_slice.len() > 1 {
            let m = statistics::mean(&ret_slice);
            let s = statistics::std_dev(&ret_slice);
            if s > 1e-8 { m / s * (252.0_f64).sqrt() } else { 0.0 }
        } else {
            0.0
        };

        let wins = trade_returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = if !trade_returns.is_empty() {
            wins as f64 / trade_returns.len() as f64
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annualized_return: annualized,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            win_rate,
            n_trades: trade_returns.len(),
            equity_curve,
        }
    }

    /// Generate signals from Monte Carlo GAN paths
    pub fn generate_signal_from_paths(
        forward_returns: &[Vec<f64>],
        threshold: f64,
    ) -> TradingSignal {
        let cumulative: Vec<f64> = forward_returns.iter()
            .map(|path| path.iter().sum::<f64>())
            .collect();

        let expected_return = statistics::mean(&cumulative);
        let risk = statistics::std_dev(&cumulative);
        let sharpe = if risk > 1e-8 { expected_return / risk } else { 0.0 };

        let p_win = cumulative.iter().filter(|&&c| c > 0.0).count() as f64
            / cumulative.len() as f64;

        let (direction, confidence) = if sharpe > threshold {
            (SignalDirection::Long, (sharpe / 2.0).min(1.0))
        } else if sharpe < -threshold {
            (SignalDirection::Short, (sharpe.abs() / 2.0).min(1.0))
        } else {
            (SignalDirection::Neutral, 0.0)
        };

        // Half-Kelly position sizing
        let wins: Vec<f64> = cumulative.iter().filter(|&&c| c > 0.0).cloned().collect();
        let losses: Vec<f64> = cumulative.iter().filter(|&&c| c <= 0.0).map(|c| c.abs()).collect();
        let avg_win = if !wins.is_empty() { statistics::mean(&wins) } else { 0.0 };
        let avg_loss = if !losses.is_empty() { statistics::mean(&losses) } else { 1e-8 };
        let b = avg_win / avg_loss;
        let kelly = ((p_win * b - (1.0 - p_win)) / b).max(0.0).min(1.0);

        TradingSignal {
            direction,
            confidence,
            position_size: kelly * 0.5,
            expected_return,
            expected_risk: risk,
            regime: MarketRegime::Sideways,
            vol_level: VolatilityLevel::Medium,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let returns = compute_log_returns(&prices);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_statistics() {
        let data = vec![0.01, -0.02, 0.03, -0.01, 0.005, 0.02, -0.015, 0.01];
        let m = statistics::mean(&data);
        let s = statistics::std_dev(&data);
        assert!(m.abs() < 0.1);
        assert!(s > 0.0);
    }

    #[test]
    fn test_heston_generation() {
        let params = heston::HestonParams::default();
        let path = heston::generate_heston_path(&params, 100);
        assert_eq!(path.len(), 100);
        // Returns should be finite and not too extreme
        for &r in &path {
            assert!(r.is_finite());
            assert!(r.abs() < 1.0);
        }
    }

    #[test]
    fn test_physics_evaluation() {
        let params = heston::HestonParams::default();
        let returns = heston::generate_heston_path(&params, 1000);
        let config = physics::PhysicsConfig::default();
        let report = physics::evaluate_physics(&returns, &config);
        assert!(report.total_physics_loss.is_finite());
        assert!(report.statistics.std > 0.0);
    }

    #[test]
    fn test_regime_detection() {
        let bull_returns = vec![0.01; 60]; // Consistently positive
        let (regime, _vol) = detect_regime(&bull_returns);
        assert_eq!(regime, MarketRegime::Bull);

        let bear_returns = vec![-0.01; 60]; // Consistently negative
        let (regime, _vol) = detect_regime(&bear_returns);
        assert_eq!(regime, MarketRegime::Bear);
    }

    #[test]
    fn test_simple_generator() {
        let gen = gan::SimpleGenerator::new(32, 64, 100);
        let paths = gen.generate_batch(10);
        assert_eq!(paths.len(), 10);
        assert_eq!(paths[0].len(), 100);
    }

    #[test]
    fn test_backtest() {
        let returns = vec![0.01, -0.005, 0.008, -0.003, 0.012, -0.002, 0.005];
        let signals = vec![1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0];
        let result = trading::run_backtest(&returns, &signals, 0.001, 100000.0);
        assert!(result.equity_curve.len() > 0);
        assert!(result.sharpe_ratio.is_finite());
    }
}
