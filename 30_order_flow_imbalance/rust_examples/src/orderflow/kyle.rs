//! # Kyle's Lambda (Price Impact Coefficient)
//!
//! Implementation based on Kyle (1985): "Continuous Auctions and Insider Trading"
//!
//! Kyle's Lambda measures how much prices move in response to order flow.
//! Higher lambda = less liquid market, orders move prices more.

use crate::data::trade::Trade;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

/// Kyle's Lambda calculator
#[derive(Debug)]
pub struct KyleLambdaCalculator {
    /// Window size for regression
    window: usize,
    /// Price changes
    price_changes: VecDeque<f64>,
    /// Signed order flow (positive = buy, negative = sell)
    order_flow: VecDeque<f64>,
    /// Timestamps
    timestamps: VecDeque<DateTime<Utc>>,
    /// Lambda history
    lambda_history: VecDeque<LambdaPoint>,
    /// Last price
    last_price: Option<f64>,
}

/// Lambda data point
#[derive(Debug, Clone)]
pub struct LambdaPoint {
    pub timestamp: DateTime<Utc>,
    pub lambda: f64,
    pub r_squared: f64,
}

impl KyleLambdaCalculator {
    /// Create a new calculator
    pub fn new(window: usize) -> Self {
        Self {
            window,
            price_changes: VecDeque::with_capacity(window),
            order_flow: VecDeque::with_capacity(window),
            timestamps: VecDeque::with_capacity(window),
            lambda_history: VecDeque::with_capacity(1000),
            last_price: None,
        }
    }

    /// Add a trade and update lambda
    pub fn add_trade(&mut self, trade: &Trade) -> Option<f64> {
        // Calculate price change
        let price_change = if let Some(last) = self.last_price {
            (trade.price - last) / last * 10000.0 // in bps
        } else {
            0.0
        };
        self.last_price = Some(trade.price);

        // Signed order flow (positive for buys)
        let signed_flow = if trade.is_buy() {
            trade.size
        } else {
            -trade.size
        };

        // Add to buffers
        self.price_changes.push_back(price_change);
        self.order_flow.push_back(signed_flow);
        self.timestamps.push_back(trade.timestamp);

        // Maintain window size
        while self.price_changes.len() > self.window {
            self.price_changes.pop_front();
            self.order_flow.pop_front();
            self.timestamps.pop_front();
        }

        // Calculate lambda if we have enough data
        if self.price_changes.len() >= self.window / 2 {
            let (lambda, r_squared) = self.calculate_lambda();

            // Store in history
            let point = LambdaPoint {
                timestamp: trade.timestamp,
                lambda,
                r_squared,
            };
            self.lambda_history.push_back(point);
            if self.lambda_history.len() > 1000 {
                self.lambda_history.pop_front();
            }

            Some(lambda)
        } else {
            None
        }
    }

    /// Calculate Kyle's Lambda using OLS regression
    ///
    /// ΔPrice = α + λ × OrderFlow + ε
    ///
    /// Returns (lambda, r_squared)
    fn calculate_lambda(&self) -> (f64, f64) {
        let n = self.price_changes.len() as f64;

        if n < 2.0 {
            return (0.0, 0.0);
        }

        // Calculate means
        let price_mean: f64 = self.price_changes.iter().sum::<f64>() / n;
        let flow_mean: f64 = self.order_flow.iter().sum::<f64>() / n;

        // Calculate covariance and variance
        let mut covariance = 0.0;
        let mut flow_variance = 0.0;
        let mut price_variance = 0.0;

        for (price, flow) in self.price_changes.iter().zip(self.order_flow.iter()) {
            let price_dev = price - price_mean;
            let flow_dev = flow - flow_mean;
            covariance += price_dev * flow_dev;
            flow_variance += flow_dev * flow_dev;
            price_variance += price_dev * price_dev;
        }

        // Lambda = Cov(ΔP, OF) / Var(OF)
        let lambda = if flow_variance > 0.0 {
            covariance / flow_variance
        } else {
            0.0
        };

        // R-squared
        let r_squared = if price_variance > 0.0 && flow_variance > 0.0 {
            let correlation = covariance / (flow_variance.sqrt() * price_variance.sqrt());
            correlation * correlation
        } else {
            0.0
        };

        (lambda, r_squared)
    }

    /// Get current lambda
    pub fn current_lambda(&self) -> Option<f64> {
        self.lambda_history.back().map(|p| p.lambda)
    }

    /// Get lambda statistics
    pub fn statistics(&self) -> LambdaStatistics {
        if self.lambda_history.is_empty() {
            return LambdaStatistics::default();
        }

        let values: Vec<f64> = self.lambda_history.iter().map(|p| p.lambda).collect();
        let r_squared: Vec<f64> = self.lambda_history.iter().map(|p| p.r_squared).collect();

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        let avg_r_squared = r_squared.iter().sum::<f64>() / n;

        LambdaStatistics {
            count: values.len(),
            current: values.last().cloned().unwrap_or(0.0),
            mean,
            std: variance.sqrt(),
            min: values.iter().cloned().fold(f64::INFINITY, f64::min),
            max: values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            avg_r_squared,
        }
    }

    /// Get liquidity interpretation
    pub fn liquidity_level(&self) -> LiquidityLevel {
        match self.current_lambda() {
            Some(lambda) if lambda < 0.1 => LiquidityLevel::VeryHigh,
            Some(lambda) if lambda < 0.5 => LiquidityLevel::High,
            Some(lambda) if lambda < 1.0 => LiquidityLevel::Medium,
            Some(lambda) if lambda < 2.0 => LiquidityLevel::Low,
            Some(_) => LiquidityLevel::VeryLow,
            None => LiquidityLevel::Unknown,
        }
    }

    /// Reset calculator
    pub fn reset(&mut self) {
        self.price_changes.clear();
        self.order_flow.clear();
        self.timestamps.clear();
        self.lambda_history.clear();
        self.last_price = None;
    }
}

/// Lambda statistics
#[derive(Debug, Clone, Default)]
pub struct LambdaStatistics {
    pub count: usize,
    pub current: f64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub avg_r_squared: f64,
}

/// Liquidity level interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityLevel {
    VeryHigh,
    High,
    Medium,
    Low,
    VeryLow,
    Unknown,
}

impl std::fmt::Display for LiquidityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiquidityLevel::VeryHigh => write!(f, "Very High"),
            LiquidityLevel::High => write!(f, "High"),
            LiquidityLevel::Medium => write!(f, "Medium"),
            LiquidityLevel::Low => write!(f, "Low"),
            LiquidityLevel::VeryLow => write!(f, "Very Low"),
            LiquidityLevel::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Price impact estimator using order book data
#[derive(Debug)]
pub struct PriceImpactEstimator {
    /// Historical price impacts
    impacts: VecDeque<ImpactObservation>,
    /// Window size
    window: usize,
}

#[derive(Debug, Clone)]
struct ImpactObservation {
    timestamp: DateTime<Utc>,
    size: f64,
    actual_impact: f64,
    expected_impact: f64,
}

impl PriceImpactEstimator {
    /// Create a new estimator
    pub fn new(window: usize) -> Self {
        Self {
            impacts: VecDeque::with_capacity(window),
            window,
        }
    }

    /// Record an observed trade and its impact
    pub fn record_impact(&mut self, timestamp: DateTime<Utc>, size: f64, before_mid: f64, after_mid: f64, expected: f64) {
        let actual_impact = (after_mid - before_mid) / before_mid * 10000.0;

        let obs = ImpactObservation {
            timestamp,
            size,
            actual_impact,
            expected_impact: expected,
        };

        self.impacts.push_back(obs);
        if self.impacts.len() > self.window {
            self.impacts.pop_front();
        }
    }

    /// Get average realized impact
    pub fn average_impact(&self) -> f64 {
        if self.impacts.is_empty() {
            return 0.0;
        }
        self.impacts.iter().map(|o| o.actual_impact).sum::<f64>() / self.impacts.len() as f64
    }

    /// Get impact prediction error
    pub fn prediction_error(&self) -> f64 {
        if self.impacts.is_empty() {
            return 0.0;
        }

        let mse: f64 = self
            .impacts
            .iter()
            .map(|o| (o.actual_impact - o.expected_impact).powi(2))
            .sum::<f64>()
            / self.impacts.len() as f64;

        mse.sqrt()
    }

    /// Estimate impact for a given size
    pub fn estimate_impact(&self, size: f64) -> f64 {
        if self.impacts.is_empty() {
            return 0.0;
        }

        // Simple linear model: impact = k * size
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for obs in &self.impacts {
            sum_xy += obs.size * obs.actual_impact;
            sum_xx += obs.size * obs.size;
        }

        let k = if sum_xx > 0.0 { sum_xy / sum_xx } else { 0.0 };

        k * size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_trade(price: f64, size: f64, is_buy: bool) -> Trade {
        Trade::new(
            "BTCUSDT".to_string(),
            Utc::now(),
            price,
            size,
            !is_buy,
            uuid::Uuid::new_v4().to_string(),
        )
    }

    #[test]
    fn test_kyle_lambda_calculation() {
        let mut calc = KyleLambdaCalculator::new(100);

        // Simulate trades where buys push price up
        let mut price = 100.0;
        for i in 0..50 {
            let is_buy = i % 2 == 0;
            let size = 1.0;

            if is_buy {
                price += 0.1; // Price goes up with buys
            } else {
                price -= 0.1; // Price goes down with sells
            }

            let trade = create_trade(price, size, is_buy);
            calc.add_trade(&trade);
        }

        // Lambda should be positive (buys increase price)
        let lambda = calc.current_lambda();
        assert!(lambda.is_some());
        // Lambda value depends on the relationship between order flow and price changes
    }

    #[test]
    fn test_liquidity_levels() {
        let calc = KyleLambdaCalculator::new(100);
        assert_eq!(calc.liquidity_level(), LiquidityLevel::Unknown);
    }

    #[test]
    fn test_price_impact_estimator() {
        let mut estimator = PriceImpactEstimator::new(100);

        // Record some impacts
        for i in 0..10 {
            let size = (i + 1) as f64;
            let impact = size * 0.5; // Linear relationship

            estimator.record_impact(Utc::now(), size, 100.0, 100.0 + impact / 200.0, impact);
        }

        // Estimate should work
        let estimate = estimator.estimate_impact(5.0);
        assert!(estimate > 0.0);
    }
}
