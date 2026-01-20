//! Market impact estimator with calibration.

use crate::data::{OrderBook, Trade};
use crate::impact::{AlmgrenChrissModel, AlmgrenChrissParams, ImpactComponent, MarketImpactModel};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Market impact estimation errors
#[derive(Error, Debug)]
pub enum MarketImpactError {
    #[error("Insufficient data for calibration: {0}")]
    InsufficientData(String),

    #[error("Calibration failed: {0}")]
    CalibrationFailed(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("No model available")]
    NoModel,
}

/// Impact estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactEstimate {
    /// Estimated impact component
    pub impact: ImpactComponent,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Estimated execution cost
    pub expected_cost: f64,
    /// Execution risk (variance)
    pub execution_risk: f64,
    /// Model used for estimation
    pub model_name: String,
}

/// Market impact estimator
#[derive(Debug, Clone)]
pub struct MarketImpactEstimator {
    model: AlmgrenChrissModel,
    calibration_data: CalibrationData,
    use_orderbook_depth: bool,
}

/// Calibration data for impact estimation
#[derive(Debug, Clone, Default)]
struct CalibrationData {
    /// Historical trades for calibration
    trades: Vec<Trade>,
    /// Estimated daily volume
    daily_volume: Option<f64>,
    /// Estimated volatility
    volatility: Option<f64>,
    /// Last calibration timestamp
    last_calibration: Option<chrono::DateTime<chrono::Utc>>,
}

impl MarketImpactEstimator {
    /// Create a new impact estimator with default parameters
    pub fn new() -> Self {
        Self {
            model: AlmgrenChrissModel::default_params(),
            calibration_data: CalibrationData::default(),
            use_orderbook_depth: true,
        }
    }

    /// Create with custom parameters
    pub fn with_params(params: AlmgrenChrissParams) -> Self {
        Self {
            model: AlmgrenChrissModel::new(params),
            calibration_data: CalibrationData::default(),
            use_orderbook_depth: true,
        }
    }

    /// Create for liquid crypto markets
    pub fn crypto() -> Self {
        Self::with_params(AlmgrenChrissParams::crypto())
    }

    /// Create for liquid equity markets
    pub fn liquid_equity() -> Self {
        Self::with_params(AlmgrenChrissParams::liquid())
    }

    /// Create for illiquid markets
    pub fn illiquid() -> Self {
        Self::with_params(AlmgrenChrissParams::illiquid())
    }

    /// Set whether to use order book depth for adjustment
    pub fn with_orderbook_depth(mut self, use_depth: bool) -> Self {
        self.use_orderbook_depth = use_depth;
        self
    }

    /// Estimate impact for a given quantity and time horizon
    pub fn estimate(
        &self,
        quantity: f64,
        time_horizon: f64,
        orderbook: Option<&OrderBook>,
    ) -> ImpactEstimate {
        let mut impact = self.model.calculate_impact(quantity, time_horizon);

        // Adjust based on order book depth if available
        if self.use_orderbook_depth {
            if let Some(book) = orderbook {
                impact = self.adjust_for_orderbook(impact, quantity, book);
            }
        }

        let expected_cost = self.model.expected_cost(quantity, time_horizon);
        let execution_risk = self.model.execution_risk(quantity, time_horizon);

        // Confidence based on calibration data availability
        let confidence = if self.calibration_data.daily_volume.is_some() {
            0.8
        } else {
            0.5
        };

        ImpactEstimate {
            impact,
            confidence,
            expected_cost,
            execution_risk,
            model_name: "AlmgrenChriss".to_string(),
        }
    }

    /// Adjust impact estimate based on order book depth
    fn adjust_for_orderbook(
        &self,
        impact: ImpactComponent,
        quantity: f64,
        orderbook: &OrderBook,
    ) -> ImpactComponent {
        // Get available depth
        let bid_depth = orderbook.bid_depth(20);
        let ask_depth = orderbook.ask_depth(20);
        let total_depth = bid_depth + ask_depth;

        if total_depth <= 0.0 {
            return impact;
        }

        // Depth ratio: how much of available depth we're consuming
        let depth_ratio = quantity / total_depth;

        // Adjust temporary impact based on depth
        // If we're consuming a large fraction of depth, impact is higher
        let depth_multiplier = if depth_ratio > 0.5 {
            2.0 // Very large order relative to book
        } else if depth_ratio > 0.2 {
            1.5 // Large order
        } else if depth_ratio > 0.1 {
            1.2 // Medium order
        } else {
            1.0 // Small order
        };

        ImpactComponent::new(
            impact.permanent,
            impact.temporary * depth_multiplier,
        )
    }

    /// Calibrate the model using historical trades
    pub fn calibrate(&mut self, trades: &[Trade]) -> Result<(), MarketImpactError> {
        if trades.len() < 100 {
            return Err(MarketImpactError::InsufficientData(
                "Need at least 100 trades for calibration".to_string(),
            ));
        }

        // Calculate daily volume (assuming trades span at least one day)
        let total_volume: f64 = trades.iter().map(|t| t.quantity).sum();
        let time_span = if let (Some(first), Some(last)) = (trades.first(), trades.last()) {
            (last.timestamp - first.timestamp).num_hours() as f64 / 24.0
        } else {
            1.0
        };

        let daily_volume = if time_span > 0.0 {
            total_volume / time_span
        } else {
            total_volume
        };

        // Calculate volatility from price returns
        let prices: Vec<f64> = trades.iter().map(|t| t.price).collect();
        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.is_empty() {
            return Err(MarketImpactError::CalibrationFailed(
                "Could not calculate returns".to_string(),
            ));
        }

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        // Annualize (assuming minute data, ~252 trading days * ~390 minutes)
        let volatility = variance.sqrt() * (252.0 * 390.0_f64).sqrt();

        // Update calibration data
        self.calibration_data.trades = trades.to_vec();
        self.calibration_data.daily_volume = Some(daily_volume);
        self.calibration_data.volatility = Some(volatility);
        self.calibration_data.last_calibration = Some(chrono::Utc::now());

        // Update model parameters
        let mut params = self.model.params().clone();
        params.daily_volume = daily_volume;
        params.sigma = volatility;
        self.model = AlmgrenChrissModel::new(params);

        Ok(())
    }

    /// Get the optimal execution trajectory
    pub fn optimal_trajectory(&self, total_quantity: f64, num_periods: usize) -> Vec<f64> {
        self.model.optimal_trajectory(total_quantity, num_periods)
    }

    /// Estimate impact of buying through the order book
    pub fn estimate_book_impact(
        &self,
        quantity: f64,
        orderbook: &OrderBook,
        is_buy: bool,
    ) -> Option<ImpactComponent> {
        let (avg_price, price_impact) = if is_buy {
            orderbook.buy_impact(quantity)?
        } else {
            orderbook.sell_impact(quantity)?
        };

        let mid_price = orderbook.mid_price()?;

        // Convert to impact fraction
        let impact_fraction = (avg_price - mid_price).abs() / mid_price;

        // Split into permanent and temporary
        // Assume 30% permanent, 70% temporary for immediate execution
        Some(ImpactComponent::new(
            impact_fraction * 0.3,
            impact_fraction * 0.7,
        ))
    }

    /// Get the current model parameters
    pub fn params(&self) -> &AlmgrenChrissParams {
        self.model.params()
    }

    /// Check if the estimator has been calibrated
    pub fn is_calibrated(&self) -> bool {
        self.calibration_data.last_calibration.is_some()
    }
}

impl Default for MarketImpactEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate the optimal execution time given risk aversion
pub fn optimal_execution_time(
    quantity: f64,
    daily_volume: f64,
    volatility: f64,
    risk_aversion: f64,
    eta: f64,
) -> f64 {
    // From Almgren-Chriss: T* = sqrt(eta / (lambda * sigma^2)) * sqrt(X / V)
    let fraction = quantity / daily_volume;
    (eta / (risk_aversion * volatility.powi(2))).sqrt() * fraction.sqrt()
}

/// Calculate participation rate needed to complete in given time
pub fn required_participation_rate(
    quantity: f64,
    daily_volume: f64,
    time_horizon: f64,
) -> f64 {
    // Participation rate = (X / T) / (V / trading_hours)
    // Assuming 6.5 trading hours per day
    let volume_per_hour = daily_volume / 6.5;
    (quantity / time_horizon) / volume_per_hour
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TradeDirection;
    use chrono::Utc;

    fn create_test_trades(count: usize) -> Vec<Trade> {
        let mut trades = Vec::with_capacity(count);
        let base_price = 100.0;
        let base_time = Utc::now();

        for i in 0..count {
            trades.push(Trade {
                id: format!("trade_{}", i),
                timestamp: base_time + chrono::Duration::seconds(i as i64),
                price: base_price + (i as f64 * 0.01).sin() * 0.5,
                quantity: 10.0 + (i as f64 * 0.1).cos() * 5.0,
                direction: if i % 2 == 0 {
                    TradeDirection::Buy
                } else {
                    TradeDirection::Sell
                },
            });
        }

        trades
    }

    #[test]
    fn test_impact_estimate() {
        let estimator = MarketImpactEstimator::new();
        let estimate = estimator.estimate(10000.0, 1.0, None);

        assert!(estimate.impact.total > 0.0);
        assert!(estimate.confidence > 0.0);
        assert!(estimate.expected_cost > 0.0);
    }

    #[test]
    fn test_calibration() {
        let mut estimator = MarketImpactEstimator::new();
        let trades = create_test_trades(200);

        let result = estimator.calibrate(&trades);
        assert!(result.is_ok());
        assert!(estimator.is_calibrated());
    }

    #[test]
    fn test_insufficient_data() {
        let mut estimator = MarketImpactEstimator::new();
        let trades = create_test_trades(50); // Not enough

        let result = estimator.calibrate(&trades);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimal_trajectory() {
        let estimator = MarketImpactEstimator::new();
        let trajectory = estimator.optimal_trajectory(1000.0, 10);

        assert_eq!(trajectory.len(), 10);

        let total: f64 = trajectory.iter().sum();
        assert!((total - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_book_impact() {
        let estimator = MarketImpactEstimator::new();
        let mut book = OrderBook::new("TEST".to_string());

        // Add some depth
        for i in 1..=10 {
            book.update_bid(100.0 - i as f64 * 0.1, 10.0);
            book.update_ask(100.0 + i as f64 * 0.1, 10.0);
        }

        let impact = estimator.estimate_book_impact(5.0, &book, true);
        assert!(impact.is_some());

        let impact = impact.unwrap();
        assert!(impact.total > 0.0);
    }

    #[test]
    fn test_crypto_params() {
        let estimator = MarketImpactEstimator::crypto();
        let params = estimator.params();

        // Crypto has higher volatility
        assert!(params.sigma > 0.02);
    }

    #[test]
    fn test_participation_rate() {
        let rate = required_participation_rate(10000.0, 1000000.0, 1.0);

        // Should be reasonable (< 100% of volume)
        assert!(rate > 0.0);
        assert!(rate < 1.0);
    }
}
