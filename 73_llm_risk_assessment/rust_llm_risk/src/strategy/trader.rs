//! Risk-based trading strategy implementation.

use tracing::{debug, info};

use crate::risk::RiskScore;
use super::signal::{Position, TradingSignal};

/// Configuration for the risk-based trader.
#[derive(Debug, Clone)]
pub struct TraderConfig {
    /// Risk score threshold for entering long positions.
    pub risk_threshold_long: f64,
    /// Risk score threshold for entering short positions.
    pub risk_threshold_short: f64,
    /// Maximum position size as a fraction of capital.
    pub max_position_size: f64,
    /// Stop loss percentage.
    pub stop_loss_pct: f64,
    /// Take profit percentage.
    pub take_profit_pct: f64,
    /// Whether to allow short positions.
    pub allow_shorts: bool,
}

impl Default for TraderConfig {
    fn default() -> Self {
        Self {
            risk_threshold_long: 4.0,
            risk_threshold_short: 7.0,
            max_position_size: 1.0,
            stop_loss_pct: 0.05, // 5%
            take_profit_pct: 0.10, // 10%
            allow_shorts: true,
        }
    }
}

/// Risk-based trading strategy.
pub struct RiskBasedTrader {
    config: TraderConfig,
    risk_history: Vec<RiskScore>,
    current_position: Option<Position>,
}

impl RiskBasedTrader {
    /// Create a new risk-based trader with default config.
    pub fn new() -> Self {
        Self {
            config: TraderConfig::default(),
            risk_history: Vec::new(),
            current_position: None,
        }
    }

    /// Create a new risk-based trader with custom config.
    pub fn with_config(config: TraderConfig) -> Self {
        Self {
            config,
            risk_history: Vec::new(),
            current_position: None,
        }
    }

    /// Generate a trading signal based on risk score.
    pub fn generate_signal(
        &mut self,
        risk_score: RiskScore,
        current_price: f64,
        symbol: &str,
    ) -> TradingSignal {
        // Store risk score in history
        self.risk_history.push(risk_score.clone());

        // Update current position if exists
        if let Some(ref mut pos) = self.current_position {
            pos.update_pnl(current_price);

            // Check stop loss
            if pos.stop_loss_hit(current_price) {
                info!("Stop loss hit for {}", symbol);
                self.current_position = None;
                return TradingSignal::close("Stop loss triggered")
                    .with_risk_score(risk_score.overall_score)
                    .with_symbol(symbol.to_string());
            }

            // Check take profit
            if pos.take_profit_hit(current_price) {
                info!("Take profit hit for {}", symbol);
                self.current_position = None;
                return TradingSignal::close("Take profit triggered")
                    .with_risk_score(risk_score.overall_score)
                    .with_symbol(symbol.to_string());
            }
        }

        // Calculate position size based on risk
        let position_size = self.calculate_position_size(&risk_score);

        // Generate signal based on risk level and trend
        let signal = self.evaluate_risk_for_signal(&risk_score, position_size, current_price, symbol);

        debug!(
            "Generated signal: {} for {} (risk: {:.1})",
            signal.signal_type, symbol, risk_score.overall_score
        );

        signal
    }

    /// Calculate position size based on risk score.
    fn calculate_position_size(&self, risk: &RiskScore) -> f64 {
        // Base size from risk level
        let base_size = risk.position_multiplier();

        // Apply maximum position size constraint
        let size = base_size * self.config.max_position_size;

        size.clamp(0.0, self.config.max_position_size)
    }

    /// Evaluate risk score and generate appropriate signal.
    fn evaluate_risk_for_signal(
        &mut self,
        risk: &RiskScore,
        position_size: f64,
        current_price: f64,
        symbol: &str,
    ) -> TradingSignal {
        let score = risk.overall_score;

        // Check for existing position
        if let Some(ref pos) = self.current_position {
            // Already in a position - check if we should exit
            if pos.is_long() && score >= self.config.risk_threshold_short {
                // Risk too high, close long position
                self.current_position = None;
                return TradingSignal::close("Risk increased, closing long")
                    .with_risk_score(score)
                    .with_symbol(symbol.to_string());
            }

            if pos.is_short() && score <= self.config.risk_threshold_long {
                // Risk decreased, close short position
                self.current_position = None;
                return TradingSignal::close("Risk decreased, closing short")
                    .with_risk_score(score)
                    .with_symbol(symbol.to_string());
            }

            // Stay in current position
            return TradingSignal::neutral("Holding current position")
                .with_risk_score(score)
                .with_symbol(symbol.to_string());
        }

        // No current position - evaluate for entry
        let trend = self.get_risk_trend(5);

        // Low risk - consider going long
        if score <= self.config.risk_threshold_long {
            if matches!(risk.direction, crate::risk::RiskDirection::Stable | crate::risk::RiskDirection::Decreasing)
                || trend == RiskTrend::Decreasing
            {
                let stop_loss = current_price * (1.0 - self.config.stop_loss_pct);
                let take_profit = current_price * (1.0 + self.config.take_profit_pct);

                // Create position
                let position = Position::new(symbol.to_string(), position_size, current_price)
                    .with_stop_loss(stop_loss)
                    .with_take_profit(take_profit);

                self.current_position = Some(position);

                return TradingSignal::long(position_size, "Low risk, favorable conditions")
                    .with_confidence(self.risk_to_confidence(&risk))
                    .with_risk_score(score)
                    .with_symbol(symbol.to_string())
                    .with_stop_loss(stop_loss)
                    .with_take_profit(take_profit);
            }
        }

        // High risk - consider going short
        if self.config.allow_shorts && score >= self.config.risk_threshold_short {
            if matches!(risk.direction, crate::risk::RiskDirection::Stable | crate::risk::RiskDirection::Increasing)
                || trend == RiskTrend::Increasing
            {
                let position_size = position_size * 0.5; // Smaller short positions
                let stop_loss = current_price * (1.0 + self.config.stop_loss_pct);
                let take_profit = current_price * (1.0 - self.config.take_profit_pct);

                let position = Position::new(symbol.to_string(), -position_size, current_price)
                    .with_stop_loss(stop_loss)
                    .with_take_profit(take_profit);

                self.current_position = Some(position);

                return TradingSignal::short(position_size, "High risk, bearish conditions")
                    .with_confidence(self.risk_to_confidence(&risk))
                    .with_risk_score(score)
                    .with_symbol(symbol.to_string())
                    .with_stop_loss(stop_loss)
                    .with_take_profit(take_profit);
            }
        }

        // Neutral - no clear signal
        TradingSignal::neutral("Risk in neutral zone, waiting")
            .with_risk_score(score)
            .with_symbol(symbol.to_string())
    }

    /// Convert risk score to confidence level.
    fn risk_to_confidence(&self, risk: &RiskScore) -> f64 {
        let base_confidence: f64 = match risk.confidence {
            crate::risk::Confidence::High => 0.9,
            crate::risk::Confidence::Medium => 0.7,
            crate::risk::Confidence::Low => 0.5,
        };

        // Adjust based on how extreme the risk score is
        let score = risk.overall_score;
        let extremity: f64 = if score <= 3.0 || score >= 8.0 {
            1.1 // More confident at extremes
        } else {
            0.9 // Less confident in middle range
        };

        (base_confidence * extremity).clamp(0.0, 1.0)
    }

    /// Analyze recent risk trend.
    fn get_risk_trend(&self, window: usize) -> RiskTrend {
        if self.risk_history.len() < window {
            return RiskTrend::Unknown;
        }

        let recent: Vec<f64> = self.risk_history
            .iter()
            .rev()
            .take(window)
            .map(|r| r.overall_score)
            .collect();

        // Simple linear regression slope
        let n = recent.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = recent.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, y) in recent.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator == 0.0 {
            return RiskTrend::Stable;
        }

        let slope = numerator / denominator;

        if slope > 0.3 {
            RiskTrend::Increasing
        } else if slope < -0.3 {
            RiskTrend::Decreasing
        } else {
            RiskTrend::Stable
        }
    }

    /// Get the current position.
    pub fn current_position(&self) -> Option<&Position> {
        self.current_position.as_ref()
    }

    /// Get risk history.
    pub fn risk_history(&self) -> &[RiskScore] {
        &self.risk_history
    }

    /// Clear risk history.
    pub fn clear_history(&mut self) {
        self.risk_history.clear();
    }

    /// Close current position.
    pub fn close_position(&mut self) {
        self.current_position = None;
    }
}

impl Default for RiskBasedTrader {
    fn default() -> Self {
        Self::new()
    }
}

/// Risk trend direction.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RiskTrend {
    Increasing,
    Stable,
    Decreasing,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trader_creation() {
        let trader = RiskBasedTrader::new();
        assert!(trader.current_position().is_none());
        assert!(trader.risk_history().is_empty());
    }

    #[test]
    fn test_low_risk_long_signal() {
        let mut trader = RiskBasedTrader::new();

        let risk = RiskScore::new(2.0, 3.0, 2.0, 2.0, 3.0, 2.0);
        let signal = trader.generate_signal(risk, 50000.0, "BTCUSDT");

        assert!(signal.is_buy());
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_high_risk_short_signal() {
        let mut trader = RiskBasedTrader::new();

        let risk = RiskScore::new(8.0, 7.0, 8.0, 9.0, 8.0, 8.0);
        let signal = trader.generate_signal(risk, 50000.0, "BTCUSDT");

        // May be short or neutral depending on direction
        assert!(matches!(signal.signal_type, SignalType::Short | SignalType::Neutral));
    }

    #[test]
    fn test_neutral_zone() {
        let mut trader = RiskBasedTrader::new();

        let risk = RiskScore::new(5.0, 5.0, 5.0, 5.0, 5.0, 5.0);
        let signal = trader.generate_signal(risk, 50000.0, "BTCUSDT");

        assert!(!signal.is_actionable());
    }

    #[test]
    fn test_position_tracking() {
        let mut trader = RiskBasedTrader::new();

        // Enter long position
        let low_risk = RiskScore::new(2.0, 2.0, 2.0, 2.0, 2.0, 2.0);
        let signal = trader.generate_signal(low_risk, 50000.0, "BTCUSDT");

        if signal.is_buy() {
            assert!(trader.current_position().is_some());
        }
    }
}
