//! Position Sizing with Chain-of-Thought Reasoning
//!
//! Risk-aware position sizing that explains its decisions.

use serde::{Deserialize, Serialize};
use crate::signals::Signal;

/// Position sizing result with reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionResult {
    /// Recommended position size in base currency
    pub position_value: f64,
    /// Number of units to trade
    pub units: f64,
    /// Amount at risk
    pub risk_amount: f64,
    /// Position as percentage of portfolio
    pub position_pct: f64,
    /// Reasoning chain for the position size
    pub reasoning_chain: Vec<String>,
}

/// Position sizer with CoT reasoning.
pub struct PositionSizer {
    /// Maximum position as percentage of portfolio
    max_position_pct: f64,
    /// Maximum risk per trade as percentage of portfolio
    max_risk_pct: f64,
    /// Whether to use Kelly criterion
    use_kelly: bool,
    /// Historical win rate (for Kelly)
    win_rate: f64,
    /// Average win/loss ratio (for Kelly)
    win_loss_ratio: f64,
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self {
            max_position_pct: 0.1,
            max_risk_pct: 0.02,
            use_kelly: false,
            win_rate: 0.55,
            win_loss_ratio: 1.5,
        }
    }
}

impl PositionSizer {
    /// Create a new position sizer.
    pub fn new(max_position_pct: f64, max_risk_pct: f64) -> Self {
        Self {
            max_position_pct,
            max_risk_pct,
            ..Default::default()
        }
    }

    /// Enable Kelly criterion sizing.
    pub fn with_kelly(mut self, win_rate: f64, win_loss_ratio: f64) -> Self {
        self.use_kelly = true;
        self.win_rate = win_rate;
        self.win_loss_ratio = win_loss_ratio;
        self
    }

    /// Calculate position size with reasoning.
    pub fn calculate(
        &self,
        signal: Signal,
        confidence: f64,
        entry_price: f64,
        stop_loss: f64,
        portfolio_value: f64,
        volatility: Option<f64>,
    ) -> PositionResult {
        let mut reasoning = Vec::new();

        // Step 1: Determine base position from signal strength
        let signal_multiplier = self.get_signal_multiplier(signal, confidence);
        reasoning.push(format!(
            "Step 1: Signal {:?} with {:.0}% confidence -> multiplier {:.2}",
            signal, confidence * 100.0, signal_multiplier
        ));

        // Step 2: Calculate risk per unit
        let risk_per_unit = (entry_price - stop_loss).abs();
        let risk_pct = risk_per_unit / entry_price;
        reasoning.push(format!(
            "Step 2: Risk per unit = ${:.2} ({:.2}% of entry)",
            risk_per_unit, risk_pct * 100.0
        ));

        // Step 3: Calculate maximum position from risk limit
        let max_risk_amount = portfolio_value * self.max_risk_pct;
        let risk_limited_units = max_risk_amount / risk_per_unit;
        let risk_limited_value = risk_limited_units * entry_price;
        reasoning.push(format!(
            "Step 3: Risk limit ${:.2} ({:.1}% of ${:.0}) allows {:.2} units (${:.2})",
            max_risk_amount, self.max_risk_pct * 100.0, portfolio_value,
            risk_limited_units, risk_limited_value
        ));

        // Step 4: Apply position size limit
        let max_position_value = portfolio_value * self.max_position_pct;
        let position_limited_units = max_position_value / entry_price;
        reasoning.push(format!(
            "Step 4: Position limit {:.1}% of portfolio = ${:.2} ({:.2} units)",
            self.max_position_pct * 100.0, max_position_value, position_limited_units
        ));

        // Step 5: Kelly criterion (optional)
        let kelly_units = if self.use_kelly {
            let kelly_fraction = self.calculate_kelly();
            let kelly_value = portfolio_value * kelly_fraction * signal_multiplier;
            let units = kelly_value / entry_price;
            reasoning.push(format!(
                "Step 5: Kelly criterion ({:.1}% base) suggests {:.2} units after signal adjustment",
                kelly_fraction * 100.0, units
            ));
            Some(units)
        } else {
            reasoning.push("Step 5: Kelly criterion not used".to_string());
            None
        };

        // Step 6: Volatility adjustment (optional)
        let volatility_multiplier = if let Some(vol) = volatility {
            let mult = if vol > 0.03 {
                0.5  // High volatility - reduce size
            } else if vol < 0.01 {
                1.2  // Low volatility - can increase slightly
            } else {
                1.0  // Normal volatility
            };
            reasoning.push(format!(
                "Step 6: Volatility {:.2}% -> multiplier {:.2}",
                vol * 100.0, mult
            ));
            mult
        } else {
            reasoning.push("Step 6: No volatility adjustment".to_string());
            1.0
        };

        // Step 7: Final calculation
        let base_units = risk_limited_units.min(position_limited_units);
        let adjusted_units = if let Some(kelly) = kelly_units {
            base_units.min(kelly)
        } else {
            base_units
        };
        let final_units = adjusted_units * signal_multiplier * volatility_multiplier;

        // Ensure non-negative and within limits
        let final_units = final_units.max(0.0);
        let position_value = final_units * entry_price;
        let position_pct = position_value / portfolio_value;
        let risk_amount = final_units * risk_per_unit;

        reasoning.push(format!(
            "Step 7: Final position = {:.4} units (${:.2}, {:.2}% of portfolio, ${:.2} at risk)",
            final_units, position_value, position_pct * 100.0, risk_amount
        ));

        PositionResult {
            position_value,
            units: final_units,
            risk_amount,
            position_pct,
            reasoning_chain: reasoning,
        }
    }

    fn get_signal_multiplier(&self, signal: Signal, confidence: f64) -> f64 {
        let base = match signal {
            Signal::StrongBuy | Signal::StrongSell => 1.0,
            Signal::Buy | Signal::Sell => 0.7,
            Signal::Hold => 0.0,
        };

        // Scale by confidence (0.5 to 1.0 range)
        base * (0.5 + 0.5 * confidence)
    }

    fn calculate_kelly(&self) -> f64 {
        // Kelly formula: f* = (bp - q) / b
        // where b = win/loss ratio, p = win rate, q = 1 - p
        let b = self.win_loss_ratio;
        let p = self.win_rate;
        let q = 1.0 - p;

        let kelly = (b * p - q) / b;

        // Apply half-Kelly for safety
        (kelly * 0.5).max(0.0).min(self.max_position_pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_sizing() {
        let sizer = PositionSizer::new(0.1, 0.02);

        let result = sizer.calculate(
            Signal::Buy,
            0.75,
            100.0,  // entry
            95.0,   // stop loss
            100000.0, // portfolio
            Some(0.02),
        );

        assert!(result.position_value > 0.0);
        assert!(result.position_pct <= 0.1);
        assert!(result.risk_amount <= 100000.0 * 0.02);
        assert!(!result.reasoning_chain.is_empty());
    }

    #[test]
    fn test_hold_signal() {
        let sizer = PositionSizer::new(0.1, 0.02);

        let result = sizer.calculate(
            Signal::Hold,
            0.5,
            100.0,
            95.0,
            100000.0,
            None,
        );

        assert_eq!(result.position_value, 0.0);
        assert_eq!(result.units, 0.0);
    }

    #[test]
    fn test_kelly_sizing() {
        let sizer = PositionSizer::new(0.1, 0.02)
            .with_kelly(0.6, 1.5);

        let result = sizer.calculate(
            Signal::StrongBuy,
            0.9,
            100.0,
            95.0,
            100000.0,
            None,
        );

        assert!(result.position_value > 0.0);
        assert!(!result.reasoning_chain.is_empty());
    }
}
