//! Trading signal generation based on RDD analysis.

use crate::model::rdd::RDDResults;

/// Signal strength levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalStrength {
    /// Strong signal (high confidence)
    Strong,
    /// Medium signal
    Medium,
    /// Weak signal (low confidence)
    Weak,
    /// No signal
    None,
}

impl SignalStrength {
    /// Get numeric strength (0.0 to 1.0).
    pub fn as_f64(&self) -> f64 {
        match self {
            SignalStrength::Strong => 1.0,
            SignalStrength::Medium => 0.66,
            SignalStrength::Weak => 0.33,
            SignalStrength::None => 0.0,
        }
    }

    /// Create from effect size and standard error.
    pub fn from_effect(effect: f64, se: f64, threshold: f64) -> Self {
        if se <= 0.0 || effect.abs() < threshold {
            return SignalStrength::None;
        }

        let t_stat = effect.abs() / se;
        if t_stat > 2.58 {
            // 99% confidence
            SignalStrength::Strong
        } else if t_stat > 1.96 {
            // 95% confidence
            SignalStrength::Medium
        } else if t_stat > 1.645 {
            // 90% confidence
            SignalStrength::Weak
        } else {
            SignalStrength::None
        }
    }
}

/// Trading signal.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal direction: positive = long, negative = short
    pub direction: f64,
    /// Signal strength
    pub strength: SignalStrength,
    /// Estimated effect size
    pub effect_size: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Distance to threshold
    pub distance_to_threshold: f64,
    /// Suggested position size (0.0 to 1.0)
    pub suggested_size: f64,
}

impl TradingSignal {
    /// Create a new trading signal from RDD results.
    ///
    /// # Arguments
    /// * `results` - RDD estimation results
    /// * `running_value` - Current value of the running variable
    /// * `cutoff` - RDD cutoff/threshold
    pub fn from_rdd(results: &RDDResults, running_value: f64, cutoff: f64) -> Self {
        let distance = running_value - cutoff;
        let is_treated = distance >= 0.0;

        // Effect size and direction
        let effect_size = results.tau;
        let direction = if is_treated { effect_size } else { -effect_size };

        // Signal strength based on statistical significance
        let strength = SignalStrength::from_effect(effect_size, results.se, 0.001);

        // Suggested position size based on distance and confidence
        let distance_factor = (-distance.abs() / results.bandwidth.max(1.0)).exp();
        let confidence_factor = strength.as_f64();
        let suggested_size = (distance_factor * confidence_factor).min(1.0);

        TradingSignal {
            direction,
            strength,
            effect_size,
            confidence_interval: (results.ci_lower, results.ci_upper),
            distance_to_threshold: distance,
            suggested_size,
        }
    }

    /// Create a long (buy) signal.
    pub fn long(effect_size: f64, strength: SignalStrength) -> Self {
        TradingSignal {
            direction: effect_size.abs(),
            strength,
            effect_size,
            confidence_interval: (0.0, 0.0),
            distance_to_threshold: 0.0,
            suggested_size: strength.as_f64(),
        }
    }

    /// Create a short (sell) signal.
    pub fn short(effect_size: f64, strength: SignalStrength) -> Self {
        TradingSignal {
            direction: -effect_size.abs(),
            strength,
            effect_size,
            confidence_interval: (0.0, 0.0),
            distance_to_threshold: 0.0,
            suggested_size: strength.as_f64(),
        }
    }

    /// Create a neutral (no position) signal.
    pub fn neutral() -> Self {
        TradingSignal {
            direction: 0.0,
            strength: SignalStrength::None,
            effect_size: 0.0,
            confidence_interval: (0.0, 0.0),
            distance_to_threshold: 0.0,
            suggested_size: 0.0,
        }
    }

    /// Check if signal suggests going long.
    pub fn is_long(&self) -> bool {
        self.direction > 0.0 && self.strength != SignalStrength::None
    }

    /// Check if signal suggests going short.
    pub fn is_short(&self) -> bool {
        self.direction < 0.0 && self.strength != SignalStrength::None
    }

    /// Check if signal is actionable.
    pub fn is_actionable(&self) -> bool {
        self.strength != SignalStrength::None
    }
}

/// Signal generator for threshold-based strategies.
pub struct ThresholdSignalGenerator {
    /// Lower threshold (e.g., RSI 30)
    lower_threshold: f64,
    /// Upper threshold (e.g., RSI 70)
    upper_threshold: f64,
    /// Bandwidth around threshold
    bandwidth: f64,
    /// Required effect size to generate signal
    min_effect: f64,
}

impl ThresholdSignalGenerator {
    /// Create a new signal generator for RSI strategy.
    pub fn new_rsi() -> Self {
        Self {
            lower_threshold: 30.0,
            upper_threshold: 70.0,
            bandwidth: 5.0,
            min_effect: 0.001,
        }
    }

    /// Create a custom signal generator.
    pub fn new(
        lower_threshold: f64,
        upper_threshold: f64,
        bandwidth: f64,
        min_effect: f64,
    ) -> Self {
        Self {
            lower_threshold,
            upper_threshold,
            bandwidth,
            min_effect,
        }
    }

    /// Generate signal based on current indicator value and RDD results.
    pub fn generate(
        &self,
        indicator_value: f64,
        lower_rdd: Option<&RDDResults>,
        upper_rdd: Option<&RDDResults>,
    ) -> TradingSignal {
        // Check lower threshold (oversold)
        let lower_distance = indicator_value - self.lower_threshold;
        if lower_distance.abs() <= self.bandwidth {
            if let Some(results) = lower_rdd {
                if results.tau > self.min_effect && results.p_value < 0.1 {
                    let strength = SignalStrength::from_effect(results.tau, results.se, self.min_effect);
                    return TradingSignal {
                        direction: results.tau, // Positive = expect price increase
                        strength,
                        effect_size: results.tau,
                        confidence_interval: (results.ci_lower, results.ci_upper),
                        distance_to_threshold: lower_distance,
                        suggested_size: strength.as_f64() * (1.0 - lower_distance.abs() / self.bandwidth),
                    };
                }
            }
        }

        // Check upper threshold (overbought)
        let upper_distance = indicator_value - self.upper_threshold;
        if upper_distance.abs() <= self.bandwidth {
            if let Some(results) = upper_rdd {
                if results.tau < -self.min_effect && results.p_value < 0.1 {
                    let strength = SignalStrength::from_effect(results.tau, results.se, self.min_effect);
                    return TradingSignal {
                        direction: results.tau, // Negative = expect price decrease
                        strength,
                        effect_size: results.tau,
                        confidence_interval: (results.ci_lower, results.ci_upper),
                        distance_to_threshold: upper_distance,
                        suggested_size: strength.as_f64() * (1.0 - upper_distance.abs() / self.bandwidth),
                    };
                }
            }
        }

        TradingSignal::neutral()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_strength() {
        assert_eq!(
            SignalStrength::from_effect(0.05, 0.01, 0.001),
            SignalStrength::Strong
        );
        assert_eq!(
            SignalStrength::from_effect(0.02, 0.01, 0.001),
            SignalStrength::Medium
        );
        assert_eq!(
            SignalStrength::from_effect(0.0001, 0.01, 0.001),
            SignalStrength::None
        );
    }

    #[test]
    fn test_signal_directions() {
        let long = TradingSignal::long(0.05, SignalStrength::Strong);
        assert!(long.is_long());
        assert!(!long.is_short());

        let short = TradingSignal::short(0.05, SignalStrength::Strong);
        assert!(!short.is_long());
        assert!(short.is_short());

        let neutral = TradingSignal::neutral();
        assert!(!neutral.is_actionable());
    }
}
