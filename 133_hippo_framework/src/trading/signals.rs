//! Trading signal types and generation from HiPPO coefficients.

/// A trading signal with direction and strength.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradingSignal {
    /// Position direction: 1.0 = long, -1.0 = short, 0.0 = flat.
    pub position: f64,
    /// Signal confidence in [0, 1].
    pub confidence: f64,
    /// Trend coefficient (c1) value.
    pub trend: f64,
    /// Acceleration coefficient (c2) value.
    pub acceleration: f64,
}

impl TradingSignal {
    /// Create a new signal.
    pub fn new(position: f64, confidence: f64, trend: f64, acceleration: f64) -> Self {
        Self {
            position: position.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            trend,
            acceleration,
        }
    }

    /// Flat (no position) signal.
    pub fn flat() -> Self {
        Self {
            position: 0.0,
            confidence: 0.0,
            trend: 0.0,
            acceleration: 0.0,
        }
    }

    /// Whether the signal indicates a long position.
    pub fn is_long(&self) -> bool {
        self.position > 0.0
    }

    /// Whether the signal indicates a short position.
    pub fn is_short(&self) -> bool {
        self.position < 0.0
    }

    /// Whether the signal is flat (no position).
    pub fn is_flat(&self) -> bool {
        self.position.abs() < 1e-10
    }
}

/// Generate trading signals from HiPPO coefficient history.
///
/// Uses the trend coefficient (c1) for direction and higher-order
/// coefficients for confidence estimation.
///
/// # Arguments
/// * `coefficient_history` - Vec of state vectors from HiPPO processing
/// * `trend_threshold` - Minimum |c1| to generate a non-zero signal
pub fn generate_signals(
    coefficient_history: &[Vec<f64>],
    trend_threshold: f64,
) -> Vec<TradingSignal> {
    coefficient_history
        .iter()
        .map(|coeffs| {
            if coeffs.len() < 2 {
                return TradingSignal::flat();
            }

            let c1 = coeffs[1]; // trend coefficient
            let c2 = if coeffs.len() > 2 { coeffs[2] } else { 0.0 };

            // Compute volatility proxy from higher-order coefficients
            let high_var = if coeffs.len() > 4 {
                let half = coeffs.len() / 2;
                let high_coeffs = &coeffs[half..];
                let mean: f64 = high_coeffs.iter().sum::<f64>() / high_coeffs.len() as f64;
                high_coeffs.iter().map(|&c| (c - mean).powi(2)).sum::<f64>()
                    / high_coeffs.len() as f64
            } else {
                0.0
            };

            // Confidence based on signal strength relative to noise
            let confidence = if high_var > 1e-10 {
                (c1.abs() / (high_var.sqrt() + 1e-10)).min(1.0)
            } else {
                0.5
            };

            let position = if c1 > trend_threshold {
                1.0
            } else if c1 < -trend_threshold {
                -1.0
            } else {
                0.0
            };

            TradingSignal::new(position, confidence, c1, c2)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_signal_creation() {
        let signal = TradingSignal::new(1.0, 0.8, 0.5, 0.1);
        assert!(signal.is_long());
        assert!(!signal.is_short());
        assert!(!signal.is_flat());
    }

    #[test]
    fn test_flat_signal() {
        let signal = TradingSignal::flat();
        assert!(signal.is_flat());
        assert_eq!(signal.confidence, 0.0);
    }

    #[test]
    fn test_generate_signals() {
        let history = vec![
            vec![0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let signals = generate_signals(&history, 0.1);
        assert_eq!(signals.len(), 3);
        assert!(signals[0].is_long());
        assert!(signals[1].is_short());
        assert!(signals[2].is_flat());
    }
}
