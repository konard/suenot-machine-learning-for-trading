//! Trading signal generation based on volatility predictions.

/// Trading signal types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Increase position size (low volatility regime).
    ScaleUp { factor: f64 },
    /// Decrease position size (high volatility regime).
    ScaleDown { factor: f64 },
    /// Exit position (volatility spike).
    Exit,
    /// Hold current position.
    Hold,
}

/// Generate trading signal from predicted volatility.
///
/// # Arguments
/// * `predicted_vol` - Current predicted volatility
/// * `historical_vol` - Historical average volatility
/// * `vol_change_rate` - Rate of change in volatility
pub fn generate_signal(
    predicted_vol: f64,
    historical_vol: f64,
    vol_change_rate: f64,
) -> TradingSignal {
    // Volatility spike detection
    if vol_change_rate > 0.5 {
        return TradingSignal::Exit;
    }

    let vol_ratio = if historical_vol > 1e-10 {
        predicted_vol / historical_vol
    } else {
        1.0
    };

    if vol_ratio < 0.7 {
        // Low vol regime: scale up positions
        TradingSignal::ScaleUp { factor: 1.0 / vol_ratio.max(0.3) }
    } else if vol_ratio > 1.5 {
        // High vol regime: scale down positions
        TradingSignal::ScaleDown { factor: vol_ratio.min(3.0) }
    } else {
        TradingSignal::Hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_vol_scales_up() {
        let signal = generate_signal(0.01, 0.02, 0.0);
        match signal {
            TradingSignal::ScaleUp { factor } => assert!(factor > 1.0),
            _ => panic!("Expected ScaleUp signal"),
        }
    }

    #[test]
    fn test_high_vol_scales_down() {
        let signal = generate_signal(0.04, 0.02, 0.0);
        match signal {
            TradingSignal::ScaleDown { factor } => assert!(factor > 1.0),
            _ => panic!("Expected ScaleDown signal"),
        }
    }

    #[test]
    fn test_spike_exits() {
        let signal = generate_signal(0.03, 0.02, 0.8);
        assert_eq!(signal, TradingSignal::Exit);
    }

    #[test]
    fn test_normal_holds() {
        let signal = generate_signal(0.02, 0.02, 0.0);
        assert_eq!(signal, TradingSignal::Hold);
    }
}
