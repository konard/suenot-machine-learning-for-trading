//! Trading signal generation from anomaly detection results.

use crate::types::{AnomalyResult, AnomalyType, Features, SignalType, TradingSignal};

/// Strategy for generating signals from anomalies.
#[derive(Debug, Clone, Copy)]
pub enum SignalStrategy {
    /// Trade against anomalies (mean reversion)
    Contrarian,
    /// Trade with anomalies (momentum)
    Momentum,
    /// Risk management (reduce exposure on anomalies)
    Risk,
}

/// Signal generator configuration.
pub struct SignalGeneratorConfig {
    pub strategy: SignalStrategy,
    pub min_anomaly_score: f64,
    pub min_confidence: f64,
    pub base_position_size: f64,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            strategy: SignalStrategy::Contrarian,
            min_anomaly_score: 0.6,
            min_confidence: 0.5,
            base_position_size: 1.0,
        }
    }
}

/// Trading signal generator based on anomaly detection.
pub struct SignalGenerator {
    config: SignalGeneratorConfig,
}

impl SignalGenerator {
    /// Create a new signal generator with default config.
    pub fn new() -> Self {
        Self {
            config: SignalGeneratorConfig::default(),
        }
    }

    /// Create a signal generator with custom config.
    pub fn with_config(config: SignalGeneratorConfig) -> Self {
        Self { config }
    }

    /// Create a signal generator with specified strategy.
    pub fn with_strategy(strategy: SignalStrategy) -> Self {
        Self {
            config: SignalGeneratorConfig {
                strategy,
                ..Default::default()
            },
        }
    }

    /// Compute position size based on anomaly score.
    fn compute_position_size(&self, anomaly_result: &AnomalyResult) -> f64 {
        // Higher anomaly score = smaller position (more risk)
        self.config.base_position_size * (1.0 - anomaly_result.score * 0.5)
    }

    /// Generate contrarian signal (trade against anomaly).
    fn generate_contrarian(
        &self,
        anomaly_result: &AnomalyResult,
        features: &Features,
    ) -> TradingSignal {
        if !anomaly_result.is_anomaly {
            return TradingSignal::hold("No anomaly detected");
        }

        let position_size = self.compute_position_size(anomaly_result);

        match anomaly_result.anomaly_type {
            AnomalyType::PriceSpike => {
                if features.returns > 0.0 {
                    TradingSignal::new(
                        SignalType::Sell,
                        anomaly_result.confidence,
                        anomaly_result.score,
                        "Price spike up - expect reversion",
                    )
                    .with_position_size(position_size)
                } else {
                    TradingSignal::new(
                        SignalType::Buy,
                        anomaly_result.confidence,
                        anomaly_result.score,
                        "Price spike down - expect reversion",
                    )
                    .with_position_size(position_size)
                }
            }
            AnomalyType::FlashCrash => TradingSignal::new(
                SignalType::Buy,
                anomaly_result.confidence,
                anomaly_result.score,
                "Flash crash detected - expect reversion",
            )
            .with_position_size(position_size),
            AnomalyType::PumpAndDump => TradingSignal::new(
                SignalType::Sell,
                anomaly_result.confidence,
                anomaly_result.score,
                "Pump and dump pattern - expect dump phase",
            )
            .with_position_size(position_size),
            _ => TradingSignal::hold("Unknown anomaly type"),
        }
    }

    /// Generate momentum signal (trade with anomaly).
    fn generate_momentum(
        &self,
        anomaly_result: &AnomalyResult,
        features: &Features,
    ) -> TradingSignal {
        if !anomaly_result.is_anomaly {
            return TradingSignal::hold("No anomaly detected");
        }

        let position_size = self.compute_position_size(anomaly_result);

        match anomaly_result.anomaly_type {
            AnomalyType::VolumeAnomaly | AnomalyType::PatternBreak => {
                if features.returns > 0.0 {
                    TradingSignal::new(
                        SignalType::Buy,
                        anomaly_result.confidence,
                        anomaly_result.score,
                        "Breakout detected - momentum buy",
                    )
                    .with_position_size(position_size)
                } else {
                    TradingSignal::new(
                        SignalType::Sell,
                        anomaly_result.confidence,
                        anomaly_result.score,
                        "Breakdown detected - momentum sell",
                    )
                    .with_position_size(position_size)
                }
            }
            _ => TradingSignal::hold("No momentum signal"),
        }
    }

    /// Generate risk management signal.
    fn generate_risk(
        &self,
        anomaly_result: &AnomalyResult,
        current_position: f64,
    ) -> TradingSignal {
        if !anomaly_result.is_anomaly {
            return TradingSignal::hold("Risk levels normal");
        }

        if anomaly_result.score > 0.8 {
            if current_position > 0.0 {
                TradingSignal::new(
                    SignalType::ExitLong,
                    anomaly_result.confidence,
                    anomaly_result.score,
                    "High risk - exit long position",
                )
            } else if current_position < 0.0 {
                TradingSignal::new(
                    SignalType::ExitShort,
                    anomaly_result.confidence,
                    anomaly_result.score,
                    "High risk - exit short position",
                )
            } else {
                TradingSignal::hold("High risk - no new positions")
            }
        } else if anomaly_result.score > 0.6 {
            let new_size = self.config.base_position_size * (1.0 - anomaly_result.score);
            TradingSignal::new(
                SignalType::ReducePosition,
                anomaly_result.confidence,
                anomaly_result.score,
                "Elevated risk - reduce position",
            )
            .with_position_size(new_size)
        } else {
            TradingSignal::hold("Moderate anomaly - hold position")
        }
    }

    /// Generate trading signal based on anomaly result.
    pub fn generate(
        &self,
        anomaly_result: &AnomalyResult,
        features: &Features,
        current_position: f64,
    ) -> TradingSignal {
        // Filter by thresholds
        if anomaly_result.score < self.config.min_anomaly_score {
            return TradingSignal::hold("Anomaly score below threshold");
        }

        if anomaly_result.confidence < self.config.min_confidence {
            return TradingSignal::hold("Confidence below threshold");
        }

        match self.config.strategy {
            SignalStrategy::Contrarian => self.generate_contrarian(anomaly_result, features),
            SignalStrategy::Momentum => self.generate_momentum(anomaly_result, features),
            SignalStrategy::Risk => self.generate_risk(anomaly_result, current_position),
        }
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contrarian_signal() {
        let generator = SignalGenerator::with_strategy(SignalStrategy::Contrarian);

        let anomaly = AnomalyResult::anomaly(
            0.85,
            AnomalyType::PriceSpike,
            0.9,
            "Test anomaly".to_string(),
        );

        let features = Features {
            returns: 0.05, // Positive return
            ..Default::default()
        };

        let signal = generator.generate(&anomaly, &features, 0.0);
        assert_eq!(signal.signal_type, SignalType::Sell);
    }

    #[test]
    fn test_momentum_signal() {
        let generator = SignalGenerator::with_strategy(SignalStrategy::Momentum);

        let anomaly = AnomalyResult::anomaly(
            0.75,
            AnomalyType::VolumeAnomaly,
            0.8,
            "Volume spike".to_string(),
        );

        let features = Features {
            returns: 0.03, // Positive return
            ..Default::default()
        };

        let signal = generator.generate(&anomaly, &features, 0.0);
        assert_eq!(signal.signal_type, SignalType::Buy);
    }

    #[test]
    fn test_risk_signal() {
        let generator = SignalGenerator::with_strategy(SignalStrategy::Risk);

        let anomaly = AnomalyResult::anomaly(
            0.85,
            AnomalyType::Unknown,
            0.9,
            "High risk".to_string(),
        );

        let features = Features::default();

        // With long position
        let signal = generator.generate(&anomaly, &features, 1.0);
        assert_eq!(signal.signal_type, SignalType::ExitLong);

        // With short position
        let signal = generator.generate(&anomaly, &features, -1.0);
        assert_eq!(signal.signal_type, SignalType::ExitShort);
    }

    #[test]
    fn test_below_threshold() {
        let generator = SignalGenerator::new();

        let anomaly = AnomalyResult::anomaly(
            0.3, // Below threshold
            AnomalyType::PriceSpike,
            0.9,
            "Low score anomaly".to_string(),
        );

        let features = Features::default();
        let signal = generator.generate(&anomaly, &features, 0.0);

        assert_eq!(signal.signal_type, SignalType::Hold);
    }
}
