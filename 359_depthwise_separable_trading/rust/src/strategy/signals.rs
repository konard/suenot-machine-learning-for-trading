//! Trading signals and strategy implementation
//!
//! Generates trading signals using depthwise separable convolution model.

use ndarray::{Array1, Array2};

use crate::convolution::DepthwiseSeparableConv1d;
use crate::data::Candle;
use crate::indicators::TechnicalIndicators;

use super::StrategyError;

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold/Neutral
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl Signal {
    /// Convert from prediction class
    pub fn from_class(class: usize, confidence: f64) -> Self {
        match class {
            0 if confidence > 0.7 => Signal::StrongSell,
            0 => Signal::Sell,
            1 => Signal::Hold,
            2 if confidence > 0.7 => Signal::StrongBuy,
            2 => Signal::Buy,
            _ => Signal::Hold,
        }
    }

    /// Convert to position multiplier
    pub fn to_position_multiplier(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Hold => 0.0,
            Signal::Sell => -0.5,
            Signal::StrongSell => -1.0,
        }
    }

    /// Is this a buy signal
    pub fn is_buy(&self) -> bool {
        matches!(self, Signal::Buy | Signal::StrongBuy)
    }

    /// Is this a sell signal
    pub fn is_sell(&self) -> bool {
        matches!(self, Signal::Sell | Signal::StrongSell)
    }
}

/// Signal generator trait
pub trait SignalGenerator {
    /// Generate signal for current market state
    fn generate_signal(&self, features: &Array2<f64>) -> Result<(Signal, f64), StrategyError>;

    /// Generate signals for a series
    fn generate_signals(
        &self,
        candles: &[Candle],
    ) -> Result<Vec<(Signal, f64)>, StrategyError>;
}

/// Trading strategy using DSC model
#[derive(Debug)]
pub struct TradingStrategy {
    /// DSC model for feature extraction
    model: DepthwiseSeparableConv1d,
    /// Technical indicators calculator
    indicators: TechnicalIndicators,
    /// Lookback window size
    window_size: usize,
    /// Confidence threshold for trading
    confidence_threshold: f64,
}

impl TradingStrategy {
    /// Create new trading strategy
    pub fn new(model: DepthwiseSeparableConv1d) -> Self {
        Self {
            model,
            indicators: TechnicalIndicators::new(),
            window_size: 100,
            confidence_threshold: 0.6,
        }
    }

    /// Set window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Prepare features from candles
    pub fn prepare_features(&self, candles: &[Candle]) -> Result<Array2<f64>, StrategyError> {
        if candles.len() < self.window_size {
            return Err(StrategyError::InsufficientData {
                needed: self.window_size,
                got: candles.len(),
            });
        }

        let n = candles.len();

        // Extract OHLCV arrays
        let open: Array1<f64> = Array1::from_iter(candles.iter().map(|c| c.open));
        let high: Array1<f64> = Array1::from_iter(candles.iter().map(|c| c.high));
        let low: Array1<f64> = Array1::from_iter(candles.iter().map(|c| c.low));
        let close: Array1<f64> = Array1::from_iter(candles.iter().map(|c| c.close));
        let volume: Array1<f64> = Array1::from_iter(candles.iter().map(|c| c.volume));

        // Calculate indicators
        let indicators = self.indicators.calculate_all(&open, &high, &low, &close, &volume);

        // Build feature matrix
        let num_features = 5 + indicators.len(); // OHLCV + indicators
        let mut features = Array2::zeros((num_features, n));

        // Add OHLCV (normalized)
        let close_mean = close.mean().unwrap_or(1.0);
        for i in 0..n {
            features[[0, i]] = open[i] / close_mean;
            features[[1, i]] = high[i] / close_mean;
            features[[2, i]] = low[i] / close_mean;
            features[[3, i]] = close[i] / close_mean;
            features[[4, i]] = volume[i] / volume.mean().unwrap_or(1.0);
        }

        // Add indicators
        for (idx, (_, indicator_values)) in indicators.iter().enumerate() {
            for i in 0..n {
                features[[5 + idx, i]] = indicator_values[i];
            }
        }

        // Normalize features (z-score)
        for mut row in features.rows_mut() {
            let mean = row.mean().unwrap_or(0.0);
            let std = row.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0).sqrt();
            if std > 1e-10 {
                row.mapv_inplace(|x| (x - mean) / std);
            }
        }

        Ok(features)
    }

    /// Apply model to features
    fn apply_model(&self, features: &Array2<f64>) -> Array2<f64> {
        self.model.forward(features)
    }

    /// Convert model output to signal
    fn output_to_signal(&self, output: &Array2<f64>) -> (Signal, f64) {
        // Global average pooling
        let pooled: Vec<f64> = output
            .rows()
            .into_iter()
            .map(|row| row.mean().unwrap_or(0.0))
            .collect();

        // Simple classification based on pooled features
        let score: f64 = pooled.iter().sum::<f64>() / pooled.len() as f64;

        // Convert score to signal
        let (signal, confidence) = if score > 0.5 {
            (Signal::StrongBuy, (score - 0.5) * 2.0)
        } else if score > 0.2 {
            (Signal::Buy, (score - 0.2) / 0.3)
        } else if score < -0.5 {
            (Signal::StrongSell, (-score - 0.5) * 2.0)
        } else if score < -0.2 {
            (Signal::Sell, (-score - 0.2) / 0.3)
        } else {
            (Signal::Hold, 1.0 - score.abs() * 5.0)
        };

        (signal, confidence.clamp(0.0, 1.0))
    }
}

impl SignalGenerator for TradingStrategy {
    fn generate_signal(&self, features: &Array2<f64>) -> Result<(Signal, f64), StrategyError> {
        let output = self.apply_model(features);
        Ok(self.output_to_signal(&output))
    }

    fn generate_signals(
        &self,
        candles: &[Candle],
    ) -> Result<Vec<(Signal, f64)>, StrategyError> {
        if candles.len() < self.window_size {
            return Err(StrategyError::InsufficientData {
                needed: self.window_size,
                got: candles.len(),
            });
        }

        let full_features = self.prepare_features(candles)?;
        let mut signals = Vec::new();

        // Generate signals for each window
        for i in 0..=candles.len() - self.window_size {
            let window = full_features
                .slice(ndarray::s![.., i..i + self.window_size])
                .to_owned();

            let (signal, confidence) = self.generate_signal(&window)?;

            // Only signal if confidence is above threshold
            if confidence >= self.confidence_threshold {
                signals.push((signal, confidence));
            } else {
                signals.push((Signal::Hold, confidence));
            }
        }

        Ok(signals)
    }
}

/// Ensemble strategy combining multiple models
pub struct EnsembleStrategy {
    strategies: Vec<TradingStrategy>,
    weights: Vec<f64>,
}

impl EnsembleStrategy {
    /// Create ensemble from multiple strategies
    pub fn new(strategies: Vec<TradingStrategy>, weights: Option<Vec<f64>>) -> Self {
        let n = strategies.len();
        let weights = weights.unwrap_or_else(|| vec![1.0 / n as f64; n]);

        Self { strategies, weights }
    }

    /// Generate ensemble signal
    pub fn generate_signal(
        &self,
        candles: &[Candle],
    ) -> Result<(Signal, f64), StrategyError> {
        let mut total_score = 0.0;
        let mut total_confidence = 0.0;

        for (strategy, weight) in self.strategies.iter().zip(self.weights.iter()) {
            let features = strategy.prepare_features(candles)?;
            let window = features
                .slice(ndarray::s![.., features.dim().1 - strategy.window_size..])
                .to_owned();

            let (signal, confidence) = strategy.generate_signal(&window)?;

            total_score += signal.to_position_multiplier() * weight * confidence;
            total_confidence += confidence * weight;
        }

        // Convert aggregate score to signal
        let signal = if total_score > 0.5 {
            Signal::StrongBuy
        } else if total_score > 0.2 {
            Signal::Buy
        } else if total_score < -0.5 {
            Signal::StrongSell
        } else if total_score < -0.2 {
            Signal::Sell
        } else {
            Signal::Hold
        };

        Ok((signal, total_confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::data::Timeframe;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base_price = 100.0 + (i as f64 * 0.1);
                Candle::new(
                    "BTCUSDT",
                    Utc::now(),
                    Timeframe::H1,
                    base_price,
                    base_price + 2.0,
                    base_price - 1.0,
                    base_price + 1.0,
                    1000.0 + (i as f64 * 10.0),
                )
            })
            .collect()
    }

    #[test]
    fn test_signal_from_class() {
        assert_eq!(Signal::from_class(0, 0.8), Signal::StrongSell);
        assert_eq!(Signal::from_class(0, 0.5), Signal::Sell);
        assert_eq!(Signal::from_class(1, 0.9), Signal::Hold);
        assert_eq!(Signal::from_class(2, 0.8), Signal::StrongBuy);
        assert_eq!(Signal::from_class(2, 0.5), Signal::Buy);
    }

    #[test]
    fn test_signal_properties() {
        assert!(Signal::Buy.is_buy());
        assert!(Signal::StrongBuy.is_buy());
        assert!(!Signal::Sell.is_buy());

        assert!(Signal::Sell.is_sell());
        assert!(Signal::StrongSell.is_sell());
        assert!(!Signal::Buy.is_sell());
    }

    #[test]
    fn test_strategy_creation() {
        let model = DepthwiseSeparableConv1d::new(22, 64, 3).unwrap();
        let strategy = TradingStrategy::new(model)
            .with_window_size(50)
            .with_confidence_threshold(0.7);

        assert_eq!(strategy.window_size, 50);
        assert_eq!(strategy.confidence_threshold, 0.7);
    }

    #[test]
    fn test_prepare_features() {
        let model = DepthwiseSeparableConv1d::new(22, 64, 3).unwrap();
        let strategy = TradingStrategy::new(model).with_window_size(50);

        let candles = create_test_candles(100);
        let features = strategy.prepare_features(&candles);

        assert!(features.is_ok());
        let features = features.unwrap();
        assert_eq!(features.dim().1, 100);
    }

    #[test]
    fn test_insufficient_data() {
        let model = DepthwiseSeparableConv1d::new(22, 64, 3).unwrap();
        let strategy = TradingStrategy::new(model).with_window_size(100);

        let candles = create_test_candles(50);
        let result = strategy.prepare_features(&candles);

        assert!(matches!(result, Err(StrategyError::InsufficientData { .. })));
    }
}
