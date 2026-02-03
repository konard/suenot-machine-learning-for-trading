//! Trading signal generation with SHAP-based confidence.

use crate::model::shap::ShapValues;

/// Trading signal with SHAP explanation.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Predicted probability (0-1).
    pub probability: f64,
    /// Signal direction (-1 = short, 0 = neutral, 1 = long).
    pub direction: i32,
    /// Confidence score (0-1).
    pub confidence: f64,
    /// SHAP explanation (optional).
    pub shap_values: Option<ShapValues>,
    /// SHAP stability score (0-1).
    pub shap_stability: f64,
}

impl TradingSignal {
    /// Create a new trading signal.
    pub fn new(probability: f64, shap_values: Option<ShapValues>) -> Self {
        let direction = if probability > 0.55 {
            1
        } else if probability < 0.45 {
            -1
        } else {
            0
        };

        let confidence = (probability - 0.5).abs() * 2.0;

        Self {
            probability,
            direction,
            confidence,
            shap_values,
            shap_stability: 1.0,
        }
    }

    /// Set SHAP stability score.
    pub fn with_stability(mut self, stability: f64) -> Self {
        self.shap_stability = stability.clamp(0.0, 1.0);
        self
    }

    /// Get position size based on signal strength and stability.
    pub fn position_size(&self, max_position: f64) -> f64 {
        let base_size = self.confidence * max_position;
        // Reduce position when SHAP explanations are unstable
        base_size * self.shap_stability
    }

    /// Check if signal is actionable.
    pub fn is_actionable(&self, min_confidence: f64, min_stability: f64) -> bool {
        self.confidence >= min_confidence && self.shap_stability >= min_stability
    }

    /// Get top features driving this signal.
    pub fn top_drivers(&self, n: usize) -> Vec<(String, f64)> {
        match &self.shap_values {
            Some(sv) => sv.top_contributors(n),
            None => vec![],
        }
    }
}

/// Signal generator with feature computation.
pub struct SignalGenerator {
    /// Lookback period for feature computation.
    pub lookback: usize,
    /// Minimum confidence threshold.
    pub min_confidence: f64,
}

impl SignalGenerator {
    /// Create a new signal generator.
    pub fn new(lookback: usize, min_confidence: f64) -> Self {
        Self {
            lookback,
            min_confidence,
        }
    }

    /// Compute trading features from price data.
    pub fn compute_features(&self, prices: &[f64], volumes: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.lookback {
            return vec![];
        }

        let mut features = Vec::new();

        // Returns at different horizons
        let returns_1 = (prices[n - 1] - prices[n - 2]) / prices[n - 2];
        let returns_5 = (prices[n - 1] - prices[n - 6]) / prices[n - 6];
        let returns_10 = (prices[n - 1] - prices[n - 11]) / prices[n - 11];

        features.push(returns_1);
        features.push(returns_5);
        features.push(returns_10);

        // Volatility (simple rolling std)
        let volatility = self.rolling_std(&prices[n - 20..], 20);
        features.push(volatility);

        // RSI
        let rsi = self.compute_rsi(&prices[n - 15..], 14);
        features.push(rsi / 100.0); // Normalize to 0-1

        // Volume ratio
        let vol_ma: f64 = volumes[n - 20..].iter().sum::<f64>() / 20.0;
        let vol_ratio = volumes[n - 1] / vol_ma.max(1.0);
        features.push(vol_ratio);

        // Price relative to MA
        let ma_20: f64 = prices[n - 20..].iter().sum::<f64>() / 20.0;
        let price_ma_ratio = prices[n - 1] / ma_20;
        features.push(price_ma_ratio);

        features
    }

    /// Rolling standard deviation.
    fn rolling_std(&self, data: &[f64], window: usize) -> f64 {
        let n = data.len().min(window);
        if n < 2 {
            return 0.0;
        }

        let mean: f64 = data[..n].iter().sum::<f64>() / n as f64;
        let variance: f64 = data[..n].iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        variance.sqrt()
    }

    /// Compute RSI indicator.
    fn compute_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..=period {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(20, 0.1)
    }
}
