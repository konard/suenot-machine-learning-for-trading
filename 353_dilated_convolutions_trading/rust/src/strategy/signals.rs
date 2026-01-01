//! Trading Signal Generation

use ndarray::Array2;

use crate::api::Kline;
use crate::conv::DilatedConvStack;
use crate::features::{Normalizer, TechnicalFeatures};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold / No action
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl Signal {
    /// Convert signal to position multiplier (-1.0 to 1.0)
    pub fn to_multiplier(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Hold => 0.0,
            Signal::Sell => -0.5,
            Signal::StrongSell => -1.0,
        }
    }

    /// Create signal from predicted direction
    pub fn from_prediction(direction: f64, confidence: f64) -> Self {
        let threshold_high = 0.6;
        let threshold_low = 0.4;

        if direction > 0.0 {
            if confidence > threshold_high {
                Signal::StrongBuy
            } else if confidence > threshold_low {
                Signal::Buy
            } else {
                Signal::Hold
            }
        } else if direction < 0.0 {
            if confidence > threshold_high {
                Signal::StrongSell
            } else if confidence > threshold_low {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }
}

/// Prediction output from the model
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted direction (-1, 0, 1)
    pub direction: f64,
    /// Predicted magnitude (absolute return)
    pub magnitude: f64,
    /// Predicted volatility
    pub volatility: f64,
    /// Confidence score (0 to 1)
    pub confidence: f64,
    /// Generated signal
    pub signal: Signal,
}

impl Prediction {
    /// Create a new prediction
    pub fn new(direction: f64, magnitude: f64, volatility: f64) -> Self {
        // Confidence based on magnitude relative to volatility
        let confidence = if volatility > 0.0 {
            (magnitude / volatility).min(1.0)
        } else {
            0.5
        };

        let signal = Signal::from_prediction(direction, confidence);

        Self {
            direction,
            magnitude,
            volatility,
            confidence,
            signal,
        }
    }
}

/// Trading strategy using dilated convolutions
#[derive(Debug)]
pub struct TradingStrategy {
    /// Dilated convolution model
    model: DilatedConvStack,
    /// Feature calculator
    features: TechnicalFeatures,
    /// Data normalizer
    normalizer: Normalizer,
    /// Minimum bars required for prediction
    min_bars: usize,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(model: DilatedConvStack) -> Self {
        let min_bars = model.receptive_field() + 10; // Add some buffer

        Self {
            model,
            features: TechnicalFeatures::new(),
            normalizer: Normalizer::new(),
            min_bars,
        }
    }

    /// Create with default model configuration
    pub fn default_model() -> Self {
        let model = DilatedConvStack::new(
            5,                           // 5 input features
            32,                          // 32 residual channels
            &[1, 2, 4, 8, 16, 32, 64],  // Dilation rates
        );
        Self::new(model)
    }

    /// Get minimum required bars
    pub fn min_bars(&self) -> usize {
        self.min_bars
    }

    /// Generate a prediction from klines
    pub fn predict(&self, klines: &[Kline]) -> Option<Prediction> {
        if klines.len() < self.min_bars {
            return None;
        }

        // Calculate features
        let features = self.features.calculate(klines);

        // Normalize (using simple z-score if not fitted)
        let normalized = if self.normalizer.is_fitted() {
            self.normalizer.transform(&features)
        } else {
            self.simple_normalize(&features)
        };

        // Get model prediction
        let output = self.model.predict_last(&normalized);

        // Parse output [direction, magnitude, volatility]
        let direction = output[0].tanh(); // Bound to [-1, 1]
        let magnitude = output[1].abs();
        let volatility = output[2].abs();

        Some(Prediction::new(direction, magnitude, volatility))
    }

    /// Generate signals for a sequence of klines
    pub fn generate_signals(&self, klines: &[Kline]) -> Vec<Signal> {
        let mut signals = Vec::new();

        for i in self.min_bars..=klines.len() {
            if let Some(pred) = self.predict(&klines[..i]) {
                signals.push(pred.signal);
            } else {
                signals.push(Signal::Hold);
            }
        }

        // Pad with Hold for initial bars
        let mut result = vec![Signal::Hold; self.min_bars];
        result.extend(signals);
        result.truncate(klines.len());

        result
    }

    /// Simple normalization (z-score per feature)
    fn simple_normalize(&self, data: &Array2<f64>) -> Array2<f64> {
        let (n_features, n_samples) = data.dim();
        let mut normalized = Array2::zeros((n_features, n_samples));

        for i in 0..n_features {
            let row = data.row(i);
            let mean: f64 = row.iter().sum::<f64>() / n_samples as f64;
            let variance: f64 = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            let std = variance.sqrt().max(1e-8);

            for j in 0..n_samples {
                normalized[[i, j]] = (data[[i, j]] - mean) / std;
            }
        }

        normalized
    }

    /// Backtest the strategy on historical data
    pub fn backtest(&self, klines: &[Kline]) -> BacktestResult {
        let signals = self.generate_signals(klines);
        let mut equity = 1.0;
        let mut max_equity = 1.0;
        let mut max_drawdown = 0.0;
        let mut trades = 0;
        let mut wins = 0;
        let mut returns = Vec::new();

        for i in 1..klines.len() {
            let signal = &signals[i - 1];
            let ret = (klines[i].close - klines[i - 1].close) / klines[i - 1].close;
            let position_return = ret * signal.to_multiplier();

            equity *= 1.0 + position_return;
            returns.push(position_return);

            if equity > max_equity {
                max_equity = equity;
            }
            let drawdown = (max_equity - equity) / max_equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }

            if *signal != Signal::Hold {
                trades += 1;
                if position_return > 0.0 {
                    wins += 1;
                }
            }
        }

        let total_return = equity - 1.0;
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_return = variance.sqrt();
        let sharpe = if std_return > 0.0 {
            mean_return / std_return * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            sharpe_ratio: sharpe,
            max_drawdown,
            total_trades: trades,
            win_rate: if trades > 0 { wins as f64 / trades as f64 } else { 0.0 },
            final_equity: equity,
        }
    }
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return (as decimal, e.g., 0.5 = 50%)
    pub total_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Maximum drawdown (as decimal)
    pub max_drawdown: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Final equity (starting from 1.0)
    pub final_equity: f64,
}

impl std::fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Backtest Results:")?;
        writeln!(f, "  Total Return: {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "  Sharpe Ratio: {:.2}", self.sharpe_ratio)?;
        writeln!(f, "  Max Drawdown: {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Total Trades: {}", self.total_trades)?;
        writeln!(f, "  Win Rate: {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  Final Equity: {:.4}", self.final_equity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_multiplier() {
        assert_eq!(Signal::StrongBuy.to_multiplier(), 1.0);
        assert_eq!(Signal::Hold.to_multiplier(), 0.0);
        assert_eq!(Signal::StrongSell.to_multiplier(), -1.0);
    }

    #[test]
    fn test_signal_from_prediction() {
        assert_eq!(Signal::from_prediction(1.0, 0.8), Signal::StrongBuy);
        assert_eq!(Signal::from_prediction(-1.0, 0.8), Signal::StrongSell);
        assert_eq!(Signal::from_prediction(0.0, 0.8), Signal::Hold);
    }

    #[test]
    fn test_prediction_creation() {
        let pred = Prediction::new(1.0, 0.02, 0.01);
        assert!(pred.confidence > 0.0);
        assert!(matches!(pred.signal, Signal::StrongBuy | Signal::Buy));
    }

    #[test]
    fn test_strategy_min_bars() {
        let strategy = TradingStrategy::default_model();
        assert!(strategy.min_bars() > 0);
    }
}
