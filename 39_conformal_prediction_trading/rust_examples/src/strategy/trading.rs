//! Trading strategy using conformal prediction intervals
//!
//! Key principle: Trade only when the prediction interval is narrow
//! (high confidence) and the direction is clear.

use crate::conformal::PredictionInterval;
use crate::strategy::sizing::PositionSizer;

/// Trading signal with direction, size, and confidence
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Point prediction
    pub prediction: f64,
    /// Lower bound of prediction interval
    pub lower: f64,
    /// Upper bound of prediction interval
    pub upper: f64,
    /// Width of prediction interval
    pub interval_width: f64,
    /// Confidence score (0-1, higher = more confident)
    pub confidence: f64,
    /// Whether to take a trade
    pub trade: bool,
    /// Direction: 1 for long, -1 for short, 0 for no trade
    pub direction: i32,
    /// Position size (0.0 to 1.0)
    pub size: f64,
    /// Expected edge (worst-case expected return)
    pub edge: f64,
    /// Reason for skipping trade (if trade = false)
    pub skip_reason: Option<String>,
}

impl TradingSignal {
    /// Create a no-trade signal
    pub fn no_trade(interval: &PredictionInterval, reason: &str) -> Self {
        Self {
            prediction: interval.prediction,
            lower: interval.lower,
            upper: interval.upper,
            interval_width: interval.width,
            confidence: Self::width_to_confidence(interval.width),
            trade: false,
            direction: 0,
            size: 0.0,
            edge: 0.0,
            skip_reason: Some(reason.to_string()),
        }
    }

    /// Convert interval width to confidence score
    fn width_to_confidence(width: f64) -> f64 {
        1.0 / (1.0 + width * 10.0)
    }
}

/// Conformal prediction-based trading strategy
#[derive(Debug, Clone)]
pub struct ConformalTradingStrategy {
    /// Maximum interval width to take a trade
    width_threshold: f64,
    /// Minimum expected edge to trade
    min_edge: f64,
    /// Position sizer
    sizer: PositionSizer,
    /// Maximum position size
    max_size: f64,
}

impl Default for ConformalTradingStrategy {
    fn default() -> Self {
        Self::new(0.02, 0.005)
    }
}

impl ConformalTradingStrategy {
    /// Create a new trading strategy
    ///
    /// # Arguments
    /// * `width_threshold` - Maximum interval width to take a trade (e.g., 0.02 for 2%)
    /// * `min_edge` - Minimum expected edge to trade (e.g., 0.005 for 0.5%)
    pub fn new(width_threshold: f64, min_edge: f64) -> Self {
        Self {
            width_threshold,
            min_edge,
            sizer: PositionSizer::inverse(),
            max_size: 1.0,
        }
    }

    /// Create strategy with custom position sizer
    pub fn with_sizer(width_threshold: f64, min_edge: f64, sizer: PositionSizer) -> Self {
        Self {
            width_threshold,
            min_edge,
            sizer,
            max_size: 1.0,
        }
    }

    /// Set maximum position size
    pub fn with_max_size(mut self, max_size: f64) -> Self {
        self.max_size = max_size.max(0.0).min(1.0);
        self
    }

    /// Generate trading signal from prediction interval
    pub fn generate_signal(&self, interval: &PredictionInterval) -> TradingSignal {
        // Condition 1: Interval must be narrow enough
        if interval.width >= self.width_threshold {
            return TradingSignal::no_trade(interval, "interval_too_wide");
        }

        // Condition 2: Direction must be clear with sufficient edge
        let (direction, edge) = if interval.lower > self.min_edge {
            // Entire interval is positive with sufficient magnitude
            (1, interval.lower)
        } else if interval.upper < -self.min_edge {
            // Entire interval is negative with sufficient magnitude
            (-1, -interval.upper)
        } else {
            return TradingSignal::no_trade(interval, "unclear_direction");
        };

        // Compute position size
        let raw_size = self.sizer.compute_size(interval.width, edge);
        let size = raw_size.min(self.max_size);

        TradingSignal {
            prediction: interval.prediction,
            lower: interval.lower,
            upper: interval.upper,
            interval_width: interval.width,
            confidence: TradingSignal::width_to_confidence(interval.width),
            trade: true,
            direction,
            size,
            edge,
            skip_reason: None,
        }
    }

    /// Process multiple intervals
    pub fn generate_signals(&self, intervals: &[PredictionInterval]) -> Vec<TradingSignal> {
        intervals.iter().map(|i| self.generate_signal(i)).collect()
    }

    /// Get width threshold
    pub fn width_threshold(&self) -> f64 {
        self.width_threshold
    }

    /// Get minimum edge
    pub fn min_edge(&self) -> f64 {
        self.min_edge
    }
}

/// Simple backtest result for a single trade
#[derive(Debug, Clone)]
pub struct TradeResult {
    /// Trading signal that generated this trade
    pub signal: TradingSignal,
    /// Actual return that occurred
    pub actual_return: f64,
    /// PnL from this trade (direction * size * actual_return)
    pub pnl: f64,
    /// Whether the actual value was covered by the interval
    pub covered: bool,
}

impl TradeResult {
    /// Create a new trade result
    pub fn new(signal: TradingSignal, actual_return: f64) -> Self {
        let pnl = signal.direction as f64 * signal.size * actual_return;
        let covered = actual_return >= signal.lower && actual_return <= signal.upper;

        Self {
            signal,
            actual_return,
            pnl,
            covered,
        }
    }
}

/// Backtester for conformal trading strategy
pub struct Backtester {
    strategy: ConformalTradingStrategy,
    results: Vec<TradeResult>,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(strategy: ConformalTradingStrategy) -> Self {
        Self {
            strategy,
            results: Vec::new(),
        }
    }

    /// Run backtest on prediction intervals and actual returns
    pub fn run(&mut self, intervals: &[PredictionInterval], actual_returns: &[f64]) {
        self.results.clear();

        for (interval, &actual) in intervals.iter().zip(actual_returns.iter()) {
            let signal = self.strategy.generate_signal(interval);
            let result = TradeResult::new(signal, actual);
            self.results.push(result);
        }
    }

    /// Get all results
    pub fn results(&self) -> &[TradeResult] {
        &self.results
    }

    /// Get only trade results (where trade = true)
    pub fn trades(&self) -> Vec<&TradeResult> {
        self.results.iter().filter(|r| r.signal.trade).collect()
    }

    /// Total PnL
    pub fn total_pnl(&self) -> f64 {
        self.results.iter().map(|r| r.pnl).sum()
    }

    /// Number of trades taken
    pub fn n_trades(&self) -> usize {
        self.results.iter().filter(|r| r.signal.trade).count()
    }

    /// Trade frequency (fraction of periods with trades)
    pub fn trade_frequency(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        self.n_trades() as f64 / self.results.len() as f64
    }

    /// Win rate (fraction of profitable trades)
    pub fn win_rate(&self) -> f64 {
        let trades: Vec<_> = self.trades();
        if trades.is_empty() {
            return 0.0;
        }
        let n_wins = trades.iter().filter(|t| t.pnl > 0.0).count();
        n_wins as f64 / trades.len() as f64
    }

    /// Average PnL per trade
    pub fn avg_pnl(&self) -> f64 {
        let trades: Vec<_> = self.trades();
        if trades.is_empty() {
            return 0.0;
        }
        let total: f64 = trades.iter().map(|t| t.pnl).sum();
        total / trades.len() as f64
    }

    /// Sharpe ratio (annualized, assuming daily data)
    pub fn sharpe_ratio(&self) -> f64 {
        let trades: Vec<_> = self.trades();
        if trades.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        if std.abs() < 1e-10 {
            return 0.0;
        }

        (mean / std) * (252.0_f64).sqrt()
    }

    /// Coverage on trades (fraction of trades where actual was in interval)
    pub fn coverage(&self) -> f64 {
        let trades: Vec<_> = self.trades();
        if trades.is_empty() {
            return 0.0;
        }
        let n_covered = trades.iter().filter(|t| t.covered).count();
        n_covered as f64 / trades.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation_long() {
        let strategy = ConformalTradingStrategy::new(0.05, 0.01);

        // Interval entirely above min_edge
        let interval = PredictionInterval::new(0.025, 0.015, 0.035);
        let signal = strategy.generate_signal(&interval);

        assert!(signal.trade);
        assert_eq!(signal.direction, 1);
        assert!(signal.size > 0.0);
    }

    #[test]
    fn test_signal_generation_short() {
        let strategy = ConformalTradingStrategy::new(0.05, 0.01);

        // Interval entirely below -min_edge
        let interval = PredictionInterval::new(-0.025, -0.035, -0.015);
        let signal = strategy.generate_signal(&interval);

        assert!(signal.trade);
        assert_eq!(signal.direction, -1);
        assert!(signal.size > 0.0);
    }

    #[test]
    fn test_signal_generation_no_trade_wide() {
        let strategy = ConformalTradingStrategy::new(0.02, 0.01);

        // Interval too wide
        let interval = PredictionInterval::new(0.025, 0.0, 0.05);
        let signal = strategy.generate_signal(&interval);

        assert!(!signal.trade);
        assert_eq!(signal.direction, 0);
        assert_eq!(signal.skip_reason, Some("interval_too_wide".to_string()));
    }

    #[test]
    fn test_signal_generation_no_trade_unclear() {
        let strategy = ConformalTradingStrategy::new(0.05, 0.01);

        // Interval crosses zero
        let interval = PredictionInterval::new(0.0, -0.01, 0.01);
        let signal = strategy.generate_signal(&interval);

        assert!(!signal.trade);
        assert_eq!(signal.direction, 0);
        assert_eq!(signal.skip_reason, Some("unclear_direction".to_string()));
    }

    #[test]
    fn test_backtester() {
        let strategy = ConformalTradingStrategy::new(0.05, 0.005);
        let mut backtester = Backtester::new(strategy);

        let intervals = vec![
            PredictionInterval::new(0.02, 0.01, 0.03),  // Long signal
            PredictionInterval::new(-0.02, -0.03, -0.01), // Short signal
            PredictionInterval::new(0.0, -0.02, 0.02),  // No trade (unclear)
        ];

        let actual_returns = vec![0.015, -0.025, 0.01];

        backtester.run(&intervals, &actual_returns);

        assert_eq!(backtester.n_trades(), 2);
        assert!(backtester.total_pnl() > 0.0); // Both trades profitable
    }
}
