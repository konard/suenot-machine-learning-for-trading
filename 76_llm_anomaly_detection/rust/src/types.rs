//! Common types for anomaly detection.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    /// Create a new candle.
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Calculate the price range as a percentage of close.
    pub fn range_pct(&self) -> f64 {
        (self.high - self.low) / self.close
    }

    /// Calculate the body size (open to close) as a percentage.
    pub fn body_pct(&self) -> f64 {
        (self.close - self.open).abs() / self.close
    }
}

/// Computed features for anomaly detection.
#[derive(Debug, Clone, Default)]
pub struct Features {
    pub returns: f64,
    pub log_returns: f64,
    pub volatility: f64,
    pub volume_ratio: f64,
    pub range_ratio: f64,
    pub returns_zscore: f64,
    pub volume_zscore: f64,
}

impl Features {
    /// Create features from raw values.
    pub fn new(
        returns: f64,
        log_returns: f64,
        volatility: f64,
        volume_ratio: f64,
        range_ratio: f64,
        returns_zscore: f64,
        volume_zscore: f64,
    ) -> Self {
        Self {
            returns,
            log_returns,
            volatility,
            volume_ratio,
            range_ratio,
            returns_zscore,
            volume_zscore,
        }
    }

    /// Convert features to a vector for numerical operations.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.returns,
            self.volume_ratio,
            self.range_ratio,
            self.returns_zscore,
        ]
    }
}

/// Type of anomaly detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyType {
    PriceSpike,
    VolumeAnomaly,
    PatternBreak,
    PumpAndDump,
    FlashCrash,
    UnusualCorrelation,
    NewsMismatch,
    Unknown,
}

impl std::fmt::Display for AnomalyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyType::PriceSpike => write!(f, "price_spike"),
            AnomalyType::VolumeAnomaly => write!(f, "volume_anomaly"),
            AnomalyType::PatternBreak => write!(f, "pattern_break"),
            AnomalyType::PumpAndDump => write!(f, "pump_and_dump"),
            AnomalyType::FlashCrash => write!(f, "flash_crash"),
            AnomalyType::UnusualCorrelation => write!(f, "unusual_correlation"),
            AnomalyType::NewsMismatch => write!(f, "news_mismatch"),
            AnomalyType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Result of anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub is_anomaly: bool,
    pub score: f64,
    pub anomaly_type: AnomalyType,
    pub confidence: f64,
    pub explanation: String,
    pub details: HashMap<String, f64>,
}

impl AnomalyResult {
    /// Create a normal (non-anomaly) result.
    pub fn normal() -> Self {
        Self {
            is_anomaly: false,
            score: 0.0,
            anomaly_type: AnomalyType::Unknown,
            confidence: 1.0,
            explanation: "Normal behavior".to_string(),
            details: HashMap::new(),
        }
    }

    /// Create an anomaly result.
    pub fn anomaly(
        score: f64,
        anomaly_type: AnomalyType,
        confidence: f64,
        explanation: String,
    ) -> Self {
        Self {
            is_anomaly: true,
            score,
            anomaly_type,
            confidence,
            explanation,
            details: HashMap::new(),
        }
    }

    /// Add a detail to the result.
    pub fn with_detail(mut self, key: &str, value: f64) -> Self {
        self.details.insert(key.to_string(), value);
        self
    }
}

/// Trading signal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
    ExitLong,
    ExitShort,
    ReducePosition,
    IncreasePosition,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::Hold => write!(f, "HOLD"),
            SignalType::ExitLong => write!(f, "EXIT_LONG"),
            SignalType::ExitShort => write!(f, "EXIT_SHORT"),
            SignalType::ReducePosition => write!(f, "REDUCE"),
            SignalType::IncreasePosition => write!(f, "INCREASE"),
        }
    }
}

/// Trading signal with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub signal_type: SignalType,
    pub confidence: f64,
    pub strength: f64,
    pub reason: String,
    pub timestamp: Option<DateTime<Utc>>,
    pub price: Option<f64>,
    pub suggested_position_size: Option<f64>,
}

impl TradingSignal {
    /// Create a hold signal.
    pub fn hold(reason: &str) -> Self {
        Self {
            signal_type: SignalType::Hold,
            confidence: 1.0,
            strength: 0.0,
            reason: reason.to_string(),
            timestamp: None,
            price: None,
            suggested_position_size: None,
        }
    }

    /// Create a new signal.
    pub fn new(
        signal_type: SignalType,
        confidence: f64,
        strength: f64,
        reason: &str,
    ) -> Self {
        Self {
            signal_type,
            confidence,
            strength,
            reason: reason.to_string(),
            timestamp: None,
            price: None,
            suggested_position_size: None,
        }
    }

    /// Set the timestamp.
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Set the price.
    pub fn with_price(mut self, price: f64) -> Self {
        self.price = Some(price);
        self
    }

    /// Set the suggested position size.
    pub fn with_position_size(mut self, size: f64) -> Self {
        self.suggested_position_size = Some(size);
        self
    }
}

/// Trade record for backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_time: DateTime<Utc>,
    pub entry_price: f64,
    pub direction: i32, // 1 for long, -1 for short
    pub size: f64,
    pub exit_time: Option<DateTime<Utc>>,
    pub exit_price: Option<f64>,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub reason: String,
}

impl Trade {
    /// Create a new open trade.
    pub fn new(
        entry_time: DateTime<Utc>,
        entry_price: f64,
        direction: i32,
        size: f64,
        reason: &str,
    ) -> Self {
        Self {
            entry_time,
            entry_price,
            direction,
            size,
            exit_time: None,
            exit_price: None,
            pnl: 0.0,
            pnl_pct: 0.0,
            reason: reason.to_string(),
        }
    }

    /// Check if the trade is still open.
    pub fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }

    /// Close the trade.
    pub fn close(&mut self, exit_time: DateTime<Utc>, exit_price: f64) {
        self.exit_time = Some(exit_time);
        self.exit_price = Some(exit_price);
        self.pnl = (exit_price - self.entry_price) * self.direction as f64 * self.size;
        self.pnl_pct = (exit_price / self.entry_price - 1.0) * self.direction as f64 * 100.0;
    }
}

/// Backtest result metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_return: f64,
    pub total_return_pct: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
    pub num_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
    pub anomaly_count: usize,
}

impl BacktestResult {
    /// Print a summary of the backtest results.
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(50));
        println!("BACKTEST RESULTS");
        println!("{}", "=".repeat(50));
        println!(
            "Total Return: ${:.2} ({:.2}%)",
            self.total_return, self.total_return_pct
        );
        println!("Sharpe Ratio: {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio: {:.2}", self.sortino_ratio);
        println!(
            "Max Drawdown: ${:.2} ({:.2}%)",
            self.max_drawdown, self.max_drawdown_pct
        );
        println!("\nNumber of Trades: {}", self.num_trades);
        println!("Win Rate: {:.1}%", self.win_rate);
        println!("Average Win: ${:.2}", self.avg_win);
        println!("Average Loss: ${:.2}", self.avg_loss);
        println!("Profit Factor: {:.2}", self.profit_factor);
        println!("Anomalies Detected: {}", self.anomaly_count);
        println!("{}\n", "=".repeat(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_range_pct() {
        let candle = Candle::new(
            Utc::now(),
            100.0,
            105.0,
            95.0,
            102.0,
            1000.0,
        );
        let range = candle.range_pct();
        assert!((range - 0.098).abs() < 0.01);
    }

    #[test]
    fn test_anomaly_result() {
        let result = AnomalyResult::anomaly(
            0.85,
            AnomalyType::PriceSpike,
            0.9,
            "Unusual price movement".to_string(),
        )
        .with_detail("zscore", 4.2);

        assert!(result.is_anomaly);
        assert_eq!(result.anomaly_type, AnomalyType::PriceSpike);
        assert!(result.details.contains_key("zscore"));
    }

    #[test]
    fn test_trade_close() {
        let mut trade = Trade::new(
            Utc::now(),
            100.0,
            1,
            10.0,
            "Test trade",
        );

        trade.close(Utc::now(), 110.0);

        assert!(!trade.is_open());
        assert_eq!(trade.pnl, 100.0);
        assert!((trade.pnl_pct - 10.0).abs() < 0.01);
    }
}
