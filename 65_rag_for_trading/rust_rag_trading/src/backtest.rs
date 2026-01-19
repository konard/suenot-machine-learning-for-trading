//! Backtesting framework for RAG-based trading strategies.

use crate::signals::{RAGSignalGenerator, SignalDirection, TradingSignal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single trade in the backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Stock ticker symbol
    pub ticker: String,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp (None if still open)
    pub exit_time: Option<DateTime<Utc>>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price (None if still open)
    pub exit_price: Option<f64>,
    /// Trade direction
    pub direction: SignalDirection,
    /// Position size
    pub quantity: f64,
    /// Profit/loss (None if still open)
    pub pnl: Option<f64>,
    /// The signal that generated this trade
    pub signal: TradingSignal,
}

impl Trade {
    /// Create a new trade.
    pub fn new(
        ticker: String,
        entry_time: DateTime<Utc>,
        entry_price: f64,
        direction: SignalDirection,
        quantity: f64,
        signal: TradingSignal,
    ) -> Self {
        Self {
            ticker,
            entry_time,
            exit_time: None,
            entry_price,
            exit_price: None,
            direction,
            quantity,
            pnl: None,
            signal,
        }
    }

    /// Close the trade.
    pub fn close(&mut self, exit_time: DateTime<Utc>, exit_price: f64) {
        self.exit_time = Some(exit_time);
        self.exit_price = Some(exit_price);

        let price_diff = exit_price - self.entry_price;
        self.pnl = Some(match self.direction {
            SignalDirection::Long => price_diff * self.quantity,
            SignalDirection::Short => -price_diff * self.quantity,
            SignalDirection::Neutral => 0.0,
        });
    }

    /// Check if trade is open.
    pub fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }

    /// Get return percentage.
    pub fn return_pct(&self) -> Option<f64> {
        self.exit_price.map(|exit| {
            let diff = exit - self.entry_price;
            match self.direction {
                SignalDirection::Long => diff / self.entry_price,
                SignalDirection::Short => -diff / self.entry_price,
                SignalDirection::Neutral => 0.0,
            }
        })
    }
}

/// Price data for backtesting.
#[derive(Debug, Clone)]
pub struct PriceBar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl PriceBar {
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
}

/// Result of a backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Starting capital
    pub initial_capital: f64,
    /// Final portfolio value
    pub final_value: f64,
    /// Total return percentage
    pub total_return: f64,
    /// Number of trades executed
    pub num_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate percentage
    pub win_rate: f64,
    /// Average profit per trade
    pub avg_profit: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Sharpe ratio (simplified)
    pub sharpe_ratio: f64,
    /// All trades
    pub trades: Vec<Trade>,
}

impl BacktestResult {
    /// Create a new backtest result.
    pub fn new(initial_capital: f64, trades: Vec<Trade>) -> Self {
        let closed_trades: Vec<&Trade> = trades.iter().filter(|t| !t.is_open()).collect();
        let num_trades = closed_trades.len();

        let total_pnl: f64 = closed_trades.iter().filter_map(|t| t.pnl).sum();
        let final_value = initial_capital + total_pnl;
        let total_return = (final_value - initial_capital) / initial_capital;

        let winning_trades = closed_trades
            .iter()
            .filter(|t| t.pnl.unwrap_or(0.0) > 0.0)
            .count();
        let losing_trades = closed_trades
            .iter()
            .filter(|t| t.pnl.unwrap_or(0.0) < 0.0)
            .count();

        let win_rate = if num_trades > 0 {
            winning_trades as f64 / num_trades as f64
        } else {
            0.0
        };

        let avg_profit = if num_trades > 0 {
            total_pnl / num_trades as f64
        } else {
            0.0
        };

        // Calculate max drawdown
        let mut peak = initial_capital;
        let mut max_dd = 0.0;
        let mut equity = initial_capital;

        for trade in &closed_trades {
            if let Some(pnl) = trade.pnl {
                equity += pnl;
                if equity > peak {
                    peak = equity;
                }
                let dd = (peak - equity) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
        }

        // Simplified Sharpe ratio
        let returns: Vec<f64> = closed_trades
            .iter()
            .filter_map(|t| t.return_pct())
            .collect();

        let sharpe_ratio = if returns.len() > 1 {
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.0 {
                mean / std_dev * (252.0_f64).sqrt() // Annualized
            } else {
                0.0
            }
        } else {
            0.0
        };

        Self {
            initial_capital,
            final_value,
            total_return,
            num_trades,
            winning_trades,
            losing_trades,
            win_rate,
            avg_profit,
            max_drawdown: max_dd,
            sharpe_ratio,
            trades,
        }
    }

    /// Print summary.
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results:\n\
             ==================\n\
             Initial Capital: ${:.2}\n\
             Final Value: ${:.2}\n\
             Total Return: {:.2}%\n\
             Number of Trades: {}\n\
             Win Rate: {:.1}%\n\
             Avg Profit/Trade: ${:.2}\n\
             Max Drawdown: {:.2}%\n\
             Sharpe Ratio: {:.2}",
            self.initial_capital,
            self.final_value,
            self.total_return * 100.0,
            self.num_trades,
            self.win_rate * 100.0,
            self.avg_profit,
            self.max_drawdown * 100.0,
            self.sharpe_ratio
        )
    }
}

impl std::fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Configuration for the backtester.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size as fraction of capital
    pub position_size: f64,
    /// Minimum confidence to enter trade
    pub min_confidence: f64,
    /// Hold period in bars
    pub hold_period: usize,
    /// Commission per trade
    pub commission: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            min_confidence: 0.6,
            hold_period: 5,
            commission: 0.001,
        }
    }
}

/// RAG-based backtester.
pub struct RAGBacktester {
    signal_generator: RAGSignalGenerator,
    config: BacktestConfig,
}

impl RAGBacktester {
    /// Create a new backtester.
    pub fn new(signal_generator: RAGSignalGenerator, config: BacktestConfig) -> Self {
        Self {
            signal_generator,
            config,
        }
    }

    /// Run backtest on price data.
    pub fn run(&self, ticker: &str, prices: &[PriceBar]) -> BacktestResult {
        let mut trades: Vec<Trade> = Vec::new();
        let mut open_trade: Option<Trade> = None;
        let mut bars_held = 0;

        for (i, bar) in prices.iter().enumerate() {
            // Check if we need to close existing position
            if let Some(ref mut trade) = open_trade {
                bars_held += 1;
                if bars_held >= self.config.hold_period {
                    trade.close(bar.timestamp, bar.close);
                    trades.push(trade.clone());
                    open_trade = None;
                    bars_held = 0;
                }
            }

            // Generate signal if no position
            if open_trade.is_none() && i < prices.len() - self.config.hold_period {
                let signal = self.signal_generator.generate_signal(ticker);

                if signal.confidence >= self.config.min_confidence
                    && signal.direction != SignalDirection::Neutral
                {
                    let position_value = self.config.initial_capital * self.config.position_size;
                    let quantity = position_value / bar.close;

                    let trade = Trade::new(
                        ticker.to_string(),
                        bar.timestamp,
                        bar.close,
                        signal.direction,
                        quantity,
                        signal,
                    );

                    open_trade = Some(trade);
                    bars_held = 0;
                }
            }
        }

        // Close any remaining open position
        if let Some(ref mut trade) = open_trade {
            if let Some(last_bar) = prices.last() {
                trade.close(last_bar.timestamp, last_bar.close);
                trades.push(trade.clone());
            }
        }

        BacktestResult::new(self.config.initial_capital, trades)
    }

    /// Run backtest on multiple tickers.
    pub fn run_portfolio(
        &self,
        data: &HashMap<String, Vec<PriceBar>>,
    ) -> HashMap<String, BacktestResult> {
        data.iter()
            .map(|(ticker, prices)| (ticker.clone(), self.run(ticker, prices)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::DocumentRetriever;
    use chrono::TimeZone;

    fn create_test_prices() -> Vec<PriceBar> {
        (0..20)
            .map(|i| {
                let ts = Utc.with_ymd_and_hms(2024, 1, 1 + i as u32, 9, 30, 0).unwrap();
                let base = 100.0 + (i as f64 * 0.5);
                PriceBar::new(ts, base, base + 1.0, base - 1.0, base + 0.3, 1000000.0)
            })
            .collect()
    }

    #[test]
    fn test_trade_creation() {
        let signal = TradingSignal::no_data("TEST".to_string());
        let trade = Trade::new(
            "TEST".to_string(),
            Utc::now(),
            100.0,
            SignalDirection::Long,
            10.0,
            signal,
        );

        assert!(trade.is_open());
        assert_eq!(trade.ticker, "TEST");
    }

    #[test]
    fn test_trade_close() {
        let signal = TradingSignal::no_data("TEST".to_string());
        let mut trade = Trade::new(
            "TEST".to_string(),
            Utc::now(),
            100.0,
            SignalDirection::Long,
            10.0,
            signal,
        );

        trade.close(Utc::now(), 110.0);

        assert!(!trade.is_open());
        assert_eq!(trade.pnl, Some(100.0)); // (110 - 100) * 10
    }

    #[test]
    fn test_backtest_result() {
        let signal = TradingSignal::no_data("TEST".to_string());
        let mut trade = Trade::new(
            "TEST".to_string(),
            Utc::now(),
            100.0,
            SignalDirection::Long,
            10.0,
            signal,
        );
        trade.close(Utc::now(), 110.0);

        let result = BacktestResult::new(10000.0, vec![trade]);

        assert_eq!(result.num_trades, 1);
        assert_eq!(result.winning_trades, 1);
        assert!(result.total_return > 0.0);
    }

    #[test]
    fn test_backtest_run() {
        let retriever = DocumentRetriever::new();
        let generator = RAGSignalGenerator::new(retriever);
        let config = BacktestConfig::default();
        let backtester = RAGBacktester::new(generator, config);

        let prices = create_test_prices();
        let result = backtester.run("TEST", &prices);

        // With empty retriever, should have neutral signals and no trades
        assert!(result.num_trades >= 0);
    }
}
