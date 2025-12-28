//! Backtesting engine

use super::metrics::PerformanceMetrics;
use crate::data::Candle;
use serde::{Deserialize, Serialize};

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Signal {
    /// Buy / Long signal
    Long,
    /// Sell / Short signal
    Short,
    /// No position / Exit
    Flat,
}

impl Signal {
    pub fn from_prediction(pred: f64, long_threshold: f64, short_threshold: f64) -> Self {
        if pred > long_threshold {
            Signal::Long
        } else if pred < short_threshold {
            Signal::Short
        } else {
            Signal::Flat
        }
    }
}

/// Position state
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Position {
    Long,
    Short,
    Flat,
}

impl Position {
    pub fn from_signal(signal: Signal) -> Self {
        match signal {
            Signal::Long => Position::Long,
            Signal::Short => Position::Short,
            Signal::Flat => Position::Flat,
        }
    }
}

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size (fraction of capital)
    pub position_size: f64,
    /// Transaction cost (as fraction)
    pub transaction_cost: f64,
    /// Slippage (as fraction)
    pub slippage: f64,
    /// Long threshold for predictions
    pub long_threshold: f64,
    /// Short threshold for predictions
    pub short_threshold: f64,
    /// Allow short selling
    pub allow_short: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            position_size: 1.0,
            transaction_cost: 0.001, // 0.1%
            slippage: 0.0005,        // 0.05%
            long_threshold: 0.0,
            short_threshold: 0.0,
            allow_short: true,
        }
    }
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_time: i64,
    pub exit_time: i64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub position: Position,
    pub pnl: f64,
    pub return_pct: f64,
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub equity_curve: Vec<f64>,
    pub returns: Vec<f64>,
    pub positions: Vec<Position>,
    pub trades: Vec<Trade>,
    pub metrics: PerformanceMetrics,
    pub final_capital: f64,
}

/// Backtesting engine
pub struct Backtest {
    config: BacktestConfig,
}

impl Backtest {
    /// Create new backtest with config
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(BacktestConfig::default())
    }

    /// Run backtest with predictions
    pub fn run(&self, candles: &[Candle], predictions: &[f64]) -> BacktestResult {
        assert_eq!(
            candles.len(),
            predictions.len(),
            "Candles and predictions must have same length"
        );

        let n = candles.len();
        if n < 2 {
            return BacktestResult {
                equity_curve: vec![self.config.initial_capital],
                returns: vec![],
                positions: vec![],
                trades: vec![],
                metrics: PerformanceMetrics::default(),
                final_capital: self.config.initial_capital,
            };
        }

        let mut capital = self.config.initial_capital;
        let mut equity_curve = vec![capital];
        let mut returns = Vec::new();
        let mut positions = Vec::new();
        let mut trades = Vec::new();

        let mut current_position = Position::Flat;
        let mut entry_price = 0.0;
        let mut entry_time = 0i64;

        for i in 0..(n - 1) {
            let signal = Signal::from_prediction(
                predictions[i],
                self.config.long_threshold,
                self.config.short_threshold,
            );

            let target_position = if self.config.allow_short {
                Position::from_signal(signal)
            } else {
                match signal {
                    Signal::Long => Position::Long,
                    _ => Position::Flat,
                }
            };

            let current_price = candles[i].close;
            let next_price = candles[i + 1].close;

            // Close existing position if changing
            if current_position != target_position && current_position != Position::Flat {
                let exit_price = current_price * (1.0 - self.config.slippage);
                let cost = exit_price * self.config.transaction_cost;

                let pnl = match current_position {
                    Position::Long => {
                        (exit_price - entry_price) / entry_price - self.config.transaction_cost
                    }
                    Position::Short => {
                        (entry_price - exit_price) / entry_price - self.config.transaction_cost
                    }
                    Position::Flat => 0.0,
                };

                trades.push(Trade {
                    entry_time,
                    exit_time: candles[i].timestamp,
                    entry_price,
                    exit_price,
                    position: current_position,
                    pnl: pnl * capital * self.config.position_size,
                    return_pct: pnl,
                });

                capital *= 1.0 + pnl * self.config.position_size;
                current_position = Position::Flat;
            }

            // Open new position
            if target_position != Position::Flat && current_position == Position::Flat {
                entry_price = current_price * (1.0 + self.config.slippage);
                entry_time = candles[i].timestamp;
                capital *= 1.0 - self.config.transaction_cost * self.config.position_size;
                current_position = target_position;
            }

            // Calculate period return
            let period_return = match current_position {
                Position::Long => (next_price - current_price) / current_price,
                Position::Short => (current_price - next_price) / current_price,
                Position::Flat => 0.0,
            };

            let adjusted_return = period_return * self.config.position_size;
            capital *= 1.0 + adjusted_return;

            equity_curve.push(capital);
            returns.push(adjusted_return);
            positions.push(current_position);
        }

        // Close any remaining position
        if current_position != Position::Flat {
            let exit_price = candles[n - 1].close * (1.0 - self.config.slippage);

            let pnl = match current_position {
                Position::Long => {
                    (exit_price - entry_price) / entry_price - self.config.transaction_cost
                }
                Position::Short => {
                    (entry_price - exit_price) / entry_price - self.config.transaction_cost
                }
                Position::Flat => 0.0,
            };

            trades.push(Trade {
                entry_time,
                exit_time: candles[n - 1].timestamp,
                entry_price,
                exit_price,
                position: current_position,
                pnl: pnl * capital * self.config.position_size,
                return_pct: pnl,
            });
        }

        // Calculate metrics (assuming daily data)
        let periods_per_year = 365.0; // For crypto (24/7 trading)
        let metrics = PerformanceMetrics::from_returns(&returns, periods_per_year);

        BacktestResult {
            equity_curve,
            returns,
            positions,
            trades,
            metrics,
            final_capital: capital,
        }
    }

    /// Run backtest with signals
    pub fn run_with_signals(&self, candles: &[Candle], signals: &[Signal]) -> BacktestResult {
        let predictions: Vec<f64> = signals
            .iter()
            .map(|s| match s {
                Signal::Long => 1.0,
                Signal::Short => -1.0,
                Signal::Flat => 0.0,
            })
            .collect();

        let config = BacktestConfig {
            long_threshold: 0.5,
            short_threshold: -0.5,
            ..self.config.clone()
        };

        let bt = Backtest::new(config);
        bt.run(candles, &predictions)
    }
}

impl BacktestResult {
    /// Print summary
    pub fn print_summary(&self) {
        println!("\nBacktest Results");
        println!("================");
        println!(
            "Final Capital: ${:.2}",
            self.final_capital
        );
        println!(
            "Total Return:  {:.2}%",
            self.metrics.total_return * 100.0
        );
        println!("Total Trades:  {}", self.trades.len());
        println!();
        self.metrics.print_summary();
    }

    /// Get trade statistics
    pub fn trade_stats(&self) -> TradeStats {
        if self.trades.is_empty() {
            return TradeStats::default();
        }

        let wins: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losses: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl < 0.0).collect();

        let avg_win = if !wins.is_empty() {
            wins.iter().map(|t| t.return_pct).sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().map(|t| t.return_pct.abs()).sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };

        let long_trades: Vec<&Trade> = self
            .trades
            .iter()
            .filter(|t| t.position == Position::Long)
            .collect();

        let short_trades: Vec<&Trade> = self
            .trades
            .iter()
            .filter(|t| t.position == Position::Short)
            .collect();

        TradeStats {
            total_trades: self.trades.len(),
            winning_trades: wins.len(),
            losing_trades: losses.len(),
            win_rate: wins.len() as f64 / self.trades.len() as f64,
            avg_win,
            avg_loss,
            risk_reward: if avg_loss > 0.0 {
                avg_win / avg_loss
            } else {
                0.0
            },
            long_trades: long_trades.len(),
            short_trades: short_trades.len(),
        }
    }
}

/// Trade statistics
#[derive(Debug, Clone, Default)]
pub struct TradeStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub risk_reward: f64,
    pub long_trades: usize,
    pub short_trades: usize,
}

impl TradeStats {
    pub fn print(&self) {
        println!("\nTrade Statistics");
        println!("================");
        println!("Total Trades:    {}", self.total_trades);
        println!("Winning Trades:  {}", self.winning_trades);
        println!("Losing Trades:   {}", self.losing_trades);
        println!("Win Rate:        {:.1}%", self.win_rate * 100.0);
        println!("Avg Win:         {:.2}%", self.avg_win * 100.0);
        println!("Avg Loss:        {:.2}%", self.avg_loss * 100.0);
        println!("Risk/Reward:     {:.2}", self.risk_reward);
        println!("Long Trades:     {}", self.long_trades);
        println!("Short Trades:    {}", self.short_trades);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest() {
        let candles: Vec<Candle> = (0..100)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                Candle::new(i * 3600000, price, price + 1.0, price - 1.0, price, 1000.0)
            })
            .collect();

        // Alternate signals
        let predictions: Vec<f64> = (0..100).map(|i| if i % 10 < 5 { 0.5 } else { -0.5 }).collect();

        let backtest = Backtest::default_config();
        let result = backtest.run(&candles, &predictions);

        assert!(!result.equity_curve.is_empty());
        assert!(!result.returns.is_empty());
    }
}
