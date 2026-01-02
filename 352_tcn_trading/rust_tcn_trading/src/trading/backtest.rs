//! Backtesting Engine

use chrono::{DateTime, Utc};

use super::risk::{PortfolioState, RiskManager};
use super::signal::{SignalType, TradingSignal};
use crate::api::Candle;

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission per trade (as fraction)
    pub commission: f64,
    /// Slippage (as fraction of price)
    pub slippage: f64,
    /// Allow short selling
    pub allow_short: bool,
    /// Use margin trading
    pub use_margin: bool,
    /// Margin requirement
    pub margin_requirement: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission: 0.001,     // 0.1% commission
            slippage: 0.0005,      // 0.05% slippage
            allow_short: true,
            use_margin: false,
            margin_requirement: 0.2,
        }
    }
}

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: Option<DateTime<Utc>>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: Option<f64>,
    /// Position size (positive for long, negative for short)
    pub position: f64,
    /// Realized P&L
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Trade type
    pub trade_type: SignalType,
    /// Symbol
    pub symbol: String,
}

impl Trade {
    /// Check if trade is still open
    pub fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }

    /// Get duration in seconds
    pub fn duration_secs(&self) -> Option<i64> {
        self.exit_time.map(|et| (et - self.entry_time).num_seconds())
    }

    /// Check if trade was profitable
    pub fn is_profitable(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Equity point for equity curve
#[derive(Debug, Clone)]
pub struct EquityPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Total equity
    pub equity: f64,
    /// Current position
    pub position: f64,
    /// Unrealized P&L
    pub unrealized_pnl: f64,
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Average trade duration in seconds
    pub avg_trade_duration: f64,
    /// Equity curve
    pub equity_curve: Vec<EquityPoint>,
    /// All trades
    pub trades: Vec<Trade>,
    /// Final equity
    pub final_equity: f64,
}

impl BacktestResult {
    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            r#"Backtest Results
================
Total Return:      {:.2}%
Annualized Return: {:.2}%
Sharpe Ratio:      {:.2}
Sortino Ratio:     {:.2}
Max Drawdown:      {:.2}%
Win Rate:          {:.2}%
Profit Factor:     {:.2}
Total Trades:      {}
Final Equity:      ${:.2}
"#,
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.total_trades,
            self.final_equity
        )
    }
}

/// Backtest engine
#[derive(Debug)]
pub struct BacktestEngine {
    /// Configuration
    pub config: BacktestConfig,
    /// Risk manager
    pub risk_manager: RiskManager,
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self {
            config: BacktestConfig::default(),
            risk_manager: RiskManager::default(),
        }
    }
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig, risk_manager: RiskManager) -> Self {
        Self {
            config,
            risk_manager,
        }
    }

    /// Run backtest
    pub fn run(
        &self,
        signals: &[TradingSignal],
        candles: &[Candle],
        symbol: &str,
    ) -> BacktestResult {
        assert_eq!(
            signals.len(),
            candles.len(),
            "Signals and candles must have same length"
        );

        let mut capital = self.config.initial_capital;
        let mut position = 0.0;
        let mut position_value = 0.0;
        let mut entry_price = 0.0;
        let mut trades = Vec::new();
        let mut equity_curve = Vec::new();
        let mut peak_equity = capital;
        let mut portfolio_state = PortfolioState::new(capital);

        for (i, (signal, candle)) in signals.iter().zip(candles.iter()).enumerate() {
            let current_price = candle.close;

            // Calculate current equity
            let unrealized_pnl = if position != 0.0 {
                position * (current_price - entry_price)
            } else {
                0.0
            };
            let equity = capital + unrealized_pnl;

            // Update peak and drawdown
            if equity > peak_equity {
                peak_equity = equity;
            }
            portfolio_state.update_drawdown(peak_equity);

            // Record equity point
            equity_curve.push(EquityPoint {
                timestamp: candle.timestamp,
                equity,
                position,
                unrealized_pnl,
            });

            // Process signal
            let validated = self.risk_manager.validate_signal(signal, &portfolio_state, Some(symbol));

            if let Some(validated_signal) = validated.get_signal() {
                // Check for position change
                let target_position = validated_signal.signal_type.to_position()
                    * validated_signal.position_size
                    * capital
                    / current_price;

                // Skip if no change needed
                if (target_position - position).abs() < 0.0001 {
                    continue;
                }

                // Close existing position if changing direction
                if position != 0.0
                    && (position > 0.0) != (target_position > 0.0)
                {
                    // Apply slippage
                    let exit_price = if position > 0.0 {
                        current_price * (1.0 - self.config.slippage)
                    } else {
                        current_price * (1.0 + self.config.slippage)
                    };

                    let pnl = position * (exit_price - entry_price);
                    let commission = position.abs() * exit_price * self.config.commission;
                    let net_pnl = pnl - commission;

                    capital += position_value + net_pnl;

                    // Record trade
                    if let Some(last_trade) = trades.last_mut() {
                        if last_trade.is_open() {
                            last_trade.exit_time = Some(candle.timestamp);
                            last_trade.exit_price = Some(exit_price);
                            last_trade.pnl = net_pnl;
                            last_trade.return_pct = net_pnl / position_value;
                        }
                    }

                    portfolio_state.update_pnl(net_pnl);
                    position = 0.0;
                    position_value = 0.0;
                }

                // Open new position
                if target_position.abs() > 0.0001 && validated_signal.signal_type != SignalType::Neutral {
                    // Check if we have enough capital
                    let required_capital = target_position.abs() * current_price;
                    if required_capital <= capital {
                        // Apply slippage
                        entry_price = if target_position > 0.0 {
                            current_price * (1.0 + self.config.slippage)
                        } else {
                            current_price * (1.0 - self.config.slippage)
                        };

                        position = target_position;
                        position_value = position.abs() * entry_price;
                        let commission = position_value * self.config.commission;
                        capital -= position_value + commission;

                        trades.push(Trade {
                            entry_time: candle.timestamp,
                            exit_time: None,
                            entry_price,
                            exit_price: None,
                            position,
                            pnl: 0.0,
                            return_pct: 0.0,
                            trade_type: validated_signal.signal_type,
                            symbol: symbol.to_string(),
                        });
                    }
                }
            }
        }

        // Close any remaining position at the end
        if position != 0.0 {
            if let Some(last_candle) = candles.last() {
                let exit_price = last_candle.close;
                let pnl = position * (exit_price - entry_price);
                let commission = position.abs() * exit_price * self.config.commission;
                let net_pnl = pnl - commission;
                capital += position_value + net_pnl;

                if let Some(last_trade) = trades.last_mut() {
                    if last_trade.is_open() {
                        last_trade.exit_time = Some(last_candle.timestamp);
                        last_trade.exit_price = Some(exit_price);
                        last_trade.pnl = net_pnl;
                        last_trade.return_pct = net_pnl / position_value;
                    }
                }
            }
        }

        // Calculate metrics
        self.calculate_metrics(capital, &trades, &equity_curve)
    }

    /// Calculate backtest metrics
    fn calculate_metrics(
        &self,
        final_capital: f64,
        trades: &[Trade],
        equity_curve: &[EquityPoint],
    ) -> BacktestResult {
        let total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital;

        // Calculate returns for Sharpe/Sortino
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
            .collect();

        let mean_return = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let std_return = if returns.len() > 1 {
            let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            1.0
        };

        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if downside_returns.len() > 1 {
            let mean_down = downside_returns.iter().sum::<f64>() / downside_returns.len() as f64;
            let variance = downside_returns
                .iter()
                .map(|r| (r - mean_down).powi(2))
                .sum::<f64>()
                / (downside_returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            1.0
        };

        // Annualize (assuming daily returns, 252 trading days)
        let annualization_factor = 252.0_f64.sqrt();
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * annualization_factor
        } else {
            0.0
        };

        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * annualization_factor
        } else {
            0.0
        };

        // Calculate max drawdown
        let mut peak = self.config.initial_capital;
        let mut max_drawdown = 0.0;
        for point in equity_curve {
            if point.equity > peak {
                peak = point.equity;
            }
            let drawdown = (peak - point.equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Trade statistics
        let completed_trades: Vec<&Trade> = trades.iter().filter(|t| !t.is_open()).collect();
        let winning_trades = completed_trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if !completed_trades.is_empty() {
            winning_trades as f64 / completed_trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = completed_trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f64 = completed_trades
            .iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_duration = if !completed_trades.is_empty() {
            completed_trades
                .iter()
                .filter_map(|t| t.duration_secs())
                .sum::<i64>() as f64
                / completed_trades.len() as f64
        } else {
            0.0
        };

        // Calculate annualized return
        let days = equity_curve.len() as f64;
        let annualized_return = if days > 0.0 {
            (1.0 + total_return).powf(365.0 / days) - 1.0
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades: trades.len(),
            avg_trade_duration,
            equity_curve: equity_curve.to_vec(),
            trades: trades.to_vec(),
            final_equity: final_capital,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize, start_price: f64) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let price = start_price * (1.0 + (i as f64 * 0.001));
                Candle::new(
                    Utc::now(),
                    price - 0.5,
                    price + 1.0,
                    price - 1.0,
                    price,
                    1000.0,
                    price * 1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert!(config.initial_capital > 0.0);
        assert!(config.commission >= 0.0);
    }

    #[test]
    fn test_backtest_no_trades() {
        let engine = BacktestEngine::default();
        let candles = create_test_candles(100, 100.0);
        let signals: Vec<TradingSignal> = (0..100).map(|_| TradingSignal::neutral()).collect();

        let result = engine.run(&signals, &candles, "BTCUSDT");

        assert_eq!(result.total_trades, 0);
        assert!((result.total_return).abs() < 0.001);
    }

    #[test]
    fn test_backtest_with_trades() {
        let engine = BacktestEngine::default();
        let candles = create_test_candles(100, 100.0);

        // Create alternating buy/sell signals
        let mut signals = Vec::new();
        for i in 0..100 {
            if i % 20 == 0 && i < 80 {
                signals.push(TradingSignal::new(SignalType::Long, 0.7, 0.1, 0.02));
            } else if i % 20 == 10 {
                signals.push(TradingSignal::new(SignalType::Short, 0.7, 0.1, 0.02));
            } else {
                signals.push(TradingSignal::neutral());
            }
        }

        let result = engine.run(&signals, &candles, "BTCUSDT");

        assert!(result.total_trades > 0);
        assert!(result.final_equity > 0.0);
    }

    #[test]
    fn test_trade_record() {
        let trade = Trade {
            entry_time: Utc::now(),
            exit_time: None,
            entry_price: 100.0,
            exit_price: None,
            position: 1.0,
            pnl: 0.0,
            return_pct: 0.0,
            trade_type: SignalType::Long,
            symbol: "BTCUSDT".to_string(),
        };

        assert!(trade.is_open());
        assert!(!trade.is_profitable());
    }

    #[test]
    fn test_result_summary() {
        let result = BacktestResult {
            total_return: 0.15,
            annualized_return: 0.45,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            max_drawdown: 0.08,
            win_rate: 0.55,
            profit_factor: 1.8,
            total_trades: 50,
            avg_trade_duration: 3600.0,
            equity_curve: vec![],
            trades: vec![],
            final_equity: 115_000.0,
        };

        let summary = result.summary();
        assert!(summary.contains("15.00%"));
        assert!(summary.contains("Sharpe Ratio"));
    }
}
