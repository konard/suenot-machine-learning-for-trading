//! Backtesting framework for risk-based trading strategies.

use serde::{Deserialize, Serialize};

use crate::data::OHLCV;
use crate::risk::RiskScore;
use super::signal::{TradingSignal, SignalType};
use super::trader::RiskBasedTrader;

/// Configuration for backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital in USD.
    pub initial_capital: f64,
    /// Trading fee as a fraction (e.g., 0.001 for 0.1%).
    pub trading_fee: f64,
    /// Slippage as a fraction (e.g., 0.0005 for 0.05%).
    pub slippage: f64,
    /// Whether to use compound returns.
    pub compound: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            trading_fee: 0.001,  // 0.1% fee
            slippage: 0.0005,   // 0.05% slippage
            compound: true,
        }
    }
}

/// A single trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp.
    pub entry_time: i64,
    /// Exit timestamp.
    pub exit_time: i64,
    /// Entry price.
    pub entry_price: f64,
    /// Exit price.
    pub exit_price: f64,
    /// Position size.
    pub size: f64,
    /// Direction (1 for long, -1 for short).
    pub direction: i8,
    /// Gross PnL before fees.
    pub gross_pnl: f64,
    /// Net PnL after fees.
    pub net_pnl: f64,
    /// Return percentage.
    pub return_pct: f64,
    /// Risk score at entry.
    pub entry_risk_score: f64,
}

/// Results from a backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// All trades executed.
    pub trades: Vec<Trade>,
    /// Final portfolio value.
    pub final_value: f64,
    /// Total return percentage.
    pub total_return_pct: f64,
    /// Annualized return percentage.
    pub annualized_return_pct: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Sortino ratio.
    pub sortino_ratio: f64,
    /// Maximum drawdown percentage.
    pub max_drawdown_pct: f64,
    /// Win rate (percentage of profitable trades).
    pub win_rate: f64,
    /// Average win percentage.
    pub avg_win_pct: f64,
    /// Average loss percentage.
    pub avg_loss_pct: f64,
    /// Profit factor (gross profits / gross losses).
    pub profit_factor: f64,
    /// Total number of trades.
    pub total_trades: usize,
    /// Number of winning trades.
    pub winning_trades: usize,
    /// Number of losing trades.
    pub losing_trades: usize,
    /// Average holding period in hours.
    pub avg_holding_hours: f64,
    /// Initial capital.
    pub initial_capital: f64,
}

impl BacktestResult {
    /// Create an empty result.
    fn empty(initial_capital: f64) -> Self {
        Self {
            trades: Vec::new(),
            final_value: initial_capital,
            total_return_pct: 0.0,
            annualized_return_pct: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown_pct: 0.0,
            win_rate: 0.0,
            avg_win_pct: 0.0,
            avg_loss_pct: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            avg_holding_hours: 0.0,
            initial_capital,
        }
    }

    /// Print a summary of the backtest results.
    pub fn print_summary(&self) {
        println!("\n=== Backtest Results ===");
        println!("Initial Capital:    ${:.2}", self.initial_capital);
        println!("Final Value:        ${:.2}", self.final_value);
        println!("Total Return:       {:.2}%", self.total_return_pct);
        println!("Annualized Return:  {:.2}%", self.annualized_return_pct);
        println!();
        println!("Sharpe Ratio:       {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio:      {:.2}", self.sortino_ratio);
        println!("Max Drawdown:       {:.2}%", self.max_drawdown_pct);
        println!();
        println!("Total Trades:       {}", self.total_trades);
        println!("Win Rate:           {:.1}%", self.win_rate * 100.0);
        println!("Profit Factor:      {:.2}", self.profit_factor);
        println!();
        println!("Avg Win:            {:.2}%", self.avg_win_pct);
        println!("Avg Loss:           {:.2}%", self.avg_loss_pct);
        println!("Avg Holding Period: {:.1} hours", self.avg_holding_hours);
        println!("========================\n");
    }
}

/// Backtester for running strategy simulations.
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester with default config.
    pub fn new() -> Self {
        Self {
            config: BacktestConfig::default(),
        }
    }

    /// Create a new backtester with custom config.
    pub fn with_config(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run a backtest with pre-generated risk scores.
    ///
    /// # Arguments
    /// * `price_data` - OHLCV price data
    /// * `risk_scores` - Risk scores aligned with price data
    /// * `trader` - The trading strategy to test
    pub fn run(
        &self,
        price_data: &[OHLCV],
        risk_scores: &[RiskScore],
        trader: &mut RiskBasedTrader,
    ) -> BacktestResult {
        if price_data.is_empty() || risk_scores.is_empty() {
            return BacktestResult::empty(self.config.initial_capital);
        }

        let mut capital = self.config.initial_capital;
        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<f64> = Vec::new();
        let mut current_trade: Option<(TradingSignal, usize)> = None;

        // Iterate through the data
        for (i, (candle, risk)) in price_data.iter().zip(risk_scores.iter()).enumerate() {
            let price = candle.close;

            // Generate signal
            let signal = trader.generate_signal(risk.clone(), price, "BACKTEST");

            // Handle signal
            match signal.signal_type {
                SignalType::Long | SignalType::Short => {
                    // Open a new position if not already in one
                    if current_trade.is_none() {
                        current_trade = Some((signal, i));
                    }
                }
                SignalType::Close => {
                    // Close current position
                    if let Some((entry_signal, entry_idx)) = current_trade.take() {
                        let entry_price = price_data[entry_idx].close;
                        let trade = self.create_trade(
                            &entry_signal,
                            entry_idx,
                            i,
                            entry_price,
                            price,
                            capital,
                        );

                        if self.config.compound {
                            capital += trade.net_pnl;
                        }

                        trades.push(trade);
                    }
                }
                SignalType::Neutral => {
                    // Do nothing
                }
            }

            // Track equity
            if let Some((ref entry_signal, entry_idx)) = current_trade {
                let entry_price = price_data[entry_idx].close;
                let direction = if entry_signal.signal_type == SignalType::Long { 1.0 } else { -1.0 };
                let position_value = entry_signal.position_size * capital;
                let unrealized_pnl = direction * position_value * (price - entry_price) / entry_price;
                equity_curve.push(capital + unrealized_pnl);
            } else {
                equity_curve.push(capital);
            }
        }

        // Close any remaining position at the end
        if let Some((entry_signal, entry_idx)) = current_trade.take() {
            let entry_price = price_data[entry_idx].close;
            let exit_price = price_data.last().unwrap().close;
            let trade = self.create_trade(
                &entry_signal,
                entry_idx,
                price_data.len() - 1,
                entry_price,
                exit_price,
                capital,
            );

            if self.config.compound {
                capital += trade.net_pnl;
            }

            trades.push(trade);
        }

        // Calculate final metrics
        self.calculate_results(trades, equity_curve, capital)
    }

    /// Create a trade record.
    fn create_trade(
        &self,
        entry_signal: &TradingSignal,
        entry_idx: usize,
        exit_idx: usize,
        entry_price: f64,
        exit_price: f64,
        capital: f64,
    ) -> Trade {
        let direction: i8 = if entry_signal.signal_type == SignalType::Long { 1 } else { -1 };
        let position_value = entry_signal.position_size * capital;

        // Apply slippage
        let adj_entry_price = entry_price * (1.0 + self.config.slippage * direction as f64);
        let adj_exit_price = exit_price * (1.0 - self.config.slippage * direction as f64);

        // Calculate PnL
        let price_change = (adj_exit_price - adj_entry_price) / adj_entry_price;
        let gross_pnl = position_value * price_change * direction as f64;

        // Apply fees (both entry and exit)
        let fees = position_value * self.config.trading_fee * 2.0;
        let net_pnl = gross_pnl - fees;

        let return_pct = (net_pnl / position_value) * 100.0;

        Trade {
            entry_time: entry_idx as i64,  // Using index as time placeholder
            exit_time: exit_idx as i64,
            entry_price: adj_entry_price,
            exit_price: adj_exit_price,
            size: entry_signal.position_size,
            direction,
            gross_pnl,
            net_pnl,
            return_pct,
            entry_risk_score: entry_signal.risk_score,
        }
    }

    /// Calculate final backtest results.
    fn calculate_results(
        &self,
        trades: Vec<Trade>,
        equity_curve: Vec<f64>,
        final_capital: f64,
    ) -> BacktestResult {
        let initial = self.config.initial_capital;

        if trades.is_empty() {
            return BacktestResult::empty(initial);
        }

        let total_trades = trades.len();
        let winning_trades = trades.iter().filter(|t| t.net_pnl > 0.0).count();
        let losing_trades = trades.iter().filter(|t| t.net_pnl <= 0.0).count();

        let win_rate = winning_trades as f64 / total_trades as f64;

        let wins: Vec<f64> = trades.iter().filter(|t| t.net_pnl > 0.0).map(|t| t.return_pct).collect();
        let losses: Vec<f64> = trades.iter().filter(|t| t.net_pnl <= 0.0).map(|t| t.return_pct).collect();

        let avg_win_pct = if wins.is_empty() { 0.0 } else { wins.iter().sum::<f64>() / wins.len() as f64 };
        let avg_loss_pct = if losses.is_empty() { 0.0 } else { losses.iter().sum::<f64>() / losses.len() as f64 };

        let gross_profits: f64 = trades.iter().filter(|t| t.net_pnl > 0.0).map(|t| t.net_pnl).sum();
        let gross_losses: f64 = trades.iter().filter(|t| t.net_pnl < 0.0).map(|t| t.net_pnl.abs()).sum();

        let profit_factor = if gross_losses > 0.0 { gross_profits / gross_losses } else { f64::INFINITY };

        let total_return_pct = (final_capital - initial) / initial * 100.0;

        // Calculate max drawdown
        let max_drawdown_pct = self.calculate_max_drawdown(&equity_curve);

        // Calculate Sharpe ratio (simplified)
        let returns: Vec<f64> = trades.iter().map(|t| t.return_pct / 100.0).collect();
        let sharpe_ratio = self.calculate_sharpe(&returns);
        let sortino_ratio = self.calculate_sortino(&returns);

        // Average holding period (using indices as hours)
        let total_holding: f64 = trades.iter().map(|t| (t.exit_time - t.entry_time) as f64).sum();
        let avg_holding_hours = if total_trades > 0 { total_holding / total_trades as f64 } else { 0.0 };

        // Annualized return (assuming hourly data)
        let hours_in_year = 8760.0;
        let total_hours = equity_curve.len() as f64;
        let annualized_return_pct = if total_hours > 0.0 {
            ((final_capital / initial).powf(hours_in_year / total_hours) - 1.0) * 100.0
        } else {
            0.0
        };

        BacktestResult {
            trades,
            final_value: final_capital,
            total_return_pct,
            annualized_return_pct,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown_pct,
            win_rate,
            avg_win_pct,
            avg_loss_pct,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
            avg_holding_hours,
            initial_capital: initial,
        }
    }

    /// Calculate maximum drawdown percentage.
    fn calculate_max_drawdown(&self, equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_value = equity_curve[0];
        let mut max_drawdown = 0.0;

        for &value in equity_curve {
            if value > max_value {
                max_value = value;
            }
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown * 100.0
    }

    /// Calculate Sharpe ratio.
    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            // Annualize (assuming daily returns)
            let annualized_mean = mean * 252.0;
            let annualized_std = std_dev * (252.0_f64).sqrt();
            annualized_mean / annualized_std
        } else {
            0.0
        }
    }

    /// Calculate Sortino ratio (only considers downside volatility).
    fn calculate_sortino(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;

        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / n;
        let downside_std = downside_variance.sqrt();

        if downside_std > 0.0 {
            let annualized_mean = mean * 252.0;
            let annualized_downside_std = downside_std * (252.0_f64).sqrt();
            annualized_mean / annualized_downside_std
        } else {
            0.0
        }
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> (Vec<OHLCV>, Vec<RiskScore>) {
        let prices: Vec<OHLCV> = (0..100)
            .map(|i| OHLCV {
                timestamp: i,
                open: 100.0 + i as f64 * 0.5,
                high: 101.0 + i as f64 * 0.5,
                low: 99.0 + i as f64 * 0.5,
                close: 100.0 + i as f64 * 0.5,
                volume: 1000.0,
            })
            .collect();

        let risks: Vec<RiskScore> = (0..100)
            .map(|i| {
                let base_risk = 3.0 + (i as f64 * 0.05).sin() * 2.0;
                RiskScore::new(base_risk, base_risk, base_risk, base_risk, base_risk, base_risk)
            })
            .collect();

        (prices, risks)
    }

    #[test]
    fn test_backtest_empty_data() {
        let backtester = Backtester::new();
        let mut trader = RiskBasedTrader::new();

        let result = backtester.run(&[], &[], &mut trader);
        assert_eq!(result.total_trades, 0);
    }

    #[test]
    fn test_backtest_with_data() {
        let backtester = Backtester::new();
        let mut trader = RiskBasedTrader::new();

        let (prices, risks) = create_sample_data();
        let result = backtester.run(&prices, &risks, &mut trader);

        assert!(result.final_value > 0.0);
    }

    #[test]
    fn test_max_drawdown_calculation() {
        let backtester = Backtester::new();

        let equity_curve = vec![100.0, 110.0, 105.0, 90.0, 95.0, 100.0];
        let max_dd = backtester.calculate_max_drawdown(&equity_curve);

        // Max drawdown should be (110 - 90) / 110 = 18.18%
        assert!((max_dd - 18.18).abs() < 0.1);
    }
}
