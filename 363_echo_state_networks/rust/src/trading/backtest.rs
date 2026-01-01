//! Backtesting engine

use crate::api::Kline;
use crate::trading::{Position, PositionSide, TradingSignal};
use serde::{Deserialize, Serialize};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission rate (e.g., 0.0004 = 0.04%)
    pub commission: f64,
    /// Slippage rate
    pub slippage: f64,
    /// Position size as fraction of capital
    pub position_size_pct: f64,
    /// Use leverage
    pub leverage: f64,
    /// Stop loss percentage
    pub stop_loss: Option<f64>,
    /// Take profit percentage
    pub take_profit: Option<f64>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission: 0.0004,
            slippage: 0.0001,
            position_size_pct: 0.1,
            leverage: 1.0,
            stop_loss: Some(0.02),
            take_profit: Some(0.04),
        }
    }
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Volatility (annualized)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Average trade return
    pub avg_trade: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Total commission paid
    pub total_commission: f64,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Trade history
    pub trades: Vec<TradeRecord>,
}

/// Single trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    /// Entry time
    pub entry_time: i64,
    /// Exit time
    pub exit_time: i64,
    /// Side (long/short)
    pub side: String,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Size
    pub size: f64,
    /// PnL
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Commission paid
    pub commission: f64,
}

/// Backtesting engine
pub struct Backtest {
    config: BacktestConfig,
}

impl Backtest {
    /// Create new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest with signals
    pub fn run(&self, klines: &[Kline], signals: &[TradingSignal]) -> BacktestResult {
        let n = klines.len().min(signals.len());
        if n == 0 {
            return self.empty_result();
        }

        let mut capital = self.config.initial_capital;
        let mut peak_capital = capital;
        let mut max_drawdown = 0.0;
        let mut equity_curve = vec![capital];
        let mut trades: Vec<TradeRecord> = Vec::new();
        let mut current_position: Option<Position> = None;
        let mut total_commission = 0.0;

        for i in 0..n {
            let kline = &klines[i];
            let signal = signals[i];
            let price = kline.close;

            // Check exit conditions for open position
            if let Some(ref mut pos) = current_position {
                let mut should_close = false;
                let mut exit_reason = "";

                // Check stop loss / take profit
                let return_pct = pos.calculate_return(price);
                if let Some(sl) = self.config.stop_loss {
                    if return_pct <= -sl {
                        should_close = true;
                        exit_reason = "stop_loss";
                    }
                }
                if let Some(tp) = self.config.take_profit {
                    if return_pct >= tp {
                        should_close = true;
                        exit_reason = "take_profit";
                    }
                }

                // Check signal reversal
                let should_reverse = match (pos.side, signal) {
                    (PositionSide::Long, TradingSignal::Sell | TradingSignal::StrongSell) => true,
                    (PositionSide::Short, TradingSignal::Buy | TradingSignal::StrongBuy) => true,
                    _ => false,
                };

                if should_reverse || should_close {
                    // Close position
                    let exit_price = self.apply_slippage(price, pos.side, false);
                    let pnl = pos.calculate_pnl(exit_price);
                    let commission = pos.size * exit_price * self.config.commission;
                    total_commission += commission;

                    let net_pnl = pnl - commission;
                    capital += net_pnl;

                    trades.push(TradeRecord {
                        entry_time: pos.entry_time,
                        exit_time: kline.start_time,
                        side: format!("{:?}", pos.side),
                        entry_price: pos.entry_price,
                        exit_price,
                        size: pos.size,
                        pnl: net_pnl,
                        return_pct: pos.calculate_return(exit_price),
                        commission,
                    });

                    current_position = None;
                }
            }

            // Open new position if no position
            if current_position.is_none() {
                let side = match signal {
                    TradingSignal::Buy | TradingSignal::StrongBuy => Some(PositionSide::Long),
                    TradingSignal::Sell | TradingSignal::StrongSell => Some(PositionSide::Short),
                    TradingSignal::Hold => None,
                };

                if let Some(side) = side {
                    let entry_price = self.apply_slippage(price, side, true);
                    let size = (capital * self.config.position_size_pct * self.config.leverage) / entry_price;
                    let commission = size * entry_price * self.config.commission;
                    total_commission += commission;
                    capital -= commission;

                    let mut pos = Position::new("");
                    pos.open(side, size, entry_price, kline.start_time);
                    current_position = Some(pos);
                }
            }

            // Update equity
            let current_equity = if let Some(ref pos) = current_position {
                capital + pos.calculate_pnl(price)
            } else {
                capital
            };

            equity_curve.push(current_equity);

            // Update drawdown
            peak_capital = peak_capital.max(current_equity);
            let drawdown = (peak_capital - current_equity) / peak_capital;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Close any remaining position
        if let Some(pos) = current_position {
            let exit_price = klines.last().unwrap().close;
            let pnl = pos.calculate_pnl(exit_price);
            capital += pnl;
        }

        // Calculate metrics
        self.calculate_metrics(capital, &equity_curve, &trades, max_drawdown, total_commission)
    }

    fn apply_slippage(&self, price: f64, side: PositionSide, is_entry: bool) -> f64 {
        let direction = match (side, is_entry) {
            (PositionSide::Long, true) | (PositionSide::Short, false) => 1.0,
            (PositionSide::Long, false) | (PositionSide::Short, true) => -1.0,
            _ => 0.0,
        };
        price * (1.0 + direction * self.config.slippage)
    }

    fn calculate_metrics(
        &self,
        final_capital: f64,
        equity_curve: &[f64],
        trades: &[TradeRecord],
        max_drawdown: f64,
        total_commission: f64,
    ) -> BacktestResult {
        let total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital;

        // Calculate returns
        let returns: Vec<f64> = equity_curve.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Volatility (annualized, assuming daily)
        let mean_return = if returns.is_empty() { 0.0 } else {
            returns.iter().sum::<f64>() / returns.len() as f64
        };
        let variance = if returns.len() > 1 {
            returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / (returns.len() - 1) as f64
        } else {
            0.0
        };
        let volatility = variance.sqrt() * (252_f64).sqrt();

        // Annual return (assuming 252 trading days)
        let n_periods = equity_curve.len() as f64;
        let annual_return = (1.0 + total_return).powf(252.0 / n_periods) - 1.0;

        // Sharpe ratio (assuming 0 risk-free rate)
        let sharpe_ratio = if volatility > 0.0 { annual_return / volatility } else { 0.0 };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_deviation = downside_variance.sqrt() * (252_f64).sqrt();
        let sortino_ratio = if downside_deviation > 0.0 { annual_return / downside_deviation } else { 0.0 };

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 { annual_return / max_drawdown } else { 0.0 };

        // Trade metrics
        let winning_trades: Vec<&TradeRecord> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&TradeRecord> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };

        let avg_trade = if !trades.is_empty() {
            trades.iter().map(|t| t.pnl).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annual_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            avg_trade,
            num_trades: trades.len(),
            total_commission,
            equity_curve: equity_curve.to_vec(),
            trades: trades.to_vec(),
        }
    }

    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            total_return: 0.0,
            annual_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_trade: 0.0,
            num_trades: 0,
            total_commission: 0.0,
            equity_curve: vec![self.config.initial_capital],
            trades: Vec::new(),
        }
    }
}

impl BacktestResult {
    /// Print summary
    pub fn print_summary(&self) {
        println!("=== Backtest Results ===");
        println!("Total Return:     {:.2}%", self.total_return * 100.0);
        println!("Annual Return:    {:.2}%", self.annual_return * 100.0);
        println!("Volatility:       {:.2}%", self.volatility * 100.0);
        println!("Sharpe Ratio:     {:.3}", self.sharpe_ratio);
        println!("Sortino Ratio:    {:.3}", self.sortino_ratio);
        println!("Calmar Ratio:     {:.3}", self.calmar_ratio);
        println!("Max Drawdown:     {:.2}%", self.max_drawdown * 100.0);
        println!("Win Rate:         {:.2}%", self.win_rate * 100.0);
        println!("Profit Factor:    {:.2}", self.profit_factor);
        println!("Avg Trade:        ${:.2}", self.avg_trade);
        println!("Num Trades:       {}", self.num_trades);
        println!("Total Commission: ${:.2}", self.total_commission);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines() -> Vec<Kline> {
        (0..100).map(|i| Kline {
            start_time: i * 60000,
            open: 100.0 + (i as f64 * 0.1),
            high: 102.0 + (i as f64 * 0.1),
            low: 98.0 + (i as f64 * 0.1),
            close: 101.0 + (i as f64 * 0.1),
            volume: 1000.0,
            turnover: 100000.0,
        }).collect()
    }

    #[test]
    fn test_backtest_basic() {
        let config = BacktestConfig::default();
        let backtest = Backtest::new(config);

        let klines = create_test_klines();
        let signals: Vec<TradingSignal> = (0..100).map(|i| {
            if i < 10 { TradingSignal::Buy }
            else if i < 50 { TradingSignal::Hold }
            else if i < 60 { TradingSignal::Sell }
            else { TradingSignal::Hold }
        }).collect();

        let result = backtest.run(&klines, &signals);

        assert!(result.num_trades > 0);
        assert!(result.equity_curve.len() > 1);
    }

    #[test]
    fn test_empty_backtest() {
        let config = BacktestConfig::default();
        let backtest = Backtest::new(config);

        let result = backtest.run(&[], &[]);
        assert_eq!(result.num_trades, 0);
    }
}
