//! Backtesting Engine with Audit Trails
//!
//! Backtest CoT trading strategies with full reasoning documentation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::signals::{SignalGenerator, Signal};
use crate::position::PositionSizer;
use crate::error::Result;

/// Trade direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    Long,
    Short,
}

/// A single trade with its reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade entry time
    pub entry_time: DateTime<Utc>,
    /// Trade exit time
    pub exit_time: DateTime<Utc>,
    /// Trade direction
    pub direction: TradeDirection,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size in units
    pub size: f64,
    /// Profit/loss amount
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Signal confidence at entry
    pub confidence: f64,
    /// Reasoning chain for the trade
    pub reasoning_chain: Vec<String>,
}

/// Backtest configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission per trade (percentage)
    pub commission_pct: f64,
    /// Slippage (percentage)
    pub slippage_pct: f64,
    /// Maximum position as percentage of portfolio
    pub max_position_pct: f64,
    /// Minimum bars between trades
    pub min_bars_between_trades: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            commission_pct: 0.001,
            slippage_pct: 0.0005,
            max_position_pct: 0.2,
            min_bars_between_trades: 5,
        }
    }
}

/// Backtest results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Final capital
    pub final_capital: f64,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// All trades with reasoning
    pub trades: Vec<Trade>,
}

/// Backtester with CoT signal generation.
pub struct Backtester {
    signal_generator: SignalGenerator,
    position_sizer: PositionSizer,
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester.
    pub fn new(
        signal_generator: SignalGenerator,
        position_sizer: PositionSizer,
        config: BacktestConfig,
    ) -> Self {
        Self {
            signal_generator,
            position_sizer,
            config,
        }
    }

    /// Create a backtester with default mock components.
    pub fn new_mock() -> Self {
        Self::new(
            SignalGenerator::new_mock(),
            PositionSizer::default(),
            BacktestConfig::default(),
        )
    }

    /// Run backtest on price data.
    pub async fn run(
        &self,
        prices: &[f64],
        timestamps: &[DateTime<Utc>],
        symbol: &str,
    ) -> Result<BacktestResult> {
        let n = prices.len();
        if n < 50 {
            return Err(crate::error::Error::InvalidInput(
                "Need at least 50 price points for backtest".to_string()
            ));
        }

        let mut capital = self.config.initial_capital;
        let mut equity_curve = Vec::with_capacity(n);
        let mut trades = Vec::new();
        let mut position: Option<(TradeDirection, f64, f64, Vec<String>, f64)> = None;
        let mut bars_since_trade = 0;

        // Calculate indicators for all bars
        let indicators = self.calculate_indicators(prices);

        for i in 50..n {
            let price = prices[i];
            let timestamp = timestamps[i];

            // Update equity curve
            let current_equity = if let Some((dir, entry_price, size, _, _)) = &position {
                let pnl = match dir {
                    TradeDirection::Long => (price - entry_price) * size,
                    TradeDirection::Short => (entry_price - price) * size,
                };
                capital + pnl
            } else {
                capital
            };
            equity_curve.push(current_equity);

            // Check for exit conditions if in position
            if let Some((dir, entry_price, size, reasoning, stop_loss)) = position.clone() {
                let should_exit = match dir {
                    TradeDirection::Long => price <= stop_loss || i == n - 1,
                    TradeDirection::Short => price >= stop_loss || i == n - 1,
                };

                if should_exit {
                    let exit_price = self.apply_slippage(price, dir == TradeDirection::Short);
                    let pnl = match dir {
                        TradeDirection::Long => (exit_price - entry_price) * size,
                        TradeDirection::Short => (entry_price - exit_price) * size,
                    };
                    let commission = (entry_price + exit_price) * size * self.config.commission_pct;
                    let net_pnl = pnl - commission;

                    let trade = Trade {
                        entry_time: timestamps[i.saturating_sub(bars_since_trade)],
                        exit_time: timestamp,
                        direction: dir,
                        entry_price,
                        exit_price,
                        size,
                        pnl: net_pnl,
                        return_pct: net_pnl / (entry_price * size),
                        confidence: 0.75,  // Would be from signal
                        reasoning_chain: reasoning,
                    };

                    capital += net_pnl;
                    trades.push(trade);
                    position = None;
                    bars_since_trade = 0;
                }
            }

            // Generate new signal if not in position and enough bars passed
            if position.is_none() && bars_since_trade >= self.config.min_bars_between_trades {
                let ind = &indicators[i - 50];

                let signal = self.signal_generator.generate(
                    symbol,
                    price,
                    ind.rsi,
                    ind.macd,
                    ind.macd_signal,
                    ind.sma_20,
                    ind.sma_50,
                    ind.volume_ratio,
                    ind.atr,
                ).await?;

                if signal.signal_type != Signal::Hold {
                    let direction = match signal.signal_type {
                        Signal::StrongBuy | Signal::Buy => TradeDirection::Long,
                        Signal::StrongSell | Signal::Sell => TradeDirection::Short,
                        Signal::Hold => unreachable!(),
                    };

                    let pos_result = self.position_sizer.calculate(
                        signal.signal_type,
                        signal.confidence,
                        price,
                        signal.stop_loss,
                        capital,
                        Some(ind.atr / price),
                    );

                    if pos_result.units > 0.0 {
                        let entry_price = self.apply_slippage(price, direction == TradeDirection::Long);
                        position = Some((
                            direction,
                            entry_price,
                            pos_result.units,
                            signal.reasoning_chain,
                            signal.stop_loss,
                        ));
                    }
                }
            }

            bars_since_trade += 1;
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&equity_curve, &trades);

        Ok(BacktestResult {
            total_return: metrics.total_return,
            annual_return: metrics.annual_return,
            sharpe_ratio: metrics.sharpe_ratio,
            max_drawdown: metrics.max_drawdown,
            win_rate: metrics.win_rate,
            profit_factor: metrics.profit_factor,
            total_trades: trades.len(),
            winning_trades: trades.iter().filter(|t| t.pnl > 0.0).count(),
            losing_trades: trades.iter().filter(|t| t.pnl <= 0.0).count(),
            final_capital: capital,
            equity_curve,
            trades,
        })
    }

    fn apply_slippage(&self, price: f64, is_buy: bool) -> f64 {
        if is_buy {
            price * (1.0 + self.config.slippage_pct)
        } else {
            price * (1.0 - self.config.slippage_pct)
        }
    }

    fn calculate_indicators(&self, prices: &[f64]) -> Vec<Indicators> {
        let n = prices.len();
        let mut indicators = Vec::with_capacity(n.saturating_sub(50));

        for i in 50..n {
            let window = &prices[i.saturating_sub(50)..=i];

            // SMA 20
            let sma_20: f64 = prices[i.saturating_sub(20)..=i].iter().sum::<f64>() / 20.0;

            // SMA 50
            let sma_50: f64 = window.iter().sum::<f64>() / window.len() as f64;

            // RSI
            let rsi = self.calculate_rsi(&prices[i.saturating_sub(14)..=i]);

            // MACD
            let ema_12 = self.calculate_ema(&prices[i.saturating_sub(12)..=i], 12);
            let ema_26 = self.calculate_ema(&prices[i.saturating_sub(26)..=i], 26);
            let macd = ema_12 - ema_26;
            let macd_signal = macd * 0.9; // Simplified

            // ATR
            let atr = self.calculate_atr(&prices[i.saturating_sub(14)..=i]);

            // Volume ratio (mock - would need actual volume data)
            let volume_ratio = 1.0;

            indicators.push(Indicators {
                sma_20,
                sma_50,
                rsi,
                macd,
                macd_signal,
                atr,
                volume_ratio,
            });
        }

        indicators
    }

    fn calculate_rsi(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 50.0;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }

        let periods = (prices.len() - 1) as f64;
        let avg_gain = gains / periods;
        let avg_loss = losses / periods;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];

        for price in prices.iter().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }

        ema
    }

    fn calculate_atr(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let mut tr_sum = 0.0;
        for i in 1..prices.len() {
            let tr = (prices[i] - prices[i - 1]).abs();
            tr_sum += tr;
        }

        tr_sum / (prices.len() - 1) as f64
    }

    fn calculate_metrics(&self, equity: &[f64], trades: &[Trade]) -> Metrics {
        if equity.is_empty() {
            return Metrics::default();
        }

        let initial = self.config.initial_capital;
        let final_equity = *equity.last().unwrap_or(&initial);
        let total_return = (final_equity - initial) / initial;

        // Annualized return (assuming 252 trading days)
        let days = equity.len() as f64;
        let years = days / 252.0;
        let annual_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Calculate daily returns
        let returns: Vec<f64> = equity.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Sharpe ratio (annualized, assuming 0% risk-free rate)
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        let sharpe_ratio = if std_dev > 0.0 {
            (mean_return * 252.0_f64.sqrt()) / (std_dev * 252.0_f64.sqrt())
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = initial;
        let mut max_dd = 0.0;
        for &eq in equity {
            if eq > peak {
                peak = eq;
            }
            let dd = (peak - eq) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        // Win rate and profit factor
        let winning = trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if trades.is_empty() {
            0.0
        } else {
            winning as f64 / trades.len() as f64
        };

        let gross_profit: f64 = trades.iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .sum();
        let gross_loss: f64 = trades.iter()
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

        Metrics {
            total_return,
            annual_return,
            sharpe_ratio,
            max_drawdown: max_dd,
            win_rate,
            profit_factor,
        }
    }
}

#[derive(Debug)]
struct Indicators {
    sma_20: f64,
    sma_50: f64,
    rsi: f64,
    macd: f64,
    macd_signal: f64,
    atr: f64,
    volume_ratio: f64,
}

#[derive(Debug, Default)]
struct Metrics {
    total_return: f64,
    annual_return: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    profit_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_mock_prices(n: usize, seed: u64) -> Vec<f64> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut prices = Vec::with_capacity(n);
        let mut price = 100.0;

        for _ in 0..n {
            let change = rng.gen_range(-0.02..0.02);
            price *= 1.0 + change;
            prices.push(price);
        }

        prices
    }

    fn generate_mock_timestamps(n: usize) -> Vec<DateTime<Utc>> {
        use chrono::Duration;
        let start = Utc::now() - Duration::days(n as i64);
        (0..n).map(|i| start + Duration::days(i as i64)).collect()
    }

    #[tokio::test]
    async fn test_backtest() {
        let backtester = Backtester::new_mock();
        let prices = generate_mock_prices(200, 42);
        let timestamps = generate_mock_timestamps(200);

        let result = backtester.run(&prices, &timestamps, "TEST").await.unwrap();

        assert!(!result.equity_curve.is_empty());
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
    }
}
