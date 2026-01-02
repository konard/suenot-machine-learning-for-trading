//! Backtesting engine for graph-based trading strategies.

use super::portfolio::Portfolio;
use crate::data::OHLCV;
use std::collections::HashMap;

/// A single trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: usize,
    pub pnl: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Backtest result summary
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: usize,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<Trade>,
}

impl BacktestResult {
    pub fn new() -> Self {
        Self {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            num_trades: 0,
            equity_curve: Vec::new(),
            trades: Vec::new(),
        }
    }
}

impl Default for BacktestResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    /// Initial capital
    initial_capital: f64,
    /// Commission rate (e.g., 0.001 = 0.1%)
    commission: f64,
    /// Slippage rate
    slippage: f64,
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(10000.0)
    }
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            commission: 0.001,
            slippage: 0.0005,
        }
    }

    /// Set commission rate
    pub fn with_commission(mut self, commission: f64) -> Self {
        self.commission = commission;
        self
    }

    /// Set slippage
    pub fn with_slippage(mut self, slippage: f64) -> Self {
        self.slippage = slippage;
        self
    }

    /// Run backtest with given signals
    pub fn run(
        &self,
        data: &HashMap<String, Vec<OHLCV>>,
        signals: &[HashMap<String, f64>],
    ) -> BacktestResult {
        let mut portfolio = Portfolio::new(self.initial_capital);
        let mut result = BacktestResult::new();
        let mut max_equity = self.initial_capital;

        // Get minimum length
        let min_len = data
            .values()
            .map(|v| v.len())
            .min()
            .unwrap_or(0)
            .min(signals.len());

        for t in 0..min_len {
            // Get current prices
            let prices: HashMap<String, f64> = data
                .iter()
                .filter_map(|(symbol, candles)| {
                    candles.get(t).map(|c| (symbol.clone(), c.close))
                })
                .collect();

            // Get current signals
            let current_signals = &signals[t];

            // Execute trades based on signals
            for (symbol, &signal) in current_signals {
                if let Some(&price) = prices.get(symbol) {
                    let current_pos = portfolio.positions.get(symbol).copied().unwrap_or(0.0);

                    // Calculate target position
                    let equity = portfolio.total_value(&prices);
                    let target_value = equity * signal * 0.1; // Use 10% of equity per unit signal
                    let target_pos = target_value / price;

                    // Trade if significant difference
                    let diff = target_pos - current_pos;
                    if diff.abs() * price > equity * 0.01 {
                        // Min 1% trade
                        let exec_price = if diff > 0.0 {
                            price * (1.0 + self.slippage)
                        } else {
                            price * (1.0 - self.slippage)
                        };

                        // Apply commission
                        let trade_value = diff.abs() * exec_price;
                        let commission_cost = trade_value * self.commission;

                        portfolio.cash -= commission_cost;
                        portfolio.update_position(symbol, target_pos, exec_price);

                        result.trades.push(Trade {
                            symbol: symbol.clone(),
                            side: if diff > 0.0 {
                                TradeSide::Buy
                            } else {
                                TradeSide::Sell
                            },
                            quantity: diff.abs(),
                            price: exec_price,
                            timestamp: t,
                            pnl: None,
                        });
                    }
                }
            }

            // Record equity
            let equity = portfolio.total_value(&prices);
            result.equity_curve.push(equity);

            // Update max drawdown
            if equity > max_equity {
                max_equity = equity;
            }
            let drawdown = (max_equity - equity) / max_equity;
            if drawdown > result.max_drawdown {
                result.max_drawdown = drawdown;
            }
        }

        // Calculate final metrics
        if let (Some(&first), Some(&last)) = (
            result.equity_curve.first(),
            result.equity_curve.last(),
        ) {
            result.total_return = (last - first) / first;
        }

        result.sharpe_ratio = self.calculate_sharpe(&result.equity_curve);
        result.num_trades = result.trades.len();
        result.win_rate = self.calculate_win_rate(&result.trades);

        result
    }

    /// Calculate Sharpe ratio from equity curve
    fn calculate_sharpe(&self, equity: &[f64]) -> f64 {
        if equity.len() < 2 {
            return 0.0;
        }

        // Calculate returns
        let returns: Vec<f64> = equity
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        // Annualize (assuming hourly data)
        let annual_factor = (24.0 * 365.0_f64).sqrt();
        (mean / std) * annual_factor
    }

    /// Calculate win rate
    fn calculate_win_rate(&self, trades: &[Trade]) -> f64 {
        if trades.is_empty() {
            return 0.0;
        }

        let winning = trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).count();
        winning as f64 / trades.len() as f64
    }
}

/// Simple strategy that trades based on signal threshold
pub fn threshold_strategy(
    signals: &HashMap<String, f64>,
    long_threshold: f64,
    short_threshold: f64,
) -> HashMap<String, f64> {
    signals
        .iter()
        .map(|(symbol, &signal)| {
            let position = if signal > long_threshold {
                1.0
            } else if signal < short_threshold {
                -1.0
            } else {
                0.0
            };
            (symbol.clone(), position)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_data() -> HashMap<String, Vec<OHLCV>> {
        let mut data = HashMap::new();

        let btc_candles: Vec<OHLCV> = (0..100)
            .map(|i| {
                let price = 40000.0 + (i as f64 * 10.0).sin() * 1000.0;
                OHLCV::new(Utc::now(), price, price + 100.0, price - 100.0, price, 1000.0)
            })
            .collect();

        let eth_candles: Vec<OHLCV> = (0..100)
            .map(|i| {
                let price = 2000.0 + (i as f64 * 10.0).sin() * 50.0;
                OHLCV::new(Utc::now(), price, price + 10.0, price - 10.0, price, 5000.0)
            })
            .collect();

        data.insert("BTCUSDT".to_string(), btc_candles);
        data.insert("ETHUSDT".to_string(), eth_candles);

        data
    }

    fn create_test_signals() -> Vec<HashMap<String, f64>> {
        (0..100)
            .map(|i| {
                let mut signals = HashMap::new();
                let btc_signal = ((i as f64 * 0.1).sin() * 0.5).max(-1.0).min(1.0);
                let eth_signal = ((i as f64 * 0.15).cos() * 0.5).max(-1.0).min(1.0);

                signals.insert("BTCUSDT".to_string(), btc_signal);
                signals.insert("ETHUSDT".to_string(), eth_signal);
                signals
            })
            .collect()
    }

    #[test]
    fn test_backtest_engine() {
        let engine = BacktestEngine::new(10000.0).with_commission(0.001);
        let data = create_test_data();
        let signals = create_test_signals();

        let result = engine.run(&data, &signals);

        assert!(!result.equity_curve.is_empty());
        assert!(result.max_drawdown >= 0.0);
        assert!(result.max_drawdown <= 1.0);
    }

    #[test]
    fn test_threshold_strategy() {
        let mut signals = HashMap::new();
        signals.insert("BTC".to_string(), 0.8);
        signals.insert("ETH".to_string(), -0.6);
        signals.insert("SOL".to_string(), 0.2);

        let positions = threshold_strategy(&signals, 0.5, -0.5);

        assert_eq!(positions["BTC"], 1.0);
        assert_eq!(positions["ETH"], -1.0);
        assert_eq!(positions["SOL"], 0.0);
    }
}
