//! Backtesting engine

use crate::strategy::signal::{Signal, SignalType};
use super::metrics::{PerformanceMetrics, CoverageMetrics};

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub timestamp: i64,
    pub signal_type: SignalType,
    pub position_change: f64,
    pub execution_price: f64,
    pub position: f64,
    pub cost: f64,
    pub capital: f64,
}

/// Portfolio state
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub capital: f64,
    pub position: f64,
    pub entry_price: f64,
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub max_leverage: f64,
    pub transaction_cost: f64,
    pub slippage: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            max_leverage: 1.0,
            transaction_cost: 0.001, // 0.1%
            slippage: 0.0005,        // 0.05%
        }
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
    portfolio: Portfolio,
    trades: Vec<Trade>,
    equity_curve: Vec<f64>,
}

impl BacktestEngine {
    /// Create new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            portfolio: Portfolio {
                capital: config.initial_capital,
                position: 0.0,
                entry_price: 0.0,
            },
            equity_curve: vec![config.initial_capital],
            trades: Vec::new(),
            config,
        }
    }

    /// Execute a trading signal
    pub fn execute_signal(&mut self, signal: &Signal, current_price: f64) -> Option<Trade> {
        let target_position = signal
            .position_size
            .clamp(-self.config.max_leverage, self.config.max_leverage);

        let position_change = target_position - self.portfolio.position;

        // Minimum position change threshold
        if position_change.abs() < 0.01 {
            return None;
        }

        // Apply slippage
        let execution_price = if position_change > 0.0 {
            current_price * (1.0 + self.config.slippage)
        } else {
            current_price * (1.0 - self.config.slippage)
        };

        // Calculate transaction cost
        let trade_value = position_change.abs() * self.portfolio.capital;
        let cost = trade_value * self.config.transaction_cost;

        // Update capital for costs
        self.portfolio.capital -= cost;

        // Close existing position P&L
        if self.portfolio.position != 0.0 && self.portfolio.entry_price > 0.0 {
            let pnl = self.portfolio.position
                * (current_price / self.portfolio.entry_price - 1.0)
                * self.portfolio.capital;
            self.portfolio.capital += pnl;
        }

        // Update position
        self.portfolio.position = target_position;
        self.portfolio.entry_price = execution_price;

        let trade = Trade {
            timestamp: signal.timestamp,
            signal_type: signal.signal_type,
            position_change,
            execution_price,
            position: self.portfolio.position,
            cost,
            capital: self.portfolio.capital,
        };

        self.trades.push(trade.clone());
        Some(trade)
    }

    /// Update equity based on current price
    pub fn update_equity(&mut self, current_price: f64) -> f64 {
        let equity = if self.portfolio.position != 0.0 && self.portfolio.entry_price > 0.0 {
            let unrealized_pnl = self.portfolio.position
                * (current_price / self.portfolio.entry_price - 1.0)
                * self.portfolio.capital;
            self.portfolio.capital + unrealized_pnl
        } else {
            self.portfolio.capital
        };

        self.equity_curve.push(equity);
        equity
    }

    /// Run backtest with signals and prices
    pub fn run(&mut self, signals: &[Signal], prices: &[f64]) -> PerformanceMetrics {
        for (signal, &price) in signals.iter().zip(prices.iter()) {
            self.execute_signal(signal, price);
            self.update_equity(price);
        }

        self.get_performance_metrics()
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let win_rate = self.calculate_win_rate();
        PerformanceMetrics::from_equity_curve(&self.equity_curve, self.trades.len(), win_rate)
    }

    /// Calculate win rate
    fn calculate_win_rate(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }

        let mut wins = 0;
        for i in 1..self.trades.len() {
            if self.trades[i].capital > self.trades[i - 1].capital {
                wins += 1;
            }
        }

        wins as f64 / self.trades.len() as f64
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    /// Get trades
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Reset backtest state
    pub fn reset(&mut self) {
        self.portfolio = Portfolio {
            capital: self.config.initial_capital,
            position: 0.0,
            entry_price: 0.0,
        };
        self.equity_curve = vec![self.config.initial_capital];
        self.trades.clear();
    }
}

/// Backtest result
#[derive(Debug)]
pub struct BacktestResult {
    pub performance: PerformanceMetrics,
    pub coverage: CoverageMetrics,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<Trade>,
    pub signals: Vec<Signal>,
}

impl BacktestResult {
    /// Create new backtest result
    pub fn new(
        performance: PerformanceMetrics,
        coverage: CoverageMetrics,
        equity_curve: Vec<f64>,
        trades: Vec<Trade>,
        signals: Vec<Signal>,
    ) -> Self {
        Self {
            performance,
            coverage,
            equity_curve,
            trades,
            signals,
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(50));
        println!("BACKTEST RESULTS");
        println!("{}", "=".repeat(50));
        println!("\n{}", self.performance);
        println!("{}", self.coverage);

        // Signal distribution
        let long_count = self.signals.iter().filter(|s| s.signal_type == SignalType::Long).count();
        let short_count = self.signals.iter().filter(|s| s.signal_type == SignalType::Short).count();
        let hold_count = self.signals.iter().filter(|s| s.signal_type == SignalType::Hold).count();

        println!("Signal Distribution");
        println!("===================");
        println!("LONG:   {:>6} ({:.1}%)", long_count, 100.0 * long_count as f64 / self.signals.len() as f64);
        println!("SHORT:  {:>6} ({:.1}%)", short_count, 100.0 * short_count as f64 / self.signals.len() as f64);
        println!("HOLD:   {:>6} ({:.1}%)", hold_count, 100.0 * hold_count as f64 / self.signals.len() as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformal::predictor::PredictionInterval;

    #[test]
    fn test_backtest_engine() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);

        // Create test signals
        let signals: Vec<Signal> = (0..10)
            .map(|i| {
                let interval = PredictionInterval::new(0.01, 0.005, 0.015, 0.1);
                Signal::new(
                    i * 3600000,
                    SignalType::Long,
                    &interval,
                    0.8,
                    0.5,
                )
            })
            .collect();

        let prices: Vec<f64> = (0..10)
            .map(|i| 100.0 + i as f64 * 0.5)
            .collect();

        let metrics = engine.run(&signals, &prices);

        assert!(metrics.final_capital > 0.0);
        assert!(!engine.trades().is_empty());
    }
}
