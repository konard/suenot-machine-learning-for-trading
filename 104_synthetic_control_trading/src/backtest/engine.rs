//! Backtesting engine for SCM-based trading strategies.

use anyhow::Result;

use crate::model::SyntheticControlModel;

/// Results from a single trade.
#[derive(Debug, Clone)]
pub struct TradeResult {
    /// Entry date index
    pub entry_idx: usize,
    /// Exit date index
    pub exit_idx: usize,
    /// Symbol traded
    pub symbol: String,
    /// Direction (1=long, -1=short)
    pub direction: i32,
    /// Position size
    pub position_size: f64,
    /// Profit/loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// CAR at entry
    pub car_at_entry: f64,
    /// Holding period
    pub holding_days: usize,
}

/// Backtest performance results.
#[derive(Debug, Clone)]
pub struct BacktestResults {
    /// All trades executed
    pub trades: Vec<TradeResult>,
    /// Total return
    pub total_return: f64,
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
    /// Average CAR
    pub avg_car: f64,
    /// CAR t-statistic
    pub car_tstat: f64,
    /// Number of events
    pub n_events: usize,
    /// Number of trades
    pub n_trades: usize,
    /// Equity curve
    pub equity_curve: Vec<f64>,
}

/// Strategy configuration.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// CAR threshold for entry
    pub entry_threshold: f64,
    /// Days to hold position
    pub exit_days: usize,
    /// Stop loss level (negative)
    pub stop_loss: f64,
    /// Take profit level (positive)
    pub take_profit: f64,
    /// Trading direction: "long", "short", or "both"
    pub direction: String,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            entry_threshold: 0.02,
            exit_days: 5,
            stop_loss: -0.05,
            take_profit: 0.10,
            direction: "long".to_string(),
        }
    }
}

/// Event data for backtesting.
#[derive(Debug, Clone)]
pub struct EventData {
    /// Symbol
    pub symbol: String,
    /// Pre-treatment returns for treated
    pub treated_pre: Vec<f64>,
    /// Pre-treatment returns for donors
    pub donors_pre: Vec<Vec<f64>>,
    /// Post-treatment returns for treated
    pub treated_post: Vec<f64>,
    /// Post-treatment returns for donors
    pub donors_post: Vec<Vec<f64>>,
}

/// Backtesting engine for SCM-based strategies.
#[derive(Debug, Clone)]
pub struct BacktestEngine {
    /// Initial capital
    initial_capital: f64,
    /// Transaction cost
    transaction_cost: f64,
    /// Position size as fraction of capital
    position_size: f64,
    /// Maximum concurrent positions
    max_positions: usize,
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(100_000.0, 0.001)
    }
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(initial_capital: f64, transaction_cost: f64) -> Self {
        Self {
            initial_capital,
            transaction_cost,
            position_size: 0.1,
            max_positions: 5,
        }
    }

    /// Set position size.
    pub fn with_position_size(mut self, size: f64) -> Self {
        self.position_size = size;
        self
    }

    /// Set max positions.
    pub fn with_max_positions(mut self, max: usize) -> Self {
        self.max_positions = max;
        self
    }

    /// Run backtest on a list of events.
    pub fn run(&self, events: &[EventData], strategy: &StrategyConfig) -> Result<BacktestResults> {
        let mut trades: Vec<TradeResult> = Vec::new();
        let mut cars: Vec<f64> = Vec::new();
        let mut capital = self.initial_capital;
        let mut equity_curve = vec![capital];

        for event in events {
            // Validate data
            if event.treated_pre.len() < 20 || event.treated_post.len() < 3 {
                continue;
            }

            // Fit SCM
            let mut model = SyntheticControlModel::new();
            if model.fit(&event.treated_pre, &event.donors_pre).is_err() {
                continue;
            }

            // Check pre-treatment fit
            if model.pre_treatment_rmse().unwrap_or(f64::MAX) > 0.05 {
                continue;
            }

            // Estimate effects
            let effects = match model.estimate_effects(&event.treated_post, &event.donors_post) {
                Ok(e) => e,
                Err(_) => continue,
            };

            let car = effects.car;
            cars.push(car);

            // Check entry signal
            let (should_enter, direction) = self.check_entry_signal(car, strategy);

            if !should_enter {
                continue;
            }

            // Execute trade
            if let Some(trade) = self.execute_trade(event, &effects, direction, strategy, capital) {
                capital += trade.pnl;
                equity_curve.push(capital);
                trades.push(trade);
            }
        }

        // Calculate metrics
        let results = self.calculate_metrics(&trades, &cars, &equity_curve, events.len());

        Ok(results)
    }

    /// Check if entry signal is triggered.
    fn check_entry_signal(&self, car: f64, strategy: &StrategyConfig) -> (bool, i32) {
        match strategy.direction.as_str() {
            "long" if car > strategy.entry_threshold => (true, 1),
            "short" if car < -strategy.entry_threshold => (true, -1),
            "both" if car.abs() > strategy.entry_threshold => (true, if car > 0.0 { 1 } else { -1 }),
            _ => (false, 0),
        }
    }

    /// Execute a single trade.
    fn execute_trade(
        &self,
        event: &EventData,
        effects: &crate::model::synthetic_control::SCMEffects,
        direction: i32,
        strategy: &StrategyConfig,
        capital: f64,
    ) -> Option<TradeResult> {
        let treated_post = &effects.treated_post;
        let car = effects.car;

        let exit_days = strategy.exit_days.min(treated_post.len().saturating_sub(1));
        if exit_days < 1 {
            return None;
        }

        let position_value = capital * self.position_size;

        // Calculate exit
        let mut cumulative_return = 0.0;
        let mut actual_exit_day = exit_days;

        for i in 1..=exit_days.min(treated_post.len() - 1) {
            let daily_return = treated_post[i];
            cumulative_return += daily_return * direction as f64;

            // Check stop loss / take profit
            if cumulative_return <= strategy.stop_loss {
                actual_exit_day = i;
                break;
            }
            if cumulative_return >= strategy.take_profit {
                actual_exit_day = i;
                break;
            }
        }

        // Calculate PnL
        let gross_return = cumulative_return;
        let net_return = gross_return - 2.0 * self.transaction_cost;
        let pnl = position_value * net_return;

        Some(TradeResult {
            entry_idx: 0,
            exit_idx: actual_exit_day,
            symbol: event.symbol.clone(),
            direction,
            position_size: position_value,
            pnl,
            return_pct: net_return,
            car_at_entry: car,
            holding_days: actual_exit_day,
        })
    }

    /// Calculate backtest metrics.
    fn calculate_metrics(
        &self,
        trades: &[TradeResult],
        cars: &[f64],
        equity_curve: &[f64],
        n_events: usize,
    ) -> BacktestResults {
        if trades.is_empty() {
            return BacktestResults {
                trades: vec![],
                total_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                avg_car: if cars.is_empty() {
                    0.0
                } else {
                    cars.iter().sum::<f64>() / cars.len() as f64
                },
                car_tstat: 0.0,
                n_events,
                n_trades: 0,
                equity_curve: equity_curve.to_vec(),
            };
        }

        let returns: Vec<f64> = trades.iter().map(|t| t.return_pct).collect();
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();

        // Total return
        let total_return = (equity_curve.last().unwrap_or(&self.initial_capital)
            - equity_curve.first().unwrap_or(&self.initial_capital))
            / equity_curve.first().unwrap_or(&self.initial_capital);

        // Sharpe ratio
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_return = std_dev(&returns);
        let avg_holding = trades.iter().map(|t| t.holding_days).sum::<usize>() as f64
            / trades.len().max(1) as f64;
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (252.0 / avg_holding.max(1.0)).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = std_dev(&negative_returns);
        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (252.0 / avg_holding.max(1.0)).sqrt()
        } else {
            sharpe_ratio
        };

        // Maximum drawdown
        let mut peak = equity_curve[0];
        let mut max_dd = 0.0_f64;
        for &equity in equity_curve {
            peak = peak.max(equity);
            let dd = (equity - peak) / peak;
            max_dd = max_dd.min(dd);
        }

        // Win rate
        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = wins as f64 / returns.len().max(1) as f64;

        // Profit factor
        let gross_profit: f64 = pnls.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        };

        // CAR statistics
        let avg_car = if cars.is_empty() {
            0.0
        } else {
            cars.iter().sum::<f64>() / cars.len() as f64
        };
        let car_std = std_dev(cars);
        let car_tstat = if car_std > 0.0 && !cars.is_empty() {
            avg_car / (car_std / (cars.len() as f64).sqrt())
        } else {
            0.0
        };

        BacktestResults {
            trades: trades.to_vec(),
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: max_dd,
            win_rate,
            profit_factor,
            avg_car,
            car_tstat,
            n_events,
            n_trades: trades.len(),
            equity_curve: equity_curve.to_vec(),
        }
    }
}

/// Calculate standard deviation.
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Generate sample events for testing.
pub fn generate_sample_events(
    n_events: usize,
    pre_days: usize,
    post_days: usize,
    n_donors: usize,
    treatment_effect: f64,
) -> Vec<EventData> {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 0.01).unwrap();

    (0..n_events)
        .map(|i| {
            // Common factor
            let common: Vec<f64> = (0..pre_days + post_days)
                .map(|_| normal.sample(&mut rng))
                .collect();

            // Treated returns
            let mut treated: Vec<f64> = common
                .iter()
                .map(|&c| c + normal.sample(&mut rng) * 0.5)
                .collect();

            // Add treatment effect
            let effect = treatment_effect * (1.0 + rng.gen::<f64>() * 0.5);
            for t in pre_days..pre_days + post_days {
                treated[t] += effect / post_days as f64;
            }

            // Donors
            let donors: Vec<Vec<f64>> = (0..n_donors)
                .map(|_| {
                    common
                        .iter()
                        .map(|&c| c + normal.sample(&mut rng) * 0.5)
                        .collect()
                })
                .collect();

            EventData {
                symbol: format!("STOCK_{}", i),
                treated_pre: treated[..pre_days].to_vec(),
                donors_pre: donors.iter().map(|d| d[..pre_days].to_vec()).collect(),
                treated_post: treated[pre_days..].to_vec(),
                donors_post: donors.iter().map(|d| d[pre_days..].to_vec()).collect(),
            }
        })
        .collect()
}

/// Print backtest report.
pub fn print_report(results: &BacktestResults) {
    println!("\n{}", "=".repeat(50));
    println!("BACKTEST RESULTS");
    println!("{}", "=".repeat(50));
    println!("Events Analyzed:     {}", results.n_events);
    println!("Trades Executed:     {}", results.n_trades);
    println!("{}", "-".repeat(50));
    println!("Total Return:        {:.2}%", results.total_return * 100.0);
    println!("Sharpe Ratio:        {:.3}", results.sharpe_ratio);
    println!("Sortino Ratio:       {:.3}", results.sortino_ratio);
    println!("Max Drawdown:        {:.2}%", results.max_drawdown * 100.0);
    println!("Win Rate:            {:.2}%", results.win_rate * 100.0);
    println!("Profit Factor:       {:.2}", results.profit_factor);
    println!("{}", "-".repeat(50));
    println!("Average CAR:         {:.4}", results.avg_car);
    println!("CAR t-statistic:     {:.3}", results.car_tstat);
    println!("{}", "=".repeat(50));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sample_events() {
        let events = generate_sample_events(10, 30, 10, 3, 0.03);
        assert_eq!(events.len(), 10);
        assert_eq!(events[0].treated_pre.len(), 30);
        assert_eq!(events[0].treated_post.len(), 10);
    }

    #[test]
    fn test_backtest_engine() {
        let events = generate_sample_events(50, 30, 10, 3, 0.03);
        let engine = BacktestEngine::new(100_000.0, 0.001);
        let strategy = StrategyConfig::default();

        let results = engine.run(&events, &strategy).unwrap();
        assert!(results.n_events == 50);
    }
}
