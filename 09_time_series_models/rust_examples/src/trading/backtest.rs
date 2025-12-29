//! Бэктестинг торговых стратегий

use crate::trading::signals::{Signal, Position, Trade, TradeStats, extract_trades};
use crate::trading::pairs::PairsTradingStrategy;

/// Результат бэктеста
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub trades: Vec<Trade>,
    pub stats: TradeStats,
    pub equity_curve: Vec<f64>,
    pub returns: Vec<f64>,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: usize,
}

impl BacktestResult {
    pub fn display(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.stats.display());
        s.push_str(&format!("\n\nPerformance Metrics\n"));
        s.push_str(&"=".repeat(20));
        s.push_str(&format!("\nSharpe Ratio: {:.3}", self.sharpe_ratio));
        s.push_str(&format!("\nSortino Ratio: {:.3}", self.sortino_ratio));
        s.push_str(&format!("\nCalmar Ratio: {:.3}", self.calmar_ratio));
        s.push_str(&format!("\nMax Drawdown: {:.2}%", self.max_drawdown * 100.0));
        s.push_str(&format!("\nMax DD Duration: {} periods", self.max_drawdown_duration));
        s
    }
}

/// Параметры бэктеста
#[derive(Debug, Clone)]
pub struct BacktestParams {
    /// Начальный капитал
    pub initial_capital: f64,
    /// Комиссия за сделку (в процентах)
    pub commission: f64,
    /// Проскальзывание (в процентах)
    pub slippage: f64,
    /// Размер позиции (доля от капитала)
    pub position_size: f64,
}

impl Default for BacktestParams {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission: 0.001,  // 0.1%
            slippage: 0.0005,   // 0.05%
            position_size: 1.0,
        }
    }
}

/// Запуск бэктеста
pub fn run_backtest(
    strategy: &mut PairsTradingStrategy,
    prices1: &[f64],
    prices2: &[f64],
    params: &BacktestParams,
) -> BacktestResult {
    let signals = strategy.generate_signals(prices1, prices2);

    // Применяем комиссии и проскальзывание
    let adjusted_prices1: Vec<f64> = prices1
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            match signals.get(i) {
                Some(Signal::BuySpread) => p * (1.0 + params.slippage + params.commission),
                Some(Signal::SellSpread) => p * (1.0 - params.slippage - params.commission),
                Some(Signal::ExitLong) => p * (1.0 - params.slippage - params.commission),
                Some(Signal::ExitShort) => p * (1.0 + params.slippage + params.commission),
                _ => p,
            }
        })
        .collect();

    let adjusted_prices2: Vec<f64> = prices2
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            match signals.get(i) {
                Some(Signal::BuySpread) => p * (1.0 - params.slippage - params.commission),
                Some(Signal::SellSpread) => p * (1.0 + params.slippage + params.commission),
                Some(Signal::ExitLong) => p * (1.0 + params.slippage + params.commission),
                Some(Signal::ExitShort) => p * (1.0 - params.slippage - params.commission),
                _ => p,
            }
        })
        .collect();

    let trades = extract_trades(&signals, &adjusted_prices1, &adjusted_prices2, strategy.hedge_ratio);
    let stats = TradeStats::from_trades(&trades);

    // Equity curve
    let mut equity_curve = vec![params.initial_capital];
    let mut current_equity = params.initial_capital;

    for trade in &trades {
        let position_value = current_equity * params.position_size;
        let pnl = trade.return_pct() * position_value;
        current_equity += pnl;
        equity_curve.push(current_equity);
    }

    // Returns
    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    // Performance metrics
    let (max_drawdown, max_dd_duration) = compute_drawdown(&equity_curve);
    let sharpe_ratio = compute_sharpe(&returns, 0.0);
    let sortino_ratio = compute_sortino(&returns, 0.0);
    let calmar_ratio = if max_drawdown > 0.0 {
        let annual_return = returns.iter().sum::<f64>() * 252.0 / returns.len().max(1) as f64;
        annual_return / max_drawdown
    } else {
        0.0
    };

    BacktestResult {
        trades,
        stats,
        equity_curve,
        returns,
        sharpe_ratio,
        sortino_ratio,
        calmar_ratio,
        max_drawdown,
        max_drawdown_duration: max_dd_duration,
    }
}

/// Вычисление максимальной просадки
fn compute_drawdown(equity: &[f64]) -> (f64, usize) {
    if equity.is_empty() {
        return (0.0, 0);
    }

    let mut peak = equity[0];
    let mut max_drawdown = 0.0;
    let mut max_duration = 0;
    let mut current_duration = 0;
    let mut in_drawdown = false;

    for &value in equity {
        if value > peak {
            peak = value;
            if in_drawdown {
                in_drawdown = false;
                if current_duration > max_duration {
                    max_duration = current_duration;
                }
                current_duration = 0;
            }
        } else {
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
            in_drawdown = true;
            current_duration += 1;
        }
    }

    if current_duration > max_duration {
        max_duration = current_duration;
    }

    (max_drawdown, max_duration)
}

/// Sharpe ratio
fn compute_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean_return: f64 = returns.iter().sum::<f64>() / n;
    let excess_return = mean_return - risk_free_rate / 252.0;

    let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev > 0.0 {
        excess_return / std_dev * (252.0_f64).sqrt()
    } else {
        0.0
    }
}

/// Sortino ratio (использует только отрицательные returns)
fn compute_sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean_return: f64 = returns.iter().sum::<f64>() / n;
    let excess_return = mean_return - risk_free_rate / 252.0;

    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

    if downside_returns.is_empty() {
        return f64::INFINITY;
    }

    let downside_variance: f64 =
        downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
    let downside_dev = downside_variance.sqrt();

    if downside_dev > 0.0 {
        excess_return / downside_dev * (252.0_f64).sqrt()
    } else {
        0.0
    }
}

/// Walk-forward анализ
pub struct WalkForwardAnalysis {
    pub in_sample_periods: usize,
    pub out_sample_periods: usize,
    pub results: Vec<WalkForwardResult>,
}

#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    pub period: usize,
    pub in_sample_sharpe: f64,
    pub out_sample_sharpe: f64,
    pub out_sample_return: f64,
    pub out_sample_trades: usize,
}

impl WalkForwardAnalysis {
    pub fn new(in_sample: usize, out_sample: usize) -> Self {
        Self {
            in_sample_periods: in_sample,
            out_sample_periods: out_sample,
            results: Vec::new(),
        }
    }

    pub fn run(
        &mut self,
        prices1: &[f64],
        prices2: &[f64],
        strategy_params: &crate::trading::pairs::PairsTradingParams,
        backtest_params: &BacktestParams,
    ) {
        let total_periods = prices1.len();
        let step = self.out_sample_periods;
        let mut period = 0;

        let mut start = 0;
        while start + self.in_sample_periods + self.out_sample_periods <= total_periods {
            let is_end = start + self.in_sample_periods;
            let os_end = is_end + self.out_sample_periods;

            // In-sample
            let is_prices1 = &prices1[start..is_end];
            let is_prices2 = &prices2[start..is_end];

            // Оптимизируем hedge ratio на in-sample данных
            let result = crate::trading::cointegration::engle_granger_test(is_prices1, is_prices2);
            let hedge_ratio = result.map(|r| r.hedge_ratio).unwrap_or(1.0);

            let mut is_strategy = PairsTradingStrategy::new(
                "ASSET1",
                "ASSET2",
                hedge_ratio,
                strategy_params.clone(),
            );

            let is_result = run_backtest(&mut is_strategy, is_prices1, is_prices2, backtest_params);

            // Out-of-sample
            let os_prices1 = &prices1[is_end..os_end];
            let os_prices2 = &prices2[is_end..os_end];

            let mut os_strategy = PairsTradingStrategy::new(
                "ASSET1",
                "ASSET2",
                hedge_ratio,
                strategy_params.clone(),
            );

            let os_result = run_backtest(&mut os_strategy, os_prices1, os_prices2, backtest_params);

            self.results.push(WalkForwardResult {
                period,
                in_sample_sharpe: is_result.sharpe_ratio,
                out_sample_sharpe: os_result.sharpe_ratio,
                out_sample_return: os_result.returns.iter().sum(),
                out_sample_trades: os_result.trades.len(),
            });

            start += step;
            period += 1;
        }
    }

    pub fn efficiency_ratio(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let avg_is: f64 = self.results.iter().map(|r| r.in_sample_sharpe).sum::<f64>()
            / self.results.len() as f64;
        let avg_os: f64 = self.results.iter().map(|r| r.out_sample_sharpe).sum::<f64>()
            / self.results.len() as f64;

        if avg_is != 0.0 {
            avg_os / avg_is
        } else {
            0.0
        }
    }

    pub fn display(&self) -> String {
        let mut s = String::from("Walk-Forward Analysis Results\n");
        s.push_str(&"=".repeat(40));
        s.push('\n');

        for result in &self.results {
            s.push_str(&format!(
                "Period {}: IS Sharpe={:.3}, OS Sharpe={:.3}, OS Return={:.2}%, Trades={}\n",
                result.period,
                result.in_sample_sharpe,
                result.out_sample_sharpe,
                result.out_sample_return * 100.0,
                result.out_sample_trades
            ));
        }

        s.push_str(&format!("\nEfficiency Ratio: {:.2}", self.efficiency_ratio()));
        s
    }
}
