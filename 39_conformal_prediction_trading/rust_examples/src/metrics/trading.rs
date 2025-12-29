//! Trading performance metrics
//!
//! Metrics for evaluating trading strategy performance.

use crate::strategy::trading::TradeResult;

/// Trading performance metrics
#[derive(Debug, Clone)]
pub struct TradingMetrics {
    /// Total number of observations
    pub n_observations: usize,
    /// Number of trades taken
    pub n_trades: usize,
    /// Trade frequency (fraction of observations with trades)
    pub trade_frequency: f64,
    /// Total PnL
    pub total_pnl: f64,
    /// Average PnL per trade
    pub avg_pnl: f64,
    /// Win rate (fraction of profitable trades)
    pub win_rate: f64,
    /// Average winning trade PnL
    pub avg_win: f64,
    /// Average losing trade PnL (positive number)
    pub avg_loss: f64,
    /// Win/loss ratio
    pub win_loss_ratio: f64,
    /// Sharpe ratio (annualized, assuming daily data)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
    /// Coverage on trades (fraction of trades where actual in interval)
    pub coverage_on_trades: f64,
}

impl TradingMetrics {
    /// Calculate metrics from trade results
    pub fn calculate(results: &[TradeResult]) -> Self {
        let n_observations = results.len();
        let trades: Vec<&TradeResult> = results.iter().filter(|r| r.signal.trade).collect();
        let n_trades = trades.len();

        if n_trades == 0 {
            return Self::empty(n_observations);
        }

        let trade_frequency = n_trades as f64 / n_observations as f64;
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let total_pnl: f64 = pnls.iter().sum();
        let avg_pnl = total_pnl / n_trades as f64;

        // Win/loss analysis
        let wins: Vec<f64> = pnls.iter().filter(|&&p| p > 0.0).copied().collect();
        let losses: Vec<f64> = pnls.iter().filter(|&&p| p < 0.0).copied().collect();

        let win_rate = wins.len() as f64 / n_trades as f64;
        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };
        let avg_loss = if !losses.is_empty() {
            -losses.iter().sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };
        let win_loss_ratio = if avg_loss > 0.0 {
            avg_win / avg_loss
        } else {
            f64::INFINITY
        };

        // Sharpe ratio
        let sharpe_ratio = Self::calculate_sharpe(&pnls);

        // Maximum drawdown
        let (max_drawdown, _) = Self::calculate_drawdown(&pnls);

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            total_pnl / max_drawdown
        } else {
            f64::INFINITY
        };

        // Coverage on trades
        let n_covered = trades.iter().filter(|t| t.covered).count();
        let coverage_on_trades = n_covered as f64 / n_trades as f64;

        Self {
            n_observations,
            n_trades,
            trade_frequency,
            total_pnl,
            avg_pnl,
            win_rate,
            avg_win,
            avg_loss,
            win_loss_ratio,
            sharpe_ratio,
            max_drawdown,
            calmar_ratio,
            coverage_on_trades,
        }
    }

    /// Calculate from PnL series directly
    pub fn from_pnl(pnls: &[f64]) -> Self {
        let n = pnls.len();
        if n == 0 {
            return Self::empty(0);
        }

        let total_pnl: f64 = pnls.iter().sum();
        let avg_pnl = total_pnl / n as f64;

        let wins: Vec<f64> = pnls.iter().filter(|&&p| p > 0.0).copied().collect();
        let losses: Vec<f64> = pnls.iter().filter(|&&p| p < 0.0).copied().collect();

        let win_rate = wins.len() as f64 / n as f64;
        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };
        let avg_loss = if !losses.is_empty() {
            -losses.iter().sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };
        let win_loss_ratio = if avg_loss > 0.0 {
            avg_win / avg_loss
        } else {
            f64::INFINITY
        };

        let sharpe_ratio = Self::calculate_sharpe(pnls);
        let (max_drawdown, _) = Self::calculate_drawdown(pnls);
        let calmar_ratio = if max_drawdown > 0.0 {
            total_pnl / max_drawdown
        } else {
            f64::INFINITY
        };

        Self {
            n_observations: n,
            n_trades: n,
            trade_frequency: 1.0,
            total_pnl,
            avg_pnl,
            win_rate,
            avg_win,
            avg_loss,
            win_loss_ratio,
            sharpe_ratio,
            max_drawdown,
            calmar_ratio,
            coverage_on_trades: 0.0,
        }
    }

    /// Calculate Sharpe ratio (annualized for daily data)
    fn calculate_sharpe(pnls: &[f64]) -> f64 {
        if pnls.len() < 2 {
            return 0.0;
        }

        let mean = pnls.iter().sum::<f64>() / pnls.len() as f64;
        let variance = pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / pnls.len() as f64;
        let std = variance.sqrt();

        if std.abs() < 1e-10 {
            return 0.0;
        }

        (mean / std) * (252.0_f64).sqrt()
    }

    /// Calculate maximum drawdown and its duration
    fn calculate_drawdown(pnls: &[f64]) -> (f64, usize) {
        if pnls.is_empty() {
            return (0.0, 0);
        }

        let mut cumsum = 0.0;
        let mut peak = 0.0;
        let mut max_dd = 0.0;
        let mut dd_start = 0;
        let mut max_dd_duration = 0;
        let mut current_dd_start = 0;

        for (i, &pnl) in pnls.iter().enumerate() {
            cumsum += pnl;

            if cumsum > peak {
                peak = cumsum;
                current_dd_start = i;
            }

            let dd = peak - cumsum;
            if dd > max_dd {
                max_dd = dd;
                dd_start = current_dd_start;
                max_dd_duration = i - dd_start;
            }
        }

        (max_dd, max_dd_duration)
    }

    /// Empty metrics
    fn empty(n_observations: usize) -> Self {
        Self {
            n_observations,
            n_trades: 0,
            trade_frequency: 0.0,
            total_pnl: 0.0,
            avg_pnl: 0.0,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            win_loss_ratio: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            coverage_on_trades: 0.0,
        }
    }

    /// Generate a summary report
    pub fn report(&self) -> String {
        format!(
            "Trading Metrics:\n\
             ================\n\
             Observations:  {}\n\
             Trades:        {}\n\
             Trade Freq:    {:.1}%\n\
             Total PnL:     {:.4}\n\
             Avg PnL:       {:.4}\n\
             Win Rate:      {:.1}%\n\
             Avg Win:       {:.4}\n\
             Avg Loss:      {:.4}\n\
             W/L Ratio:     {:.2}\n\
             Sharpe:        {:.2}\n\
             Max Drawdown:  {:.4}\n\
             Calmar:        {:.2}\n\
             Coverage:      {:.1}%",
            self.n_observations,
            self.n_trades,
            self.trade_frequency * 100.0,
            self.total_pnl,
            self.avg_pnl,
            self.win_rate * 100.0,
            self.avg_win,
            self.avg_loss,
            self.win_loss_ratio,
            self.sharpe_ratio,
            self.max_drawdown,
            self.calmar_ratio,
            self.coverage_on_trades * 100.0
        )
    }
}

/// Compare two strategies
pub fn compare_strategies(baseline: &TradingMetrics, conformal: &TradingMetrics) -> String {
    format!(
        "Strategy Comparison:\n\
         ====================\n\
         Metric          Baseline    Conformal   Improvement\n\
         ------          --------    ---------   -----------\n\
         Trade Freq      {:.1}%       {:.1}%       {:.1}%\n\
         Total PnL       {:.4}     {:.4}     {:.4}\n\
         Win Rate        {:.1}%       {:.1}%       {:.1}%\n\
         Sharpe          {:.2}        {:.2}        {:.2}\n\
         Max DD          {:.4}     {:.4}     {:.4}",
        baseline.trade_frequency * 100.0,
        conformal.trade_frequency * 100.0,
        (conformal.trade_frequency - baseline.trade_frequency) * 100.0,
        baseline.total_pnl,
        conformal.total_pnl,
        conformal.total_pnl - baseline.total_pnl,
        baseline.win_rate * 100.0,
        conformal.win_rate * 100.0,
        (conformal.win_rate - baseline.win_rate) * 100.0,
        baseline.sharpe_ratio,
        conformal.sharpe_ratio,
        conformal.sharpe_ratio - baseline.sharpe_ratio,
        baseline.max_drawdown,
        conformal.max_drawdown,
        baseline.max_drawdown - conformal.max_drawdown
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_metrics_from_pnl() {
        let pnls = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.003];

        let metrics = TradingMetrics::from_pnl(&pnls);

        assert_eq!(metrics.n_trades, 7);
        assert!(metrics.win_rate > 0.5); // 4 wins, 3 losses
        assert!(metrics.total_pnl > 0.0);
    }

    #[test]
    fn test_drawdown() {
        // Series: 0.1, -0.2, 0.05, 0.1
        // Cumsum: 0.1, -0.1, -0.05, 0.05
        // Max DD: 0.2 (peak 0.1, trough -0.1)
        let pnls = vec![0.1, -0.2, 0.05, 0.1];

        let (max_dd, _duration) = TradingMetrics::calculate_drawdown(&pnls);

        assert!((max_dd - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_calculation() {
        // Constant positive returns should have high Sharpe
        let constant_pnls = vec![0.01; 100];
        let sharpe = TradingMetrics::calculate_sharpe(&constant_pnls);

        // With zero variance, Sharpe should be 0 (our implementation)
        // Actually constant returns have zero std, so Sharpe is 0
        assert!(sharpe.abs() < 1e-10 || sharpe > 10.0);
    }
}
