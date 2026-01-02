//! Backtesting Engine

use std::collections::HashMap;
use super::{Position, TradingMetrics, TradingSignal, TradeDirection, RiskManager};
use crate::data::Candle;

/// Backtesting engine for E-GNN trading strategies
pub struct Backtester {
    /// Fee rate per trade
    fee_rate: f64,
    /// Slippage estimate
    slippage: f64,
    /// Risk manager
    risk_manager: RiskManager,
    /// Initial capital
    initial_capital: f64,
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub metrics: TradingMetrics,
    pub equity_curve: Vec<f64>,
    pub trades: Vec<TradeRecord>,
    pub final_capital: f64,
}

/// Record of a completed trade
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub symbol: String,
    pub direction: TradeDirection,
    pub entry_price: f64,
    pub exit_price: f64,
    pub entry_time: u64,
    pub exit_time: u64,
    pub pnl: f64,
    pub size: f64,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(fee_rate: f64, slippage: f64, initial_capital: f64) -> Self {
        Self {
            fee_rate,
            slippage,
            risk_manager: RiskManager::default(),
            initial_capital,
        }
    }

    /// Run backtest with given signals and price data
    pub fn run(
        &self,
        signals: &[Vec<TradingSignal>],
        prices: &HashMap<String, Vec<Candle>>,
    ) -> BacktestResult {
        let mut capital = self.initial_capital;
        let mut positions: HashMap<String, Position> = HashMap::new();
        let mut equity_curve = vec![capital];
        let mut trades = Vec::new();
        let mut returns = Vec::new();

        for (t, signals_at_t) in signals.iter().enumerate() {
            let prev_capital = capital;

            for signal in signals_at_t {
                let candles = match prices.get(&signal.symbol) {
                    Some(c) if t < c.len() => c,
                    _ => continue,
                };
                let current_price = candles[t].close;

                // Close existing position if signal changed
                if let Some(pos) = positions.remove(&signal.symbol) {
                    if pos.direction != signal.direction {
                        let pnl = pos.calculate_close_pnl(current_price, self.fee_rate);
                        capital *= 1.0 + pnl;

                        trades.push(TradeRecord {
                            symbol: signal.symbol.clone(),
                            direction: pos.direction,
                            entry_price: pos.entry_price,
                            exit_price: current_price,
                            entry_time: pos.entry_time,
                            exit_time: signal.timestamp,
                            pnl,
                            size: pos.size,
                        });
                    } else {
                        positions.insert(signal.symbol.clone(), pos);
                    }
                }

                // Open new position
                if signal.is_actionable() && !positions.contains_key(&signal.symbol) {
                    let size = self.risk_manager.calculate_position_size(
                        signal.size, signal.volatility
                    );
                    positions.insert(
                        signal.symbol.clone(),
                        Position::new(
                            signal.symbol.clone(),
                            signal.direction,
                            size,
                            current_price,
                            signal.timestamp,
                        ),
                    );
                }
            }

            // Update unrealized PnL
            for (symbol, pos) in positions.iter_mut() {
                if let Some(candles) = prices.get(symbol) {
                    if t < candles.len() {
                        pos.update_pnl(candles[t].close);
                    }
                }
            }

            let unrealized: f64 = positions.values().map(|p| p.unrealized_pnl).sum();
            let equity = capital * (1.0 + unrealized);
            equity_curve.push(equity);

            if prev_capital > 0.0 {
                returns.push((equity - prev_capital) / prev_capital);
            }
        }

        let metrics = TradingMetrics::from_returns(&returns, 0.02);

        BacktestResult {
            metrics,
            equity_curve,
            trades,
            final_capital: capital,
        }
    }
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(0.0004, 0.0001, 10000.0)
    }
}
