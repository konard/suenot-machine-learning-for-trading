//! Trading strategy implementation.

use crate::data::attributes::{AssetAttributes, AssetRegistry};
use crate::data::bybit::Kline;
use crate::model::network::ZeroShotModel;
use crate::strategy::regime::ZeroShotRegimePredictor;
use crate::strategy::signals::{Signal, SignalFilter, TimestampedSignal};
use crate::{Result, ZeroShotError};

/// Trading strategy configuration.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Symbols to trade
    pub symbols: Vec<String>,
    /// Kline interval (e.g., "15" for 15 minutes)
    pub interval: String,
    /// Number of klines to fetch for analysis
    pub lookback: usize,
    /// Maximum position size as fraction of portfolio
    pub max_position_size: f64,
    /// Risk per trade as fraction of portfolio
    pub risk_per_trade: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Minimum confidence for trading
    pub min_confidence: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["BTCUSDT".to_string()],
            interval: "15".to_string(),
            lookback: 200,
            max_position_size: 0.1,
            risk_per_trade: 0.02,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.04,
            min_confidence: 0.5,
        }
    }
}

/// Current position information.
#[derive(Debug, Clone)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Position side (true = long, false = short)
    pub is_long: bool,
    /// Entry price
    pub entry_price: f64,
    /// Position size (in quote currency)
    pub size: f64,
    /// Stop loss price
    pub stop_loss: f64,
    /// Take profit price
    pub take_profit: f64,
    /// Entry timestamp
    pub entry_time: i64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
}

impl Position {
    /// Update unrealized PnL based on current price.
    pub fn update_pnl(&mut self, current_price: f64) {
        let price_change = if self.is_long {
            current_price - self.entry_price
        } else {
            self.entry_price - current_price
        };
        self.unrealized_pnl = (price_change / self.entry_price) * self.size;
    }

    /// Check if stop loss is hit.
    pub fn is_stop_loss_hit(&self, current_price: f64) -> bool {
        if self.is_long {
            current_price <= self.stop_loss
        } else {
            current_price >= self.stop_loss
        }
    }

    /// Check if take profit is hit.
    pub fn is_take_profit_hit(&self, current_price: f64) -> bool {
        if self.is_long {
            current_price >= self.take_profit
        } else {
            current_price <= self.take_profit
        }
    }
}

/// Zero-shot trading strategy.
pub struct ZeroShotTradingStrategy {
    /// Model for predictions
    predictor: ZeroShotRegimePredictor,
    /// Strategy configuration
    config: StrategyConfig,
    /// Asset registry
    asset_registry: AssetRegistry,
    /// Signal filter
    signal_filter: SignalFilter,
    /// Current positions
    positions: std::collections::HashMap<String, Position>,
    /// Signal history
    signal_history: Vec<TimestampedSignal>,
}

impl ZeroShotTradingStrategy {
    /// Create a new trading strategy.
    pub fn new(model: ZeroShotModel, config: StrategyConfig) -> Self {
        Self {
            predictor: ZeroShotRegimePredictor::new(model),
            config: config.clone(),
            asset_registry: AssetRegistry::with_crypto_defaults(),
            signal_filter: SignalFilter::new(config.min_confidence, 3, 300),
            positions: std::collections::HashMap::new(),
            signal_history: Vec::new(),
        }
    }

    /// Get current configuration.
    pub fn config(&self) -> &StrategyConfig {
        &self.config
    }

    /// Get current positions.
    pub fn positions(&self) -> &std::collections::HashMap<String, Position> {
        &self.positions
    }

    /// Get signal history.
    pub fn signal_history(&self) -> &[TimestampedSignal] {
        &self.signal_history
    }

    /// Analyze a symbol and generate signal.
    pub fn analyze(&mut self, klines: &[Kline], symbol: &str) -> Result<TimestampedSignal> {
        if klines.is_empty() {
            return Err(ZeroShotError::NotEnoughData {
                needed: 1,
                got: 0,
            });
        }

        // Predict regime with confidence
        let (regime, confidence) = self.predictor.predict_with_confidence(klines)?;

        // Generate signal from regime
        let raw_signal = Signal::from_regime_with_confidence(regime, confidence);

        // Apply signal filter
        let filtered_signal = self.signal_filter.filter(raw_signal, confidence);

        // Get current price
        let current_price = klines.last().unwrap().close;

        // Create timestamped signal
        let signal = TimestampedSignal::new(
            filtered_signal,
            symbol.to_string(),
            regime,
            confidence,
            current_price,
        );

        // Store in history
        self.signal_history.push(signal.clone());

        Ok(signal)
    }

    /// Open a position based on signal.
    pub fn open_position(
        &mut self,
        signal: &TimestampedSignal,
        portfolio_value: f64,
    ) -> Option<Position> {
        // Don't open if already have position
        if self.positions.contains_key(&signal.symbol) {
            return None;
        }

        // Only open on buy/sell signals
        if signal.signal == Signal::Hold {
            return None;
        }

        let is_long = signal.signal.is_buy();

        // Calculate position size
        let position_multiplier = signal.signal.position_multiplier().abs();
        let max_size = portfolio_value * self.config.max_position_size;
        let size = max_size * position_multiplier;

        // Calculate stop loss and take profit
        let (stop_loss, take_profit) = if is_long {
            (
                signal.price * (1.0 - self.config.stop_loss_pct),
                signal.price * (1.0 + self.config.take_profit_pct),
            )
        } else {
            (
                signal.price * (1.0 + self.config.stop_loss_pct),
                signal.price * (1.0 - self.config.take_profit_pct),
            )
        };

        let position = Position {
            symbol: signal.symbol.clone(),
            is_long,
            entry_price: signal.price,
            size,
            stop_loss,
            take_profit,
            entry_time: signal.timestamp,
            unrealized_pnl: 0.0,
        };

        self.positions.insert(signal.symbol.clone(), position.clone());
        Some(position)
    }

    /// Close a position.
    pub fn close_position(&mut self, symbol: &str, exit_price: f64) -> Option<f64> {
        if let Some(mut position) = self.positions.remove(symbol) {
            position.update_pnl(exit_price);
            Some(position.unrealized_pnl)
        } else {
            None
        }
    }

    /// Update all positions with current prices.
    pub fn update_positions(&mut self, prices: &std::collections::HashMap<String, f64>) {
        for (symbol, position) in self.positions.iter_mut() {
            if let Some(&price) = prices.get(symbol) {
                position.update_pnl(price);
            }
        }
    }

    /// Check for stop loss / take profit hits.
    pub fn check_exits(
        &mut self,
        prices: &std::collections::HashMap<String, f64>,
    ) -> Vec<(String, f64, String)> {
        let mut exits = Vec::new();

        let symbols: Vec<String> = self.positions.keys().cloned().collect();
        for symbol in symbols {
            if let Some(&price) = prices.get(&symbol) {
                let should_exit = {
                    let position = self.positions.get(&symbol).unwrap();
                    if position.is_stop_loss_hit(price) {
                        Some("stop_loss")
                    } else if position.is_take_profit_hit(price) {
                        Some("take_profit")
                    } else {
                        None
                    }
                };

                if let Some(reason) = should_exit {
                    if let Some(pnl) = self.close_position(&symbol, price) {
                        exits.push((symbol, pnl, reason.to_string()));
                    }
                }
            }
        }

        exits
    }

    /// Get asset attributes for a symbol.
    pub fn get_attributes(&self, symbol: &str) -> AssetAttributes {
        self.asset_registry.get_or_default(symbol)
    }

    /// Register asset attributes.
    pub fn register_asset(&mut self, symbol: &str, attributes: AssetAttributes) {
        self.asset_registry.register(symbol, attributes);
    }

    /// Get strategy summary.
    pub fn summary(&self) -> StrategySummary {
        let total_positions = self.positions.len();
        let long_positions = self.positions.values().filter(|p| p.is_long).count();
        let short_positions = total_positions - long_positions;

        let total_unrealized_pnl: f64 = self
            .positions
            .values()
            .map(|p| p.unrealized_pnl)
            .sum();

        let total_signals = self.signal_history.len();
        let buy_signals = self
            .signal_history
            .iter()
            .filter(|s| s.signal.is_buy())
            .count();
        let sell_signals = self
            .signal_history
            .iter()
            .filter(|s| s.signal.is_sell())
            .count();

        StrategySummary {
            total_positions,
            long_positions,
            short_positions,
            total_unrealized_pnl,
            total_signals,
            buy_signals,
            sell_signals,
        }
    }
}

/// Strategy summary statistics.
#[derive(Debug, Clone)]
pub struct StrategySummary {
    /// Total open positions
    pub total_positions: usize,
    /// Long positions count
    pub long_positions: usize,
    /// Short positions count
    pub short_positions: usize,
    /// Total unrealized PnL
    pub total_unrealized_pnl: f64,
    /// Total signals generated
    pub total_signals: usize,
    /// Buy signals count
    pub buy_signals: usize,
    /// Sell signals count
    pub sell_signals: usize,
}

impl std::fmt::Display for StrategySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Strategy Summary:")?;
        writeln!(
            f,
            "  Positions: {} total ({} long, {} short)",
            self.total_positions, self.long_positions, self.short_positions
        )?;
        writeln!(f, "  Unrealized PnL: ${:.2}", self.total_unrealized_pnl)?;
        writeln!(
            f,
            "  Signals: {} total ({} buy, {} sell)",
            self.total_signals, self.buy_signals, self.sell_signals
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::ModelConfig;
    use chrono::Utc;

    fn create_dummy_klines(n: usize, trend: f64) -> Vec<Kline> {
        let mut klines = Vec::with_capacity(n);
        let mut price = 100.0;

        for _ in 0..n {
            price *= 1.0 + trend + (rand::random::<f64>() - 0.5) * 0.01;
            klines.push(Kline {
                timestamp: Utc::now(),
                open: price * 0.99,
                high: price * 1.01,
                low: price * 0.98,
                close: price,
                volume: 1000.0,
                turnover: price * 1000.0,
            });
        }

        klines
    }

    #[test]
    fn test_strategy_analyze() {
        let model = ZeroShotModel::new(ModelConfig::default());
        let config = StrategyConfig::default();
        let mut strategy = ZeroShotTradingStrategy::new(model, config);

        let klines = create_dummy_klines(200, 0.002);
        let signal = strategy.analyze(&klines, "BTCUSDT").unwrap();

        assert_eq!(signal.symbol, "BTCUSDT");
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    }

    #[test]
    fn test_position_management() {
        let model = ZeroShotModel::new(ModelConfig::default());
        let config = StrategyConfig::default();
        let mut strategy = ZeroShotTradingStrategy::new(model, config);

        let signal = TimestampedSignal::new(
            Signal::Buy,
            "BTCUSDT".to_string(),
            MarketRegime::WeakUptrend,
            0.7,
            50000.0,
        );

        let position = strategy.open_position(&signal, 100000.0);
        assert!(position.is_some());

        let pos = position.unwrap();
        assert!(pos.is_long);
        assert_eq!(pos.entry_price, 50000.0);

        // Should not open duplicate position
        let dup = strategy.open_position(&signal, 100000.0);
        assert!(dup.is_none());

        // Close position
        let pnl = strategy.close_position("BTCUSDT", 51000.0);
        assert!(pnl.is_some());
    }
}
