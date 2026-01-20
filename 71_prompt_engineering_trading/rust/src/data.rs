//! Market data loading utilities.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::backtest::Bar;
use crate::error::TradingResult;

/// Market snapshot for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub price: f64,
    pub change_pct: f64,
    pub volume: f64,
    pub high_52w: f64,
    pub low_52w: f64,
    pub avg_volume: f64,
    pub market_cap: Option<f64>,
    pub pe_ratio: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Technical indicators calculated from price data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicators {
    pub sma_20: Option<f64>,
    pub sma_50: Option<f64>,
    pub sma_200: Option<f64>,
    pub rsi_14: Option<f64>,
    pub volatility_20d: Option<f64>,
    pub trend_20d: Option<f64>,
}

/// Mock data loader for testing.
pub struct MockDataLoader {
    base_prices: HashMap<String, f64>,
    volatility: f64,
}

impl MockDataLoader {
    /// Create a new mock loader with default prices.
    pub fn new() -> Self {
        let mut base_prices = HashMap::new();
        base_prices.insert("AAPL".to_string(), 185.0);
        base_prices.insert("MSFT".to_string(), 380.0);
        base_prices.insert("GOOGL".to_string(), 140.0);
        base_prices.insert("BTC/USDT".to_string(), 45000.0);
        base_prices.insert("ETH/USDT".to_string(), 2500.0);

        Self {
            base_prices,
            volatility: 0.02,
        }
    }

    /// Create with custom base prices.
    pub fn with_prices(prices: HashMap<String, f64>) -> Self {
        Self {
            base_prices: prices,
            volatility: 0.02,
        }
    }

    /// Generate mock OHLCV data.
    pub fn get_ohlcv(&self, symbol: &str, days: usize) -> TradingResult<Vec<Bar>> {
        let base_price = self.base_prices.get(symbol).copied().unwrap_or(100.0);
        let mut bars = Vec::new();
        let mut price = base_price;

        for i in 0..days {
            // Simple random walk (deterministic for testing)
            let change = ((i as f64 * 7.0).sin() * self.volatility) + 0.0001;
            price *= 1.0 + change;

            let high = price * (1.0 + ((i as f64 * 3.0).cos().abs() * 0.01));
            let low = price * (1.0 - ((i as f64 * 5.0).sin().abs() * 0.01));
            let open = price * (1.0 + ((i as f64 * 2.0).sin() * 0.002));
            let volume = 100000.0 * (1.0 + ((i as f64 * 4.0).cos() * 0.5));

            bars.push(Bar {
                timestamp: Utc::now() - Duration::days((days - i) as i64),
                open,
                high,
                low,
                close: price,
                volume,
            });
        }

        Ok(bars)
    }

    /// Get current price for symbol.
    pub fn get_current_price(&self, symbol: &str) -> f64 {
        self.base_prices.get(symbol).copied().unwrap_or(100.0)
    }

    /// Get market snapshot.
    pub fn get_snapshot(&self, symbol: &str) -> TradingResult<MarketSnapshot> {
        let bars = self.get_ohlcv(symbol, 252)?;
        let current_price = bars.last().map(|b| b.close).unwrap_or(100.0);
        let prev_price = bars.get(bars.len().saturating_sub(2)).map(|b| b.close).unwrap_or(current_price);

        Ok(MarketSnapshot {
            symbol: symbol.to_string(),
            price: current_price,
            change_pct: ((current_price - prev_price) / prev_price) * 100.0,
            volume: bars.last().map(|b| b.volume).unwrap_or(0.0),
            high_52w: bars.iter().map(|b| b.high).fold(f64::MIN, f64::max),
            low_52w: bars.iter().map(|b| b.low).fold(f64::MAX, f64::min),
            avg_volume: bars.iter().map(|b| b.volume).sum::<f64>() / bars.len() as f64,
            market_cap: None,
            pe_ratio: None,
            timestamp: Utc::now(),
        })
    }
}

impl Default for MockDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate technical indicators from price bars.
pub fn calculate_indicators(bars: &[Bar]) -> TechnicalIndicators {
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();

    // SMA calculations
    let sma_20 = if closes.len() >= 20 {
        Some(closes[closes.len() - 20..].iter().sum::<f64>() / 20.0)
    } else {
        None
    };

    let sma_50 = if closes.len() >= 50 {
        Some(closes[closes.len() - 50..].iter().sum::<f64>() / 50.0)
    } else {
        None
    };

    let sma_200 = if closes.len() >= 200 {
        Some(closes[closes.len() - 200..].iter().sum::<f64>() / 200.0)
    } else {
        None
    };

    // RSI calculation (simplified)
    let rsi_14 = if closes.len() >= 15 {
        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in (closes.len() - 14)..closes.len() {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }

        let avg_gain = gains / 14.0;
        let avg_loss = losses / 14.0;

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            Some(100.0 - (100.0 / (1.0 + rs)))
        } else {
            Some(100.0)
        }
    } else {
        None
    };

    // Volatility (20-day standard deviation as percentage)
    let volatility_20d = if closes.len() >= 20 {
        let slice = &closes[closes.len() - 20..];
        let mean = slice.iter().sum::<f64>() / 20.0;
        let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 20.0;
        Some((variance.sqrt() / mean) * 100.0)
    } else {
        None
    };

    // 20-day trend (percent change)
    let trend_20d = if closes.len() >= 20 {
        let old_price = closes[closes.len() - 20];
        let new_price = closes[closes.len() - 1];
        Some(((new_price - old_price) / old_price) * 100.0)
    } else {
        None
    };

    TechnicalIndicators {
        sma_20,
        sma_50,
        sma_200,
        rsi_14,
        volatility_20d,
        trend_20d,
    }
}

/// Prepare market data for LLM prompt.
pub fn prepare_for_prompt(snapshot: &MarketSnapshot, indicators: &TechnicalIndicators) -> HashMap<String, String> {
    let mut data = HashMap::new();

    data.insert("symbol".to_string(), snapshot.symbol.clone());
    data.insert("current_price".to_string(), format!("{:.2}", snapshot.price));
    data.insert("change_pct".to_string(), format!("{:.2}", snapshot.change_pct));
    data.insert("volume".to_string(), format!("{:.0}", snapshot.volume));
    data.insert("high_52w".to_string(), format!("{:.2}", snapshot.high_52w));
    data.insert("low_52w".to_string(), format!("{:.2}", snapshot.low_52w));

    if let Some(sma_20) = indicators.sma_20 {
        data.insert("sma_20".to_string(), format!("{:.2}", sma_20));
    }
    if let Some(sma_50) = indicators.sma_50 {
        data.insert("sma_50".to_string(), format!("{:.2}", sma_50));
    }
    if let Some(rsi_14) = indicators.rsi_14 {
        data.insert("rsi_14".to_string(), format!("{:.1}", rsi_14));
    }
    if let Some(volatility) = indicators.volatility_20d {
        data.insert("volatility_20d".to_string(), format!("{:.2}", volatility));
    }
    if let Some(trend) = indicators.trend_20d {
        data.insert("trend_20d".to_string(), format!("{:.2}", trend));
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_loader() {
        let loader = MockDataLoader::new();
        let bars = loader.get_ohlcv("AAPL", 30).unwrap();

        assert_eq!(bars.len(), 30);
        assert!(bars[0].close > 0.0);
    }

    #[test]
    fn test_indicators() {
        let loader = MockDataLoader::new();
        let bars = loader.get_ohlcv("AAPL", 60).unwrap();
        let indicators = calculate_indicators(&bars);

        assert!(indicators.sma_20.is_some());
        assert!(indicators.sma_50.is_some());
        assert!(indicators.rsi_14.is_some());
    }

    #[test]
    fn test_snapshot() {
        let loader = MockDataLoader::new();
        let snapshot = loader.get_snapshot("MSFT").unwrap();

        assert_eq!(snapshot.symbol, "MSFT");
        assert!(snapshot.price > 0.0);
        assert!(snapshot.high_52w >= snapshot.low_52w);
    }
}
