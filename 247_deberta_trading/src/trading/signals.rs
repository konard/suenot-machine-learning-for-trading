//! Signal generation utilities for DeBERTa sentiment trading.
//!
//! Provides functions to combine sentiment signals with price data
//! for generating actionable trading signals.

use crate::data::bybit::Kline;
use crate::model::deberta::{SentimentModel, SentimentLabel};

/// A trading signal with associated metadata.
#[derive(Debug, Clone)]
pub struct Signal {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub direction: SignalDirection,
    pub strength: f64,
    pub price: f64,
    pub sentiment_label: SentimentLabel,
}

/// Direction of a trading signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalDirection::Buy => write!(f, "BUY"),
            SignalDirection::Sell => write!(f, "SELL"),
            SignalDirection::Hold => write!(f, "HOLD"),
        }
    }
}

/// Generate a signal from a headline and current price.
pub fn generate_signal(
    model: &SentimentModel,
    headline: &str,
    kline: &Kline,
    threshold: f64,
) -> Signal {
    let result = model.predict(headline);

    let (direction, strength) = match result.label {
        SentimentLabel::Positive if result.score >= threshold => {
            (SignalDirection::Buy, result.score)
        }
        SentimentLabel::Negative if result.score >= threshold => {
            (SignalDirection::Sell, result.score)
        }
        _ => (SignalDirection::Hold, 0.0),
    };

    Signal {
        timestamp: kline.timestamp,
        direction,
        strength,
        price: kline.close,
        sentiment_label: result.label,
    }
}

/// Generate signals from multiple headlines for a single price bar.
pub fn generate_composite_signal(
    model: &SentimentModel,
    headlines: &[&str],
    kline: &Kline,
    threshold: f64,
) -> Signal {
    if headlines.is_empty() {
        return Signal {
            timestamp: kline.timestamp,
            direction: SignalDirection::Hold,
            strength: 0.0,
            price: kline.close,
            sentiment_label: SentimentLabel::Neutral,
        };
    }

    let trading_signal = model.get_trading_signal(headlines, threshold);

    let direction = if trading_signal.signal > 0.0 {
        SignalDirection::Buy
    } else if trading_signal.signal < 0.0 {
        SignalDirection::Sell
    } else {
        SignalDirection::Hold
    };

    let label = if trading_signal.n_positive > trading_signal.n_negative {
        SentimentLabel::Positive
    } else if trading_signal.n_negative > trading_signal.n_positive {
        SentimentLabel::Negative
    } else {
        SentimentLabel::Neutral
    };

    Signal {
        timestamp: kline.timestamp,
        direction,
        strength: trading_signal.confidence,
        price: kline.close,
        sentiment_label: label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_klines;

    #[test]
    fn test_generate_signal_positive() {
        let model = SentimentModel::new();
        let klines = generate_synthetic_klines(1, "BTCUSDT");
        let signal = generate_signal(
            &model,
            "Bitcoin surges on massive institutional buying",
            &klines[0],
            0.5,
        );
        assert_eq!(signal.direction, SignalDirection::Buy);
        assert!(signal.strength > 0.0);
    }

    #[test]
    fn test_generate_signal_negative() {
        let model = SentimentModel::new();
        let klines = generate_synthetic_klines(1, "BTCUSDT");
        let signal = generate_signal(
            &model,
            "Crypto market crashes amid fraud investigation",
            &klines[0],
            0.5,
        );
        assert_eq!(signal.direction, SignalDirection::Sell);
    }

    #[test]
    fn test_composite_signal() {
        let model = SentimentModel::new();
        let klines = generate_synthetic_klines(1, "BTCUSDT");
        let headlines = vec![
            "Bitcoin surges on ETF approval",
            "Record institutional inflows",
        ];
        let signal = generate_composite_signal(&model, &headlines, &klines[0], 0.3);
        assert_eq!(signal.direction, SignalDirection::Buy);
    }
}
