//! Event type definitions

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Type of crypto market event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Significant volume spike (>2x average)
    VolumeSpike,
    /// Price gap (overnight or session gap)
    PriceGap,
    /// Volatility expansion
    VolatilityExpansion,
    /// Large single-candle move
    LargeMove,
    /// Combination of multiple signals
    CompositeEvent,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::VolumeSpike => write!(f, "Volume Spike"),
            EventType::PriceGap => write!(f, "Price Gap"),
            EventType::VolatilityExpansion => write!(f, "Volatility Expansion"),
            EventType::LargeMove => write!(f, "Large Move"),
            EventType::CompositeEvent => write!(f, "Composite Event"),
        }
    }
}

/// A detected crypto market event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoEvent {
    /// Timestamp of the event
    pub timestamp: u64,
    /// Type of event
    pub event_type: EventType,
    /// Magnitude of the event (normalized)
    pub magnitude: f64,
    /// Direction: positive = bullish, negative = bearish
    pub direction: f64,
    /// Price at event time
    pub price: f64,
    /// Volume at event time
    pub volume: f64,
    /// Candle index in the data
    pub candle_index: usize,
    /// Additional metadata
    pub metadata: EventMetadata,
}

impl CryptoEvent {
    /// Convert timestamp to DateTime
    pub fn datetime(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.timestamp as i64).unwrap()
    }

    /// Check if this is a bullish event
    pub fn is_bullish(&self) -> bool {
        self.direction > 0.0
    }

    /// Check if this is a bearish event
    pub fn is_bearish(&self) -> bool {
        self.direction < 0.0
    }

    /// Get event strength based on magnitude
    pub fn strength(&self) -> EventStrength {
        if self.magnitude >= 3.0 {
            EventStrength::Strong
        } else if self.magnitude >= 2.0 {
            EventStrength::Moderate
        } else {
            EventStrength::Weak
        }
    }
}

/// Event strength classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventStrength {
    Weak,
    Moderate,
    Strong,
}

impl std::fmt::Display for EventStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventStrength::Weak => write!(f, "Weak"),
            EventStrength::Moderate => write!(f, "Moderate"),
            EventStrength::Strong => write!(f, "Strong"),
        }
    }
}

/// Additional event metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Volume multiple (vs average)
    pub volume_multiple: Option<f64>,
    /// Gap size in percent
    pub gap_size: Option<f64>,
    /// Volatility multiple (vs average)
    pub volatility_multiple: Option<f64>,
    /// Return at event
    pub return_pct: Option<f64>,
}

/// Summary of events over a period
#[derive(Debug, Clone)]
pub struct EventSummary {
    /// Total number of events
    pub total_events: usize,
    /// Events by type
    pub by_type: std::collections::HashMap<EventType, usize>,
    /// Bullish events count
    pub bullish_count: usize,
    /// Bearish events count
    pub bearish_count: usize,
    /// Average magnitude
    pub avg_magnitude: f64,
}

impl EventSummary {
    /// Create summary from events
    pub fn from_events(events: &[CryptoEvent]) -> Self {
        use std::collections::HashMap;

        let mut by_type: HashMap<EventType, usize> = HashMap::new();
        let mut bullish_count = 0;
        let mut bearish_count = 0;
        let mut total_magnitude = 0.0;

        for event in events {
            *by_type.entry(event.event_type).or_insert(0) += 1;
            if event.is_bullish() {
                bullish_count += 1;
            } else if event.is_bearish() {
                bearish_count += 1;
            }
            total_magnitude += event.magnitude;
        }

        let avg_magnitude = if events.is_empty() {
            0.0
        } else {
            total_magnitude / events.len() as f64
        };

        Self {
            total_events: events.len(),
            by_type,
            bullish_count,
            bearish_count,
            avg_magnitude,
        }
    }
}
