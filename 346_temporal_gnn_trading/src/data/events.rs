//! Event definitions for TGN

use serde::{Deserialize, Serialize};

/// Event types in the temporal graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Trade execution
    Trade,
    /// Order book update
    OrderBookUpdate,
    /// Liquidation event
    Liquidation,
    /// Funding rate change
    FundingUpdate,
    /// Large transfer/whale activity
    WhaleAlert,
    /// Price level break
    PriceLevelBreak,
}

/// A temporal event in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event type
    pub event_type: EventType,
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Timestamp in milliseconds
    pub timestamp: u64,
    /// Event-specific data
    pub data: EventData,
}

/// Event-specific data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventData {
    /// Price at event time
    pub price: Option<f64>,
    /// Volume/size
    pub volume: Option<f64>,
    /// Direction (positive = buy, negative = sell)
    pub direction: Option<f64>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for EventData {
    fn default() -> Self {
        Self {
            price: None,
            volume: None,
            direction: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

impl Event {
    /// Create a trade event
    pub fn trade(source: &str, target: &str, timestamp: u64, price: f64, volume: f64, is_buy: bool) -> Self {
        Self {
            event_type: EventType::Trade,
            source: source.to_string(),
            target: target.to_string(),
            timestamp,
            data: EventData {
                price: Some(price),
                volume: Some(volume),
                direction: Some(if is_buy { 1.0 } else { -1.0 }),
                metadata: std::collections::HashMap::new(),
            },
        }
    }

    /// Create a liquidation event
    pub fn liquidation(source: &str, target: &str, timestamp: u64, price: f64, volume: f64, is_long: bool) -> Self {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("position_type".to_string(), if is_long { "long" } else { "short" }.to_string());

        Self {
            event_type: EventType::Liquidation,
            source: source.to_string(),
            target: target.to_string(),
            timestamp,
            data: EventData {
                price: Some(price),
                volume: Some(volume),
                direction: Some(if is_long { -1.0 } else { 1.0 }), // Liquidation direction
                metadata,
            },
        }
    }

    /// Create a funding update event
    pub fn funding_update(symbol: &str, timestamp: u64, funding_rate: f64) -> Self {
        Self {
            event_type: EventType::FundingUpdate,
            source: "funding".to_string(),
            target: symbol.to_string(),
            timestamp,
            data: EventData {
                price: None,
                volume: None,
                direction: Some(funding_rate),
                metadata: std::collections::HashMap::new(),
            },
        }
    }
}

/// Event stream for processing multiple events
#[derive(Debug)]
pub struct EventStream {
    /// Events buffer
    events: Vec<Event>,
    /// Current position
    position: usize,
}

impl EventStream {
    /// Create new event stream
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            position: 0,
        }
    }

    /// Create from events
    pub fn from_events(mut events: Vec<Event>) -> Self {
        // Sort by timestamp
        events.sort_by_key(|e| e.timestamp);
        Self {
            events,
            position: 0,
        }
    }

    /// Add event to stream
    pub fn add(&mut self, event: Event) {
        self.events.push(event);
    }

    /// Get next event
    pub fn next(&mut self) -> Option<&Event> {
        if self.position < self.events.len() {
            let event = &self.events[self.position];
            self.position += 1;
            Some(event)
        } else {
            None
        }
    }

    /// Peek at next event without consuming
    pub fn peek(&self) -> Option<&Event> {
        self.events.get(self.position)
    }

    /// Get events in time window
    pub fn get_window(&self, start: u64, end: u64) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp < end)
            .collect()
    }

    /// Reset stream to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get total number of events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if stream is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get remaining events count
    pub fn remaining(&self) -> usize {
        self.events.len().saturating_sub(self.position)
    }
}

impl Default for EventStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for EventStream {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.events.len() {
            let event = self.events[self.position].clone();
            self.position += 1;
            Some(event)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_event() {
        let event = Event::trade("exchange", "BTCUSDT", 1000, 50000.0, 1.0, true);
        assert_eq!(event.event_type, EventType::Trade);
        assert_eq!(event.data.direction, Some(1.0));
    }

    #[test]
    fn test_event_stream() {
        let events = vec![
            Event::trade("ex", "BTC", 100, 50000.0, 1.0, true),
            Event::trade("ex", "BTC", 200, 50100.0, 0.5, false),
            Event::trade("ex", "ETH", 150, 3000.0, 2.0, true),
        ];

        let mut stream = EventStream::from_events(events);

        // Should be sorted by timestamp
        assert_eq!(stream.next().unwrap().timestamp, 100);
        assert_eq!(stream.next().unwrap().timestamp, 150);
        assert_eq!(stream.next().unwrap().timestamp, 200);
        assert!(stream.next().is_none());
    }

    #[test]
    fn test_event_stream_window() {
        let events = vec![
            Event::trade("ex", "BTC", 100, 50000.0, 1.0, true),
            Event::trade("ex", "BTC", 200, 50100.0, 0.5, false),
            Event::trade("ex", "ETH", 300, 3000.0, 2.0, true),
        ];

        let stream = EventStream::from_events(events);
        let window = stream.get_window(100, 250);

        assert_eq!(window.len(), 2);
    }
}
