//! Analysis modules for news interpretation

pub mod sentiment;
pub mod events;
pub mod aggregator;

pub use sentiment::{SentimentAnalyzer, SentimentResult};
pub use events::{EventClassifier, EventType};
pub use aggregator::{SignalAggregator, AggregatedSignal};
