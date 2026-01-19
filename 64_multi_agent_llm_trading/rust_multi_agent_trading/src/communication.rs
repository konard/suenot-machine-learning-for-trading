//! Agent communication patterns.
//!
//! This module provides communication mechanisms for multi-agent systems:
//! - Message passing
//! - Debates (Bull vs Bear)
//! - Round table discussions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

use crate::agents::{Agent, Analysis, Signal};
use crate::data::MarketData;
use crate::error::Result;

/// Message types for agent communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Analysis,
    Query,
    Response,
    DebateArgument,
    DebateRebuttal,
    Decision,
    Broadcast,
}

/// A message between agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub sender: String,
    pub receiver: String,
    pub message_type: MessageType,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

impl Message {
    /// Create a new message.
    pub fn new(
        sender: &str,
        receiver: &str,
        message_type: MessageType,
        content: &str,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            sender: sender.to_string(),
            receiver: receiver.to_string(),
            message_type,
            content: content.to_string(),
            timestamp: Utc::now(),
        }
    }

    /// Create a broadcast message (to all agents).
    pub fn broadcast(sender: &str, content: &str) -> Self {
        Self::new(sender, "broadcast", MessageType::Broadcast, content)
    }
}

/// Message bus for agent communication.
#[derive(Debug, Default)]
pub struct MessageBus {
    messages: VecDeque<Message>,
}

impl MessageBus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Send a message.
    pub fn send(&mut self, message: Message) {
        self.messages.push_back(message);
    }

    /// Get all messages for a specific receiver.
    pub fn get_messages(&self, receiver: &str) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.receiver == receiver || m.receiver == "broadcast")
            .collect()
    }

    /// Get message history.
    pub fn history(&self) -> &VecDeque<Message> {
        &self.messages
    }

    /// Clear all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

/// Debate round result.
#[derive(Debug, Clone)]
pub struct DebateRound {
    pub round: usize,
    pub bull_argument: Analysis,
    pub bear_argument: Analysis,
}

/// Result of a debate between Bull and Bear agents.
#[derive(Debug, Clone)]
pub struct DebateResult {
    pub symbol: String,
    pub rounds: Vec<DebateRound>,
    pub bull_avg_confidence: f64,
    pub bear_avg_confidence: f64,
    pub winner: String,
    pub final_signal: Signal,
    pub final_confidence: f64,
    pub conclusion: String,
}

/// A structured debate between Bull and Bear agents.
pub struct Debate<B, A>
where
    B: Agent,
    A: Agent,
{
    bull_agent: B,
    bear_agent: A,
    num_rounds: usize,
    message_bus: MessageBus,
}

impl<B, A> Debate<B, A>
where
    B: Agent,
    A: Agent,
{
    /// Create a new debate.
    pub fn new(bull_agent: B, bear_agent: A, num_rounds: usize) -> Self {
        Self {
            bull_agent,
            bear_agent,
            num_rounds,
            message_bus: MessageBus::new(),
        }
    }

    /// Conduct the debate.
    pub async fn conduct(&mut self, symbol: &str, data: &MarketData) -> Result<DebateResult> {
        let mut rounds = Vec::with_capacity(self.num_rounds);

        for round in 0..self.num_rounds {
            // Bull presents argument
            let bull_analysis = self.bull_agent.analyze(symbol, data, None).await?;

            self.message_bus.send(Message::new(
                self.bull_agent.name(),
                self.bear_agent.name(),
                MessageType::DebateArgument,
                &bull_analysis.reasoning,
            ));

            // Bear presents rebuttal
            let bear_analysis = self.bear_agent.analyze(symbol, data, None).await?;

            self.message_bus.send(Message::new(
                self.bear_agent.name(),
                self.bull_agent.name(),
                MessageType::DebateRebuttal,
                &bear_analysis.reasoning,
            ));

            rounds.push(DebateRound {
                round: round + 1,
                bull_argument: bull_analysis,
                bear_argument: bear_analysis,
            });
        }

        // Calculate scores
        let bull_avg_confidence: f64 = rounds
            .iter()
            .map(|r| r.bull_argument.confidence)
            .sum::<f64>()
            / rounds.len() as f64;

        let bear_avg_confidence: f64 = rounds
            .iter()
            .map(|r| r.bear_argument.confidence)
            .sum::<f64>()
            / rounds.len() as f64;

        let (winner, final_signal, conclusion) = if bull_avg_confidence > bear_avg_confidence + 0.1 {
            (
                "bull".to_string(),
                Signal::Buy,
                "Bull arguments significantly outweigh bear concerns".to_string(),
            )
        } else if bear_avg_confidence > bull_avg_confidence + 0.1 {
            (
                "bear".to_string(),
                Signal::Sell,
                "Bear arguments significantly outweigh bull case".to_string(),
            )
        } else {
            (
                "tie".to_string(),
                Signal::Neutral,
                "Arguments are evenly balanced".to_string(),
            )
        };

        let final_confidence = 0.5 + (bull_avg_confidence - bear_avg_confidence).abs() * 0.5;

        Ok(DebateResult {
            symbol: symbol.to_string(),
            rounds,
            bull_avg_confidence,
            bear_avg_confidence,
            winner,
            final_signal,
            final_confidence,
            conclusion,
        })
    }

    /// Get the message history from the debate.
    pub fn message_history(&self) -> &VecDeque<Message> {
        self.message_bus.history()
    }
}

/// Round table discussion result.
#[derive(Debug, Clone)]
pub struct RoundTableResult {
    pub symbol: String,
    pub analyses: Vec<Analysis>,
    pub consensus: String,
    pub buy_ratio: f64,
    pub sell_ratio: f64,
}

/// A round table discussion where agents share views sequentially.
pub struct RoundTable<'a> {
    agents: Vec<&'a dyn Agent>,
    message_bus: MessageBus,
}

impl<'a> RoundTable<'a> {
    /// Create a new round table.
    pub fn new(agents: Vec<&'a dyn Agent>) -> Self {
        Self {
            agents,
            message_bus: MessageBus::new(),
        }
    }

    /// Conduct the round table discussion.
    pub async fn conduct(&mut self, symbol: &str, data: &MarketData) -> Result<RoundTableResult> {
        let mut analyses = Vec::with_capacity(self.agents.len());

        for agent in &self.agents {
            let analysis = agent.analyze(symbol, data, None).await?;

            self.message_bus.send(Message::broadcast(
                agent.name(),
                &format!(
                    "{}: {} with {:.0}% confidence",
                    agent.name(),
                    analysis.signal,
                    analysis.confidence * 100.0
                ),
            ));

            analyses.push(analysis);
        }

        // Calculate consensus
        let buy_count = analyses
            .iter()
            .filter(|a| matches!(a.signal, Signal::Buy | Signal::StrongBuy))
            .count();
        let sell_count = analyses
            .iter()
            .filter(|a| matches!(a.signal, Signal::Sell | Signal::StrongSell))
            .count();
        let total = analyses.len();

        let buy_ratio = buy_count as f64 / total as f64;
        let sell_ratio = sell_count as f64 / total as f64;

        let consensus = if buy_ratio > 0.6 {
            "BULLISH".to_string()
        } else if sell_ratio > 0.6 {
            "BEARISH".to_string()
        } else {
            "MIXED".to_string()
        };

        Ok(RoundTableResult {
            symbol: symbol.to_string(),
            analyses,
            consensus,
            buy_ratio,
            sell_ratio,
        })
    }

    /// Get message history.
    pub fn message_history(&self) -> &VecDeque<Message> {
        self.message_bus.history()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::{BearAgent, BullAgent};
    use crate::data::create_mock_data;

    #[tokio::test]
    async fn test_debate() {
        let bull = BullAgent::new("Bull");
        let bear = BearAgent::new("Bear");
        let mut debate = Debate::new(bull, bear, 2);

        let data = create_mock_data("TEST", 100, 100.0);
        let result = debate.conduct("TEST", &data).await.unwrap();

        assert_eq!(result.rounds.len(), 2);
        assert!(!result.winner.is_empty());
    }

    #[test]
    fn test_message_bus() {
        let mut bus = MessageBus::new();

        bus.send(Message::new("A", "B", MessageType::Analysis, "test"));
        bus.send(Message::broadcast("A", "broadcast message"));

        assert_eq!(bus.get_messages("B").len(), 2); // Direct + broadcast
        assert_eq!(bus.get_messages("C").len(), 1); // Only broadcast
    }
}
