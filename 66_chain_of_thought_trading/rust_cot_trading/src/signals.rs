//! Trading Signal Generation
//!
//! Multi-step signal generation with Chain-of-Thought reasoning.

use serde::{Deserialize, Serialize};
use crate::analyzer::{CoTAnalyzer, ReasoningStep};
use crate::error::Result;

/// Trading signal types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    /// Strong buy signal
    StrongBuy,
    /// Buy signal
    Buy,
    /// Hold/neutral signal
    Hold,
    /// Sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

impl Signal {
    /// Convert signal to numeric value (-2 to +2).
    pub fn to_score(&self) -> i32 {
        match self {
            Signal::StrongBuy => 2,
            Signal::Buy => 1,
            Signal::Hold => 0,
            Signal::Sell => -1,
            Signal::StrongSell => -2,
        }
    }

    /// Create signal from numeric score.
    pub fn from_score(score: i32) -> Self {
        match score {
            s if s >= 2 => Signal::StrongBuy,
            1 => Signal::Buy,
            0 => Signal::Hold,
            -1 => Signal::Sell,
            _ => Signal::StrongSell,
        }
    }
}

/// Complete trading signal with reasoning chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoTSignal {
    /// The symbol analyzed
    pub symbol: String,
    /// Current price at signal generation
    pub current_price: f64,
    /// The trading signal
    pub signal_type: Signal,
    /// Confidence in the signal (0.0-1.0)
    pub confidence: f64,
    /// Suggested stop loss price
    pub stop_loss: f64,
    /// Suggested take profit price
    pub take_profit: f64,
    /// Complete reasoning chain
    pub reasoning_chain: Vec<String>,
    /// Detailed reasoning steps
    pub reasoning_steps: Vec<ReasoningStep>,
}

/// Multi-step signal generator with CoT reasoning.
pub struct SignalGenerator {
    analyzer: CoTAnalyzer,
    risk_multiplier: f64,
    reward_multiplier: f64,
}

impl SignalGenerator {
    /// Create a new signal generator.
    pub fn new(analyzer: CoTAnalyzer) -> Self {
        Self {
            analyzer,
            risk_multiplier: 1.5,
            reward_multiplier: 2.5,
        }
    }

    /// Create a signal generator with default mock analyzer.
    pub fn new_mock() -> Self {
        Self::new(CoTAnalyzer::new_mock())
    }

    /// Set risk multiplier for stop loss calculation.
    pub fn with_risk_multiplier(mut self, multiplier: f64) -> Self {
        self.risk_multiplier = multiplier;
        self
    }

    /// Set reward multiplier for take profit calculation.
    pub fn with_reward_multiplier(mut self, multiplier: f64) -> Self {
        self.reward_multiplier = multiplier;
        self
    }

    /// Generate a trading signal with full reasoning chain.
    #[allow(clippy::too_many_arguments)]
    pub async fn generate(
        &self,
        symbol: &str,
        current_price: f64,
        rsi: f64,
        macd: f64,
        macd_signal: f64,
        sma_20: f64,
        sma_50: f64,
        volume_ratio: f64,
        atr: f64,
    ) -> Result<CoTSignal> {
        let mut reasoning_chain = Vec::new();
        let mut reasoning_steps = Vec::new();
        let mut scores: Vec<i32> = Vec::new();
        let mut confidences: Vec<f64> = Vec::new();

        // Step 1: Trend Analysis
        let (trend_score, trend_conf, trend_reason) = self.analyze_trend(
            current_price, sma_20, sma_50,
        );
        reasoning_chain.push(format!("Trend Analysis: {}", trend_reason));
        reasoning_steps.push(ReasoningStep::new(
            "Trend Analysis",
            format!("Price: {:.2}, SMA20: {:.2}, SMA50: {:.2}", current_price, sma_20, sma_50),
            "Comparing price to moving averages to determine trend direction",
            &trend_reason,
            trend_conf,
        ));
        scores.push(trend_score);
        confidences.push(trend_conf);

        // Step 2: Momentum Analysis
        let (momentum_score, momentum_conf, momentum_reason) = self.analyze_momentum(
            rsi, macd, macd_signal,
        );
        reasoning_chain.push(format!("Momentum Analysis: {}", momentum_reason));
        reasoning_steps.push(ReasoningStep::new(
            "Momentum Analysis",
            format!("RSI: {:.1}, MACD: {:.4}, Signal: {:.4}", rsi, macd, macd_signal),
            "Analyzing RSI and MACD for momentum confirmation",
            &momentum_reason,
            momentum_conf,
        ));
        scores.push(momentum_score);
        confidences.push(momentum_conf);

        // Step 3: Volume Analysis
        let (volume_score, volume_conf, volume_reason) = self.analyze_volume(volume_ratio);
        reasoning_chain.push(format!("Volume Analysis: {}", volume_reason));
        reasoning_steps.push(ReasoningStep::new(
            "Volume Analysis",
            format!("Volume Ratio: {:.2}x average", volume_ratio),
            "Checking if volume confirms the price movement",
            &volume_reason,
            volume_conf,
        ));
        scores.push(volume_score);
        confidences.push(volume_conf);

        // Step 4: Risk/Reward Calculation
        let (stop_loss, take_profit, rr_reason) = self.calculate_risk_reward(
            current_price, atr,
        );
        reasoning_chain.push(format!("Risk Management: {}", rr_reason));
        reasoning_steps.push(ReasoningStep::new(
            "Risk/Reward Calculation",
            format!("ATR: {:.2}", atr),
            "Setting stop loss and take profit based on volatility",
            &rr_reason,
            0.85,
        ));

        // Step 5: LLM-based sentiment (optional deeper analysis)
        let query = format!(
            "Analyze {} for trading: Price ${:.2}, RSI {:.1}, MACD {:.4}, \
             above SMA20: {}, above SMA50: {}, volume {:.1}x average",
            symbol, current_price, rsi, macd,
            current_price > sma_20, current_price > sma_50, volume_ratio
        );

        if let Ok(analysis) = self.analyzer.analyze(&query).await {
            reasoning_chain.push(format!("AI Analysis: {}", analysis.final_answer));
            reasoning_steps.extend(analysis.reasoning_steps);
            confidences.push(analysis.confidence);
        }

        // Step 6: Aggregate Signals
        let total_score: i32 = scores.iter().sum();
        let avg_confidence: f64 = confidences.iter().sum::<f64>() / confidences.len() as f64;

        let signal_type = Signal::from_score(total_score);

        let aggregation_reason = format!(
            "Aggregated {} signals with total score {} -> {:?}",
            scores.len(), total_score, signal_type
        );
        reasoning_chain.push(format!("Final Decision: {}", aggregation_reason));
        reasoning_steps.push(ReasoningStep::new(
            "Signal Aggregation",
            format!("Scores: {:?}", scores),
            "Combining all analysis factors for final decision",
            &aggregation_reason,
            avg_confidence,
        ));

        Ok(CoTSignal {
            symbol: symbol.to_string(),
            current_price,
            signal_type,
            confidence: avg_confidence,
            stop_loss,
            take_profit,
            reasoning_chain,
            reasoning_steps,
        })
    }

    fn analyze_trend(
        &self,
        price: f64,
        sma_20: f64,
        sma_50: f64,
    ) -> (i32, f64, String) {
        let above_sma20 = price > sma_20;
        let above_sma50 = price > sma_50;
        let sma20_above_sma50 = sma_20 > sma_50;

        let (score, confidence, reason) = match (above_sma20, above_sma50, sma20_above_sma50) {
            (true, true, true) => (2, 0.85, "Strong uptrend: price above both SMAs, golden cross"),
            (true, true, false) => (1, 0.7, "Moderate uptrend: price above SMAs but no golden cross"),
            (true, false, _) => (0, 0.5, "Mixed trend: price between SMAs"),
            (false, false, false) => (-2, 0.85, "Strong downtrend: price below both SMAs, death cross"),
            (false, false, true) => (-1, 0.7, "Moderate downtrend: price below SMAs but golden cross exists"),
            (false, true, _) => (0, 0.5, "Unusual pattern: below SMA20 but above SMA50"),
        };

        (score, confidence, reason.to_string())
    }

    fn analyze_momentum(
        &self,
        rsi: f64,
        macd: f64,
        macd_signal: f64,
    ) -> (i32, f64, String) {
        let rsi_oversold = rsi < 30.0;
        let rsi_overbought = rsi > 70.0;
        let macd_bullish = macd > macd_signal;

        let (score, confidence, reason) = if rsi_oversold && macd_bullish {
            (2, 0.8, format!("Strong bullish: RSI oversold ({:.1}), MACD bullish crossover", rsi))
        } else if rsi_overbought && !macd_bullish {
            (-2, 0.8, format!("Strong bearish: RSI overbought ({:.1}), MACD bearish", rsi))
        } else if rsi < 40.0 && macd_bullish {
            (1, 0.7, format!("Bullish: RSI low ({:.1}) with MACD confirmation", rsi))
        } else if rsi > 60.0 && !macd_bullish {
            (-1, 0.7, format!("Bearish: RSI high ({:.1}) with MACD weakness", rsi))
        } else {
            (0, 0.5, format!("Neutral momentum: RSI {:.1}, MACD {}", rsi, if macd_bullish { "bullish" } else { "bearish" }))
        };

        (score, confidence, reason)
    }

    fn analyze_volume(&self, volume_ratio: f64) -> (i32, f64, String) {
        let (score, confidence, reason) = if volume_ratio > 2.0 {
            (1, 0.8, format!("High volume ({:.1}x) confirms price action", volume_ratio))
        } else if volume_ratio > 1.2 {
            (1, 0.6, format!("Above average volume ({:.1}x) supports move", volume_ratio))
        } else if volume_ratio < 0.5 {
            (0, 0.4, format!("Low volume ({:.1}x) - weak confirmation", volume_ratio))
        } else {
            (0, 0.5, format!("Normal volume ({:.1}x)", volume_ratio))
        };

        (score, confidence, reason)
    }

    fn calculate_risk_reward(
        &self,
        price: f64,
        atr: f64,
    ) -> (f64, f64, String) {
        let stop_loss = price - (atr * self.risk_multiplier);
        let take_profit = price + (atr * self.reward_multiplier);

        let risk = price - stop_loss;
        let reward = take_profit - price;
        let ratio = reward / risk;

        let reason = format!(
            "Stop loss at {:.2} (risk {:.2}), Take profit at {:.2} (reward {:.2}), R:R = 1:{:.1}",
            stop_loss, risk, take_profit, reward, ratio
        );

        (stop_loss, take_profit, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_signal_generation() {
        let generator = SignalGenerator::new_mock();

        let signal = generator.generate(
            "AAPL",
            150.0,
            45.0,   // RSI
            0.5,    // MACD
            0.3,    // MACD signal
            148.0,  // SMA 20
            145.0,  // SMA 50
            1.2,    // volume ratio
            2.5,    // ATR
        ).await.unwrap();

        assert_eq!(signal.symbol, "AAPL");
        assert_eq!(signal.current_price, 150.0);
        assert!(!signal.reasoning_chain.is_empty());
        assert!(signal.stop_loss < signal.current_price);
        assert!(signal.take_profit > signal.current_price);
    }

    #[test]
    fn test_signal_score_conversion() {
        assert_eq!(Signal::StrongBuy.to_score(), 2);
        assert_eq!(Signal::from_score(2), Signal::StrongBuy);
        assert_eq!(Signal::from_score(0), Signal::Hold);
        assert_eq!(Signal::from_score(-5), Signal::StrongSell);
    }
}
