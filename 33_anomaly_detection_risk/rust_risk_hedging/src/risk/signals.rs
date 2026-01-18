//! Risk signals and alerts
//!
//! Generates trading signals based on anomaly detection

use crate::anomaly::AnomalyLevel;

/// Risk signal type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskSignal {
    /// No action needed
    None,
    /// Reduce exposure
    ReduceExposure,
    /// Activate hedge
    ActivateHedge,
    /// Emergency exit
    EmergencyExit,
}

impl RiskSignal {
    /// Create from anomaly level
    pub fn from_level(level: AnomalyLevel) -> Self {
        match level {
            AnomalyLevel::Normal => RiskSignal::None,
            AnomalyLevel::Elevated => RiskSignal::ReduceExposure,
            AnomalyLevel::High => RiskSignal::ActivateHedge,
            AnomalyLevel::Extreme => RiskSignal::EmergencyExit,
        }
    }

    /// Get action description
    pub fn description(&self) -> &'static str {
        match self {
            RiskSignal::None => "Market conditions normal. Continue regular trading.",
            RiskSignal::ReduceExposure => "Elevated risk detected. Consider reducing position sizes by 25-50%.",
            RiskSignal::ActivateHedge => "High risk detected. Activate hedging positions (5-10% of portfolio).",
            RiskSignal::EmergencyExit => "Extreme risk! Exit risky positions, move to stablecoins.",
        }
    }

    /// Get urgency level (0-10)
    pub fn urgency(&self) -> u8 {
        match self {
            RiskSignal::None => 0,
            RiskSignal::ReduceExposure => 4,
            RiskSignal::ActivateHedge => 7,
            RiskSignal::EmergencyExit => 10,
        }
    }
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Minimum level to generate alert
    pub min_level: AnomalyLevel,
    /// Cooldown period between alerts (seconds)
    pub cooldown_seconds: u64,
    /// Whether to escalate repeated alerts
    pub escalate_repeated: bool,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            min_level: AnomalyLevel::Elevated,
            cooldown_seconds: 300, // 5 minutes
            escalate_repeated: true,
        }
    }
}

/// Alert generator
#[derive(Debug, Clone)]
pub struct AlertGenerator {
    config: AlertConfig,
    last_alert_time: Option<std::time::Instant>,
    consecutive_alerts: u32,
}

impl AlertGenerator {
    /// Create new alert generator
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            last_alert_time: None,
            consecutive_alerts: 0,
        }
    }

    /// Check if alert should be generated
    pub fn should_alert(&mut self, level: AnomalyLevel) -> bool {
        // Check minimum level
        if self.level_to_number(level) < self.level_to_number(self.config.min_level) {
            self.consecutive_alerts = 0;
            return false;
        }

        // Check cooldown
        if let Some(last_time) = self.last_alert_time {
            let elapsed = last_time.elapsed().as_secs();
            if elapsed < self.config.cooldown_seconds {
                // Reduce cooldown for escalating situations
                if self.config.escalate_repeated && level == AnomalyLevel::Extreme {
                    // Always alert for extreme
                } else {
                    return false;
                }
            }
        }

        self.last_alert_time = Some(std::time::Instant::now());
        self.consecutive_alerts += 1;
        true
    }

    /// Generate alert message
    pub fn generate_alert(&self, level: AnomalyLevel, score: f64) -> Alert {
        let signal = RiskSignal::from_level(level);
        let escalation = if self.consecutive_alerts > 3 {
            " [ESCALATED - Repeated alert]"
        } else {
            ""
        };

        Alert {
            level,
            score,
            signal,
            message: format!(
                "Risk Alert: {} (score: {:.2}){}\n{}",
                self.level_name(level),
                score,
                escalation,
                signal.description()
            ),
            consecutive_count: self.consecutive_alerts,
        }
    }

    fn level_to_number(&self, level: AnomalyLevel) -> u8 {
        match level {
            AnomalyLevel::Normal => 0,
            AnomalyLevel::Elevated => 1,
            AnomalyLevel::High => 2,
            AnomalyLevel::Extreme => 3,
        }
    }

    fn level_name(&self, level: AnomalyLevel) -> &'static str {
        match level {
            AnomalyLevel::Normal => "NORMAL",
            AnomalyLevel::Elevated => "ELEVATED",
            AnomalyLevel::High => "HIGH",
            AnomalyLevel::Extreme => "EXTREME",
        }
    }
}

impl Default for AlertGenerator {
    fn default() -> Self {
        Self::new(AlertConfig::default())
    }
}

/// Alert message
#[derive(Debug, Clone)]
pub struct Alert {
    pub level: AnomalyLevel,
    pub score: f64,
    pub signal: RiskSignal,
    pub message: String,
    pub consecutive_count: u32,
}

impl Alert {
    /// Format for display
    pub fn format_colored(&self) -> String {
        let color = match self.level {
            AnomalyLevel::Normal => "\x1b[32m",    // Green
            AnomalyLevel::Elevated => "\x1b[33m", // Yellow
            AnomalyLevel::High => "\x1b[38;5;208m", // Orange
            AnomalyLevel::Extreme => "\x1b[31m",  // Red
        };
        let reset = "\x1b[0m";

        format!("{}{}{}", color, self.message, reset)
    }
}

/// Signal history for trend analysis
#[derive(Debug, Clone)]
pub struct SignalHistory {
    signals: Vec<(std::time::Instant, RiskSignal, f64)>,
    max_history: usize,
}

impl SignalHistory {
    /// Create new history
    pub fn new(max_history: usize) -> Self {
        Self {
            signals: Vec::new(),
            max_history,
        }
    }

    /// Add a signal
    pub fn add(&mut self, signal: RiskSignal, score: f64) {
        self.signals.push((std::time::Instant::now(), signal, score));

        if self.signals.len() > self.max_history {
            self.signals.remove(0);
        }
    }

    /// Check if risk is escalating
    pub fn is_escalating(&self) -> bool {
        if self.signals.len() < 3 {
            return false;
        }

        let recent: Vec<&(std::time::Instant, RiskSignal, f64)> =
            self.signals.iter().rev().take(5).collect();

        // Check if scores are increasing
        let scores: Vec<f64> = recent.iter().map(|(_, _, s)| *s).collect();
        let mut is_increasing = true;
        for i in 1..scores.len() {
            if scores[i] <= scores[i - 1] {
                is_increasing = false;
                break;
            }
        }

        is_increasing
    }

    /// Get average score over last N signals
    pub fn average_score(&self, n: usize) -> f64 {
        let recent: Vec<f64> = self.signals.iter().rev().take(n).map(|(_, _, s)| *s).collect();

        if recent.is_empty() {
            return 0.0;
        }

        recent.iter().sum::<f64>() / recent.len() as f64
    }

    /// Get trend direction
    pub fn trend(&self) -> SignalTrend {
        if self.signals.len() < 3 {
            return SignalTrend::Stable;
        }

        let old_avg = self.average_score(3);
        let new_avg = self
            .signals
            .iter()
            .rev()
            .take(3)
            .map(|(_, _, s)| *s)
            .sum::<f64>()
            / 3.0;

        let diff = new_avg - old_avg;

        if diff > 0.1 {
            SignalTrend::Worsening
        } else if diff < -0.1 {
            SignalTrend::Improving
        } else {
            SignalTrend::Stable
        }
    }
}

impl Default for SignalHistory {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Signal trend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalTrend {
    Improving,
    Stable,
    Worsening,
}

impl SignalTrend {
    pub fn description(&self) -> &'static str {
        match self {
            SignalTrend::Improving => "Risk levels improving",
            SignalTrend::Stable => "Risk levels stable",
            SignalTrend::Worsening => "Risk levels worsening - stay alert",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_signal() {
        let signal = RiskSignal::from_level(AnomalyLevel::High);
        assert_eq!(signal, RiskSignal::ActivateHedge);
        assert!(signal.urgency() > 5);
    }

    #[test]
    fn test_alert_generator() {
        let mut gen = AlertGenerator::default();

        // First alert should pass
        assert!(gen.should_alert(AnomalyLevel::High));

        // Immediate second should be blocked by cooldown
        assert!(!gen.should_alert(AnomalyLevel::High));

        // But extreme should always pass
        assert!(gen.should_alert(AnomalyLevel::Extreme));
    }
}
