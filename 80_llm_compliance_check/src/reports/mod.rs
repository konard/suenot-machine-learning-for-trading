//! Report generation for compliance results

use crate::data::{ComplianceResult, ComplianceStatus, ViolationSeverity};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compliance metrics for monitoring and reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMetrics {
    /// Total activities checked
    pub total_checks: u64,

    /// Auto-approved count
    pub auto_approved: u64,

    /// Rejected count
    pub rejected: u64,

    /// Manual review required count
    pub review_required: u64,

    /// Average confidence score
    pub avg_confidence: f64,

    /// Average check latency (milliseconds)
    pub avg_latency_ms: f64,

    /// Violations by severity
    pub violations_by_severity: HashMap<String, u64>,

    /// Violations by rule
    pub violations_by_rule: HashMap<String, u64>,

    /// Period start
    pub period_start: DateTime<Utc>,

    /// Period end
    pub period_end: DateTime<Utc>,
}

impl ComplianceMetrics {
    /// Create new metrics for a period
    pub fn new(period_start: DateTime<Utc>, period_end: DateTime<Utc>) -> Self {
        Self {
            total_checks: 0,
            auto_approved: 0,
            rejected: 0,
            review_required: 0,
            avg_confidence: 0.0,
            avg_latency_ms: 0.0,
            violations_by_severity: HashMap::new(),
            violations_by_rule: HashMap::new(),
            period_start,
            period_end,
        }
    }

    /// Record a compliance result
    pub fn record(&mut self, result: &ComplianceResult, latency_ms: f64) {
        self.total_checks += 1;

        // Update status counts
        match result.status {
            ComplianceStatus::Approved => self.auto_approved += 1,
            ComplianceStatus::Rejected => self.rejected += 1,
            ComplianceStatus::ReviewRequired => self.review_required += 1,
            ComplianceStatus::Pending => {}
        }

        // Update average confidence (rolling average)
        let n = self.total_checks as f64;
        self.avg_confidence = self.avg_confidence * (n - 1.0) / n + result.confidence / n;

        // Update average latency (rolling average)
        self.avg_latency_ms = self.avg_latency_ms * (n - 1.0) / n + latency_ms / n;

        // Count violations
        for violation in &result.violations {
            let severity_key = format!("{:?}", violation.severity);
            *self.violations_by_severity.entry(severity_key).or_insert(0) += 1;
            *self
                .violations_by_rule
                .entry(violation.rule.clone())
                .or_insert(0) += 1;
        }
    }

    /// Calculate approval rate
    pub fn approval_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.auto_approved as f64 / self.total_checks as f64
    }

    /// Calculate rejection rate
    pub fn rejection_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.rejected as f64 / self.total_checks as f64
    }

    /// Get top violations by count
    pub fn top_violations(&self, n: usize) -> Vec<(&String, &u64)> {
        let mut violations: Vec<_> = self.violations_by_rule.iter().collect();
        violations.sort_by(|a, b| b.1.cmp(a.1));
        violations.into_iter().take(n).collect()
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let top_violations: String = self
            .top_violations(5)
            .iter()
            .map(|(rule, count)| format!("  - {}: {}", rule, count))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"
Compliance Summary ({} to {})
================================
Total Checks: {}
Auto-Approved: {} ({:.1}%)
Rejected: {} ({:.1}%)
Review Required: {} ({:.1}%)
Average Confidence: {:.2}
Average Latency: {:.1}ms

Top Violations:
{}
"#,
            self.period_start.format("%Y-%m-%d"),
            self.period_end.format("%Y-%m-%d"),
            self.total_checks,
            self.auto_approved,
            self.approval_rate() * 100.0,
            self.rejected,
            self.rejection_rate() * 100.0,
            self.review_required,
            (self.review_required as f64 / self.total_checks.max(1) as f64) * 100.0,
            self.avg_confidence,
            self.avg_latency_ms,
            if top_violations.is_empty() {
                "  (none)".to_string()
            } else {
                top_violations
            },
        )
    }
}

/// Report generator for compliance results
#[derive(Debug, Clone)]
pub struct ReportGenerator;

impl ReportGenerator {
    /// Create a new report generator
    pub fn new() -> Self {
        Self
    }

    /// Generate a detailed report for a single compliance result
    pub fn generate_detail_report(&self, result: &ComplianceResult) -> String {
        let status_symbol = match result.status {
            ComplianceStatus::Approved => "✅",
            ComplianceStatus::Rejected => "🚫",
            ComplianceStatus::ReviewRequired => "⚠️",
            ComplianceStatus::Pending => "⏳",
        };

        let violations_text = if result.violations.is_empty() {
            "  (none)".to_string()
        } else {
            result
                .violations
                .iter()
                .map(|v| {
                    let severity_symbol = match v.severity {
                        ViolationSeverity::Low => "🟡",
                        ViolationSeverity::Medium => "🟠",
                        ViolationSeverity::High => "🔴",
                        ViolationSeverity::Critical => "⛔",
                    };
                    format!(
                        "  {} [{}] {}: {}\n      Evidence: {:?}",
                        severity_symbol, v.rule, v.severity as i32, v.description, v.evidence
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let recommendations_text = if result.recommendations.is_empty() {
            "  (none)".to_string()
        } else {
            result
                .recommendations
                .iter()
                .map(|r| format!("  • {}", r))
                .collect::<Vec<_>>()
                .join("\n")
        };

        format!(
            r#"
================================================================================
                        COMPLIANCE CHECK REPORT
================================================================================

Activity ID: {}
Audit ID: {}
Checked At: {}

Status: {} {:?}
Confidence: {:.1}%

--------------------------------------------------------------------------------
EXPLANATION
--------------------------------------------------------------------------------
{}

--------------------------------------------------------------------------------
VIOLATIONS
--------------------------------------------------------------------------------
{}

--------------------------------------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------------------------------------
{}

--------------------------------------------------------------------------------
REGULATIONS CHECKED
--------------------------------------------------------------------------------
{}

================================================================================
"#,
            result.activity_id,
            result.audit_id,
            result.checked_at.format("%Y-%m-%d %H:%M:%S UTC"),
            status_symbol,
            result.status,
            result.confidence * 100.0,
            result.explanation,
            violations_text,
            recommendations_text,
            result.regulations_checked.join(", "),
        )
    }

    /// Generate JSON report for a compliance result
    pub fn generate_json_report(&self, result: &ComplianceResult) -> String {
        serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generate CSV row for a compliance result
    pub fn generate_csv_row(&self, result: &ComplianceResult) -> String {
        format!(
            "{},{},{:?},{},{},{},{},{}",
            result.activity_id,
            result.audit_id,
            result.status,
            result.confidence,
            result.violations.len(),
            result
                .violations
                .iter()
                .any(|v| v.severity == ViolationSeverity::Critical),
            result.checked_at.to_rfc3339(),
            result.regulations_checked.join(";"),
        )
    }

    /// Get CSV header
    pub fn csv_header() -> &'static str {
        "activity_id,audit_id,status,confidence,violation_count,has_critical,checked_at,regulations"
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Violation;

    #[test]
    fn test_metrics_recording() {
        let mut metrics = ComplianceMetrics::new(Utc::now(), Utc::now());

        let result1 = ComplianceResult::approved("order_001", "Approved");
        let result2 = ComplianceResult::rejected(
            "order_002",
            "Rejected",
            vec![Violation::new("RULE_1", "Violation", ViolationSeverity::High)],
        );

        metrics.record(&result1, 50.0);
        metrics.record(&result2, 100.0);

        assert_eq!(metrics.total_checks, 2);
        assert_eq!(metrics.auto_approved, 1);
        assert_eq!(metrics.rejected, 1);
        assert_eq!(metrics.approval_rate(), 0.5);
    }

    #[test]
    fn test_report_generation() {
        let generator = ReportGenerator::new();
        let result = ComplianceResult::approved("order_001", "Trade approved");

        let report = generator.generate_detail_report(&result);
        assert!(report.contains("order_001"));
        assert!(report.contains("✅"));

        let json = generator.generate_json_report(&result);
        assert!(json.contains("order_001"));

        let csv = generator.generate_csv_row(&result);
        assert!(csv.contains("order_001"));
    }
}
