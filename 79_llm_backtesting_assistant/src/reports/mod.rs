//! Reports module
//!
//! Provides functionality for generating formatted reports from backtesting results
//! and LLM analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::analysis::AnalysisResult;
use crate::backtesting::BacktestResults;
use crate::error::Result;

/// Report format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// Plain text format
    Text,
    /// Markdown format
    Markdown,
    /// JSON format
    Json,
    /// HTML format
    Html,
}

/// Complete report containing backtest results and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Report title
    pub title: String,
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Backtest results
    pub results: BacktestResults,
    /// LLM analysis (if available)
    pub analysis: Option<AnalysisResult>,
}

impl Report {
    /// Create a new report
    pub fn new(results: BacktestResults) -> Self {
        Self {
            title: format!("{} Backtest Report", results.strategy_name),
            generated_at: Utc::now(),
            results,
            analysis: None,
        }
    }

    /// Add LLM analysis to the report
    pub fn with_analysis(mut self, analysis: AnalysisResult) -> Self {
        self.analysis = Some(analysis);
        self
    }

    /// Set a custom title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    /// Generate the report in the specified format
    pub fn generate(&self, format: ReportFormat) -> Result<String> {
        match format {
            ReportFormat::Text => self.generate_text(),
            ReportFormat::Markdown => self.generate_markdown(),
            ReportFormat::Json => self.generate_json(),
            ReportFormat::Html => self.generate_html(),
        }
    }

    fn generate_text(&self) -> Result<String> {
        let mut output = String::new();

        output.push_str(&format!("{}\n", "=".repeat(60)));
        output.push_str(&format!("{}\n", self.title));
        output.push_str(&format!("Generated: {}\n", self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")));
        output.push_str(&format!("{}\n\n", "=".repeat(60)));

        // Strategy info
        output.push_str("STRATEGY INFORMATION\n");
        output.push_str(&format!("{}\n", "-".repeat(40)));
        output.push_str(&format!("Name: {}\n", self.results.strategy_name));
        output.push_str(&format!("Symbol: {}\n", self.results.symbol));
        output.push_str(&format!("Market: {}\n", self.results.market_type));
        output.push_str(&format!("Period: {} to {}\n",
            self.results.start_date.format("%Y-%m-%d"),
            self.results.end_date.format("%Y-%m-%d")
        ));
        output.push_str(&format!("Initial Capital: ${:.2}\n", self.results.initial_capital));
        output.push_str(&format!("Final Capital: ${:.2}\n", self.results.final_capital));
        output.push_str("\n");

        // Performance metrics
        output.push_str("PERFORMANCE METRICS\n");
        output.push_str(&format!("{}\n", "-".repeat(40)));
        let m = &self.results.metrics;
        output.push_str(&format!("Total Return: {:.2}%\n", m.total_return * 100.0));
        output.push_str(&format!("Annualized Return: {:.2}%\n", m.annualized_return * 100.0));
        output.push_str(&format!("Sharpe Ratio: {:.2}\n", m.sharpe_ratio));
        output.push_str(&format!("Sortino Ratio: {:.2}\n", m.sortino_ratio));
        output.push_str(&format!("Calmar Ratio: {:.2}\n", m.calmar_ratio));
        output.push_str(&format!("Max Drawdown: {:.2}%\n", m.max_drawdown * 100.0));
        output.push_str(&format!("Volatility: {:.2}%\n", m.volatility * 100.0));
        output.push_str(&format!("Win Rate: {:.2}%\n", m.win_rate * 100.0));
        output.push_str(&format!("Profit Factor: {:.2}\n", m.profit_factor));
        output.push_str(&format!("Total Trades: {}\n", m.total_trades));
        output.push_str("\n");

        // Analysis
        if let Some(ref analysis) = self.analysis {
            output.push_str("LLM ANALYSIS\n");
            output.push_str(&format!("{}\n", "-".repeat(40)));
            output.push_str(&format!("Provider: {}\n\n", analysis.provider));
            output.push_str(&analysis.analysis);
            output.push_str("\n");
        }

        Ok(output)
    }

    fn generate_markdown(&self) -> Result<String> {
        let mut output = String::new();

        output.push_str(&format!("# {}\n\n", self.title));
        output.push_str(&format!("*Generated: {}*\n\n", self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")));

        // Strategy info
        output.push_str("## Strategy Information\n\n");
        output.push_str(&format!("| Property | Value |\n"));
        output.push_str(&format!("|----------|-------|\n"));
        output.push_str(&format!("| Name | {} |\n", self.results.strategy_name));
        output.push_str(&format!("| Symbol | {} |\n", self.results.symbol));
        output.push_str(&format!("| Market | {} |\n", self.results.market_type));
        output.push_str(&format!("| Period | {} to {} |\n",
            self.results.start_date.format("%Y-%m-%d"),
            self.results.end_date.format("%Y-%m-%d")
        ));
        output.push_str(&format!("| Initial Capital | ${:.2} |\n", self.results.initial_capital));
        output.push_str(&format!("| Final Capital | ${:.2} |\n", self.results.final_capital));
        output.push_str("\n");

        // Performance metrics
        output.push_str("## Performance Metrics\n\n");
        let m = &self.results.metrics;
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Total Return | {:.2}% |\n", m.total_return * 100.0));
        output.push_str(&format!("| Annualized Return | {:.2}% |\n", m.annualized_return * 100.0));
        output.push_str(&format!("| Sharpe Ratio | {:.2} |\n", m.sharpe_ratio));
        output.push_str(&format!("| Sortino Ratio | {:.2} |\n", m.sortino_ratio));
        output.push_str(&format!("| Calmar Ratio | {:.2} |\n", m.calmar_ratio));
        output.push_str(&format!("| Max Drawdown | {:.2}% |\n", m.max_drawdown * 100.0));
        output.push_str(&format!("| Volatility | {:.2}% |\n", m.volatility * 100.0));
        output.push_str(&format!("| Win Rate | {:.2}% |\n", m.win_rate * 100.0));
        output.push_str(&format!("| Profit Factor | {:.2} |\n", m.profit_factor));
        output.push_str(&format!("| Total Trades | {} |\n", m.total_trades));
        output.push_str("\n");

        // Trade stats
        output.push_str("### Trade Statistics\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Winning Trades | {} |\n", m.winning_trades));
        output.push_str(&format!("| Losing Trades | {} |\n", m.losing_trades));
        output.push_str(&format!("| Avg Trade Return | ${:.2} |\n", m.avg_trade_return));
        output.push_str(&format!("| Avg Win | ${:.2} |\n", m.avg_win));
        output.push_str(&format!("| Avg Loss | ${:.2} |\n", m.avg_loss));
        output.push_str(&format!("| Largest Win | ${:.2} |\n", m.largest_win));
        output.push_str(&format!("| Largest Loss | ${:.2} |\n", m.largest_loss));
        output.push_str("\n");

        // Parameters
        output.push_str("## Strategy Parameters\n\n");
        output.push_str("```json\n");
        output.push_str(&serde_json::to_string_pretty(&self.results.parameters).unwrap_or_default());
        output.push_str("\n```\n\n");

        // Analysis
        if let Some(ref analysis) = self.analysis {
            output.push_str("## LLM Analysis\n\n");
            output.push_str(&format!("*Provider: {} | Analyzed: {}*\n\n",
                analysis.provider,
                analysis.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
            ));
            output.push_str(&analysis.analysis);
            output.push_str("\n");
        }

        Ok(output)
    }

    fn generate_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self)?)
    }

    fn generate_html(&self) -> Result<String> {
        let mut output = String::new();

        output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        output.push_str("<meta charset=\"UTF-8\">\n");
        output.push_str(&format!("<title>{}</title>\n", self.title));
        output.push_str("<style>\n");
        output.push_str("body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }\n");
        output.push_str("h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }\n");
        output.push_str("h2 { color: #555; margin-top: 30px; }\n");
        output.push_str("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n");
        output.push_str("th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }\n");
        output.push_str("th { background-color: #4CAF50; color: white; }\n");
        output.push_str("tr:nth-child(even) { background-color: #f2f2f2; }\n");
        output.push_str(".metric-positive { color: green; }\n");
        output.push_str(".metric-negative { color: red; }\n");
        output.push_str(".analysis { background-color: #f9f9f9; padding: 20px; border-radius: 5px; }\n");
        output.push_str("pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }\n");
        output.push_str("</style>\n</head>\n<body>\n");

        output.push_str(&format!("<h1>{}</h1>\n", self.title));
        output.push_str(&format!("<p><em>Generated: {}</em></p>\n",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Strategy info
        output.push_str("<h2>Strategy Information</h2>\n");
        output.push_str("<table>\n<tr><th>Property</th><th>Value</th></tr>\n");
        output.push_str(&format!("<tr><td>Name</td><td>{}</td></tr>\n", self.results.strategy_name));
        output.push_str(&format!("<tr><td>Symbol</td><td>{}</td></tr>\n", self.results.symbol));
        output.push_str(&format!("<tr><td>Market</td><td>{}</td></tr>\n", self.results.market_type));
        output.push_str(&format!("<tr><td>Period</td><td>{} to {}</td></tr>\n",
            self.results.start_date.format("%Y-%m-%d"),
            self.results.end_date.format("%Y-%m-%d")
        ));
        output.push_str(&format!("<tr><td>Initial Capital</td><td>${:.2}</td></tr>\n", self.results.initial_capital));
        output.push_str(&format!("<tr><td>Final Capital</td><td>${:.2}</td></tr>\n", self.results.final_capital));
        output.push_str("</table>\n");

        // Performance metrics
        let m = &self.results.metrics;
        output.push_str("<h2>Performance Metrics</h2>\n");
        output.push_str("<table>\n<tr><th>Metric</th><th>Value</th></tr>\n");

        let return_class = if m.total_return >= 0.0 { "metric-positive" } else { "metric-negative" };
        output.push_str(&format!("<tr><td>Total Return</td><td class=\"{}\">{:.2}%</td></tr>\n",
            return_class, m.total_return * 100.0));
        output.push_str(&format!("<tr><td>Annualized Return</td><td class=\"{}\">{:.2}%</td></tr>\n",
            return_class, m.annualized_return * 100.0));
        output.push_str(&format!("<tr><td>Sharpe Ratio</td><td>{:.2}</td></tr>\n", m.sharpe_ratio));
        output.push_str(&format!("<tr><td>Sortino Ratio</td><td>{:.2}</td></tr>\n", m.sortino_ratio));
        output.push_str(&format!("<tr><td>Calmar Ratio</td><td>{:.2}</td></tr>\n", m.calmar_ratio));
        output.push_str(&format!("<tr><td>Max Drawdown</td><td class=\"metric-negative\">{:.2}%</td></tr>\n",
            m.max_drawdown * 100.0));
        output.push_str(&format!("<tr><td>Volatility</td><td>{:.2}%</td></tr>\n", m.volatility * 100.0));
        output.push_str(&format!("<tr><td>Win Rate</td><td>{:.2}%</td></tr>\n", m.win_rate * 100.0));
        output.push_str(&format!("<tr><td>Profit Factor</td><td>{:.2}</td></tr>\n", m.profit_factor));
        output.push_str(&format!("<tr><td>Total Trades</td><td>{}</td></tr>\n", m.total_trades));
        output.push_str("</table>\n");

        // Analysis
        if let Some(ref analysis) = self.analysis {
            output.push_str("<h2>LLM Analysis</h2>\n");
            output.push_str(&format!("<p><em>Provider: {} | Analyzed: {}</em></p>\n",
                analysis.provider,
                analysis.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
            ));
            output.push_str("<div class=\"analysis\">\n");
            // Simple markdown to HTML conversion for analysis
            let html_analysis = analysis.analysis
                .replace("##", "<h3>")
                .replace("\n\n", "</p><p>")
                .replace("**", "<strong>")
                .replace("- ", "<li>");
            output.push_str(&format!("<p>{}</p>\n", html_analysis));
            output.push_str("</div>\n");
        }

        output.push_str("</body>\n</html>\n");

        Ok(output)
    }
}

/// Report builder for creating customized reports
pub struct ReportBuilder {
    results: BacktestResults,
    analysis: Option<AnalysisResult>,
    title: Option<String>,
    include_trades: bool,
    include_equity_curve: bool,
}

impl ReportBuilder {
    /// Create a new report builder
    pub fn new(results: BacktestResults) -> Self {
        Self {
            results,
            analysis: None,
            title: None,
            include_trades: true,
            include_equity_curve: false,
        }
    }

    /// Add LLM analysis
    pub fn with_analysis(mut self, analysis: AnalysisResult) -> Self {
        self.analysis = Some(analysis);
        self
    }

    /// Set custom title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    /// Include individual trades in report
    pub fn with_trades(mut self, include: bool) -> Self {
        self.include_trades = include;
        self
    }

    /// Include equity curve data
    pub fn with_equity_curve(mut self, include: bool) -> Self {
        self.include_equity_curve = include;
        self
    }

    /// Build the report
    pub fn build(self) -> Report {
        let mut report = Report::new(self.results);

        if let Some(analysis) = self.analysis {
            report = report.with_analysis(analysis);
        }

        if let Some(title) = self.title {
            report = report.with_title(title);
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_report_generation() {
        let results = BacktestResults::sample();
        let report = Report::new(results);
        let text = report.generate(ReportFormat::Text).unwrap();

        assert!(text.contains("STRATEGY INFORMATION"));
        assert!(text.contains("PERFORMANCE METRICS"));
    }

    #[test]
    fn test_markdown_report_generation() {
        let results = BacktestResults::sample();
        let report = Report::new(results);
        let md = report.generate(ReportFormat::Markdown).unwrap();

        assert!(md.contains("## Strategy Information"));
        assert!(md.contains("| Metric | Value |"));
    }

    #[test]
    fn test_json_report_generation() {
        let results = BacktestResults::sample();
        let report = Report::new(results);
        let json = report.generate(ReportFormat::Json).unwrap();

        // Should be valid JSON
        let _: Report = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_report_builder() {
        let results = BacktestResults::sample();
        let report = ReportBuilder::new(results)
            .with_title("Custom Report".to_string())
            .build();

        assert_eq!(report.title, "Custom Report");
    }
}
