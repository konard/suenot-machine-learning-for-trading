//! Basic earnings call analysis example
//!
//! This example demonstrates how to analyze an earnings call transcript
//! and generate trading signals.

use earnings_call_analyzer::{SignalGenerator, EarningsAnalyzer};

fn main() {
    println!("=== Earnings Call Analysis Example ===\n");

    // Sample earnings call transcript
    let bullish_transcript = r#"
John Smith - CEO:
Good afternoon, and thank you for joining us today.

I am pleased to report that we delivered exceptional results this quarter.
Revenue grew 28% year over year, significantly exceeding our guidance.
Our core business showed strong momentum across all segments.

We are raising our full-year guidance based on this performance.
The team executed flawlessly on our strategic initiatives.

Jane Doe - CFO:
Thank you, John. Let me walk through the financials.

Total revenue came in at $2.4 billion, up 28% year over year.
Gross margin expanded 200 basis points to 72%.
Operating income increased 35% driven by strong leverage.

We are confident in our ability to sustain this growth trajectory.

Question-and-Answer Session

Analyst - Goldman Sachs:
Congratulations on the strong quarter. Can you talk about the pipeline?

John Smith - CEO:
Absolutely. Our pipeline is the strongest it has ever been.
We are seeing robust demand across all customer segments.
We are committed to continued execution and delivery.

Analyst - Morgan Stanley:
What gives you confidence in raising guidance?

Jane Doe - CFO:
We have clear visibility into our business for the remainder of the year.
Backlog is at record levels, and churn remains at historic lows.
We are well positioned to deliver on our raised targets.
    "#;

    let bearish_transcript = r#"
Mike Johnson - CEO:
Good afternoon. This was a challenging quarter for the company.

We faced significant headwinds from macroeconomic uncertainty.
Revenue declined 8% year over year, below our expectations.
Market conditions were more difficult than we anticipated.

We are taking a cautious approach to the outlook given the environment.

Sarah Williams - CFO:
I'll provide additional detail on the quarter.

Revenue was $1.8 billion, down 8% year over year.
Gross margin contracted 150 basis points due to cost pressures.
We incurred restructuring charges related to workforce reductions.

We are lowering our full-year guidance to reflect current conditions.

Question-and-Answer Session

Analyst - JP Morgan:
Can you help us understand the drivers of the weakness?

Mike Johnson - CEO:
We saw softness across several markets, particularly in enterprise.
The macro environment created unexpected challenges.
We are uncertain about the timing of recovery.

Analyst - Bank of America:
How should we think about margins going forward?

Sarah Williams - CFO:
We expect continued pressure in the near term.
We are implementing cost reduction measures but results may take time.
The situation remains volatile and difficult to predict.
    "#;

    // Create analyzer and signal generator
    let analyzer = EarningsAnalyzer::new();
    let signal_gen = SignalGenerator::new();

    // Analyze bullish transcript
    println!("--- Bullish Transcript Analysis ---\n");
    let bullish_analysis = analyzer.analyze(bullish_transcript);

    println!("Overall Sentiment:");
    println!("  Net Score: {:.3}", bullish_analysis.overall_sentiment.net_sentiment);
    println!("  Positive: {:.3}", bullish_analysis.overall_sentiment.positive_score);
    println!("  Negative: {:.3}", bullish_analysis.overall_sentiment.negative_score);
    println!("  Hedging: {:.3}", bullish_analysis.overall_sentiment.hedging_score);

    println!("\nConfidence: {:.1}%", bullish_analysis.confidence.overall * 100.0);
    println!("Guidance: {}", bullish_analysis.guidance.direction);

    let bullish_signal = signal_gen.generate_signal(bullish_transcript);
    println!("\nTrading Signal:");
    println!("{}", bullish_signal);

    // Analyze bearish transcript
    println!("\n--- Bearish Transcript Analysis ---\n");
    let bearish_analysis = analyzer.analyze(bearish_transcript);

    println!("Overall Sentiment:");
    println!("  Net Score: {:.3}", bearish_analysis.overall_sentiment.net_sentiment);
    println!("  Positive: {:.3}", bearish_analysis.overall_sentiment.positive_score);
    println!("  Negative: {:.3}", bearish_analysis.overall_sentiment.negative_score);
    println!("  Hedging: {:.3}", bearish_analysis.overall_sentiment.hedging_score);

    println!("\nConfidence: {:.1}%", bearish_analysis.confidence.overall * 100.0);
    println!("Guidance: {}", bearish_analysis.guidance.direction);

    let bearish_signal = signal_gen.generate_signal(bearish_transcript);
    println!("\nTrading Signal:");
    println!("{}", bearish_signal);

    println!("\n=== Analysis Complete ===");
}
