//! Decision Fusion Example
//!
//! This example demonstrates how different fusion methods combine
//! multi-task predictions into trading decisions.
//!
//! Run with: cargo run --example decision_fusion

use task_agnostic_trading::tasks::{
    MultiTaskPrediction,
    Direction, DirectionPrediction,
    VolatilityPrediction, VolatilityLevel,
    RegimePrediction, MarketRegime,
    ReturnsPrediction,
};
use task_agnostic_trading::fusion::{DecisionFusion, FusionConfig, FusionMethod};

fn main() {
    println!("=== Task-Agnostic Trading: Decision Fusion Example ===\n");

    // Define market scenarios
    let scenarios = vec![
        ("Strong Bullish", create_bullish_scenario()),
        ("Strong Bearish", create_bearish_scenario()),
        ("Mixed Signals", create_mixed_scenario()),
        ("High Volatility", create_volatile_scenario()),
        ("Crash Warning", create_crash_scenario()),
        ("Low Confidence", create_low_confidence_scenario()),
    ];

    // Define fusion methods to compare
    let fusion_methods = [
        (FusionMethod::Voting, "Voting"),
        (FusionMethod::WeightedConfidence, "Weighted Confidence"),
        (FusionMethod::Bayesian, "Bayesian"),
        (FusionMethod::RuleBased, "Rule-Based"),
    ];

    // Test each scenario with each fusion method
    for (scenario_name, prediction) in &scenarios {
        println!("=== Scenario: {} ===\n", scenario_name);
        print_prediction(prediction);
        println!();

        for (method, method_name) in fusion_methods {
            let fusion = DecisionFusion::new(FusionConfig {
                method,
                min_confidence: 0.5,
                ..Default::default()
            });

            let result = fusion.fuse(prediction);

            println!("{:20} | {} | Size: {:5.1}% | Confidence: {:5.1}%",
                method_name,
                format!("{:5}", format!("{}", result.decision)),
                result.position_size * 100.0,
                result.confidence.overall * 100.0
            );
        }

        println!("\n{}\n", "-".repeat(70));
    }

    // Demonstrate confidence breakdown
    println!("=== Confidence Breakdown Analysis ===\n");

    let bullish = create_bullish_scenario();
    let fusion = DecisionFusion::new(FusionConfig::default());
    let result = fusion.fuse(&bullish);

    println!("Scenario: Strong Bullish");
    println!("Decision: {}", result.decision);
    println!("\nConfidence Breakdown:");
    println!("  Overall:           {:.1}%", result.confidence.overall * 100.0);
    println!("  Direction:         {:.1}%", result.confidence.direction_confidence * 100.0);
    println!("  Volatility:        {:.1}%", result.confidence.volatility_confidence * 100.0);
    println!("  Regime:            {:.1}%", result.confidence.regime_confidence * 100.0);
    println!("  Returns:           {:.1}%", result.confidence.returns_confidence * 100.0);
    println!("  Task Agreement:    {:.1}%", result.confidence.task_agreement * 100.0);
    println!("  Risk-Adjusted:     {:.1}%", result.confidence.risk_adjusted * 100.0);

    println!("\nReasoning:");
    for reason in &result.reasoning {
        println!("  - {}", reason);
    }

    // Position sizing analysis
    println!("\n=== Position Sizing Across Scenarios ===\n");

    let fusion = DecisionFusion::new(FusionConfig {
        method: FusionMethod::WeightedConfidence,
        min_confidence: 0.4,
        ..Default::default()
    });

    println!("{:20} | {:6} | {:>8} | {:>8} | {:>10}",
        "Scenario", "Signal", "Size", "Conf", "Risk-Adj");
    println!("{}", "-".repeat(60));

    for (name, prediction) in &scenarios {
        let result = fusion.fuse(prediction);
        println!("{:20} | {:6} | {:>7.1}% | {:>7.1}% | {:>9.1}%",
            name,
            format!("{}", result.decision),
            result.position_size * 100.0,
            result.confidence.overall * 100.0,
            result.confidence.risk_adjusted * 100.0
        );
    }

    println!("\n=== Example Complete ===");
}

fn print_prediction(prediction: &MultiTaskPrediction) {
    if let Some(ref dir) = prediction.direction {
        println!("Direction: {} ({:.1}%)", dir.direction, dir.confidence * 100.0);
    }
    if let Some(ref vol) = prediction.volatility {
        println!("Volatility: {:.2}% - {} ({:.1}%)",
            vol.volatility_pct, vol.level, vol.confidence * 100.0);
    }
    if let Some(ref regime) = prediction.regime {
        println!("Regime: {} (risk {}) ({:.1}%)",
            regime.regime, regime.risk_level, regime.confidence * 100.0);
    }
    if let Some(ref ret) = prediction.returns {
        println!("Returns: {:.2}% ({:.1}%)", ret.return_pct, ret.confidence * 100.0);
    }
}

fn create_bullish_scenario() -> MultiTaskPrediction {
    let mut pred = MultiTaskPrediction::new();

    pred.direction = Some(DirectionPrediction {
        direction: Direction::Up,
        confidence: 0.85,
        probabilities: [0.85, 0.10, 0.05],
    });

    pred.volatility = Some(VolatilityPrediction {
        volatility_pct: 1.5,
        level: VolatilityLevel::Low,
        confidence: 0.75,
        lower_bound: 1.0,
        upper_bound: 2.0,
    });

    pred.regime = Some(RegimePrediction {
        regime: MarketRegime::Trending,
        confidence: 0.80,
        probabilities: vec![0.80, 0.10, 0.05, 0.03, 0.02],
        risk_level: 2,
        recommendation: "Follow the trend".to_string(),
    });

    pred.returns = Some(ReturnsPrediction {
        return_pct: 3.5,
        confidence: 0.70,
        lower_bound: 1.5,
        upper_bound: 5.5,
        risk_adjusted: 1.8,
    });

    pred
}

fn create_bearish_scenario() -> MultiTaskPrediction {
    let mut pred = MultiTaskPrediction::new();

    pred.direction = Some(DirectionPrediction {
        direction: Direction::Down,
        confidence: 0.82,
        probabilities: [0.10, 0.82, 0.08],
    });

    pred.volatility = Some(VolatilityPrediction {
        volatility_pct: 2.5,
        level: VolatilityLevel::Medium,
        confidence: 0.70,
        lower_bound: 1.5,
        upper_bound: 3.5,
    });

    pred.regime = Some(RegimePrediction {
        regime: MarketRegime::Trending,
        confidence: 0.75,
        probabilities: vec![0.75, 0.10, 0.08, 0.05, 0.02],
        risk_level: 2,
        recommendation: "Follow the trend".to_string(),
    });

    pred.returns = Some(ReturnsPrediction {
        return_pct: -2.8,
        confidence: 0.68,
        lower_bound: -4.5,
        upper_bound: -1.0,
        risk_adjusted: -1.5,
    });

    pred
}

fn create_mixed_scenario() -> MultiTaskPrediction {
    let mut pred = MultiTaskPrediction::new();

    pred.direction = Some(DirectionPrediction {
        direction: Direction::Up,
        confidence: 0.55, // Low confidence
        probabilities: [0.55, 0.30, 0.15],
    });

    pred.volatility = Some(VolatilityPrediction {
        volatility_pct: 2.0,
        level: VolatilityLevel::Medium,
        confidence: 0.60,
        lower_bound: 1.0,
        upper_bound: 3.0,
    });

    pred.regime = Some(RegimePrediction {
        regime: MarketRegime::Ranging,
        confidence: 0.65,
        probabilities: vec![0.20, 0.65, 0.10, 0.03, 0.02],
        risk_level: 1,
        recommendation: "Trade range bounds".to_string(),
    });

    pred.returns = Some(ReturnsPrediction {
        return_pct: -0.5, // Conflicting with direction
        confidence: 0.50,
        lower_bound: -2.0,
        upper_bound: 1.0,
        risk_adjusted: -0.3,
    });

    pred
}

fn create_volatile_scenario() -> MultiTaskPrediction {
    let mut pred = MultiTaskPrediction::new();

    pred.direction = Some(DirectionPrediction {
        direction: Direction::Up,
        confidence: 0.70,
        probabilities: [0.70, 0.20, 0.10],
    });

    pred.volatility = Some(VolatilityPrediction {
        volatility_pct: 5.5,
        level: VolatilityLevel::Extreme,
        confidence: 0.80,
        lower_bound: 3.5,
        upper_bound: 8.0,
    });

    pred.regime = Some(RegimePrediction {
        regime: MarketRegime::Volatile,
        confidence: 0.85,
        probabilities: vec![0.05, 0.05, 0.85, 0.03, 0.02],
        risk_level: 4,
        recommendation: "Reduce position size".to_string(),
    });

    pred.returns = Some(ReturnsPrediction {
        return_pct: 2.0,
        confidence: 0.40, // Low confidence due to volatility
        lower_bound: -5.0,
        upper_bound: 9.0,
        risk_adjusted: 0.5,
    });

    pred
}

fn create_crash_scenario() -> MultiTaskPrediction {
    let mut pred = MultiTaskPrediction::new();

    pred.direction = Some(DirectionPrediction {
        direction: Direction::Down,
        confidence: 0.90,
        probabilities: [0.05, 0.90, 0.05],
    });

    pred.volatility = Some(VolatilityPrediction {
        volatility_pct: 8.0,
        level: VolatilityLevel::Extreme,
        confidence: 0.85,
        lower_bound: 5.0,
        upper_bound: 12.0,
    });

    pred.regime = Some(RegimePrediction {
        regime: MarketRegime::Crash,
        confidence: 0.88,
        probabilities: vec![0.02, 0.02, 0.05, 0.88, 0.03],
        risk_level: 5,
        recommendation: "Stay out or hedge".to_string(),
    });

    pred.returns = Some(ReturnsPrediction {
        return_pct: -8.5,
        confidence: 0.75,
        lower_bound: -15.0,
        upper_bound: -2.0,
        risk_adjusted: -2.5,
    });

    pred
}

fn create_low_confidence_scenario() -> MultiTaskPrediction {
    let mut pred = MultiTaskPrediction::new();

    pred.direction = Some(DirectionPrediction {
        direction: Direction::Sideways,
        confidence: 0.40,
        probabilities: [0.30, 0.30, 0.40],
    });

    pred.volatility = Some(VolatilityPrediction {
        volatility_pct: 2.0,
        level: VolatilityLevel::Medium,
        confidence: 0.35,
        lower_bound: 0.5,
        upper_bound: 4.0,
    });

    pred.regime = Some(RegimePrediction {
        regime: MarketRegime::Ranging,
        confidence: 0.38,
        probabilities: vec![0.25, 0.38, 0.22, 0.10, 0.05],
        risk_level: 2,
        recommendation: "Uncertain - wait for clarity".to_string(),
    });

    pred.returns = Some(ReturnsPrediction {
        return_pct: 0.2,
        confidence: 0.30,
        lower_bound: -3.0,
        upper_bound: 3.5,
        risk_adjusted: 0.1,
    });

    pred
}
