//! Bybit Real-Time Multi-Task Prediction Example
//!
//! This example demonstrates using the task-agnostic model with real
//! market data from Bybit exchange.
//!
//! Run with: cargo run --example bybit_realtime

use task_agnostic_trading::data::{BybitClient, BybitConfig, FeatureExtractor, FeatureConfig};
use task_agnostic_trading::encoder::{EncoderConfig, EncoderType};
use task_agnostic_trading::training::{TrainerConfig, MultiTaskTrainer, HarmonizerType};
use task_agnostic_trading::fusion::{DecisionFusion, FusionConfig, FusionMethod};

#[tokio::main]
async fn main() {
    println!("=== Task-Agnostic Trading: Bybit Real-Time Example ===\n");

    // Initialize Bybit client
    let client = BybitClient::new(BybitConfig::default());
    println!("Connected to Bybit API: {}", client.base_url());
    println!("Testnet: {}\n", client.is_testnet());

    // Define symbols to analyze
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    // Initialize feature extractor
    let feature_config = FeatureConfig {
        ma_windows: vec![5, 10, 20],
        rsi_period: 14,
        atr_period: 14,
        orderbook_depth: 10,
        normalize: true,
    };
    let extractor = FeatureExtractor::new(feature_config);

    // Initialize multi-task model
    let model_config = TrainerConfig::default()
        .with_encoder(
            EncoderConfig::default()
                .with_encoder_type(EncoderType::Transformer)
                .with_input_dim(extractor.kline_feature_dim())
                .with_embedding_dim(64)
        )
        .with_harmonizer(HarmonizerType::PCGrad);

    let model = MultiTaskTrainer::new(model_config);

    // Initialize decision fusion
    let fusion = DecisionFusion::new(FusionConfig {
        method: FusionMethod::WeightedConfidence,
        min_confidence: 0.6,
        ..Default::default()
    });

    println!("Model initialized with {} input features", extractor.kline_feature_dim());
    println!("Analyzing {} symbols...\n", symbols.len());

    // Analyze each symbol
    for symbol in symbols {
        println!("--- {} ---", symbol);

        // Fetch market data
        match fetch_and_analyze(&client, &extractor, &model, &fusion, symbol).await {
            Ok(()) => {}
            Err(e) => {
                println!("Error analyzing {}: {}\n", symbol, e);
            }
        }
    }

    // Multi-timeframe analysis for BTC
    println!("\n--- Multi-Timeframe Analysis: BTCUSDT ---\n");

    let timeframes = [("1", "1 minute"), ("5", "5 minutes"), ("15", "15 minutes"), ("60", "1 hour")];

    for (interval, name) in timeframes {
        match analyze_timeframe(&client, &extractor, &model, &fusion, "BTCUSDT", interval).await {
            Ok(decision) => {
                println!("{}: {} (confidence: {:.1}%, size: {:.1}%)",
                    name, decision.0, decision.1 * 100.0, decision.2 * 100.0);
            }
            Err(e) => {
                println!("{}: Error - {}", name, e);
            }
        }
    }

    println!("\n=== Example Complete ===");
}

async fn fetch_and_analyze(
    client: &BybitClient,
    extractor: &FeatureExtractor,
    model: &MultiTaskTrainer,
    fusion: &DecisionFusion,
    symbol: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Fetch recent klines
    let klines = client.get_klines(symbol, "5", Some(100)).await?;
    println!("Fetched {} klines", klines.len());

    if klines.len() < 30 {
        println!("Not enough data for analysis\n");
        return Ok(());
    }

    // Extract features
    let features = extractor.extract_from_klines(&klines);
    println!("Extracted {} features", features.dim());

    // Fetch order book for additional features
    match client.get_orderbook(symbol, Some(20)).await {
        Ok(orderbook) => {
            let ob_features = extractor.extract_from_orderbook(&orderbook);
            println!("Order book features: {}", ob_features.dim());

            if let Some(spread) = orderbook.spread_pct() {
                println!("Spread: {:.4}%", spread * 100.0);
            }

            let imbalance = orderbook.imbalance(10);
            println!("Order book imbalance (depth 10): {:.2}", imbalance);
        }
        Err(e) => {
            println!("Could not fetch order book: {}", e);
        }
    }

    // Fetch ticker
    match client.get_ticker(symbol).await {
        Ok(ticker) => {
            println!("Price: ${:.2}, 24h change: {:.2}%",
                ticker.last_price, ticker.price_change_pct * 100.0);
            println!("24h volume: ${:.0}M", ticker.turnover_24h / 1_000_000.0);
        }
        Err(e) => {
            println!("Could not fetch ticker: {}", e);
        }
    }

    // Make multi-task prediction
    let prediction = model.predict(&features.features);

    println!("\nMulti-Task Predictions:");

    if let Some(ref dir) = prediction.direction {
        println!("  Direction: {} ({:.1}%)", dir.direction, dir.confidence * 100.0);
    }

    if let Some(ref vol) = prediction.volatility {
        println!("  Volatility: {:.2}% - {} ({:.1}%)",
            vol.volatility_pct, vol.level, vol.confidence * 100.0);
    }

    if let Some(ref regime) = prediction.regime {
        println!("  Regime: {} (risk: {}) ({:.1}%)",
            regime.regime, regime.risk_level, regime.confidence * 100.0);
        println!("  → {}", regime.recommendation);
    }

    if let Some(ref ret) = prediction.returns {
        println!("  Expected return: {:.2}% [{:.2}%, {:.2}%]",
            ret.return_pct, ret.lower_bound, ret.upper_bound);
        println!("  Risk-adjusted: {:.2}", ret.risk_adjusted);
    }

    // Fuse into trading decision
    let result = fusion.fuse(&prediction);

    println!("\nTrading Decision: {}", result.decision);
    println!("  Position size: {:.1}%", result.position_size * 100.0);
    println!("  Overall confidence: {:.1}%", result.confidence.overall * 100.0);
    println!("  Task agreement: {:.1}%", result.confidence.task_agreement * 100.0);
    println!("  Risk-adjusted confidence: {:.1}%", result.confidence.risk_adjusted * 100.0);

    if !result.reasoning.is_empty() {
        println!("  Reasoning:");
        for reason in &result.reasoning {
            println!("    - {}", reason);
        }
    }

    println!();
    Ok(())
}

async fn analyze_timeframe(
    client: &BybitClient,
    extractor: &FeatureExtractor,
    model: &MultiTaskTrainer,
    fusion: &DecisionFusion,
    symbol: &str,
    interval: &str,
) -> Result<(String, f64, f64), Box<dyn std::error::Error>> {
    let klines = client.get_klines(symbol, interval, Some(100)).await?;

    if klines.len() < 30 {
        return Err("Not enough data".into());
    }

    let features = extractor.extract_from_klines(&klines);
    let prediction = model.predict(&features.features);
    let result = fusion.fuse(&prediction);

    Ok((
        format!("{}", result.decision),
        result.confidence.overall,
        result.position_size,
    ))
}
