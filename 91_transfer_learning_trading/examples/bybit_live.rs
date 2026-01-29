//! Bybit Live Trading Example
//!
//! Demonstrates transfer learning with real-time data from Bybit:
//! 1. Fetch source domain data (BTC/USDT, ETH/USDT)
//! 2. Fetch target domain data (a different trading pair)
//! 3. Pre-train on source, adapt to target
//! 4. Generate trading signals

use transfer_learning_trading::{
    TransferNetwork, FineTuneStrategy, TransferTradingStrategy, TradingSignal,
};

#[cfg(feature = "bybit")]
use transfer_learning_trading::data::{BybitClient, BybitConfig, KlineInterval, DataFeatureExtractor};

#[cfg(feature = "bybit")]
use transfer_learning_trading::training::{TransferTrainer, TrainingConfig, AdaptationConfig};
use transfer_learning_trading::network::AdaptationMethod;

#[cfg(feature = "bybit")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== Bybit Live Transfer Learning Example ===\n");

    // Step 1: Initialize Bybit client
    let config = BybitConfig::default();
    let client = BybitClient::new(config);

    // Step 2: Fetch source domain data (established pairs)
    println!("Fetching source domain data...");
    let btc_bars = client.fetch_klines("BTCUSDT", KlineInterval::Hour1, 200).await?;
    let eth_bars = client.fetch_klines("ETHUSDT", KlineInterval::Hour1, 200).await?;
    println!("  BTC/USDT: {} bars", btc_bars.len());
    println!("  ETH/USDT: {} bars", eth_bars.len());

    // Step 3: Fetch target domain data (less established pair)
    println!("\nFetching target domain data...");
    let target_bars = client.fetch_klines("SOLUSDT", KlineInterval::Hour1, 200).await?;
    println!("  SOL/USDT: {} bars", target_bars.len());

    // Step 4: Extract features
    println!("\nExtracting features...");
    let feat_ext = DataFeatureExtractor::new(20);

    // Combine source data
    let mut all_source_bars = btc_bars;
    all_source_bars.extend(eth_bars);

    let source_features = feat_ext.extract_features(&all_source_bars);
    let source_labels = feat_ext.generate_labels(&all_source_bars, 3, 0.003);
    let target_features = feat_ext.extract_features(&target_bars);
    let target_labels = feat_ext.generate_labels(&target_bars, 3, 0.003);

    let n_source = source_features.nrows().min(source_labels.len());
    let n_target = target_features.nrows().min(target_labels.len());

    if n_source == 0 || n_target == 0 {
        println!("Not enough data for training.");
        return Ok(());
    }

    let source_features = source_features.slice(ndarray::s![..n_source, ..]).to_owned();
    let source_labels = &source_labels[..n_source];
    let target_features = target_features.slice(ndarray::s![..n_target, ..]).to_owned();
    let target_labels = &target_labels[..n_target];

    println!("  Source: {} samples x {} features", n_source, feat_ext.feature_dim());
    println!("  Target: {} samples x {} features", n_target, feat_ext.feature_dim());

    // Step 5: Transfer learning pipeline
    println!("\nRunning transfer learning pipeline...");
    let network = TransferNetwork::new(feat_ext.feature_dim(), 64, 32, true)
        .with_adaptation_method(AdaptationMethod::MMD);

    let mut trainer = TransferTrainer::new(network)
        .with_pretrain_config(TrainingConfig {
            epochs: 30,
            learning_rate: 0.01,
            ..Default::default()
        })
        .with_adapt_config(AdaptationConfig {
            epochs: 15,
            learning_rate: 0.001,
            method: AdaptationMethod::MMD,
            strategy: FineTuneStrategy::PartialFineTune,
            ..Default::default()
        });

    let (pretrain_result, adapt_result) = trainer.run_pipeline(
        &source_features,
        source_labels,
        &target_features,
        Some(target_labels),
    );

    println!("  Pre-training: {} epochs, loss: {:.4}",
        pretrain_result.epochs_trained, pretrain_result.final_loss);
    println!("  Adaptation: {} epochs, loss: {:.6}",
        adapt_result.epochs_trained, adapt_result.final_loss);

    // Step 6: Domain similarity assessment
    let domain_similarity = trainer.compute_domain_similarity(&source_features, &target_features);
    println!("\nDomain similarity: {:.4}", domain_similarity);

    if domain_similarity < 0.5 {
        println!("WARNING: Low domain similarity. Transfer may not be effective.");
        println!("Consider using a different source domain or adaptation method.");
    }

    // Step 7: Generate current trading signal
    println!("\nGenerating trading signal for SOL/USDT...");
    let ticker = client.fetch_ticker("SOLUSDT").await?;
    println!("  Current price: ${:.2}", ticker.last_price);
    println!("  24h volume: {:.0}", ticker.volume_24h);

    // Get latest features from target
    if let Some(latest_features) = feat_ext.extract_single(&target_bars) {
        let (class, confidence) = trainer.network().predict_single(&latest_features.features);
        let signal = TradingSignal::from_class(class);

        let signal_str = match signal {
            TradingSignal::Buy => "BUY",
            TradingSignal::Sell => "SELL",
            TradingSignal::Hold => "HOLD",
        };

        println!("\n  Signal: {} (confidence: {:.2}%)", signal_str, confidence * 100.0);
        println!("  Domain similarity: {:.2}%", domain_similarity * 100.0);

        if confidence < 0.65 {
            println!("  NOTE: Confidence below threshold (65%). Signal may not be reliable.");
        }

        // Step 8: Strategy recommendation
        let mut strategy = TransferTradingStrategy::new(100_000.0)
            .with_confidence_threshold(0.65)
            .with_stop_loss(0.015);

        let proba = trainer.network().predict_proba(
            &latest_features.features.clone().insert_axis(ndarray::Axis(0)),
        );
        let action = strategy.generate_signal(
            &proba.row(0).to_owned(),
            ticker.last_price,
            domain_similarity,
        );

        let action_str = match action {
            TradingSignal::Buy => "OPEN LONG",
            TradingSignal::Sell => "OPEN SHORT",
            TradingSignal::Hold => "NO ACTION",
        };

        println!("\n  Strategy action: {}", action_str);
        println!("  Max position size: {:.2}% of portfolio", 2.0);
        println!("  Stop-loss: {:.2}%", 1.5);
    }

    println!("\n=== Bybit Live Example Complete ===");
    Ok(())
}

#[cfg(not(feature = "bybit"))]
fn main() {
    println!("=== Bybit Live Transfer Learning Example ===\n");
    println!("This example requires the 'bybit' feature flag.");
    println!("Run with: cargo run --example bybit_live --features bybit");

    // Demo with synthetic data
    println!("\nRunning with synthetic data instead...\n");

    use transfer_learning_trading::data::StockDataLoader;
    let loader = StockDataLoader::new();
    let source_bars = loader.generate_synthetic(200, 45000.0, 0.02);
    let target_bars = loader.generate_synthetic(50, 150.0, 0.03);

    println!("Generated {} source bars (simulating BTC)", source_bars.len());
    println!("Generated {} target bars (simulating SOL)", target_bars.len());
    println!("\nTo use real Bybit data, enable the 'bybit' feature flag.");
    println!("\n=== Example Complete ===");
}
