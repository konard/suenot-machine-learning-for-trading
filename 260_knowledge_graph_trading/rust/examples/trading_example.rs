use knowledge_graph_trading::*;
use ndarray::Array1;

#[tokio::main]
async fn main() {
    println!("=== Chapter 260: Knowledge Graph Trading ===\n");

    // ── Step 1: Build Knowledge Graph ──────────────────────────────────
    println!("── Step 1: Build Financial Knowledge Graph ──");
    let kg = build_sample_financial_kg();
    println!("  Entities:  {}", kg.num_entities());
    println!("  Relations: {}", kg.num_relations());
    println!("  Triples:   {}", kg.triples().len());

    // Show some entities and their connections
    for name in &["TSMC", "AAPL", "NVDA", "MSFT"] {
        if let Some(id) = kg.entity_id(name) {
            let neighbors = kg.neighbors(id);
            let connections: Vec<String> = neighbors
                .iter()
                .map(|&(r, t)| {
                    format!(
                        "—[{}]→ {}",
                        kg.relation_names[r],
                        kg.entity_names[t]
                    )
                })
                .collect();
            println!("  {} connections: {}", name, connections.join(", "));
        }
    }

    // ── Step 2: Train TransE Embeddings ────────────────────────────────
    println!("\n── Step 2: Train TransE Embeddings ──");
    let mut model = TransEModel::new(kg.num_entities(), kg.num_relations(), 32, 1.0, 0.01);

    let loss_initial = model.train_epoch(kg.triples(), kg.num_entities());
    println!("  Epoch   1 | loss = {:.4}", loss_initial);
    for epoch in 2..=100 {
        let loss = model.train_epoch(kg.triples(), kg.num_entities());
        if epoch % 20 == 0 {
            println!("  Epoch {:3} | loss = {:.4}", epoch, loss);
        }
    }

    // Show similarities between related companies
    println!("\n  Entity similarities after training:");
    let pairs = [
        ("AAPL", "NVDA"),   // both supplied by TSMC
        ("AAPL", "Samsung"),// supplier-competitor
        ("MSFT", "GOOG"),   // competitors
        ("BTC", "ETH"),     // correlated crypto
        ("TSLA", "BTC"),    // less related
    ];
    for (a, b) in &pairs {
        if let (Some(ia), Some(ib)) = (kg.entity_id(a), kg.entity_id(b)) {
            let sim = model.entity_similarity(ia, ib);
            println!("  {} <-> {}: {:.4}", a, b, sim);
        }
    }

    // ── Step 3: Extract Graph Features ─────────────────────────────────
    println!("\n── Step 3: Extract Graph Features ──");
    let extractor = GraphFeatureExtractor::new(&kg);
    let pageranks = extractor.pagerank(20);

    println!("  PageRank scores (top entities):");
    let mut pr_indexed: Vec<(usize, f64)> = pageranks.iter().cloned().enumerate().collect();
    pr_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (id, pr) in pr_indexed.iter().take(5) {
        println!(
            "    {}: {:.4} (degree centrality: {:.4})",
            kg.entity_names[*id],
            pr,
            extractor.degree_centrality(*id)
        );
    }

    // Shortest paths
    println!("\n  Shortest paths:");
    let path_pairs = [("TSMC", "S&P500"), ("BTC", "AAPL"), ("NVDA", "Cloud")];
    for (a, b) in &path_pairs {
        if let (Some(ia), Some(ib)) = (kg.entity_id(a), kg.entity_id(b)) {
            let dist = kg.shortest_path(ia, ib);
            let label = if dist < 0 { "unreachable".to_string() } else { dist.to_string() };
            println!("    {} -> {}: {}", a, b, label);
        }
    }

    // ── Step 4: Fetch Live Market Data ─────────────────────────────────
    println!("\n── Step 4: Fetch Live Market Data (Bybit) ──");
    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "5", 10).await {
        Ok(k) => {
            println!("  Fetched {} klines for BTCUSDT (5m):", k.len());
            for kline in k.iter().take(3) {
                println!(
                    "    ts={} O={:.1} H={:.1} L={:.1} C={:.1} V={:.1}",
                    kline.timestamp, kline.open, kline.high, kline.low, kline.close, kline.volume
                );
            }
            k
        }
        Err(e) => {
            println!("  API unavailable ({}), using synthetic data", e);
            Vec::new()
        }
    };

    let (bids, asks) = match client.get_orderbook("BTCUSDT", 5).await {
        Ok((b, a)) => {
            println!("  Orderbook: {} bids, {} asks", b.len(), a.len());
            if let (Some(best_bid), Some(best_ask)) = (b.first(), a.first()) {
                println!(
                    "    Best bid: {:.1} @ {:.4}, Best ask: {:.1} @ {:.4}",
                    best_bid.0, best_bid.1, best_ask.0, best_ask.1
                );
            }
            (b, a)
        }
        Err(e) => {
            println!("  Orderbook unavailable ({}), using defaults", e);
            (vec![(50000.0, 1.5)], vec![(50010.0, 1.2)])
        }
    };

    // ── Step 5: Generate Signals with KG + Market Features ─────────────
    println!("\n── Step 5: Train Trading Signal Generator ──");
    let (features, labels) = generate_training_data(&kg, &model, 2000);

    let split = 1600;
    let (train_x, test_x) = features.split_at(split);
    let (train_y, test_y) = labels.split_at(split);

    let mut signal_gen = TradingSignalGenerator::new(3, 4);
    let acc_before = signal_gen.accuracy(test_x, test_y);
    println!("  Accuracy before training: {:.2}%", acc_before * 100.0);

    signal_gen.train(train_x, train_y, 100, 0.01);
    let acc_after = signal_gen.accuracy(test_x, test_y);
    println!("  Accuracy after training:  {:.2}%", acc_after * 100.0);

    // ── Step 6: Live Prediction ────────────────────────────────────────
    println!("\n── Step 6: Live Prediction ──");

    // Pick an entity (NVDA) and build its feature vector
    let nvda = kg.entity_id("NVDA").unwrap();
    let kg_feats = extractor.entity_features(nvda, &pageranks);

    // Market features from live data or defaults
    let momentum = if klines.len() >= 2 {
        (klines[0].close - klines[1].close) / klines[1].close
    } else {
        0.01
    };
    let volatility = if klines.len() >= 5 {
        let returns: Vec<f64> = klines.windows(2)
            .map(|w| (w[0].close - w[1].close) / w[1].close)
            .collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt()
    } else {
        0.03
    };
    let volume_ratio = if klines.len() >= 2 {
        klines[0].volume / klines[1].volume.max(1.0)
    } else {
        1.1
    };
    let spread = if !bids.is_empty() && !asks.is_empty() {
        (asks[0].0 - bids[0].0) / bids[0].0
    } else {
        0.002
    };

    let live_features = Array1::from(vec![
        kg_feats[0], kg_feats[1], kg_feats[2],
        momentum, volatility, volume_ratio, spread,
    ]);

    let (direction, confidence) = signal_gen.predict(&live_features);
    println!("  Entity:     NVDA");
    println!("  KG features: degree={:.3}, pagerank={:.4}, rel_types={:.3}",
        kg_feats[0], kg_feats[1], kg_feats[2]);
    println!("  Market:     momentum={:.4}, vol={:.4}, vol_ratio={:.2}, spread={:.5}",
        momentum, volatility, volume_ratio, spread);
    println!(
        "  Signal:     {} (confidence: {:.1}%)",
        if direction { "UP" } else { "DOWN" },
        confidence * 100.0
    );

    println!("\n=== Done ===");
}
