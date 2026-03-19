use anyhow::Result;
use trade_classification::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Chapter 266: Trade Classification ===\n");

    // ----------------------------------------------------------
    // Part 1: Demonstrate classifiers on synthetic data
    // ----------------------------------------------------------
    println!("--- Part 1: Synthetic Data Classification ---\n");

    let (trades, quotes, labels) = generate_synthetic_trades(200, 50000.0);
    println!("Generated {} synthetic trades around base price $50,000", trades.len());

    // Tick Test
    let tick_results = TickTestClassifier::classify(&trades);
    let tick_preds: Vec<TradeSide> = tick_results
        .iter()
        .skip(1)
        .map(|r| r.unwrap_or(TradeSide::Buy))
        .collect();
    let tick_acc = evaluate_accuracy(&tick_preds, &labels[1..]);
    println!("Tick Test accuracy:    {:.2}%", tick_acc * 100.0);

    // Lee-Ready
    let lr_results = LeeReadyClassifier::classify(&trades, &quotes);
    let lr_preds: Vec<TradeSide> = lr_results
        .iter()
        .skip(1)
        .map(|r| r.unwrap_or(TradeSide::Buy))
        .collect();
    let lr_acc = evaluate_accuracy(&lr_preds, &labels[1..]);
    println!("Lee-Ready accuracy:   {:.2}%", lr_acc * 100.0);

    // BVC
    let bvc = BulkVolumeClassifier::new(0.01);
    let prices: Vec<f64> = trades.iter().map(|t| t.price).collect();
    let bvc_sides = bvc.classify_sides(&prices);
    let bvc_acc = evaluate_accuracy(&bvc_sides, &labels[1..bvc_sides.len() + 1]);
    println!("BVC accuracy:         {:.2}%", bvc_acc * 100.0);

    // ML Ensemble
    let ensemble = EnsembleClassifier::new();
    let ensemble_results = ensemble.classify(&trades, &quotes);
    let ensemble_preds: Vec<TradeSide> = ensemble_results
        .iter()
        .skip(1)
        .map(|r| r.predicted_side)
        .collect();
    let ensemble_acc = evaluate_accuracy(&ensemble_preds, &labels[1..]);
    println!("ML Ensemble accuracy: {:.2}%", ensemble_acc * 100.0);

    // Show some individual classifications
    println!("\n--- Sample Classifications (first 10 trades) ---\n");
    println!(
        "{:<6} {:<12} {:<10} {:<10} {:<10} {:<10}",
        "Trade", "True Side", "Tick", "Lee-Ready", "BVC", "Ensemble"
    );
    for i in 1..11.min(trades.len()) {
        let true_side = &labels[i];
        let tick = tick_results[i].map(|s| format!("{}", s)).unwrap_or("N/A".to_string());
        let lr = lr_results[i].map(|s| format!("{}", s)).unwrap_or("N/A".to_string());
        let bvc_side = if i > 0 && i - 1 < bvc_sides.len() {
            format!("{}", bvc_sides[i - 1])
        } else {
            "N/A".to_string()
        };
        let ens = format!(
            "{} ({:.0}%)",
            ensemble_results[i].predicted_side, ensemble_results[i].confidence * 100.0
        );
        println!(
            "{:<6} {:<12} {:<10} {:<10} {:<10} {:<10}",
            i, true_side, tick, lr, bvc_side, ens
        );
    }

    // ----------------------------------------------------------
    // Part 2: Fetch live Bybit data and classify
    // ----------------------------------------------------------
    println!("\n--- Part 2: Live Bybit Trade Classification ---\n");

    let symbol = "BTCUSDT";
    println!("Fetching recent trades for {} from Bybit...", symbol);

    match fetch_bybit_trades(symbol, 50).await {
        Ok(mut live_trades) => {
            // Sort by timestamp ascending
            live_trades.sort_by_key(|t| t.timestamp);

            println!("Fetched {} trades", live_trades.len());

            // Store actual labels from exchange
            let exchange_labels: Vec<Option<TradeSide>> =
                live_trades.iter().map(|t| t.side).collect();

            // Fetch orderbook for quote data
            match fetch_bybit_orderbook(symbol).await {
                Ok(quote) => {
                    println!(
                        "Current spread: bid={:.2}, ask={:.2}, mid={:.2}, spread={:.4}",
                        quote.bid_price,
                        quote.ask_price,
                        quote.midpoint(),
                        quote.spread()
                    );

                    // Use same quote for all trades (simplification)
                    let quotes_vec: Vec<Quote> =
                        (0..live_trades.len()).map(|_| quote.clone()).collect();

                    // Classify with tick test
                    let tick_results = TickTestClassifier::classify(&live_trades);

                    // Classify with Lee-Ready
                    let lr_results = LeeReadyClassifier::classify(&live_trades, &quotes_vec);

                    // Classify with ensemble
                    let ensemble = EnsembleClassifier::new();
                    let ens_results = ensemble.classify(&live_trades, &quotes_vec);

                    // Compare with exchange-reported labels
                    println!("\n{:<6} {:<12} {:<10} {:<10} {:<12} {:<10}", "Trade", "Price", "Exchange", "Tick", "Lee-Ready", "Ensemble");
                    for i in 0..live_trades.len().min(20) {
                        let exchange = exchange_labels[i]
                            .map(|s| format!("{}", s))
                            .unwrap_or("N/A".to_string());
                        let tick = tick_results[i]
                            .map(|s| format!("{}", s))
                            .unwrap_or("N/A".to_string());
                        let lr = lr_results[i]
                            .map(|s| format!("{}", s))
                            .unwrap_or("N/A".to_string());
                        let ens = format!("{}", ens_results[i].predicted_side);
                        println!(
                            "{:<6} {:<12.2} {:<10} {:<10} {:<12} {:<10}",
                            i, live_trades[i].price, exchange, tick, lr, ens
                        );
                    }

                    // Calculate accuracy vs exchange labels
                    let mut correct_tick = 0;
                    let mut correct_lr = 0;
                    let mut correct_ens = 0;
                    let mut total = 0;

                    for i in 1..live_trades.len() {
                        if let Some(label) = exchange_labels[i] {
                            total += 1;
                            if tick_results[i] == Some(label) {
                                correct_tick += 1;
                            }
                            if lr_results[i] == Some(label) {
                                correct_lr += 1;
                            }
                            if ens_results[i].predicted_side == label {
                                correct_ens += 1;
                            }
                        }
                    }

                    if total > 0 {
                        println!("\n--- Accuracy vs Exchange Labels ({} trades) ---", total);
                        println!(
                            "Tick Test:    {:.1}%",
                            100.0 * correct_tick as f64 / total as f64
                        );
                        println!(
                            "Lee-Ready:    {:.1}%",
                            100.0 * correct_lr as f64 / total as f64
                        );
                        println!(
                            "ML Ensemble:  {:.1}%",
                            100.0 * correct_ens as f64 / total as f64
                        );
                    }
                }
                Err(e) => println!("Could not fetch orderbook: {}", e),
            }
        }
        Err(e) => println!("Could not fetch live trades: {}", e),
    }

    // ----------------------------------------------------------
    // Part 3: Feature importance analysis
    // ----------------------------------------------------------
    println!("\n--- Part 3: Feature Importance Analysis ---\n");

    let classifier = LogisticClassifier::default_trade_classifier();
    let feature_names = [
        "relative_price",
        "relative_size",
        "tick_direction",
        "spread",
        "quote_imbalance",
        "time_delta_ms",
    ];

    println!("Logistic classifier weights:");
    for (name, &weight) in feature_names.iter().zip(classifier.weights.iter()) {
        let bar_len = (weight.abs() * 10.0) as usize;
        let bar: String = std::iter::repeat('#').take(bar_len).collect();
        println!("  {:<20} {:>6.3}  {}", name, weight, bar);
    }

    println!("\nDone!");

    Ok(())
}
