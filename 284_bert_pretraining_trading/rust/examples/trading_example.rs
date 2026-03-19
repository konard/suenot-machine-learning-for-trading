use bert_pretraining_trading::*;

fn main() -> anyhow::Result<()> {
    println!("=== Chapter 284: BERT Pretraining for Trading ===\n");

    // ----------------------------------------------------------
    // Step 1: Fetch BTCUSDT data from Bybit
    // ----------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT klines from Bybit...");
    let klines = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(k) => {
            println!("  Fetched {} klines", k.len());
            k
        }
        Err(e) => {
            println!("  Failed to fetch from Bybit: {}. Using synthetic data.", e);
            generate_synthetic_klines(200)
        }
    };

    let prices = extract_close_prices(&klines);
    println!(
        "  Price range: {:.2} - {:.2}\n",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // ----------------------------------------------------------
    // Step 2: Initialize BERT model and tokenizer
    // ----------------------------------------------------------
    println!("Step 2: Initializing BERT trading model...");
    let config = BertConfig {
        vocab_size: 64,
        hidden_dim: 32,
        num_heads: 4,
        num_layers: 2,
        intermediate_dim: 64,
        max_seq_len: 256,
        mask_ratio: 0.15,
    };
    let mut model = BertTradingModel::new(config);
    model.fit_tokenizer(&prices);
    println!("  Model initialized with {} price bins\n", model.tokenizer.num_bins);

    // ----------------------------------------------------------
    // Step 3: Tokenize and mask price sequences
    // ----------------------------------------------------------
    println!("Step 3: Tokenizing and masking price sequences...");
    let returns = model.tokenizer.compute_returns(&prices);
    let tokens = model.tokenizer.tokenize(&returns);
    println!("  Tokenized {} returns into {} tokens", returns.len(), tokens.len());

    let (masked_tokens, mask_positions, original_tokens) = model.tokenizer.mask_tokens(&tokens);
    println!(
        "  Masked {} out of {} tokens ({:.1}%)",
        mask_positions.len(),
        tokens.len(),
        100.0 * mask_positions.len() as f64 / tokens.len() as f64,
    );

    // Show a few masked positions
    let show_count = mask_positions.len().min(5);
    for i in 0..show_count {
        println!(
            "    Position {}: original token {} -> masked token {}",
            mask_positions[i], original_tokens[i], masked_tokens[mask_positions[i]]
        );
    }
    println!();

    // ----------------------------------------------------------
    // Step 4: Train BERT on MLM objective
    // ----------------------------------------------------------
    println!("Step 4: Pretraining BERT with MLM objective...");
    let num_epochs = 5;
    let window_size = 50;
    let stride = 10;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let mut start = 0;
        while start + window_size < prices.len() {
            let window = &prices[start..start + window_size];
            let loss = model.pretrain_step(window);
            total_loss += loss;
            num_batches += 1;
            start += stride;
        }

        let avg_loss = if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        };
        println!("  Epoch {}/{}: avg MLM loss = {:.4}", epoch + 1, num_epochs, avg_loss);
    }
    println!();

    // ----------------------------------------------------------
    // Step 5: Fine-tune for direction prediction
    // ----------------------------------------------------------
    println!("Step 5: Fine-tuning for direction prediction...");

    // Generate labeled windows: look at future return to assign labels
    let mut correct = 0;
    let mut total = 0;

    let mut start = 0;
    while start + window_size + 1 < prices.len() {
        let window = &prices[start..start + window_size];
        let future_price = prices[start + window_size];
        let current_price = prices[start + window_size - 1];
        let future_return = (future_price - current_price) / current_price;

        let true_signal = if future_return > 0.001 {
            TradingSignal::Buy
        } else if future_return < -0.001 {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        };

        let predicted = model.predict_signal(window);

        if predicted == true_signal {
            correct += 1;
        }
        total += 1;
        start += stride;
    }

    let accuracy = if total > 0 {
        correct as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "  Direction prediction: {}/{} correct ({:.1}%)",
        correct, total, accuracy
    );
    println!("  (Note: random baseline with 3 classes = ~33.3%)\n");

    // ----------------------------------------------------------
    // Step 6: Generate live trading signals
    // ----------------------------------------------------------
    println!("Step 6: Generating trading signals for recent windows...");
    let recent_windows = 5;
    let available = prices.len();
    for i in 0..recent_windows {
        let end = available - i * stride;
        if end < window_size {
            break;
        }
        let start_idx = end - window_size;
        let window = &prices[start_idx..end];
        let signal = model.predict_signal(window);
        let last_price = window[window.len() - 1];
        println!(
            "  Window [{}-{}] last_price={:.2} -> signal={}",
            start_idx, end, last_price, signal
        );
    }

    println!("\n=== Done ===");
    Ok(())
}

/// Generate synthetic kline data for offline testing.
fn generate_synthetic_klines(n: usize) -> Vec<KlineData> {
    let mut price = 50000.0_f64;
    let mut klines = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();
    use rand::Rng;

    for i in 0..n {
        let ret: f64 = rng.gen::<f64>() * 0.04 - 0.02; // -2% to +2%
        let open = price;
        price *= 1.0 + ret;
        let close = price;
        let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.005);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.005);
        let volume = 100.0 + rng.gen::<f64>() * 500.0;

        klines.push(KlineData {
            timestamp: 1700000000000 + (i as u64) * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    klines
}
