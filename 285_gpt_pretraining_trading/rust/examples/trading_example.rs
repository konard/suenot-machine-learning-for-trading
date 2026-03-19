use gpt_pretraining_trading::*;

fn main() -> anyhow::Result<()> {
    println!("=== GPT Pretraining for Trading ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Fetch BTCUSDT data from Bybit
    // -------------------------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT data from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("  Warning: Could not fetch live data ({}), using synthetic data", e);
            generate_synthetic_candles(200)
        }
    };

    let prices = extract_close_prices(&candles);
    println!(
        "  Price range: {:.2} - {:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // -------------------------------------------------------------------------
    // Step 2: Compute log-returns and tokenize
    // -------------------------------------------------------------------------
    println!("\nStep 2: Tokenizing price series...");
    let returns = compute_log_returns(&prices);
    println!("  Computed {} log-returns", returns.len());

    let vocab_size = 32;
    let tokenizer = PriceTokenizer::fit(&returns, vocab_size);
    println!(
        "  Tokenizer: vocab_size={}, total_vocab_size={}",
        tokenizer.vocab_size,
        tokenizer.total_vocab_size()
    );

    let tokens = tokenizer.encode_sequence(&returns);
    println!("  Token sequence length: {}", tokens.len());

    // Show first few tokens and their decoded values
    println!("  First 10 tokens:");
    for (i, &tok) in tokens.iter().take(10).enumerate() {
        if let Some(val) = tokenizer.decode(tok) {
            println!("    [{}] token={} -> return={:.6}", i, tok, val);
        } else {
            let name = match tok {
                BOS_TOKEN => "BOS",
                EOS_TOKEN => "EOS",
                _ => "SPECIAL",
            };
            println!("    [{}] token={} ({})", i, tok, name);
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: Create training sequences and pretrain GPT
    // -------------------------------------------------------------------------
    println!("\nStep 3: Pretraining GPT model...");
    let seq_len = 20;
    let mut training_sequences: Vec<Vec<usize>> = Vec::new();

    // Create overlapping windows (skip BOS/EOS for windowing, re-add them)
    let inner_tokens: Vec<usize> = tokens[1..tokens.len() - 1].to_vec();
    for start in (0..inner_tokens.len().saturating_sub(seq_len)).step_by(seq_len / 2) {
        let end = (start + seq_len).min(inner_tokens.len());
        let mut seq = vec![BOS_TOKEN];
        seq.extend_from_slice(&inner_tokens[start..end]);
        seq.push(EOS_TOKEN);
        training_sequences.push(seq);
    }
    println!("  Created {} training sequences of length ~{}", training_sequences.len(), seq_len + 2);

    let config = GptConfig {
        vocab_size: tokenizer.total_vocab_size(),
        d_model: 32,
        n_heads: 4,
        n_layers: 2,
        max_seq_len: seq_len + 2,
        learning_rate: 0.001,
    };

    let mut model = GptDecoder::new(config);

    // Compute initial loss
    let initial_loss = training_sequences
        .iter()
        .map(|seq| model.compute_loss(seq))
        .sum::<f64>()
        / training_sequences.len() as f64;
    println!("  Initial loss: {:.4}", initial_loss);

    // Training loop
    let n_epochs = 3;
    for epoch in 0..n_epochs {
        let batch_size = 4.min(training_sequences.len());
        let batch = &training_sequences[..batch_size];
        let loss = model.train_step(&batch.to_vec(), 0.001);
        println!("  Epoch {}: loss = {:.4}", epoch + 1, loss);
    }

    // -------------------------------------------------------------------------
    // Step 4: Generate next price tokens (direction prediction)
    // -------------------------------------------------------------------------
    println!("\nStep 4: Generating predictions...");
    let context_len = 10.min(inner_tokens.len());
    let context: Vec<usize> = std::iter::once(BOS_TOKEN)
        .chain(inner_tokens[inner_tokens.len() - context_len..].iter().cloned())
        .collect();

    println!("  Context (last {} tokens):", context_len);
    for &tok in &context {
        if let Some(val) = tokenizer.decode(tok) {
            print!("  {:.4}", val);
        } else {
            print!("  [BOS]");
        }
    }
    println!();

    let n_predict = 5;
    println!("\n  Generating {} future tokens...", n_predict);

    // Generate with different temperatures
    for &temp in &[0.5, 1.0, 1.5] {
        let generated = model.generate(&context, n_predict, temp);
        print!("  Temperature {:.1}: ", temp);
        let mut direction_sum = 0.0;
        for &tok in &generated {
            if let Some(val) = tokenizer.decode(tok) {
                direction_sum += val;
                let arrow = if val > 0.001 {
                    "UP"
                } else if val < -0.001 {
                    "DN"
                } else {
                    "--"
                };
                print!("{:.4}({}) ", val, arrow);
            } else {
                print!("?? ");
            }
        }
        let overall = if direction_sum > 0.0 {
            "BULLISH"
        } else {
            "BEARISH"
        };
        println!(" => Net: {:.4} ({})", direction_sum, overall);
    }

    // -------------------------------------------------------------------------
    // Step 5: Fine-tuning head for trading signals
    // -------------------------------------------------------------------------
    println!("\nStep 5: Trading signal generation...");
    let head = TradingHead::new(model.config.d_model);

    // Get hidden states from the last few positions
    let logits = model.forward(&context);
    let last_idx = logits.nrows() - 1;
    let hidden_dim = model.config.d_model;

    // Use output logits as proxy for hidden state (simplified)
    let hidden_proxy = ndarray::Array1::from_vec(
        (0..hidden_dim).map(|i| logits[[last_idx, i % logits.ncols()]]).collect()
    );

    let signal = head.predict_signal(&hidden_proxy);
    let position = if signal > 0.3 {
        "LONG"
    } else if signal < -0.3 {
        "SHORT"
    } else {
        "FLAT"
    };
    println!("  Trading signal: {:.4} => Position: {}", signal, position);

    println!("\n=== Done ===");
    Ok(())
}

/// Generate synthetic candle data for testing when API is unavailable.
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let ret = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.005));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.005));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1700000000000 + (i as u64) * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });
        price = close;
    }

    candles
}
