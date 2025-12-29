# Crypto Embeddings - Rust

Word embeddings library for cryptocurrency trading analysis with Bybit exchange integration.

## Overview

This Rust library provides tools for:
- Fetching real-time cryptocurrency data from Bybit exchange
- Text preprocessing for trading-related content
- Training Word2Vec-style embeddings
- Sentiment analysis of crypto texts
- Document similarity analysis

## Project Structure

```
rust_crypto_embeddings/
├── Cargo.toml           # Project dependencies
├── src/
│   ├── lib.rs           # Library entry point
│   ├── main.rs          # CLI application
│   ├── api/             # Bybit API client
│   │   └── mod.rs
│   ├── embeddings/      # Word2Vec implementation
│   │   └── mod.rs
│   ├── preprocessing/   # Text tokenization
│   │   └── mod.rs
│   ├── analysis/        # Sentiment & similarity
│   │   └── mod.rs
│   └── utils/           # Common utilities
│       └── mod.rs
├── examples/
│   ├── fetch_trades.rs      # Bybit API usage
│   ├── train_embeddings.rs  # Training word vectors
│   └── analyze_sentiment.rs # Sentiment analysis
└── data/
    └── sample_corpus.txt    # Sample training data
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
crypto_embeddings = { path = "./rust_crypto_embeddings" }
```

Or clone and build:

```bash
cd rust_crypto_embeddings
cargo build --release
```

## Quick Start

### 1. Fetch Market Data

```rust
use crypto_embeddings::BybitClient;

#[tokio::main]
async fn main() {
    let client = BybitClient::new();

    // Get ticker
    let ticker = client.get_ticker("BTCUSDT").await.unwrap();
    println!("BTC Price: ${}", ticker.last_price);

    // Get recent trades
    let trades = client.get_recent_trades("BTCUSDT", 100).await.unwrap();
    println!("Got {} trades", trades.len());
}
```

### 2. Train Word Embeddings

```rust
use crypto_embeddings::{Word2Vec, Tokenizer};

fn main() {
    let texts = vec![
        "BTC bullish breakout momentum",
        "ETH bearish support breakdown",
    ];

    // Tokenize
    let tokenizer = Tokenizer::new();
    let sentences: Vec<Vec<String>> = texts
        .iter()
        .map(|t| tokenizer.tokenize(t))
        .collect();

    // Train
    let mut model = Word2Vec::new(100, 5, 2);
    model.build_vocab(&sentences);
    model.train(&sentences, 5).unwrap();

    // Find similar words
    let similar = model.most_similar("bullish", 5).unwrap();
    for (word, score) in similar {
        println!("{}: {:.4}", word, score);
    }
}
```

### 3. Analyze Sentiment

```rust
use crypto_embeddings::analysis::SentimentAnalyzer;
use crypto_embeddings::Word2Vec;

fn main() {
    let model = Word2Vec::new(10, 2, 1);
    let analyzer = SentimentAnalyzer::new(model);

    let result = analyzer.analyze("BTC pumping to the moon!");
    println!("Sentiment: {:?}, Score: {}", result.label, result.score);
}
```

## CLI Usage

```bash
# Fetch trades
cargo run -- fetch -s BTCUSDT -l 100 -o trades.csv

# Train embeddings
cargo run -- train -i corpus.txt -o model.vec -d 100 -w 5

# Find similar words
cargo run -- similar -m model.vec -w bullish -n 10

# Word analogies
cargo run -- analogy -m model.vec --positive btc bullish --negative eth
```

## Running Examples

```bash
# Fetch trades from Bybit
cargo run --example fetch_trades

# Train embeddings on sample data
cargo run --example train_embeddings

# Analyze sentiment
cargo run --example analyze_sentiment
```

## Module Details

### API Module (`src/api/`)

Bybit V5 API client supporting:
- Recent trades (`get_recent_trades`)
- Order book (`get_orderbook`)
- Klines/candlesticks (`get_klines`)
- Ticker information (`get_ticker`)
- Trading symbols list (`get_symbols`)

### Preprocessing Module (`src/preprocessing/`)

Text processing utilities:
- `Tokenizer` - Tokenization with stopword removal
- `CryptoVocab` - Crypto-specific vocabulary detection
- N-gram generation and phrase detection

### Embeddings Module (`src/embeddings/`)

Word2Vec implementation:
- Skip-gram architecture with negative sampling
- Vocabulary building with frequency filtering
- Model saving/loading (word2vec text format)
- Similarity search and analogies

### Analysis Module (`src/analysis/`)

Analysis tools:
- `SentimentAnalyzer` - Lexicon-based and embedding-based sentiment
- `SimilarityAnalyzer` - Document similarity using averaged embeddings
- `TrendAnalyzer` - Topic detection from text corpus

## Configuration

### Tokenizer Options

```rust
let tokenizer = Tokenizer::with_options(
    true,   // lowercase
    2,      // min token length
    Some(custom_stopwords),
);
```

### Word2Vec Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| dim | 100 | Embedding dimension |
| window | 5 | Context window size |
| min_count | 5 | Minimum word frequency |
| learning_rate | 0.025 | Initial learning rate |
| negative_samples | 5 | Number of negative samples |

## Testing

```bash
cargo test
```

## License

MIT License
