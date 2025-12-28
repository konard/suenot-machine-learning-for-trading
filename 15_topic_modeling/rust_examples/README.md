# Topic Modeling in Rust for Cryptocurrency Markets

This project provides a modular Rust implementation of topic modeling algorithms (LSI and LDA) for analyzing cryptocurrency market data from Bybit exchange.

## Features

- **Bybit API Integration**: Fetch announcements, market tickers, and historical data
- **Text Preprocessing**: Tokenization, stop word removal, TF-IDF vectorization
- **LSI (Latent Semantic Indexing)**: SVD-based topic extraction
- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic modeling with Gibbs sampling
- **Evaluation Metrics**: Perplexity, coherence, topic diversity

## Project Structure

```
rust_examples/
├── Cargo.toml                 # Project dependencies
├── README.md                  # This file
├── src/
│   ├── lib.rs                 # Library entry point
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs           # Bybit API client
│   ├── preprocessing/
│   │   ├── mod.rs
│   │   ├── tokenizer.rs       # Text tokenization
│   │   └── vectorizer.rs      # TF-IDF and count vectorization
│   ├── models/
│   │   ├── mod.rs
│   │   ├── lsi.rs             # LSI implementation
│   │   └── lda.rs             # LDA implementation
│   ├── utils/
│   │   ├── mod.rs
│   │   ├── io.rs              # Data loading/saving
│   │   └── evaluation.rs      # Model evaluation metrics
│   └── bin/
│       ├── fetch_data.rs      # Data fetching example
│       ├── lsi_example.rs     # LSI demonstration
│       ├── lda_example.rs     # LDA demonstration
│       └── analyze_market.rs  # Complete analysis pipeline
└── data/                      # Saved datasets (created at runtime)
```

## Installation

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs)
- OpenBLAS (for linear algebra operations)

### Ubuntu/Debian
```bash
sudo apt-get install libopenblas-dev
```

### macOS
```bash
brew install openblas
```

### Build
```bash
cd rust_examples
cargo build --release
```

## Usage

### 1. Fetch Data from Bybit

```bash
cargo run --release --bin fetch_data
```

This will:
- Connect to Bybit public API
- Fetch recent announcements
- Get current market tickers
- Save data for analysis

### 2. Run LSI Example

```bash
cargo run --release --bin lsi_example
```

Output includes:
- Discovered topics with top words
- Document-topic assignments
- Document similarity analysis

### 3. Run LDA Example

```bash
cargo run --release --bin lda_example
```

Output includes:
- Topic distributions
- Perplexity and coherence scores
- Document categorization

### 4. Complete Market Analysis

```bash
cargo run --release --bin analyze_market
```

This combines:
- Live market data
- Topic extraction with both LSI and LDA
- Symbol-topic correlation
- Trading insights generation

## API Reference

### Bybit Client

```rust
use topic_modeling::api::bybit::BybitClient;

let client = BybitClient::new();

// Fetch announcements
let announcements = client.get_announcements("en-US", 50)?;

// Get market tickers
let tickers = client.get_tickers("spot", Some("BTCUSDT"))?;

// Get historical klines
let klines = client.get_klines("spot", "BTCUSDT", "60", 100)?;
```

### Text Preprocessing

```rust
use topic_modeling::preprocessing::tokenizer::Tokenizer;
use topic_modeling::preprocessing::vectorizer::TfIdfVectorizer;

// Create tokenizer optimized for crypto text
let tokenizer = Tokenizer::for_crypto().min_length(3);
let tokens = tokenizer.tokenize("Bitcoin price surges as institutional investors...");

// Build TF-IDF matrix
let mut vectorizer = TfIdfVectorizer::new()
    .min_df(2)
    .max_df_ratio(0.8)
    .max_features(500);

let matrix = vectorizer.fit_transform(&tokenized_docs);
```

### LSI Model

```rust
use topic_modeling::models::lsi::LSI;

let mut lsi = LSI::new(5)?;  // 5 topics
lsi.fit(&tfidf_matrix, vocabulary, terms)?;

// Get discovered topics
let topics = lsi.get_topics(10)?;  // Top 10 words per topic

// Find similar documents
let similar = lsi.most_similar_documents(0, 5)?;
```

### LDA Model

```rust
use topic_modeling::models::lda::{LDA, LdaConfig};

let config = LdaConfig::new(5)   // 5 topics
    .alpha(0.1)                  // Document-topic prior
    .beta(0.01)                  // Topic-word prior
    .n_iterations(1000)
    .random_seed(42);

let mut lda = LDA::new(config)?;
lda.fit(&count_matrix, vocabulary, terms)?;

// Get topics with probabilities
let topics = lda.get_topics(10)?;

// Get perplexity
let perplexity = lda.perplexity(&count_matrix)?;
```

## Configuration

### LDA Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_topics` | Number of topics | 5-50 |
| `alpha` | Document-topic prior | 0.01-0.5 (lower = sparser) |
| `beta` | Topic-word prior | 0.001-0.1 (lower = sparser) |
| `n_iterations` | Gibbs sampling iterations | 500-2000 |
| `burn_in` | Iterations to discard | 50-200 |

### Tokenizer Options

```rust
let tokenizer = Tokenizer::new()
    .min_length(2)           // Minimum token length
    .max_length(50)          // Maximum token length
    .lowercase(true)         // Convert to lowercase
    .remove_numbers(false);  // Keep numbers for crypto context
```

## Examples

### Analyzing Crypto Announcements

```rust
use topic_modeling::api::bybit::{BybitClient, MarketDocument};
use topic_modeling::models::lda::{LDA, LdaConfig};

// Fetch announcements
let client = BybitClient::new();
let announcements = client.get_announcements("en-US", 50)?;

// Convert to documents
let docs: Vec<MarketDocument> = announcements
    .iter()
    .map(MarketDocument::from_announcement)
    .collect();

// Preprocess and train LDA...
```

### Tracking Topic Trends

```rust
// Group documents by time period
let mut period_docs = HashMap::new();
for doc in documents {
    let period = doc.timestamp / (24 * 3600 * 1000);  // Daily
    period_docs.entry(period).or_insert(Vec::new()).push(doc);
}

// Train LDA on each period and compare topic distributions
```

## Performance Tips

1. **Use Release Mode**: Always run with `--release` for 10-50x speedup
2. **Limit Vocabulary**: Use `max_features` to cap vocabulary size
3. **Adjust Iterations**: Start with fewer iterations, increase if needed
4. **Parallel Processing**: The library uses Rayon for parallel operations

## Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test models::lda

# Run with output
cargo test -- --nocapture
```

## Limitations

- LSI uses power iteration SVD (slower than LAPACK but no external dependencies)
- LDA Gibbs sampling can be slow for large corpora
- No GPU acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
- [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/)
