//! Crypto Embeddings CLI
//!
//! Command-line interface for cryptocurrency word embeddings analysis.

use clap::{Parser, Subcommand};
use crypto_embeddings::{BybitClient, Word2Vec, Tokenizer};
use std::path::PathBuf;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "crypto_embeddings")]
#[command(about = "Word embeddings for cryptocurrency trading analysis")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch market data from Bybit
    Fetch {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Number of records to fetch
        #[arg(short, long, default_value = "1000")]
        limit: usize,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Train word embeddings model
    Train {
        /// Input corpus file (one sentence per line)
        #[arg(short, long)]
        input: PathBuf,

        /// Output model file
        #[arg(short, long)]
        output: PathBuf,

        /// Embedding dimension
        #[arg(short, long, default_value = "100")]
        dim: usize,

        /// Context window size
        #[arg(short, long, default_value = "5")]
        window: usize,

        /// Minimum word frequency
        #[arg(short, long, default_value = "5")]
        min_count: usize,
    },

    /// Find similar words
    Similar {
        /// Model file path
        #[arg(short, long)]
        model: PathBuf,

        /// Word to find similarities for
        #[arg(short, long)]
        word: String,

        /// Number of similar words to return
        #[arg(short, long, default_value = "10")]
        top_n: usize,
    },

    /// Perform word arithmetic (king - man + woman = queen)
    Analogy {
        /// Model file path
        #[arg(short, long)]
        model: PathBuf,

        /// Positive words (add)
        #[arg(long, num_args = 1..)]
        positive: Vec<String>,

        /// Negative words (subtract)
        #[arg(long, num_args = 1..)]
        negative: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch { symbol, limit, output } => {
            println!("Fetching {} trades for {}...", limit, symbol);
            let client = BybitClient::new();
            let trades = client.get_recent_trades(&symbol, limit).await?;

            println!("Fetched {} trades", trades.len());

            if let Some(path) = output {
                // Save to CSV
                let mut wtr = csv::Writer::from_path(&path)?;
                for trade in &trades {
                    wtr.serialize(trade)?;
                }
                wtr.flush()?;
                println!("Saved to {:?}", path);
            } else {
                // Print first few
                for trade in trades.iter().take(5) {
                    println!("{:?}", trade);
                }
                if trades.len() > 5 {
                    println!("... and {} more", trades.len() - 5);
                }
            }
        }

        Commands::Train { input, output, dim, window, min_count } => {
            println!("Training embeddings...");
            println!("  Input: {:?}", input);
            println!("  Dimensions: {}", dim);
            println!("  Window: {}", window);
            println!("  Min count: {}", min_count);

            // Read corpus
            let corpus = std::fs::read_to_string(&input)?;
            let sentences: Vec<&str> = corpus.lines().collect();

            // Tokenize
            let tokenizer = Tokenizer::new();
            let tokenized: Vec<Vec<String>> = sentences
                .iter()
                .map(|s| tokenizer.tokenize(s))
                .collect();

            // Train
            let mut model = Word2Vec::new(dim, window, min_count);
            model.build_vocab(&tokenized);
            model.train(&tokenized, 5)?;

            // Save
            model.save(&output)?;
            println!("Model saved to {:?}", output);
            println!("Vocabulary size: {}", model.vocab_size());
        }

        Commands::Similar { model, word, top_n } => {
            let model = Word2Vec::load(&model)?;
            let similar = model.most_similar(&word, top_n)?;

            println!("Words most similar to '{}':", word);
            for (w, score) in similar {
                println!("  {}: {:.4}", w, score);
            }
        }

        Commands::Analogy { model, positive, negative } => {
            let model = Word2Vec::load(&model)?;
            let results = model.analogy(&positive, &negative, 10)?;

            println!("Analogy: {} - {} = ?",
                     positive.join(" + "),
                     negative.join(" - "));
            for (w, score) in results {
                println!("  {}: {:.4}", w, score);
            }
        }
    }

    Ok(())
}
