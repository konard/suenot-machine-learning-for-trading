# Chapter 70: Fine-tuning LLM for Finance — LoRA, QLoRA, and Prefix-Tuning

This chapter explores **Fine-tuning techniques for Large Language Models (LLMs)** in the financial domain. We cover Parameter-Efficient Fine-Tuning (PEFT) methods including LoRA, QLoRA, and prefix-tuning, demonstrating how to adapt foundation models for financial sentiment analysis, market prediction, and trading signal generation.

<p align="center">
<img src="https://i.imgur.com/KLMnB8v.png" width="70%">
</p>

## Contents

1. [Introduction to LLM Fine-tuning](#introduction-to-llm-fine-tuning)
    * [Why Fine-tune for Finance?](#why-fine-tune-for-finance)
    * [Full Fine-tuning vs PEFT](#full-fine-tuning-vs-peft)
    * [Key PEFT Methods](#key-peft-methods)
2. [LoRA: Low-Rank Adaptation](#lora-low-rank-adaptation)
    * [Mathematical Foundation](#mathematical-foundation)
    * [Implementation Details](#implementation-details)
    * [Hyperparameter Selection](#hyperparameter-selection)
3. [QLoRA: Quantized LoRA](#qlora-quantized-lora)
    * [4-bit Quantization](#4-bit-quantization)
    * [Double Quantization](#double-quantization)
    * [Memory Efficiency](#memory-efficiency)
4. [Prefix-Tuning](#prefix-tuning)
    * [Soft Prompts](#soft-prompts)
    * [Virtual Tokens](#virtual-tokens)
    * [Comparison with LoRA](#comparison-with-lora)
5. [Financial Applications](#financial-applications)
    * [Sentiment Analysis](#sentiment-analysis)
    * [Market Prediction](#market-prediction)
    * [Trading Signal Generation](#trading-signal-generation)
6. [Practical Examples](#practical-examples)
    * [01: Fine-tuning for Financial Sentiment](#01-fine-tuning-for-financial-sentiment)
    * [02: Crypto Market Analysis with Bybit Data](#02-crypto-market-analysis-with-bybit-data)
    * [03: Backtesting Fine-tuned Models](#03-backtesting-fine-tuned-models)
7. [Rust Implementation](#rust-implementation)
8. [Python Implementation](#python-implementation)
9. [Best Practices](#best-practices)
10. [Resources](#resources)

## Introduction to LLM Fine-tuning

Fine-tuning adapts pre-trained Large Language Models to specific domains or tasks. In finance, this enables models to understand specialized terminology, interpret market sentiment accurately, and generate actionable trading signals.

### Why Fine-tune for Finance?

Pre-trained models lack domain expertise:

```
CHALLENGES WITH GENERAL LLMs IN FINANCE:
┌──────────────────────────────────────────────────────────────────┐
│  1. DOMAIN TERMINOLOGY                                            │
│     "The stock has a forward P/E of 25x with strong FCF yield"   │
│     General LLM: May misinterpret financial ratios               │
│     Fine-tuned: Understands valuation metrics contextually       │
├──────────────────────────────────────────────────────────────────┤
│  2. SENTIMENT NUANCE                                              │
│     "Company maintained guidance despite macro headwinds"         │
│     General LLM: Neutral or negative?                            │
│     Fine-tuned: Recognizes as moderately positive                │
├──────────────────────────────────────────────────────────────────┤
│  3. TEMPORAL PATTERNS                                             │
│     "Beat consensus by 200bps, raised FY guidance"               │
│     General LLM: May miss earnings season context                │
│     Fine-tuned: Understands quarterly reporting patterns          │
├──────────────────────────────────────────────────────────────────┤
│  4. MARKET IMPACT ASSESSMENT                                      │
│     "Fed signals hawkish pivot, yields surge"                    │
│     General LLM: May not link to trading implications            │
│     Fine-tuned: Understands cross-asset relationships            │
└──────────────────────────────────────────────────────────────────┘
```

### Full Fine-tuning vs PEFT

| Aspect | Full Fine-tuning | PEFT (LoRA/QLoRA) |
|--------|------------------|-------------------|
| Parameters Updated | All (billions) | 0.1-1% of total |
| GPU Memory | 40-80GB+ per GPU | 4-16GB single GPU |
| Training Time | Days to weeks | Hours to days |
| Catastrophic Forgetting | High risk | Low risk |
| Storage per Task | Full model copy | Small adapter files |
| Deployment | Complex | Simple adapter swapping |

### Key PEFT Methods

```
PARAMETER-EFFICIENT FINE-TUNING LANDSCAPE:
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTER-BASED METHODS                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │    LoRA       │  │   QLoRA       │  │   AdaLoRA     │        │
│  │  Low-rank     │  │  4-bit quant  │  │  Adaptive     │        │
│  │  adaptation   │  │  + LoRA       │  │  rank alloc   │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                  │
│  Parameters: 0.1-1%  │  Memory: 4-8GB  │  Preserves knowledge   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT-BASED METHODS                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ Prefix-Tuning │  │ Prompt-Tuning │  │  P-Tuning v2  │        │
│  │  Virtual      │  │  Soft         │  │  Deep prompt  │        │
│  │  tokens       │  │  prompts      │  │  tuning       │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                  │
│  Parameters: <0.1%  │  Memory: 2-4GB  │  Task-specific          │
└─────────────────────────────────────────────────────────────────┘
```

## LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) is the most popular PEFT method, introducing trainable low-rank matrices that modify the behavior of frozen pre-trained weights.

### Mathematical Foundation

Instead of updating the full weight matrix W ∈ ℝ^(d×k), LoRA learns a low-rank decomposition:

```
LORA WEIGHT UPDATE MECHANISM:
═══════════════════════════════════════════════════════════════════

Original Weight Matrix:     W₀ ∈ ℝ^(d×k)     (frozen)
LoRA Decomposition:         ΔW = BA
  where: B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)

Forward Pass:               h = W₀x + ΔWx = W₀x + BAx

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Input x ──────┬──────────────────────────────┬─────── Output h │
│                │                              │                  │
│                ▼                              ▼                  │
│        ┌──────────────┐              ┌──────────────┐           │
│        │   W₀ (frozen)│              │   BA (LoRA)  │           │
│        │   d × k      │              │   d × r × k  │           │
│        └──────────────┘              └──────────────┘           │
│                │                              │                  │
│                └────────────┬─────────────────┘                  │
│                             ▼                                    │
│                       h = W₀x + αBAx                            │
│                       (α = scaling factor)                       │
└─────────────────────────────────────────────────────────────────┘

Parameter Reduction Example:
  Original: d=4096, k=4096 → 16.7M parameters
  LoRA r=8: (4096×8) + (8×4096) = 65K parameters (0.4%)
  LoRA r=16: (4096×16) + (16×4096) = 131K parameters (0.8%)
```

### Implementation Details

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    LoRA layer implementation for financial LLM fine-tuning.

    This layer adds trainable low-rank matrices to frozen pre-trained weights,
    enabling efficient adaptation to financial tasks like sentiment analysis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen pre-trained weight (simulated, normally from base model)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )

        # LoRA trainable matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming, B with zeros (start from original)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original transformation (frozen)
        result = x @ self.weight.T

        # LoRA adaptation
        lora_result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        return result + self.scaling * lora_result


class FinancialSentimentLoRA(nn.Module):
    """
    Financial sentiment classifier using LoRA-adapted transformer.

    Classifies financial text into sentiment categories:
    - Bullish (positive market outlook)
    - Bearish (negative market outlook)
    - Neutral (no clear directional signal)
    """

    def __init__(
        self,
        base_dim: int = 768,
        lora_rank: int = 8,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        # LoRA-adapted attention projection
        self.query_lora = LoRALayer(base_dim, base_dim, lora_rank)
        self.value_lora = LoRALayer(base_dim, base_dim, lora_rank)

        # Classification head (fully trainable)
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim // 2, num_classes)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply LoRA transformations
        query = self.query_lora(hidden_states)
        value = self.value_lora(hidden_states)

        # Simple aggregation (mean pooling)
        pooled = hidden_states.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)
        return logits
```

### Hyperparameter Selection

Optimal hyperparameters for financial fine-tuning:

| Hyperparameter | Recommended Range | Financial Tasks |
|---------------|-------------------|-----------------|
| Rank (r) | 4-64 | 8-16 for sentiment, 16-32 for generation |
| Alpha (α) | r to 2r | Usually 2×rank works well |
| Learning Rate | 1e-4 to 3e-4 | Lower for larger models |
| Dropout | 0.05-0.1 | 0.1 for small datasets |
| Target Modules | q_proj, v_proj | Add k_proj, o_proj for complex tasks |
| Warmup Steps | 5-10% | Critical for stability |

```
RANK SELECTION GUIDE FOR FINANCIAL TASKS:
═══════════════════════════════════════════════════════════════════

┌─────────────────┬──────────┬────────────────────────────────────┐
│ Task            │ Rank     │ Rationale                          │
├─────────────────┼──────────┼────────────────────────────────────┤
│ Binary Sentiment│ r=4-8    │ Simple classification, low rank    │
│ Multi-class     │ r=8-16   │ More nuance requires capacity      │
│ Named Entity    │ r=16-32  │ Precise boundary detection         │
│ Text Generation │ r=32-64  │ Complex output space               │
│ Multi-task      │ r=64+    │ Multiple objectives to balance     │
└─────────────────┴──────────┴────────────────────────────────────┘

Training Data Size vs Rank:
  < 1K samples   → r=4-8   (prevent overfitting)
  1K-10K samples → r=8-16  (balanced capacity)
  10K+ samples   → r=16-32 (can leverage more parameters)
```

## QLoRA: Quantized LoRA

QLoRA combines 4-bit quantization with LoRA, enabling fine-tuning of large models on consumer hardware.

### 4-bit Quantization

```
QLORA QUANTIZATION SCHEME:
═══════════════════════════════════════════════════════════════════

Base Model (FP16):        16 bits per parameter
After NF4 Quantization:    4 bits per parameter  (4× compression)

┌─────────────────────────────────────────────────────────────────┐
│              NormalFloat4 (NF4) Data Type                        │
│                                                                  │
│  Values distributed to match normal distribution quantiles:      │
│                                                                  │
│  [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,   │
│    0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, │
│    1.0]                                                          │
│                                                                  │
│  Why NF4?                                                        │
│  - Neural network weights follow ~normal distribution            │
│  - NF4 optimally covers this distribution                        │
│  - Better preservation of model quality vs uniform quantization  │
└─────────────────────────────────────────────────────────────────┘

Memory Comparison (7B parameter model):
  FP32: 28 GB
  FP16: 14 GB
  INT8:  7 GB
  NF4:   3.5 GB  ← QLoRA operates here
```

### Double Quantization

```python
# QLoRA configuration for financial fine-tuning
from transformers import BitsAndBytesConfig

qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",      # NormalFloat4 quantization
    bnb_4bit_use_double_quant=True,  # Double quantization for constants
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in BF16
)

# Double quantization saves additional memory:
# Quantization constants (32-bit) → also quantized (8-bit)
# Saves ~0.37 bits per parameter on average
```

### Memory Efficiency

```
MEMORY FOOTPRINT COMPARISON (7B Model Fine-tuning):
═══════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────┐
│ Method           │ Model │ Optimizer │ Gradients │ Total      │
├──────────────────┼───────┼───────────┼───────────┼────────────┤
│ Full FP16        │ 14GB  │ 28GB      │ 14GB      │ ~56GB      │
│ Full + ZeRO-3    │ 5GB   │ 9GB       │ 5GB       │ ~19GB/GPU  │
│ LoRA FP16        │ 14GB  │ 0.1GB     │ 0.05GB    │ ~15GB      │
│ QLoRA NF4        │ 3.5GB │ 0.1GB     │ 0.05GB    │ ~4GB       │
│ QLoRA + Gradient │ 3.5GB │ 0.1GB     │ ~0GB*     │ ~4GB       │
│ Checkpointing    │       │           │           │            │
└────────────────────────────────────────────────────────────────┘

* Gradient checkpointing trades memory for compute

Hardware Requirements:
  Full Fine-tuning 7B:  4× A100 80GB
  LoRA 7B:              1× A100 40GB
  QLoRA 7B:             1× RTX 3090/4090 (24GB)
  QLoRA 7B + 8bit opt:  1× RTX 3080 (10GB)
```

## Prefix-Tuning

Prefix-tuning prepends learnable continuous vectors (prefixes) to the input, steering model behavior without modifying weights.

### Soft Prompts

```
PREFIX-TUNING MECHANISM:
═══════════════════════════════════════════════════════════════════

Traditional Prompting (Discrete):
  Input: "Classify sentiment: [text]" → Hardcoded tokens

Prefix-Tuning (Continuous):
  Input: [P₁, P₂, ..., Pₘ] + [text tokens] → Learned embeddings

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Prefix (learned)           │ Input Text (frozen encoding)   ││
│  │ [P₁] [P₂] [P₃] ... [Pₘ]   │ [CLS] The stock ... [SEP]      ││
│  └─────────────────────────────────────────────────────────────┘│
│           │                              │                       │
│           ▼                              ▼                       │
│  ┌────────────────┐           ┌────────────────────┐            │
│  │  Prefix MLP    │           │  Frozen LLM Body   │            │
│  │  (trainable)   │           │  (no gradients)    │            │
│  └────────────────┘           └────────────────────┘            │
│           │                              │                       │
│           └──────────────────────────────┘                       │
│                       │                                          │
│                       ▼                                          │
│            ┌────────────────────┐                               │
│            │   Output / Loss    │                               │
│            └────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘

Prefix Parameters:
  m = prefix length (typically 10-100 tokens)
  Each prefix token has dimension d (hidden size)
  Total params: m × d × num_layers (for deep prefix)
```

### Virtual Tokens

```python
import torch
import torch.nn as nn

class PrefixTuningLayer(nn.Module):
    """
    Prefix-tuning implementation for financial LLM adaptation.

    Uses learned prefix embeddings to steer model behavior for
    financial tasks without modifying the base model weights.
    """

    def __init__(
        self,
        num_prefix_tokens: int = 20,
        hidden_dim: int = 768,
        num_layers: int = 12,
        prefix_projection: bool = True,
        prefix_hidden_dim: int = 512
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if prefix_projection:
            # Two-stage: embedding → MLP → prefix
            self.prefix_embedding = nn.Embedding(
                num_prefix_tokens,
                prefix_hidden_dim
            )
            self.prefix_mlp = nn.Sequential(
                nn.Linear(prefix_hidden_dim, prefix_hidden_dim),
                nn.Tanh(),
                nn.Linear(prefix_hidden_dim, num_layers * 2 * hidden_dim)
            )
        else:
            # Direct prefix parameters
            self.prefix_embedding = nn.Embedding(
                num_prefix_tokens,
                num_layers * 2 * hidden_dim
            )
            self.prefix_mlp = nn.Identity()

    def forward(self, batch_size: int) -> tuple:
        """
        Generate prefix key-value pairs for all layers.

        Returns:
            Tuple of (prefix_keys, prefix_values) for each layer
        """
        prefix_tokens = torch.arange(self.num_prefix_tokens).unsqueeze(0)
        prefix_tokens = prefix_tokens.expand(batch_size, -1)

        # Get prefix embeddings and project
        prefix_embeds = self.prefix_embedding(prefix_tokens)
        prefix_output = self.prefix_mlp(prefix_embeds)

        # Reshape to (batch, layers, 2, num_prefix, hidden)
        prefix_output = prefix_output.view(
            batch_size,
            self.num_prefix_tokens,
            self.num_layers,
            2,
            self.hidden_dim
        )
        prefix_output = prefix_output.permute(2, 3, 0, 1, 4)

        # Split into keys and values for each layer
        prefix_keys = prefix_output[:, 0]   # (layers, batch, prefix, hidden)
        prefix_values = prefix_output[:, 1]

        return prefix_keys, prefix_values


class FinancialPrefixClassifier(nn.Module):
    """
    Financial text classifier using prefix-tuning.

    Adapts a frozen transformer for financial sentiment classification
    by learning task-specific prefix tokens.
    """

    def __init__(
        self,
        base_model,  # Frozen HuggingFace model
        num_prefix_tokens: int = 20,
        num_classes: int = 3
    ):
        super().__init__()
        self.base_model = base_model

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Prefix tuning layer
        self.prefix_tuning = PrefixTuningLayer(
            num_prefix_tokens=num_prefix_tokens,
            hidden_dim=base_model.config.hidden_size,
            num_layers=base_model.config.num_hidden_layers
        )

        # Classification head
        self.classifier = nn.Linear(
            base_model.config.hidden_size,
            num_classes
        )

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)

        # Get prefix key-values
        prefix_keys, prefix_values = self.prefix_tuning(batch_size)

        # Extend attention mask for prefix
        prefix_attention = torch.ones(
            batch_size,
            self.prefix_tuning.num_prefix_tokens,
            device=attention_mask.device
        )
        extended_attention = torch.cat(
            [prefix_attention, attention_mask],
            dim=1
        )

        # Forward through model with prefix
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=extended_attention,
            past_key_values=list(zip(prefix_keys, prefix_values))
        )

        # Classify from [CLS] token
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)

        return logits
```

### Comparison with LoRA

| Aspect | LoRA | Prefix-Tuning |
|--------|------|---------------|
| Where Applied | Weight matrices | Input sequence |
| Parameters | ~0.5% of model | ~0.1% of model |
| Sequence Length Impact | None | Reduces effective context |
| Multi-task | Separate adapters | Separate prefixes |
| Generation Quality | Better | May affect fluency |
| Classification | Good | Very good |
| Best For | General adaptation | Task-specific steering |

## Financial Applications

### Sentiment Analysis

Fine-tuning for financial sentiment with LoRA:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def create_financial_sentiment_model():
    """
    Create a LoRA-adapted model for financial sentiment analysis.

    Labels:
        0: Bearish (negative sentiment, sell signal)
        1: Neutral (no clear direction)
        2: Bullish (positive sentiment, buy signal)
    """
    # Load base model
    model_name = "ProsusAI/finbert"  # or "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        problem_type="single_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                          # Rank
        lora_alpha=16,                # Scaling
        lora_dropout=0.1,             # Dropout
        target_modules=["query", "value"],  # Apply to Q and V projections
        bias="none"
    )

    # Create PEFT model
    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()
    # Output: trainable params: 294,912 || all params: 109,777,923 || trainable%: 0.27%

    return peft_model, tokenizer


# Example training data format
financial_examples = [
    {
        "text": "Apple beats earnings expectations, raises dividend by 10%",
        "label": 2  # Bullish
    },
    {
        "text": "Fed signals aggressive rate hikes, markets tumble",
        "label": 0  # Bearish
    },
    {
        "text": "Company maintains Q4 guidance amid mixed economic signals",
        "label": 1  # Neutral
    },
    {
        "text": "Bitcoin surges past $50K on institutional buying",
        "label": 2  # Bullish (crypto)
    },
    {
        "text": "Bybit reports record trading volume as BTC volatility spikes",
        "label": 1  # Neutral (market activity, not direction)
    }
]
```

### Market Prediction

```python
class MarketDirectionPredictor(nn.Module):
    """
    Fine-tuned LLM for market direction prediction.

    Combines textual signals (news, sentiment) with numerical features
    (price, volume) for next-day direction prediction.
    """

    def __init__(
        self,
        text_model,           # LoRA-adapted transformer
        numerical_features: int = 10,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.text_encoder = text_model

        # Numerical feature processor
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Fusion layer
        text_dim = text_model.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # Down, Flat, Up
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        # Encode text
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_embedding = text_output.hidden_states[-1][:, 0]  # CLS token

        # Encode numerical features
        num_embedding = self.numerical_encoder(numerical_features)

        # Fuse and predict
        combined = torch.cat([text_embedding, num_embedding], dim=-1)
        logits = self.fusion(combined)

        return logits


# Numerical features for market prediction
def prepare_market_features(df):
    """
    Prepare numerical features for market prediction.

    Features:
    - Price momentum (returns over various windows)
    - Volume indicators
    - Volatility measures
    - Technical indicators
    """
    features = {
        'return_1d': df['close'].pct_change(1),
        'return_5d': df['close'].pct_change(5),
        'return_20d': df['close'].pct_change(20),
        'volume_ratio': df['volume'] / df['volume'].rolling(20).mean(),
        'volatility_20d': df['close'].pct_change().rolling(20).std(),
        'rsi_14': compute_rsi(df['close'], 14),
        'macd': compute_macd(df['close']),
        'bb_position': compute_bollinger_position(df['close']),
        'atr_ratio': compute_atr(df) / df['close'],
        'vwap_distance': (df['close'] - compute_vwap(df)) / df['close']
    }
    return pd.DataFrame(features)
```

### Trading Signal Generation

```python
class LLMTradingSignalGenerator:
    """
    Generate trading signals using fine-tuned LLM.

    Combines sentiment analysis with confidence scoring
    to produce actionable trading signals.
    """

    def __init__(
        self,
        sentiment_model,
        tokenizer,
        confidence_threshold: float = 0.7
    ):
        self.model = sentiment_model
        self.tokenizer = tokenizer
        self.threshold = confidence_threshold
        self.label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

    def generate_signal(self, text: str) -> dict:
        """
        Generate trading signal from text.

        Args:
            text: Financial news or analysis text

        Returns:
            Dict with signal, confidence, and raw scores
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        confidence, prediction = probs.max(dim=-1)

        # Generate signal
        signal = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": self.label_map[prediction.item()],
            "confidence": confidence.item(),
            "scores": {
                "bearish": probs[0, 0].item(),
                "neutral": probs[0, 1].item(),
                "bullish": probs[0, 2].item()
            },
            "actionable": confidence.item() >= self.threshold
        }

        return signal

    def batch_signals(self, texts: list) -> list:
        """Generate signals for multiple texts."""
        return [self.generate_signal(text) for text in texts]

    def aggregate_signals(self, signals: list) -> dict:
        """
        Aggregate multiple signals into a composite signal.

        Uses confidence-weighted voting.
        """
        if not signals:
            return {"signal": "HOLD", "confidence": 0.0}

        weighted_scores = {"SELL": 0, "HOLD": 0, "BUY": 0}
        total_weight = 0

        for sig in signals:
            weight = sig["confidence"]
            weighted_scores[sig["prediction"]] += weight
            total_weight += weight

        # Normalize
        for key in weighted_scores:
            weighted_scores[key] /= total_weight

        # Get final signal
        final_signal = max(weighted_scores, key=weighted_scores.get)

        return {
            "signal": final_signal,
            "confidence": weighted_scores[final_signal],
            "score_breakdown": weighted_scores,
            "num_sources": len(signals)
        }
```

## Practical Examples

### 01: Fine-tuning for Financial Sentiment

See `python/examples/01_sentiment_finetuning.py` for complete implementation.

```python
# Quick start example
from python.trainer import FineTuningTrainer
from python.data_loader import load_financial_phrasebank

# Load data
train_data, val_data = load_financial_phrasebank()

# Create trainer with LoRA
trainer = FineTuningTrainer(
    model_name="ProsusAI/finbert",
    method="lora",
    lora_rank=8,
    learning_rate=2e-4
)

# Train
trainer.train(train_data, val_data, epochs=3)

# Evaluate
metrics = trainer.evaluate(val_data)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

### 02: Crypto Market Analysis with Bybit Data

See `python/examples/02_crypto_analysis.py` for complete implementation.

```python
# Crypto sentiment analysis with Bybit data
from python.data_loader import BybitDataLoader
from python.signals import CryptoSignalGenerator

# Initialize Bybit loader
bybit = BybitDataLoader()

# Get recent market data
btc_data = bybit.get_klines(
    symbol="BTCUSDT",
    interval="1h",
    limit=1000
)

# Load fine-tuned model
signal_gen = CryptoSignalGenerator.from_pretrained(
    "outputs/crypto_sentiment_model"
)

# Generate signals from news
news_texts = [
    "Bitcoin whales accumulate as price consolidates near support",
    "Regulatory concerns weigh on crypto market sentiment",
    "Bybit launches new perpetual contracts with reduced fees"
]

signals = signal_gen.batch_signals(news_texts)
composite = signal_gen.aggregate_signals(signals)

print(f"Composite Signal: {composite['signal']}")
print(f"Confidence: {composite['confidence']:.2%}")
```

### 03: Backtesting Fine-tuned Models

See `python/examples/03_backtest.py` for complete implementation.

```python
# Backtest LLM trading signals
from python.backtest import LLMBacktester
from python.data_loader import YahooFinanceLoader

# Load historical data
yahoo = YahooFinanceLoader()
spy_data = yahoo.get_daily("SPY", start="2020-01-01", end="2024-01-01")

# Initialize backtester with fine-tuned model
backtester = LLMBacktester(
    model_path="outputs/sentiment_model",
    initial_capital=100000,
    position_size=0.1,
    confidence_threshold=0.7
)

# Run backtest with news data
results = backtester.run(
    price_data=spy_data,
    news_data=news_headlines,  # Historical news headlines
    signal_aggregation="confidence_weighted"
)

# Print metrics
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

## Rust Implementation

The Rust implementation provides high-performance inference for production deployment. See `rust/` directory for complete code.

```rust
//! Financial LLM Fine-tuning - Rust Implementation
//!
//! This crate provides efficient inference for fine-tuned models,
//! designed for low-latency trading signal generation.

use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, Module};
use serde::{Deserialize, Serialize};

/// LoRA layer for efficient model adaptation
pub struct LoraLayer {
    lora_a: Tensor,      // (rank, in_features)
    lora_b: Tensor,      // (out_features, rank)
    scaling: f64,
    rank: usize,
}

impl LoraLayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f64,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let lora_a = vb.get((rank, in_features), "lora_a")?;
        let lora_b = vb.get((out_features, rank), "lora_b")?;

        Ok(Self {
            lora_a,
            lora_b,
            scaling: alpha / rank as f64,
            rank,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Compute BA @ x with scaling
        let intermediate = x.matmul(&self.lora_a.t()?)?;
        let output = intermediate.matmul(&self.lora_b.t()?)?;
        output.affine(self.scaling, 0.0)
    }
}

/// Trading signal generated by the fine-tuned model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub direction: SignalDirection,
    pub confidence: f64,
    pub scores: SentimentScores,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SignalDirection {
    Buy,
    Hold,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScores {
    pub bullish: f64,
    pub neutral: f64,
    pub bearish: f64,
}

/// High-performance signal generator for production use
pub struct SignalGenerator {
    model: FineTunedModel,
    tokenizer: tokenizers::Tokenizer,
    confidence_threshold: f64,
}

impl SignalGenerator {
    pub fn from_pretrained(path: &str) -> anyhow::Result<Self> {
        let model = FineTunedModel::load(path)?;
        let tokenizer = tokenizers::Tokenizer::from_file(
            format!("{}/tokenizer.json", path)
        )?;

        Ok(Self {
            model,
            tokenizer,
            confidence_threshold: 0.7,
        })
    }

    pub fn generate(&self, text: &str) -> anyhow::Result<TradingSignal> {
        // Tokenize
        let encoding = self.tokenizer.encode(text, true)?;
        let tokens = encoding.get_ids();

        // Create tensor
        let device = Device::Cpu;
        let input_ids = Tensor::new(tokens, &device)?;

        // Forward pass
        let logits = self.model.forward(&input_ids)?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;

        // Extract predictions
        let probs_vec: Vec<f64> = probs.to_vec1()?;
        let (max_idx, max_prob) = probs_vec.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let direction = match max_idx {
            0 => SignalDirection::Sell,
            1 => SignalDirection::Hold,
            _ => SignalDirection::Buy,
        };

        Ok(TradingSignal {
            direction,
            confidence: *max_prob,
            scores: SentimentScores {
                bearish: probs_vec[0],
                neutral: probs_vec[1],
                bullish: probs_vec[2],
            },
            timestamp: chrono::Utc::now().timestamp(),
        })
    }

    pub fn batch_generate(&self, texts: &[&str]) -> anyhow::Result<Vec<TradingSignal>> {
        texts.iter()
            .map(|text| self.generate(text))
            .collect()
    }
}
```

## Python Implementation

The Python implementation includes complete training and evaluation pipelines. See `python/` directory for full code.

**Key modules:**

| Module | Description |
|--------|-------------|
| `model.py` | LoRA, QLoRA, and prefix-tuning implementations |
| `trainer.py` | Training loop with early stopping and checkpointing |
| `data_loader.py` | Yahoo Finance and Bybit data loaders |
| `signals.py` | Trading signal generation and aggregation |
| `backtest.py` | Backtesting framework for LLM signals |
| `evaluate.py` | Evaluation metrics (accuracy, F1, Sharpe, etc.) |

## Best Practices

### Training Guidelines

```
FINE-TUNING BEST PRACTICES:
═══════════════════════════════════════════════════════════════════

1. DATA PREPARATION
   ✓ Balance classes (oversample minority, use focal loss)
   ✓ Clean financial jargon consistently
   ✓ Include temporal context in text
   ✓ Separate train/val/test by time (no future leakage)

2. HYPERPARAMETER SELECTION
   ✓ Start with r=8 for LoRA, increase if underfitting
   ✓ Use alpha = 2 × rank as baseline
   ✓ Learning rate: 1e-4 to 3e-4 for adapters
   ✓ Batch size: 8-32 (accumulate gradients if limited GPU)

3. REGULARIZATION
   ✓ LoRA dropout: 0.05-0.1
   ✓ Weight decay: 0.01-0.1
   ✓ Early stopping on validation loss
   ✓ Gradient clipping: max_norm=1.0

4. EVALUATION
   ✓ Use time-based train/val/test split
   ✓ Report both classification metrics AND trading metrics
   ✓ Test on multiple market regimes
   ✓ Calculate statistical significance

5. DEPLOYMENT
   ✓ Quantize model for inference (INT8)
   ✓ Batch predictions when possible
   ✓ Monitor prediction latency
   ✓ Implement confidence thresholds
```

### Common Pitfalls

```
COMMON MISTAKES TO AVOID:
═══════════════════════════════════════════════════════════════════

❌ Using future data in training
   → Always use strict temporal splits

❌ Ignoring class imbalance
   → Financial sentiment is often skewed; use weighted loss

❌ Over-relying on accuracy
   → Use F1, precision, recall for imbalanced data

❌ No out-of-sample testing
   → Test on held-out time periods

❌ Ignoring transaction costs
   → Include costs in backtest metrics

❌ Overfitting to specific market regime
   → Validate across bull/bear/sideways markets

❌ Using too high LoRA rank
   → Can overfit with small datasets; start with r=4-8

❌ Not monitoring forgetting
   → Check base capabilities periodically
```

## Resources

### Papers

1. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
   - https://arxiv.org/abs/2106.09685

2. **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)
   - https://arxiv.org/abs/2305.14314

3. **Prefix-Tuning: Optimizing Continuous Prompts** (Li & Liang, 2021)
   - https://arxiv.org/abs/2101.00190

4. **FinBERT: Financial Sentiment Analysis** (Araci, 2019)
   - https://arxiv.org/abs/1908.10063

5. **BloombergGPT: A Large Language Model for Finance** (Wu et al., 2023)
   - https://arxiv.org/abs/2303.17564

### Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| Financial PhraseBank | Sentiment-labeled financial news | 4,840 sentences |
| FiQA | Financial question answering | 17,000+ QA pairs |
| SemEval-2017 Task 5 | Sentiment in financial microblogs | 2,000+ texts |
| Crypto Sentiment | Twitter crypto sentiment | 10,000+ tweets |

### Tools & Libraries

- [HuggingFace PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 4-bit quantization
- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data
- [ccxt](https://github.com/ccxt/ccxt) - Crypto exchange data (Bybit)

### Directory Structure

```
70_fine_tuning_llm_finance/
├── README.md              # This file (English)
├── README.ru.md           # Russian translation
├── readme.simple.md       # Beginner-friendly explanation
├── readme.simple.ru.md    # Beginner-friendly (Russian)
├── python/
│   ├── __init__.py
│   ├── model.py           # LoRA/QLoRA/Prefix implementations
│   ├── trainer.py         # Training pipeline
│   ├── data_loader.py     # Yahoo Finance & Bybit loaders
│   ├── signals.py         # Signal generation
│   ├── backtest.py        # Backtesting framework
│   ├── evaluate.py        # Evaluation metrics
│   ├── requirements.txt   # Python dependencies
│   └── examples/
│       ├── 01_sentiment_finetuning.py
│       ├── 02_crypto_analysis.py
│       └── 03_backtest.py
└── rust/
    ├── Cargo.toml
    ├── README.md
    └── src/
        ├── lib.rs         # Library root
        ├── lora.rs        # LoRA implementation
        ├── model.rs       # Model loading
        ├── signals.rs     # Signal generation
        ├── data.rs        # Data loading
        └── bin/
            ├── sentiment.rs
            └── backtest.rs
```
