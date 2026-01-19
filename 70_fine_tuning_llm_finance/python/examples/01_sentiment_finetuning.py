#!/usr/bin/env python3
"""
Example 01: Financial Sentiment Analysis with LoRA Fine-tuning

This example demonstrates:
1. Loading financial news data
2. Creating a LoRA-based sentiment model
3. Training with early stopping
4. Evaluating on test data
5. Making predictions on new text

Usage:
    python 01_sentiment_finetuning.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import List, Tuple

from model import FinancialSentimentLoRA
from trainer import FineTuningTrainer, TrainingConfig
from data_loader import load_financial_phrasebank
from evaluate import evaluate_sentiment, generate_evaluation_report


def create_synthetic_dataset(n_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """
    Create a synthetic financial sentiment dataset for demonstration.

    In practice, you would load real financial news data.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (texts, labels)
    """
    # Sample financial phrases with different sentiments
    bullish_phrases = [
        "Revenue exceeded expectations with strong growth",
        "Company announces record-breaking quarterly profits",
        "Stock price rallies on positive earnings report",
        "Analysts upgrade rating to strong buy",
        "New product launch drives sales increase",
        "Merger creates significant shareholder value",
        "Expansion into new markets boosts outlook",
        "Cost reduction initiatives show positive results",
        "Customer acquisition accelerates beyond targets",
        "Innovation pipeline strengthens competitive position",
    ]

    neutral_phrases = [
        "Company reports quarterly results in line with expectations",
        "Management maintains current guidance",
        "Stock trades sideways amid market uncertainty",
        "Analysts keep hold rating unchanged",
        "Operations continue as planned",
        "Board approves standard dividend payment",
        "Company completes scheduled maintenance",
        "Industry trends remain stable",
        "Regulatory review proceeds normally",
        "Market share holds steady",
    ]

    bearish_phrases = [
        "Revenue misses analyst expectations significantly",
        "Company issues profit warning for next quarter",
        "Stock plunges on weak earnings report",
        "Analysts downgrade to sell recommendation",
        "Product recall damages brand reputation",
        "Key executive departures create uncertainty",
        "Competition intensifies in core markets",
        "Rising costs pressure profit margins",
        "Customer churn increases unexpectedly",
        "Debt levels raise concerns among investors",
    ]

    texts = []
    labels = []

    samples_per_class = n_samples // 3

    for _ in range(samples_per_class):
        # Bullish (label 0)
        texts.append(np.random.choice(bullish_phrases))
        labels.append(0)

        # Neutral (label 1)
        texts.append(np.random.choice(neutral_phrases))
        labels.append(1)

        # Bearish (label 2)
        texts.append(np.random.choice(bearish_phrases))
        labels.append(2)

    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels


def text_to_features(texts: List[str], dim: int = 768) -> torch.Tensor:
    """
    Convert text to feature vectors.

    In practice, you would use a pre-trained transformer encoder.
    This is a simplified simulation for demonstration.

    Args:
        texts: List of text strings
        dim: Feature dimension

    Returns:
        Tensor of shape (n_samples, dim)
    """
    features = []

    for text in texts:
        # Create a pseudo-random but deterministic feature vector based on text
        np.random.seed(hash(text) % (2**32))

        # Base features
        feat = np.random.randn(dim).astype(np.float32)

        # Add some structure based on sentiment keywords
        positive_words = ['growth', 'profit', 'rally', 'upgrade', 'strong', 'exceeds', 'boost']
        negative_words = ['miss', 'warning', 'plunge', 'downgrade', 'weak', 'decline', 'concern']

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Modify features based on sentiment
        feat[:50] += positive_count * 0.5
        feat[50:100] -= negative_count * 0.5

        features.append(feat)

    return torch.tensor(np.array(features), dtype=torch.float32)


def main():
    """Main function to run the sentiment fine-tuning example."""

    print("=" * 60)
    print("Financial Sentiment Analysis with LoRA Fine-tuning")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 1. Create dataset
    print("\n[1/5] Creating synthetic financial dataset...")
    texts, labels = create_synthetic_dataset(n_samples=900)

    # Split into train/val/test
    n_train = 600
    n_val = 150

    train_texts, train_labels = texts[:n_train], labels[:n_train]
    val_texts, val_labels = texts[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_texts, test_labels = texts[n_train+n_val:], labels[n_train+n_val:]

    print(f"  Train samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    print(f"  Test samples: {len(test_texts)}")

    # Convert to features
    print("\n[2/5] Converting text to features...")
    X_train = text_to_features(train_texts)
    X_val = text_to_features(val_texts)
    X_test = text_to_features(test_texts)

    y_train = torch.tensor(train_labels, dtype=torch.long)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    # 2. Create LoRA model
    print("\n[3/5] Creating LoRA-based sentiment model...")
    model = FinancialSentimentLoRA(
        base_dim=768,
        num_heads=12,
        lora_rank=8,
        lora_alpha=16.0,
        num_classes=3,
        dropout=0.1
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # 3. Configure training
    print("\n[4/5] Training the model...")
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=20,
        early_stopping_patience=5,
        weight_decay=0.01,
        warmup_steps=100,
        mixed_precision=False,  # Disable for CPU compatibility
        gradient_checkpointing=False,
    )

    # Create trainer
    trainer = FineTuningTrainer(
        model=model,
        config=config,
        device=device,
    )

    # Train
    metrics = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
    )

    print(f"\n  Training completed!")
    print(f"  Best validation loss: {metrics.best_val_loss:.4f}")
    print(f"  Best epoch: {metrics.best_epoch}")

    # 4. Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_test_device = X_test.to(device)
        outputs = model(X_test_device)
        predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

    # Calculate metrics
    results = evaluate_sentiment(
        predictions=predictions,
        labels=y_test.numpy(),
        label_names=['Bullish', 'Neutral', 'Bearish']
    )

    print(f"\n  Test Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")

    print("\n  Per-class metrics:")
    for class_name in ['Bullish', 'Neutral', 'Bearish']:
        class_metrics = results['per_class'][class_name]
        print(f"    {class_name}: P={class_metrics['precision']:.3f}, "
              f"R={class_metrics['recall']:.3f}, F1={class_metrics['f1']:.3f}")

    # 5. Example predictions
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)

    example_texts = [
        "Company announces 50% revenue growth in Q3",
        "Earnings report meets consensus estimates",
        "Stock drops 10% after missing guidance",
    ]

    label_names = ['Bullish', 'Neutral', 'Bearish']

    for text in example_texts:
        features = text_to_features([text])

        with torch.no_grad():
            output = model(features.to(device))
            probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
            pred = np.argmax(probs)

        print(f"\n  Text: \"{text}\"")
        print(f"  Prediction: {label_names[pred]} (confidence: {probs[pred]:.2f})")
        print(f"  Probabilities: Bullish={probs[0]:.2f}, Neutral={probs[1]:.2f}, Bearish={probs[2]:.2f}")

    print("\n" + "=" * 60)
    print("Fine-tuning example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
