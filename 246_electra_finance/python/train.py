"""
Training utilities for ELECTRA-based financial models.

This module provides functions for preparing financial text data,
training ELECTRA models, and evaluating classification performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import yfinance as yf


class FinancialTextDataset(Dataset):
    """
    PyTorch Dataset for financial text classification.

    Args:
        input_ids: Token ID array of shape (num_samples, max_length)
        attention_masks: Attention mask array of shape (num_samples, max_length)
        labels: Label array of shape (num_samples,)
    """

    def __init__(
        self,
        input_ids: np.ndarray,
        attention_masks: np.ndarray,
        labels: np.ndarray
    ):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_masks = torch.LongTensor(attention_masks)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]


def fetch_stock_data(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df


def generate_synthetic_news(
    price_data: pd.DataFrame,
    num_samples_per_day: int = 3
) -> pd.DataFrame:
    """
    Generate synthetic financial news based on price movements.

    This is used for demonstration when real news data is unavailable.
    In production, use actual financial news feeds.

    Args:
        price_data: DataFrame with OHLCV columns
        num_samples_per_day: Number of synthetic news items per day

    Returns:
        DataFrame with columns: date, text, label (0=neg, 1=neutral, 2=pos)
    """
    positive_templates = [
        "Stock rallied {pct:.1f}% on strong trading volume",
        "Shares surged as buyers dominated the session with {pct:.1f}% gain",
        "Bullish momentum drove prices up {pct:.1f}% today",
        "Market showed strength with {pct:.1f}% advance on heavy volume",
        "Strong buying pressure pushed the asset higher by {pct:.1f}%",
    ]

    negative_templates = [
        "Stock declined {pct:.1f}% amid heavy selling pressure",
        "Shares tumbled as sellers took control with {pct:.1f}% drop",
        "Bearish sentiment drove prices down {pct:.1f}% today",
        "Market weakness led to {pct:.1f}% decline on increased volume",
        "Selling pressure pushed the asset lower by {pct:.1f}%",
    ]

    neutral_templates = [
        "Stock traded flat with minimal price change of {pct:.1f}%",
        "Shares consolidated near previous close with {pct:.1f}% movement",
        "Market showed indecision with sideways action, moving {pct:.1f}%",
        "Trading remained range-bound with {pct:.1f}% change",
        "Mixed signals kept the asset near unchanged at {pct:.1f}%",
    ]

    rng = np.random.RandomState(42)
    records = []

    returns = price_data['Close'].pct_change().dropna()

    for date, ret in returns.items():
        pct = abs(ret * 100)
        for _ in range(num_samples_per_day):
            if ret > 0.005:
                template = rng.choice(positive_templates)
                label = 2  # positive
            elif ret < -0.005:
                template = rng.choice(negative_templates)
                label = 0  # negative
            else:
                template = rng.choice(neutral_templates)
                label = 1  # neutral

            text = template.format(pct=pct)
            records.append({
                'date': date,
                'text': text,
                'label': label
            })

    return pd.DataFrame(records)


def build_vocabulary(texts: List[str], max_vocab_size: int = 10000) -> Dict[str, int]:
    """
    Build a simple vocabulary from a list of texts.

    Args:
        texts: List of text strings
        max_vocab_size: Maximum vocabulary size

    Returns:
        Dictionary mapping tokens to integer IDs
    """
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
    word_counts = {}

    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, _ in sorted_words[:max_vocab_size - len(vocab)]:
        vocab[word] = len(vocab)

    return vocab


def tokenize_texts(
    texts: List[str],
    vocab: Dict[str, int],
    max_length: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tokenize a list of texts using a vocabulary.

    Args:
        texts: List of text strings
        vocab: Vocabulary dictionary (token -> ID)
        max_length: Maximum sequence length

    Returns:
        Tuple of (input_ids, attention_masks) arrays
    """
    all_input_ids = []
    all_attention_masks = []

    unk_id = vocab.get('[UNK]', 1)
    cls_id = vocab.get('[CLS]', 2)
    sep_id = vocab.get('[SEP]', 3)

    for text in texts:
        tokens = text.lower().split()[:max_length - 2]
        token_ids = [cls_id]
        token_ids += [vocab.get(t, unk_id) for t in tokens]
        token_ids += [sep_id]

        attention_mask = [1] * len(token_ids)

        padding_length = max_length - len(token_ids)
        token_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        all_input_ids.append(token_ids)
        all_attention_masks.append(attention_mask)

    return np.array(all_input_ids), np.array(all_attention_masks)


def prepare_text_data(
    texts: List[str],
    labels: List[int],
    max_length: int = 128,
    max_vocab_size: int = 10000,
    test_size: float = 0.2
) -> Tuple[Dict, Dict[str, int]]:
    """
    Prepare text data for training and evaluation.

    Args:
        texts: List of text strings
        labels: List of integer labels
        max_length: Maximum sequence length
        max_vocab_size: Maximum vocabulary size
        test_size: Fraction of data for testing

    Returns:
        Tuple of (data_dict, vocabulary) where data_dict contains
        train/test splits of input_ids, attention_masks, and labels
    """
    vocab = build_vocabulary(texts, max_vocab_size)
    input_ids, attention_masks = tokenize_texts(texts, vocab, max_length)
    labels = np.array(labels)

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    data = {
        'train_input_ids': input_ids[train_idx],
        'train_attention_masks': attention_masks[train_idx],
        'train_labels': labels[train_idx],
        'test_input_ids': input_ids[test_idx],
        'test_attention_masks': attention_masks[test_idx],
        'test_labels': labels[test_idx],
    }

    return data, vocab


def create_dataloaders(
    data: Dict,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from prepared data.

    Args:
        data: Dictionary with train/test data arrays
        batch_size: Batch size for DataLoaders

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = FinancialTextDataset(
        data['train_input_ids'],
        data['train_attention_masks'],
        data['train_labels']
    )
    test_dataset = FinancialTextDataset(
        data['test_input_ids'],
        data['test_attention_masks'],
        data['test_labels']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    device: str = 'cpu'
) -> List[Dict]:
    """
    Train a FinancialElectra model.

    Args:
        model: FinancialElectra model
        train_loader: Training DataLoader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for AdamW optimizer
        weight_decay: Weight decay for regularization
        device: Device for computation

    Returns:
        List of dictionaries with training history (loss, accuracy per epoch)
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for input_ids, attention_masks, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_masks)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        history.append({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_acc})

    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained FinancialElectra model
        test_loader: Test DataLoader
        device: Device for computation

    Returns:
        Dictionary with evaluation metrics (accuracy, f1_score, classification_report)
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_masks, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)

            logits = model(input_ids, attention_masks)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(
        all_labels, all_preds,
        target_names=['Negative', 'Neutral', 'Positive'],
        output_dict=True
    )

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels
    }
