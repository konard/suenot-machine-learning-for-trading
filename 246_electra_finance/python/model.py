"""
ELECTRA-based models for financial text classification.

This module provides neural network models based on the ELECTRA architecture
for financial sentiment analysis and event detection tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple


class ElectraGenerator(nn.Module):
    """
    Small transformer generator for the ELECTRA pre-training framework.

    The generator performs masked language modeling, producing plausible
    replacement tokens that the discriminator must detect.

    Args:
        vocab_size: Size of the token vocabulary
        hidden_size: Dimension of the hidden representations
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        intermediate_size: Dimension of the feed-forward intermediate layer
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        intermediate_size: int = 256,
        max_seq_length: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for masked language modeling.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)

        Returns:
            Logits over vocabulary of shape (batch_size, seq_length, vocab_size)
        """
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        embeddings = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        hidden_states = self.encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        logits = self.mlm_head(hidden_states)
        return logits


class ElectraDiscriminator(nn.Module):
    """
    Transformer discriminator for the ELECTRA pre-training framework.

    The discriminator classifies each token as original or replaced,
    learning from ALL input tokens rather than just masked positions.

    Args:
        vocab_size: Size of the token vocabulary
        hidden_size: Dimension of the hidden representations
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        intermediate_size: Dimension of the feed-forward intermediate layer
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        intermediate_size: int = 1024,
        max_seq_length: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.discriminator_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for replaced token detection.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)

        Returns:
            Logits of shape (batch_size, seq_length) indicating
            probability of each token being original
        """
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        embeddings = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        hidden_states = self.encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        logits = self.discriminator_head(hidden_states).squeeze(-1)
        return logits

    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the [CLS] token embedding for classification tasks.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)

        Returns:
            CLS embedding of shape (batch_size, hidden_size)
        """
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        embeddings = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        hidden_states = self.encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        return hidden_states[:, 0, :]


class FinancialElectra(nn.Module):
    """
    ELECTRA-based model for financial text classification.

    Uses the ELECTRA discriminator as a backbone with a classification
    head for financial sentiment analysis (positive/neutral/negative).

    Args:
        num_classes: Number of output classes (default: 3 for sentiment)
        vocab_size: Size of the token vocabulary
        hidden_size: Dimension of the hidden representations
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        intermediate_size: Dimension of the feed-forward intermediate layer
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 3,
        vocab_size: int = 30522,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        intermediate_size: int = 1024,
        max_seq_length: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        self.discriminator = ElectraDiscriminator(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        cls_output = self.discriminator.get_cls_embedding(input_ids, attention_mask)
        logits = self.classifier(cls_output)
        return logits


class ElectraTradingSignal:
    """
    Generate trading signals from financial text using ELECTRA.

    Args:
        model: Trained FinancialElectra model
        vocab: Dictionary mapping tokens to IDs
        device: Device for computation
    """

    def __init__(
        self,
        model: FinancialElectra,
        vocab: Dict[str, int],
        device: str = 'cpu'
    ):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device

    def tokenize(self, text: str, max_length: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple whitespace tokenization with vocabulary lookup.

        Args:
            text: Input text string
            max_length: Maximum sequence length

        Returns:
            Tuple of (input_ids, attention_mask) tensors
        """
        tokens = text.lower().split()[:max_length - 2]
        token_ids = [self.vocab.get('[CLS]', 0)]
        token_ids += [self.vocab.get(t, self.vocab.get('[UNK]', 1)) for t in tokens]
        token_ids += [self.vocab.get('[SEP]', 2)]

        attention_mask = [1] * len(token_ids)

        # Pad to max_length
        padding_length = max_length - len(token_ids)
        token_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        return (
            torch.tensor([token_ids], dtype=torch.long),
            torch.tensor([attention_mask], dtype=torch.long)
        )

    def predict_sentiment(self, texts: list) -> np.ndarray:
        """
        Predict sentiment for a list of financial texts.

        Args:
            texts: List of text strings

        Returns:
            Array of shape (num_texts, num_classes) with probabilities
        """
        all_probs = []
        for text in texts:
            input_ids, attention_mask = self.tokenize(text)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def generate_signal(
        self,
        news_items: list,
        threshold: float = 0.6
    ) -> Tuple[float, float]:
        """
        Generate a trading signal from multiple news items.

        Args:
            news_items: List of news text strings
            threshold: Minimum confidence to generate a signal

        Returns:
            Tuple of (signal, confidence) where signal is in [-1, 1]
            and confidence is the maximum class probability
        """
        if not news_items:
            return 0.0, 0.0

        probs = self.predict_sentiment(news_items)
        avg_probs = probs.mean(axis=0)

        # Classes: 0=negative, 1=neutral, 2=positive
        signal = float(avg_probs[2] - avg_probs[0])
        confidence = float(max(avg_probs[0], avg_probs[2]))

        if confidence < threshold:
            return 0.0, confidence

        return signal, confidence


def create_model(
    model_type: str = 'classifier',
    **kwargs
) -> nn.Module:
    """
    Factory function to create ELECTRA models.

    Args:
        model_type: Type of model ('classifier', 'generator', 'discriminator')
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Instantiated model
    """
    models = {
        'classifier': FinancialElectra,
        'generator': ElectraGenerator,
        'discriminator': ElectraDiscriminator,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](**kwargs)
