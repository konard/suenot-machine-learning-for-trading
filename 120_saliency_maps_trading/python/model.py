"""
Trading models for saliency map computation.

This module provides neural network models for predicting price direction
that are compatible with gradient-based saliency map methods.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class TradingLSTM(nn.Module):
    """
    LSTM-based model for predicting price direction.

    This model takes a sequence of OHLCV features and predicts the probability
    of price increase in the next time step.

    Args:
        input_size: Number of features per time step
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability (applied between LSTM layers)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)

        Returns:
            Probability of price increase, shape (batch, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


class TradingCNN(nn.Module):
    """
    1D CNN model for predicting price direction.

    Uses convolutional layers to capture local patterns in time series data.

    Args:
        input_size: Number of features per time step
        sequence_length: Length of input sequence
        num_filters: Number of convolutional filters
        kernel_size: Size of convolutional kernel
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        num_filters: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length

        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        conv_output_size = (sequence_length // 2) * num_filters * 2

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)

        Returns:
            Probability of price increase, shape (batch, 1)
        """
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


class TradingTransformer(nn.Module):
    """
    Transformer-based model for predicting price direction.

    Uses self-attention to capture long-range dependencies in time series.

    Args:
        input_size: Number of features per time step
        d_model: Dimension of transformer model
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, features)

        Returns:
            Probability of price increase, shape (batch, 1)
        """
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        output = self.fc(x)
        return output


def create_model(
    model_type: str,
    input_size: int,
    sequence_length: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create trading models.

    Args:
        model_type: Type of model ('lstm', 'cnn', 'transformer')
        input_size: Number of features per time step
        sequence_length: Length of input sequence (required for CNN)
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    model_type = model_type.lower()

    if model_type == 'lstm':
        return TradingLSTM(input_size=input_size, **kwargs)
    elif model_type == 'cnn':
        if sequence_length is None:
            raise ValueError("sequence_length is required for CNN model")
        return TradingCNN(
            input_size=input_size,
            sequence_length=sequence_length,
            **kwargs
        )
    elif model_type == 'transformer':
        return TradingTransformer(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
