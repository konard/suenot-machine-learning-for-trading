"""
Prediction Utilities for GQA Trading Model

This module provides functions for making predictions with trained
GQA models, including efficient inference with KV caching.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .model import GQATrader


def predict_next(
    model: GQATrader,
    sequence: Union[np.ndarray, torch.Tensor],
    return_probs: bool = False,
    device: Optional[torch.device] = None
) -> Union[int, Tuple[int, np.ndarray]]:
    """
    Predict the next price movement from a single sequence.

    Args:
        model: Trained GQATrader model
        sequence: Input sequence of shape (seq_len, 5) or (1, seq_len, 5)
        return_probs: Whether to return class probabilities
        device: Inference device

    Returns:
        If return_probs: Tuple of (prediction, probabilities)
        Else: prediction (0=down, 1=neutral, 2=up)

    Example:
        >>> model = GQATrader.load("model.pt")
        >>> data = load_bybit_data("BTCUSDT", limit=100)
        >>> pred = predict_next(model, data[-60:])
        >>> print(f"Prediction: {'UP' if pred == 2 else 'DOWN' if pred == 0 else 'NEUTRAL'}")
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Convert to tensor if needed
    if isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence).float()

    # Add batch dimension if needed
    if sequence.dim() == 2:
        sequence = sequence.unsqueeze(0)

    sequence = sequence.to(device)

    with torch.no_grad():
        logits = model(sequence)
        probs = F.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1).item()

    if return_probs:
        return prediction, probs.cpu().numpy()[0]
    return prediction


def predict_batch(
    model: GQATrader,
    sequences: Union[np.ndarray, torch.Tensor],
    batch_size: int = 32,
    return_probs: bool = False,
    device: Optional[torch.device] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions for multiple sequences.

    Args:
        model: Trained GQATrader model
        sequences: Input sequences of shape (n_samples, seq_len, 5)
        batch_size: Batch size for inference
        return_probs: Whether to return class probabilities
        device: Inference device

    Returns:
        If return_probs: Tuple of (predictions, probabilities)
        Else: predictions array of shape (n_samples,)

    Example:
        >>> model = GQATrader.load("model.pt")
        >>> X, _ = prepare_sequences(data, seq_len=60)
        >>> predictions = predict_batch(model, X)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Convert to tensor if needed
    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences).float()

    n_samples = len(sequences)
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = sequences[i:i + batch_size].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

            all_predictions.append(predictions.cpu())
            if return_probs:
                all_probs.append(probs.cpu())

    predictions = torch.cat(all_predictions).numpy()

    if return_probs:
        probs = torch.cat(all_probs).numpy()
        return predictions, probs
    return predictions


def predict_with_cache(
    model: GQATrader,
    initial_sequence: torch.Tensor,
    new_data: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Efficient incremental prediction using KV cache.

    After processing the initial sequence, new data points are
    processed incrementally using cached key-value pairs, which
    is much faster than reprocessing the entire sequence.

    Args:
        model: Trained GQATrader model
        initial_sequence: Initial sequence to build cache from
        new_data: New data points to predict incrementally
        device: Inference device

    Returns:
        Tuple of (predictions, probabilities) for each new data point

    Example:
        >>> # Initial prediction
        >>> initial_seq = data[:100]
        >>> new_points = data[100:110]
        >>> preds, probs = predict_with_cache(model, initial_seq, new_points)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Ensure batch dimension
    if initial_sequence.dim() == 2:
        initial_sequence = initial_sequence.unsqueeze(0)
    if new_data.dim() == 2:
        new_data = new_data.unsqueeze(0)

    initial_sequence = initial_sequence.to(device)
    new_data = new_data.to(device)

    predictions = []
    probabilities = []

    with torch.no_grad():
        # Process initial sequence and build cache
        _, kv_cache = model(initial_sequence, use_cache=True)

        # Process each new data point incrementally
        for i in range(new_data.shape[1]):
            new_point = new_data[:, i:i+1, :]

            logits, kv_cache = model(new_point, use_cache=True, past_kv_cache=kv_cache)
            probs = F.softmax(logits, dim=-1)

            predictions.append(logits.argmax(dim=-1).item())
            probabilities.append(probs.cpu().numpy()[0])

    return predictions, probabilities


def get_confidence(probs: np.ndarray) -> float:
    """
    Calculate prediction confidence from probabilities.

    Args:
        probs: Class probabilities array

    Returns:
        Confidence score (0 to 1)
    """
    # Confidence = difference between top two probabilities
    sorted_probs = np.sort(probs)[::-1]
    return sorted_probs[0] - sorted_probs[1]


def get_signal_strength(probs: np.ndarray, threshold: float = 0.5) -> str:
    """
    Get trading signal strength based on probabilities.

    Args:
        probs: Class probabilities [down, neutral, up]
        threshold: Confidence threshold for strong signals

    Returns:
        Signal string: "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    """
    confidence = get_confidence(probs)
    prediction = np.argmax(probs)

    if prediction == 2:  # Up
        return "STRONG_BUY" if confidence > threshold else "BUY"
    elif prediction == 0:  # Down
        return "STRONG_SELL" if confidence > threshold else "SELL"
    else:  # Neutral
        return "HOLD"


def ensemble_predict(
    models: List[GQATrader],
    sequence: torch.Tensor,
    method: str = "average",
    device: Optional[torch.device] = None
) -> Tuple[int, np.ndarray]:
    """
    Make ensemble predictions from multiple models.

    Args:
        models: List of trained GQATrader models
        sequence: Input sequence
        method: Ensemble method ("average", "vote", "weighted")
        device: Inference device

    Returns:
        Tuple of (prediction, ensemble_probabilities)
    """
    if device is None:
        device = next(models[0].parameters()).device

    all_probs = []

    for model in models:
        _, probs = predict_next(model, sequence, return_probs=True, device=device)
        all_probs.append(probs)

    all_probs = np.array(all_probs)

    if method == "average":
        ensemble_probs = all_probs.mean(axis=0)
    elif method == "vote":
        votes = all_probs.argmax(axis=1)
        vote_counts = np.bincount(votes, minlength=3)
        ensemble_probs = vote_counts / len(models)
    elif method == "weighted":
        # Weight by confidence
        confidences = np.array([get_confidence(p) for p in all_probs])
        weights = confidences / confidences.sum()
        ensemble_probs = (all_probs * weights[:, np.newaxis]).sum(axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    prediction = ensemble_probs.argmax()
    return prediction, ensemble_probs


def analyze_prediction(
    model: GQATrader,
    sequence: torch.Tensor,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Analyze a prediction with detailed metrics.

    Args:
        model: Trained GQATrader model
        sequence: Input sequence
        device: Inference device

    Returns:
        Dictionary with prediction analysis
    """
    prediction, probs = predict_next(model, sequence, return_probs=True, device=device)

    labels = ["DOWN", "NEUTRAL", "UP"]

    return {
        "prediction": prediction,
        "prediction_label": labels[prediction],
        "probabilities": {
            "down": float(probs[0]),
            "neutral": float(probs[1]),
            "up": float(probs[2])
        },
        "confidence": float(get_confidence(probs)),
        "signal_strength": get_signal_strength(probs),
        "entropy": float(-np.sum(probs * np.log(probs + 1e-10))),
        "recommended_action": labels[prediction] if get_confidence(probs) > 0.3 else "WAIT"
    }


if __name__ == "__main__":
    # Test prediction utilities
    print("Testing Prediction Utilities...")
    print("=" * 50)

    from .model import GQATrader
    from .data import _generate_synthetic_data, prepare_sequences

    # Create and initialize model
    print("\n1. Creating model...")
    model = GQATrader(
        input_dim=5,
        d_model=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2
    )

    # Generate test data
    print("\n2. Generating test data...")
    data = _generate_synthetic_data(200)
    X, y = prepare_sequences(data, seq_len=30)

    # Test single prediction
    print("\n3. Testing single prediction...")
    pred, probs = predict_next(model, X[0], return_probs=True)
    print(f"   Prediction: {pred}")
    print(f"   Probabilities: {probs}")

    # Test batch prediction
    print("\n4. Testing batch prediction...")
    predictions = predict_batch(model, X[:50])
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Distribution: {np.bincount(predictions, minlength=3)}")

    # Test prediction analysis
    print("\n5. Testing prediction analysis...")
    analysis = analyze_prediction(model, X[0])
    print(f"   Signal: {analysis['signal_strength']}")
    print(f"   Confidence: {analysis['confidence']:.3f}")
    print(f"   Recommendation: {analysis['recommended_action']}")

    # Test incremental prediction with cache
    print("\n6. Testing cached prediction...")
    initial_seq = X[0:1, :20, :]  # First 20 timesteps
    new_data = X[0:1, 20:25, :]   # Next 5 timesteps

    preds, probs_list = predict_with_cache(model, initial_seq, new_data)
    print(f"   Incremental predictions: {preds}")

    print("\n" + "=" * 50)
    print("All prediction tests passed!")
