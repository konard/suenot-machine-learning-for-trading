"""
Deep Ensemble Model Implementation

This module implements Deep Ensembles for uncertainty-aware predictions
in cryptocurrency trading.
"""

import numpy as np
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""
    mean: np.ndarray  # Mean prediction
    total_std: np.ndarray  # Total standard deviation
    epistemic_std: np.ndarray  # Epistemic uncertainty (model disagreement)
    aleatoric_std: np.ndarray  # Aleatoric uncertainty (data noise)
    individual_means: np.ndarray  # Individual model means
    individual_stds: np.ndarray  # Individual model stds


class GaussianMLP:
    """
    Multi-layer perceptron that outputs Gaussian parameters (mean and std).
    Implements manual forward/backward pass for educational purposes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize the Gaussian MLP.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate (not used in inference)
            seed: Random seed for initialization
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        if seed is not None:
            np.random.seed(seed)

        # Initialize weights using Xavier initialization
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            std = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            w = np.random.randn(dims[i], dims[i+1]) * std
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

        # Output layers for mean and log_std
        out_std = np.sqrt(2.0 / (hidden_dims[-1] + 1))
        self.w_mean = np.random.randn(hidden_dims[-1], 1) * out_std
        self.b_mean = np.zeros(1)

        self.w_log_std = np.random.randn(hidden_dims[-1], 1) * out_std
        self.b_log_std = np.zeros(1)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus activation for positive outputs."""
        return np.log(1 + np.exp(np.clip(x, -20, 20)))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Tuple of (mean, std) predictions
        """
        self.activations = [x]

        # Hidden layers
        h = x
        for w, b in zip(self.weights, self.biases):
            z = h @ w + b
            h = self.relu(z)
            self.activations.append(h)

        # Output layers
        mean = h @ self.w_mean + self.b_mean
        log_std = h @ self.w_log_std + self.b_log_std

        # Ensure std is positive with minimum value
        std = self.softplus(log_std) + 1e-6

        return mean.flatten(), std.flatten()

    def nll_loss(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Negative Log-Likelihood loss for Gaussian distribution.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            target: True target values

        Returns:
            NLL loss value
        """
        # NLL = 0.5 * log(2*pi) + log(std) + 0.5 * ((target - mean) / std)^2
        nll = 0.5 * np.log(2 * np.pi) + np.log(std) + 0.5 * ((target - mean) / std) ** 2
        return np.mean(nll)

    def backward(
        self,
        target: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        learning_rate: float = 0.001
    ):
        """
        Backward pass with gradient descent update.

        Args:
            target: True target values
            mean: Predicted mean
            std: Predicted standard deviation
            learning_rate: Learning rate for gradient descent
        """
        batch_size = len(target)

        # Gradient of NLL w.r.t. mean: (mean - target) / std^2
        d_mean = (mean - target) / (std ** 2) / batch_size

        # Gradient of NLL w.r.t. std: 1/std - (target - mean)^2 / std^3
        d_std = (1 / std - ((target - mean) ** 2) / (std ** 3)) / batch_size

        # Gradient through softplus for log_std
        log_std = np.log(std - 1e-6)
        d_log_std = d_std * std / (1 + np.exp(-log_std))

        # Reshape for matrix operations
        d_mean = d_mean.reshape(-1, 1)
        d_log_std = d_log_std.reshape(-1, 1)

        # Get last hidden activation
        h = self.activations[-1]

        # Gradients for output layers
        self.w_mean -= learning_rate * (h.T @ d_mean)
        self.b_mean -= learning_rate * np.sum(d_mean, axis=0)

        self.w_log_std -= learning_rate * (h.T @ d_log_std)
        self.b_log_std -= learning_rate * np.sum(d_log_std, axis=0)

        # Backpropagate through hidden layers
        d_h = d_mean @ self.w_mean.T + d_log_std @ self.w_log_std.T

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient through ReLU
            d_h = d_h * self.relu_derivative(self.activations[i+1])

            # Update weights and biases
            self.weights[i] -= learning_rate * (self.activations[i].T @ d_h)
            self.biases[i] -= learning_rate * np.sum(d_h, axis=0)

            # Propagate gradient
            if i > 0:
                d_h = d_h @ self.weights[i].T

    def to_dict(self) -> Dict:
        """Convert model parameters to dictionary for serialization."""
        return {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'w_mean': self.w_mean.tolist(),
            'b_mean': self.b_mean.tolist(),
            'w_log_std': self.w_log_std.tolist(),
            'b_log_std': self.b_log_std.tolist()
        }

    def from_dict(self, data: Dict):
        """Load model parameters from dictionary."""
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]
        self.w_mean = np.array(data['w_mean'])
        self.b_mean = np.array(data['b_mean'])
        self.w_log_std = np.array(data['w_log_std'])
        self.b_log_std = np.array(data['b_log_std'])


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty-aware predictions.

    Trains multiple neural networks with different random initializations
    to capture epistemic uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_models: int = 5,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the Deep Ensemble.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            num_models: Number of ensemble members
            dropout_rate: Dropout rate for each model
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_models = num_models
        self.dropout_rate = dropout_rate

        # Create ensemble members with different random seeds
        self.models = [
            GaussianMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                seed=42 + i  # Different seed for each model
            )
            for i in range(num_models)
        ]

        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train all ensemble members.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }

        n_samples = len(X_train)
        n_batches = n_samples // batch_size

        for model_idx, model in enumerate(self.models):
            if verbose:
                print(f"\nTraining model {model_idx + 1}/{self.num_models}")

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                # Shuffle data
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                epoch_loss = 0

                for batch_idx in range(n_batches):
                    start = batch_idx * batch_size
                    end = start + batch_size

                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    # Forward pass
                    mean, std = model.forward(X_batch)

                    # Compute loss
                    loss = model.nll_loss(mean, std, y_batch)
                    epoch_loss += loss

                    # Backward pass
                    model.backward(y_batch, mean, std, learning_rate)

                epoch_loss /= n_batches

                # Validation
                val_loss = 0
                if X_val is not None and y_val is not None:
                    mean_val, std_val = model.forward(X_val)
                    val_loss = model.nll_loss(mean_val, std_val, y_val)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}, "
                          f"Train Loss: {epoch_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")

            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_loss if X_val is not None else 0)

        self.is_trained = True
        return history

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Make predictions with uncertainty estimation.

        Args:
            X: Input features [n_samples, n_features]

        Returns:
            EnsemblePrediction with mean, std, and uncertainty decomposition
        """
        if not self.is_trained:
            print("Warning: Model has not been trained yet!")

        n_samples = len(X)
        individual_means = np.zeros((self.num_models, n_samples))
        individual_stds = np.zeros((self.num_models, n_samples))

        # Get predictions from each model
        for i, model in enumerate(self.models):
            mean, std = model.forward(X)
            individual_means[i] = mean
            individual_stds[i] = std

        # Ensemble mean
        ensemble_mean = np.mean(individual_means, axis=0)

        # Epistemic uncertainty: variance of means (model disagreement)
        epistemic_var = np.var(individual_means, axis=0)

        # Aleatoric uncertainty: mean of variances (data noise)
        aleatoric_var = np.mean(individual_stds ** 2, axis=0)

        # Total uncertainty
        total_var = epistemic_var + aleatoric_var

        return EnsemblePrediction(
            mean=ensemble_mean,
            total_std=np.sqrt(total_var),
            epistemic_std=np.sqrt(epistemic_var),
            aleatoric_std=np.sqrt(aleatoric_var),
            individual_means=individual_means,
            individual_stds=individual_stds
        )

    def predict_with_samples(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Generate prediction samples for visualization.

        Args:
            X: Input features
            n_samples: Number of samples to generate

        Returns:
            Array of prediction samples [n_samples, n_inputs]
        """
        pred = self.predict(X)

        # Sample from the predictive distribution
        samples = np.random.normal(
            loc=pred.mean,
            scale=pred.total_std,
            size=(n_samples, len(pred.mean))
        )

        return samples

    def save(self, filepath: str):
        """Save ensemble to JSON file."""
        data = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'num_models': self.num_models,
            'dropout_rate': self.dropout_rate,
            'models': [m.to_dict() for m in self.models],
            'is_trained': self.is_trained
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'DeepEnsemble':
        """Load ensemble from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        ensemble = cls(
            input_dim=data['input_dim'],
            hidden_dims=data['hidden_dims'],
            num_models=data['num_models'],
            dropout_rate=data['dropout_rate']
        )

        for i, model_data in enumerate(data['models']):
            ensemble.models[i].from_dict(model_data)

        ensemble.is_trained = data['is_trained']
        return ensemble


def compute_calibration_error(
    predictions: EnsemblePrediction,
    targets: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        predictions: Ensemble predictions
        targets: True target values
        num_bins: Number of bins for calibration

    Returns:
        ECE value
    """
    # Compute normalized confidence
    confidences = 1 - predictions.total_std / np.max(predictions.total_std)

    # Compute accuracy (whether target is within 1 std)
    errors = np.abs(predictions.mean - targets)
    accuracies = (errors < predictions.total_std).astype(float)

    ece = 0
    for bin_lower in np.linspace(0, 1, num_bins):
        bin_upper = bin_lower + 1 / num_bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * np.mean(in_bin)

    return ece


if __name__ == "__main__":
    # Example usage
    print("Deep Ensemble - Example")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 8

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    weights = np.random.randn(n_features)
    y = (X @ weights + 0.5 * np.random.randn(n_samples)).astype(np.float32)

    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    # Create and train ensemble
    print("\nCreating Deep Ensemble with 5 models...")
    ensemble = DeepEnsemble(
        input_dim=n_features,
        hidden_dims=[64, 32],
        num_models=5
    )

    print("\nTraining ensemble...")
    history = ensemble.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        learning_rate=0.01,
        verbose=True
    )

    # Make predictions
    print("\nMaking predictions on test set...")
    predictions = ensemble.predict(X_test)

    print(f"\nPrediction Statistics:")
    print(f"  Mean prediction range: [{predictions.mean.min():.3f}, {predictions.mean.max():.3f}]")
    print(f"  Average total std: {predictions.total_std.mean():.3f}")
    print(f"  Average epistemic std: {predictions.epistemic_std.mean():.3f}")
    print(f"  Average aleatoric std: {predictions.aleatoric_std.mean():.3f}")

    # Compute metrics
    mse = np.mean((predictions.mean - y_test) ** 2)
    mae = np.mean(np.abs(predictions.mean - y_test))
    ece = compute_calibration_error(predictions, y_test)

    print(f"\nTest Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  ECE: {ece:.4f}")

    # Show individual model predictions for first sample
    print(f"\nIndividual model predictions for first test sample:")
    print(f"  True value: {y_test[0]:.3f}")
    for i in range(ensemble.num_models):
        print(f"  Model {i+1}: {predictions.individual_means[i, 0]:.3f} +/- {predictions.individual_stds[i, 0]:.3f}")
    print(f"  Ensemble: {predictions.mean[0]:.3f} +/- {predictions.total_std[0]:.3f}")
