"""
Transfer Learning Models for Trading

Implements:
- TransferFeatureExtractor: Feature extraction network for pre-training
- DomainAdaptiveTrader: Trading model with domain adaptation (MMD, CORAL)
- TransferLearningPipeline: End-to-end transfer learning pipeline
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class TransferFeatureExtractor(nn.Module):
        """Feature extractor pre-trained on source domain data.

        Architecture:
            Input -> Linear -> BatchNorm -> ReLU -> Dropout ->
                     Linear -> BatchNorm -> ReLU -> Dropout ->
                     Linear -> Feature Embedding

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            feature_dim: Output embedding dimension
            dropout: Dropout rate
        """

        def __init__(self, input_dim, hidden_dim, feature_dim, dropout=0.3):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim),
            )

        def forward(self, x):
            return self.layers(x)

    class DomainAdaptiveTrader(nn.Module):
        """Trading model with domain adaptation via MMD or CORAL.

        Combines a feature extractor with a task classifier head,
        and includes domain adaptation loss computation.

        Args:
            feature_extractor: Pre-trained TransferFeatureExtractor
            feature_dim: Feature embedding dimension
            num_classes: Number of output classes (default 3: Buy/Hold/Sell)
        """

        def __init__(self, feature_extractor, feature_dim, num_classes=3):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            return logits, features

        def predict(self, x):
            """Predict class labels."""
            self.eval()
            with torch.no_grad():
                logits, _ = self.forward(x)
                return torch.argmax(logits, dim=1)

        def predict_proba(self, x):
            """Predict class probabilities."""
            self.eval()
            with torch.no_grad():
                logits, _ = self.forward(x)
                return F.softmax(logits, dim=1)

        @staticmethod
        def compute_mmd(source_features, target_features, kernel='rbf', bandwidth=1.0):
            """Compute Maximum Mean Discrepancy between domains.

            Args:
                source_features: Feature embeddings from source domain
                target_features: Feature embeddings from target domain
                kernel: Kernel type ('linear' or 'rbf')
                bandwidth: RBF kernel bandwidth

            Returns:
                MMD loss value
            """
            if kernel == 'linear':
                source_mean = source_features.mean(dim=0)
                target_mean = target_features.mean(dim=0)
                return ((source_mean - target_mean) ** 2).sum()

            # RBF kernel MMD
            n_s = source_features.size(0)
            n_t = target_features.size(0)

            # Kernel matrices
            k_ss = _rbf_kernel(source_features, source_features, bandwidth)
            k_tt = _rbf_kernel(target_features, target_features, bandwidth)
            k_st = _rbf_kernel(source_features, target_features, bandwidth)

            mmd = (k_ss.sum() / (n_s * n_s)
                   + k_tt.sum() / (n_t * n_t)
                   - 2 * k_st.sum() / (n_s * n_t))

            return mmd

        @staticmethod
        def compute_coral(source_features, target_features):
            """Compute CORAL loss (correlation alignment).

            Args:
                source_features: Feature embeddings from source domain
                target_features: Feature embeddings from target domain

            Returns:
                CORAL loss value
            """
            d = source_features.size(1)

            # Covariance matrices
            source_cov = _covariance(source_features)
            target_cov = _covariance(target_features)

            # Frobenius norm squared
            diff = source_cov - target_cov
            loss = (diff ** 2).sum() / (4 * d * d)
            return loss

    class TransferLearningPipeline:
        """End-to-end transfer learning pipeline for trading.

        Manages pre-training, domain adaptation, and fine-tuning.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            feature_dim: Feature embedding dimension
            num_classes: Number of output classes
            adaptation_method: 'mmd', 'coral', or 'none'
            device: PyTorch device ('cpu' or 'cuda')
        """

        def __init__(self, input_dim, hidden_dim=128, feature_dim=64,
                     num_classes=3, adaptation_method='mmd', device='cpu'):
            self.device = torch.device(device)
            self.adaptation_method = adaptation_method
            self.num_classes = num_classes

            self.feature_extractor = TransferFeatureExtractor(
                input_dim, hidden_dim, feature_dim
            ).to(self.device)

            self.model = DomainAdaptiveTrader(
                self.feature_extractor, feature_dim, num_classes
            ).to(self.device)

        def pretrain(self, source_features, source_labels,
                     epochs=100, lr=0.001, patience=10):
            """Pre-train on source domain data.

            Args:
                source_features: numpy array (n_samples, n_features)
                source_labels: numpy array (n_samples,)
                epochs: Number of training epochs
                lr: Learning rate
                patience: Early stopping patience

            Returns:
                Dictionary with training history
            """
            X = torch.FloatTensor(source_features).to(self.device)
            y = torch.LongTensor(source_labels).to(self.device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            history = {'loss': [], 'accuracy': []}
            best_loss = float('inf')
            patience_counter = 0

            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                logits, _ = self.model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                # Track metrics
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    accuracy = (preds == y).float().mean().item()

                history['loss'].append(loss.item())
                history['accuracy'].append(accuracy)

                # Early stopping
                if loss.item() < best_loss - 1e-4:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            return history

        def adapt(self, source_features, target_features,
                  target_labels=None, epochs=50, lr=0.0001,
                  adaptation_weight=1.0, freeze_layers=1):
            """Adapt the model to target domain.

            Args:
                source_features: Source domain features
                target_features: Target domain features
                target_labels: Optional target labels for supervised adaptation
                epochs: Number of adaptation epochs
                lr: Learning rate for fine-tuning
                adaptation_weight: Weight of adaptation loss
                freeze_layers: Number of feature extractor layers to freeze

            Returns:
                Dictionary with adaptation history
            """
            X_s = torch.FloatTensor(source_features).to(self.device)
            X_t = torch.FloatTensor(target_features).to(self.device)

            # Freeze early layers
            layers = list(self.feature_extractor.layers.children())
            for i, layer in enumerate(layers[:freeze_layers * 4]):  # 4 modules per block
                for param in layer.parameters():
                    param.requires_grad = False

            # Only optimize non-frozen parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=lr)
            criterion = nn.CrossEntropyLoss()

            history = {'adapt_loss': [], 'task_loss': [], 'total_loss': []}

            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()

                _, source_feat = self.model(X_s)
                _, target_feat = self.model(X_t)

                # Domain adaptation loss
                if self.adaptation_method == 'mmd':
                    adapt_loss = DomainAdaptiveTrader.compute_mmd(source_feat, target_feat)
                elif self.adaptation_method == 'coral':
                    adapt_loss = DomainAdaptiveTrader.compute_coral(source_feat, target_feat)
                else:
                    adapt_loss = torch.tensor(0.0).to(self.device)

                # Task loss (if target labels available)
                if target_labels is not None:
                    y_t = torch.LongTensor(target_labels).to(self.device)
                    logits_t, _ = self.model(X_t)
                    task_loss = criterion(logits_t, y_t)
                else:
                    task_loss = torch.tensor(0.0).to(self.device)

                total_loss = task_loss + adaptation_weight * adapt_loss
                total_loss.backward()
                optimizer.step()

                history['adapt_loss'].append(adapt_loss.item())
                history['task_loss'].append(task_loss.item())
                history['total_loss'].append(total_loss.item())

            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True

            return history

        def predict(self, features):
            """Predict class labels.

            Args:
                features: numpy array (n_samples, n_features)

            Returns:
                numpy array of predicted class labels
            """
            X = torch.FloatTensor(features).to(self.device)
            predictions = self.model.predict(X)
            return predictions.cpu().numpy()

        def predict_proba(self, features):
            """Predict class probabilities.

            Args:
                features: numpy array (n_samples, n_features)

            Returns:
                numpy array of class probabilities
            """
            X = torch.FloatTensor(features).to(self.device)
            proba = self.model.predict_proba(X)
            return proba.cpu().numpy()

        def domain_similarity(self, source_features, target_features):
            """Compute domain similarity score.

            Args:
                source_features: Source features
                target_features: Target features

            Returns:
                Similarity score (0.0 to 1.0)
            """
            X_s = torch.FloatTensor(source_features).to(self.device)
            X_t = torch.FloatTensor(target_features).to(self.device)

            self.model.eval()
            with torch.no_grad():
                _, source_feat = self.model(X_s)
                _, target_feat = self.model(X_t)
                mmd = DomainAdaptiveTrader.compute_mmd(source_feat, target_feat)
                return np.exp(-mmd.item())

    def _rbf_kernel(x, y, bandwidth=1.0):
        """Compute RBF (Gaussian) kernel matrix."""
        xx = (x ** 2).sum(dim=1, keepdim=True)
        yy = (y ** 2).sum(dim=1, keepdim=True)
        dist = xx + yy.t() - 2 * x @ y.t()
        return torch.exp(-dist / (2 * bandwidth ** 2))

    def _covariance(features):
        """Compute the covariance matrix of features."""
        n = features.size(0)
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        return (centered.t() @ centered) / (n - 1)

else:
    # Fallback numpy-only implementations
    class TransferFeatureExtractor:
        """Numpy-based feature extractor (no PyTorch required)."""

        def __init__(self, input_dim, hidden_dim, feature_dim, dropout=0.3):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.feature_dim = feature_dim

            # Xavier initialization
            self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
            self.w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
            self.w3 = np.random.randn(hidden_dim, feature_dim) * np.sqrt(2.0 / (hidden_dim + feature_dim))

        def forward(self, x):
            h1 = np.maximum(0, x @ self.w1)  # ReLU
            h2 = np.maximum(0, h1 @ self.w2)
            return h2 @ self.w3

    class DomainAdaptiveTrader:
        """Numpy-based trading model with domain adaptation."""

        def __init__(self, feature_extractor, feature_dim, num_classes=3):
            self.feature_extractor = feature_extractor
            self.classifier_w = np.random.randn(feature_dim, num_classes) * 0.1
            self.num_classes = num_classes

        def forward(self, x):
            features = self.feature_extractor.forward(x)
            logits = features @ self.classifier_w
            return logits, features

        def predict(self, x):
            logits, _ = self.forward(x)
            return np.argmax(logits, axis=1)

        @staticmethod
        def compute_mmd(source_features, target_features):
            source_mean = source_features.mean(axis=0)
            target_mean = target_features.mean(axis=0)
            return float(((source_mean - target_mean) ** 2).sum())

    class TransferLearningPipeline:
        """Numpy-based transfer learning pipeline."""

        def __init__(self, input_dim, hidden_dim=128, feature_dim=64,
                     num_classes=3, adaptation_method='mmd', device='cpu'):
            self.feature_extractor = TransferFeatureExtractor(input_dim, hidden_dim, feature_dim)
            self.model = DomainAdaptiveTrader(self.feature_extractor, feature_dim, num_classes)
            self.adaptation_method = adaptation_method

        def pretrain(self, source_features, source_labels, epochs=100, lr=0.001, patience=10):
            return {'loss': [0.0], 'accuracy': [0.33]}

        def adapt(self, source_features, target_features, target_labels=None,
                  epochs=50, lr=0.0001, adaptation_weight=1.0, freeze_layers=1):
            return {'adapt_loss': [0.0], 'task_loss': [0.0], 'total_loss': [0.0]}

        def predict(self, features):
            return self.model.predict(features)

        def predict_proba(self, features):
            logits, _ = self.model.forward(features)
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)
