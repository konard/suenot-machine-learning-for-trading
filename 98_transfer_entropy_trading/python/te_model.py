"""Transfer Entropy estimation and network analysis for trading."""

import numpy as np
from typing import Optional


class TransferEntropyEstimator:
    """Estimates Transfer Entropy between time series.

    Supports multiple estimation methods:
    - 'binning': Histogram-based discretization
    - 'symbolic': Ordinal pattern symbolization

    Args:
        method: Estimation method ('binning' or 'symbolic').
        n_bins: Number of bins for the binning method.
        history_length: Number of past time steps to condition on.
        symbolic_order: Order of ordinal patterns for symbolic method.
    """

    def __init__(
        self,
        method: str = "binning",
        n_bins: int = 8,
        history_length: int = 3,
        symbolic_order: int = 3,
    ):
        if method not in ("binning", "symbolic"):
            raise ValueError(f"Unknown method: {method}. Use 'binning' or 'symbolic'.")
        self.method = method
        self.n_bins = n_bins
        self.history_length = history_length
        self.symbolic_order = symbolic_order

    def compute_te(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute Transfer Entropy from source to target.

        Args:
            source: Source time series (1D array).
            target: Target time series (1D array).

        Returns:
            Transfer Entropy value in bits.
        """
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)

        if len(source) != len(target):
            raise ValueError("Source and target must have the same length.")
        if len(source) < self.history_length + 2:
            raise ValueError("Time series too short for given history length.")

        if self.method == "binning":
            return self._te_binning(source, target)
        else:
            return self._te_symbolic(source, target)

    def effective_te(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_shuffles: int = 100,
    ) -> tuple[float, float]:
        """Compute Effective Transfer Entropy with significance test.

        Subtracts the mean TE from shuffled surrogates and computes a p-value.

        Args:
            source: Source time series.
            target: Target time series.
            n_shuffles: Number of surrogate shuffles.

        Returns:
            Tuple of (effective_te, p_value).
        """
        te_original = self.compute_te(source, target)

        surrogate_tes = np.empty(n_shuffles)
        rng = np.random.default_rng(42)
        for i in range(n_shuffles):
            shuffled = rng.permutation(source)
            surrogate_tes[i] = self.compute_te(shuffled, target)

        ete = te_original - np.mean(surrogate_tes)
        p_value = np.mean(surrogate_tes >= te_original)
        return ete, p_value

    def net_te(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Net Transfer Entropy: TE(X→Y) - TE(Y→X).

        Positive means X is the net information source.
        """
        return self.compute_te(x, y) - self.compute_te(y, x)

    # ------------------------------------------------------------------ #
    #  Binning estimator
    # ------------------------------------------------------------------ #

    def _te_binning(self, source: np.ndarray, target: np.ndarray) -> float:
        """TE estimation using histogram binning."""
        src_binned = self._discretize(source)
        tgt_binned = self._discretize(target)

        k = self.history_length
        n = len(source)

        te_sum = 0.0
        total = 0

        # Build joint counts using a dictionary for sparse representation
        joint_counts: dict[tuple, int] = {}
        yk_counts: dict[tuple, int] = {}
        yk_yt1_counts: dict[tuple, int] = {}
        yk_xl_yt1_counts: dict[tuple, int] = {}

        for t in range(k, n - 1):
            y_past = tuple(tgt_binned[t - k : t])
            x_past = tuple(src_binned[t - k : t])
            y_next = tgt_binned[t + 1]

            key_yk = y_past
            key_yk_yt1 = (*y_past, y_next)
            key_full = (*y_past, *x_past, y_next)
            key_yk_xl = (*y_past, *x_past)

            joint_counts[key_full] = joint_counts.get(key_full, 0) + 1
            yk_counts[key_yk] = yk_counts.get(key_yk, 0) + 1
            yk_yt1_counts[key_yk_yt1] = yk_yt1_counts.get(key_yk_yt1, 0) + 1

            if key_yk_xl not in yk_xl_yt1_counts:
                yk_xl_yt1_counts[key_yk_xl] = 0
            yk_xl_yt1_counts[key_yk_xl] += 1

            total += 1

        if total == 0:
            return 0.0

        # Compute TE from counts
        for key_full, count in joint_counts.items():
            y_past = key_full[:k]
            x_past = key_full[k : 2 * k]
            y_next = key_full[2 * k]

            p_full = count / total
            key_yk_xl = (*y_past, *x_past)
            key_yk_yt1 = (*y_past, y_next)

            p_yt1_given_yk_xl = count / yk_xl_yt1_counts[key_yk_xl]
            p_yt1_given_yk = (
                yk_yt1_counts.get(key_yk_yt1, 0) / yk_counts.get(y_past, 1)
            )

            if p_yt1_given_yk > 0 and p_yt1_given_yk_xl > 0:
                te_sum += p_full * np.log2(p_yt1_given_yk_xl / p_yt1_given_yk)

        return max(te_sum, 0.0)

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Discretize a continuous time series into integer bins."""
        x_min, x_max = np.min(x), np.max(x)
        if x_max == x_min:
            return np.zeros(len(x), dtype=int)
        edges = np.linspace(x_min, x_max, self.n_bins + 1)
        edges[-1] += 1e-10  # include max value
        return np.digitize(x, edges) - 1

    # ------------------------------------------------------------------ #
    #  Symbolic estimator
    # ------------------------------------------------------------------ #

    def _te_symbolic(self, source: np.ndarray, target: np.ndarray) -> float:
        """TE estimation using ordinal pattern symbolization."""
        d = self.symbolic_order
        src_symbols = self._to_ordinal_patterns(source, d)
        tgt_symbols = self._to_ordinal_patterns(target, d)

        min_len = min(len(src_symbols), len(tgt_symbols))
        src_symbols = src_symbols[:min_len]
        tgt_symbols = tgt_symbols[:min_len]

        k = self.history_length
        n = len(src_symbols)
        if n < k + 2:
            return 0.0

        joint_counts: dict[tuple, int] = {}
        yk_counts: dict[tuple, int] = {}
        yk_yt1_counts: dict[tuple, int] = {}
        yk_xl_counts: dict[tuple, int] = {}
        total = 0

        for t in range(k, n - 1):
            y_past = tuple(tgt_symbols[t - k : t])
            x_past = tuple(src_symbols[t - k : t])
            y_next = tgt_symbols[t + 1]

            key_full = (*y_past, *x_past, y_next)
            key_yk = y_past
            key_yk_yt1 = (*y_past, y_next)
            key_yk_xl = (*y_past, *x_past)

            joint_counts[key_full] = joint_counts.get(key_full, 0) + 1
            yk_counts[key_yk] = yk_counts.get(key_yk, 0) + 1
            yk_yt1_counts[key_yk_yt1] = yk_yt1_counts.get(key_yk_yt1, 0) + 1
            yk_xl_counts[key_yk_xl] = yk_xl_counts.get(key_yk_xl, 0) + 1

            total += 1

        if total == 0:
            return 0.0

        te_sum = 0.0
        for key_full, count in joint_counts.items():
            y_past = key_full[:k]
            x_past = key_full[k : 2 * k]
            y_next = key_full[2 * k]

            p_full = count / total
            key_yk_xl = (*y_past, *x_past)
            key_yk_yt1 = (*y_past, y_next)

            p_yt1_given_yk_xl = count / yk_xl_counts[key_yk_xl]
            p_yt1_given_yk = (
                yk_yt1_counts.get(key_yk_yt1, 0) / yk_counts.get(y_past, 1)
            )

            if p_yt1_given_yk > 0 and p_yt1_given_yk_xl > 0:
                te_sum += p_full * np.log2(p_yt1_given_yk_xl / p_yt1_given_yk)

        return max(te_sum, 0.0)

    @staticmethod
    def _to_ordinal_patterns(x: np.ndarray, d: int) -> list[int]:
        """Convert time series to ordinal pattern symbols.

        Each window of length d is mapped to the rank permutation index.
        """
        import math

        n = len(x)
        symbols = []
        for i in range(n - d + 1):
            window = x[i : i + d]
            ranks = list(np.argsort(np.argsort(window)))
            # Convert rank permutation to a single integer (Lehmer code)
            index = 0
            for j in range(d):
                count = sum(1 for r in ranks[j + 1 :] if r < ranks[j])
                index += count * math.factorial(d - 1 - j)
            symbols.append(index)
        return symbols


class TENetworkBuilder:
    """Builds a directed information flow network using Transfer Entropy.

    Args:
        symbols: List of asset names/symbols.
        estimator: TransferEntropyEstimator instance.
    """

    def __init__(
        self,
        symbols: list[str],
        estimator: Optional[TransferEntropyEstimator] = None,
    ):
        self.symbols = symbols
        self.estimator = estimator or TransferEntropyEstimator()

    def compute_te_matrix(self, returns: dict[str, np.ndarray]) -> np.ndarray:
        """Compute pairwise TE matrix.

        Args:
            returns: Dict mapping symbol name to return series.

        Returns:
            n x n matrix where element [i,j] = TE(symbol_i → symbol_j).
        """
        n = len(self.symbols)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = self.estimator.compute_te(
                        returns[self.symbols[i]],
                        returns[self.symbols[j]],
                    )
        return matrix

    def compute_effective_te_matrix(
        self,
        returns: dict[str, np.ndarray],
        n_shuffles: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute effective TE matrix with p-values.

        Returns:
            Tuple of (ete_matrix, p_value_matrix).
        """
        n = len(self.symbols)
        ete_matrix = np.zeros((n, n))
        p_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    ete, p = self.estimator.effective_te(
                        returns[self.symbols[i]],
                        returns[self.symbols[j]],
                        n_shuffles,
                    )
                    ete_matrix[i, j] = ete
                    p_matrix[i, j] = p
        return ete_matrix, p_matrix

    def identify_leaders_followers(
        self, te_matrix: np.ndarray
    ) -> tuple[list[str], list[str]]:
        """Identify leader and follower assets based on net TE.

        Leaders have high net outgoing TE; followers have high net incoming TE.

        Returns:
            Tuple of (leader_symbols, follower_symbols).
        """
        n = len(self.symbols)
        net_te = np.zeros(n)
        for i in range(n):
            outgoing = np.sum(te_matrix[i, :])
            incoming = np.sum(te_matrix[:, i])
            net_te[i] = outgoing - incoming

        sorted_idx = np.argsort(net_te)[::-1]
        mid = n // 2
        leaders = [self.symbols[i] for i in sorted_idx[:mid]]
        followers = [self.symbols[i] for i in sorted_idx[mid:]]
        return leaders, followers

    def get_net_te(self, te_matrix: np.ndarray) -> dict[str, float]:
        """Get net TE (outgoing - incoming) for each symbol."""
        n = len(self.symbols)
        result = {}
        for i in range(n):
            outgoing = np.sum(te_matrix[i, :])
            incoming = np.sum(te_matrix[:, i])
            result[self.symbols[i]] = outgoing - incoming
        return result
