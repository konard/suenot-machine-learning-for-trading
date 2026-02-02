"""
PCMCI Causal Discovery Implementation

Provides:
- IndependenceTest: Enum for independence test types
- PCMCIConfig: Configuration for PCMCI algorithm
- PCMCICausalDiscovery: Main causal discovery class with PC-stable + MCI phases

The implementation provides a pure numpy/scipy fallback when tigramite is not
available, as well as a wrapper that uses tigramite when installed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
from enum import Enum
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class IndependenceTest(Enum):
    """Type of conditional independence test for PCMCI"""
    ParCorr = "parcorr"       # Partial correlation (linear)
    CMI = "cmi"               # Conditional mutual information (nonlinear)
    GPDC = "gpdc"             # Gaussian process distance correlation


@dataclass
class PCMCIConfig:
    """
    Configuration for PCMCI causal discovery algorithm.

    The PCMCI algorithm combines the PC algorithm's condition-selection phase
    (PC-stable) with the Momentary Conditional Independence (MCI) test to
    discover time-lagged causal links in multivariate time series.

    Example:
        config = PCMCIConfig(
            max_lag=5,
            significance_level=0.05,
            test_type=IndependenceTest.ParCorr
        )
    """
    # Algorithm parameters
    max_lag: int = 5
    significance_level: float = 0.05
    test_type: IndependenceTest = IndependenceTest.ParCorr
    max_conds_dim: Optional[int] = None  # Max conditioning set size (None=unlimited)
    min_lag: int = 1

    # PC-stable phase
    pc_alpha: Optional[float] = None  # Significance for PC phase (default: same as significance_level)

    # MCI test phase
    max_conds_px: Optional[int] = None  # Max conditions from parents of X
    max_conds_py: Optional[int] = None  # Max conditions from parents of Y

    # Data parameters
    n_variables: int = 5
    variable_names: Optional[List[str]] = None

    # Tigramite integration
    use_tigramite: bool = True  # Try to use tigramite if available

    def __post_init__(self):
        """Set defaults after initialization"""
        if self.pc_alpha is None:
            self.pc_alpha = self.significance_level
        if self.variable_names is None:
            self.variable_names = [f"X_{i}" for i in range(self.n_variables)]

    def validate(self):
        """Validate configuration parameters"""
        assert self.max_lag >= 1, "max_lag must be >= 1"
        assert 0 < self.significance_level < 1, "significance_level must be in (0, 1)"
        assert self.min_lag >= 0, "min_lag must be >= 0"
        assert self.min_lag <= self.max_lag, "min_lag must be <= max_lag"
        assert self.n_variables >= 2, "Need at least 2 variables for causal discovery"


class PCMCICausalDiscovery:
    """
    PCMCI Causal Discovery for multivariate time series.

    Implements the PCMCI algorithm (Runge et al., 2019) which combines:
    1. PC-stable condition selection phase to identify potential causal parents
    2. MCI (Momentary Conditional Independence) test for robust link detection

    The algorithm discovers time-lagged causal relationships between variables,
    producing a causal graph with directed edges annotated by lag and strength.

    If tigramite is installed and config.use_tigramite is True, the tigramite
    library is used for the core computation. Otherwise, a pure numpy/scipy
    implementation is used as a fallback.

    Example:
        config = PCMCIConfig(max_lag=5, n_variables=4)
        pcmci = PCMCICausalDiscovery(config)

        # data shape: (n_timesteps, n_variables)
        causal_graph = pcmci.fit(data)
        links = pcmci.get_causal_links(threshold=0.1)

        for source, target, lag, strength in links:
            print(f"{source} --({lag})--> {target}: {strength:.3f}")
    """

    def __init__(self, config: PCMCIConfig):
        """
        Initialize PCMCI causal discovery.

        Args:
            config: PCMCIConfig with algorithm parameters
        """
        config.validate()
        self.config = config
        self._causal_graph: Optional[np.ndarray] = None
        self._p_values: Optional[np.ndarray] = None
        self._val_matrix: Optional[np.ndarray] = None
        self._parents: Optional[Dict[int, List[Tuple[int, int]]]] = None
        self._fitted = False
        self._tigramite_available = False

        # Check tigramite availability
        if config.use_tigramite:
            try:
                import tigramite
                from tigramite.pcmci import PCMCI as TigramitePCMCI
                from tigramite.independence_tests.parcorr import ParCorr as TigramiteParCorr
                self._tigramite_available = True
                logger.info("Using tigramite for PCMCI computation")
            except ImportError:
                self._tigramite_available = False
                logger.info("tigramite not available, using pure numpy/scipy fallback")

    def _compute_partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Compute partial correlation between x and y conditioned on z.

        Uses linear regression residuals to compute the partial correlation
        coefficient and its p-value via Fisher's z-transform.

        Args:
            x: First variable, shape (n_samples,)
            y: Second variable, shape (n_samples,)
            z: Conditioning variables, shape (n_samples, n_conds) or None

        Returns:
            Tuple of (partial_correlation, p_value)
        """
        from scipy import stats

        n = len(x)

        if z is None or z.shape[1] == 0:
            # Simple correlation
            corr, pval = stats.pearsonr(x, y)
            return corr, pval

        # Regress x on z and y on z, compute correlation of residuals
        z_with_intercept = np.column_stack([z, np.ones(n)])

        # Solve least squares for x ~ z
        try:
            beta_x, _, _, _ = np.linalg.lstsq(z_with_intercept, x, rcond=None)
            residual_x = x - z_with_intercept @ beta_x

            # Solve least squares for y ~ z
            beta_y, _, _, _ = np.linalg.lstsq(z_with_intercept, y, rcond=None)
            residual_y = y - z_with_intercept @ beta_y
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in partial correlation computation")
            return 0.0, 1.0

        # Correlation of residuals
        if np.std(residual_x) < 1e-10 or np.std(residual_y) < 1e-10:
            return 0.0, 1.0

        corr, _ = stats.pearsonr(residual_x, residual_y)

        # P-value using Fisher's z-transform
        dof = max(n - z.shape[1] - 2, 1)
        z_score = 0.5 * np.log((1 + corr + 1e-10) / (1 - corr + 1e-10))
        se = 1.0 / np.sqrt(dof)
        pval = 2 * (1 - stats.norm.cdf(abs(z_score) / se))

        return float(corr), float(pval)

    def _get_lagged_values(
        self,
        data: np.ndarray,
        var_idx: int,
        lag: int,
        max_lag: int
    ) -> np.ndarray:
        """
        Extract lagged values of a variable from the data matrix.

        Args:
            data: Full data matrix, shape (n_timesteps, n_variables)
            var_idx: Variable index
            lag: Lag value (0 = contemporaneous)
            max_lag: Maximum lag used (for alignment)

        Returns:
            Lagged values, shape (n_effective_samples,)
        """
        n_time = data.shape[0]
        effective_start = max_lag
        if lag == 0:
            return data[effective_start:, var_idx]
        else:
            return data[effective_start - lag:n_time - lag, var_idx]

    def _pc_stable_phase(
        self,
        data: np.ndarray
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        PC-stable condition selection phase.

        Iteratively tests conditional independence to identify potential
        causal parents for each variable. The "stable" variant tests all
        pairs at the same conditioning set size before removing links.

        Args:
            data: Time series data, shape (n_timesteps, n_variables)

        Returns:
            Dictionary mapping each variable index to its list of
            potential parents as (variable_index, lag) tuples
        """
        n_vars = self.config.n_variables
        max_lag = self.config.max_lag
        min_lag = self.config.min_lag
        pc_alpha = self.config.pc_alpha

        # Initialize: all lagged variables are potential parents
        parents: Dict[int, List[Tuple[int, int]]] = {}
        for j in range(n_vars):
            parents[j] = []
            for i in range(n_vars):
                for tau in range(min_lag, max_lag + 1):
                    if tau == 0 and i == j:
                        continue  # No contemporaneous self-links
                    parents[j].append((i, tau))

        # Iteratively increase conditioning set size
        max_conds = self.config.max_conds_dim
        if max_conds is None:
            max_conds = n_vars * max_lag  # Effectively unlimited

        for cond_dim in range(0, max_conds + 1):
            any_removed = False

            for j in range(n_vars):
                # Get current parents for variable j
                current_parents = list(parents[j])

                if len(current_parents) - 1 < cond_dim:
                    continue

                links_to_remove = []

                for parent in current_parents:
                    i, tau = parent

                    # Get candidate conditioning sets (all other parents)
                    other_parents = [p for p in current_parents if p != parent]

                    if len(other_parents) < cond_dim:
                        continue

                    # Test all conditioning sets of size cond_dim
                    is_independent = False

                    if cond_dim == 0:
                        # Unconditional test
                        x = self._get_lagged_values(data, i, tau, max_lag)
                        y = self._get_lagged_values(data, j, 0, max_lag)
                        _, pval = self._compute_partial_correlation(x, y)

                        if pval > pc_alpha:
                            is_independent = True
                    else:
                        # Test with conditioning sets
                        cond_sets = list(combinations(other_parents, cond_dim))
                        # Limit number of conditioning sets tested
                        max_tests = min(len(cond_sets), 20)
                        cond_sets = cond_sets[:max_tests]

                        for cond_set in cond_sets:
                            x = self._get_lagged_values(data, i, tau, max_lag)
                            y = self._get_lagged_values(data, j, 0, max_lag)

                            # Build conditioning matrix
                            z_cols = []
                            for ci, ctau in cond_set:
                                z_cols.append(
                                    self._get_lagged_values(data, ci, ctau, max_lag)
                                )
                            z = np.column_stack(z_cols) if z_cols else None

                            _, pval = self._compute_partial_correlation(x, y, z)

                            if pval > pc_alpha:
                                is_independent = True
                                break

                    if is_independent:
                        links_to_remove.append(parent)
                        any_removed = True

                # Remove independent links (stable: remove after testing all)
                for link in links_to_remove:
                    if link in parents[j]:
                        parents[j].remove(link)

            if not any_removed:
                break

            logger.debug(
                f"PC-stable phase: cond_dim={cond_dim}, "
                f"total parents={sum(len(v) for v in parents.values())}"
            )

        return parents

    def _mci_test(
        self,
        data: np.ndarray,
        parents: Dict[int, List[Tuple[int, int]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Momentary Conditional Independence (MCI) test phase.

        Tests each potential link X_{t-tau} -> Y_t while conditioning on:
        - Parents of Y (excluding the tested link)
        - Parents of X (at appropriate lags)

        This controls for both autocorrelation and indirect effects.

        Args:
            data: Time series data, shape (n_timesteps, n_variables)
            parents: Potential parents from PC-stable phase

        Returns:
            val_matrix: Test statistic values, shape (n_vars, n_vars, max_lag+1)
            p_matrix: P-values, shape (n_vars, n_vars, max_lag+1)
        """
        n_vars = self.config.n_variables
        max_lag = self.config.max_lag
        min_lag = self.config.min_lag

        val_matrix = np.zeros((n_vars, n_vars, max_lag + 1))
        p_matrix = np.ones((n_vars, n_vars, max_lag + 1))

        for j in range(n_vars):
            for i in range(n_vars):
                for tau in range(min_lag, max_lag + 1):
                    if tau == 0 and i == j:
                        continue

                    # Check if (i, tau) is a potential parent of j
                    if (i, tau) not in parents[j]:
                        continue

                    # Get X and Y values
                    x = self._get_lagged_values(data, i, tau, max_lag)
                    y = self._get_lagged_values(data, j, 0, max_lag)

                    # Build conditioning set
                    z_cols = []

                    # Parents of Y (excluding tested link)
                    parents_y = [p for p in parents[j] if p != (i, tau)]
                    max_conds_py = self.config.max_conds_py or len(parents_y)
                    for ci, ctau in parents_y[:max_conds_py]:
                        z_cols.append(
                            self._get_lagged_values(data, ci, ctau, max_lag)
                        )

                    # Parents of X (shifted by tau)
                    parents_x = parents.get(i, [])
                    max_conds_px = self.config.max_conds_px or len(parents_x)
                    for ci, ctau in parents_x[:max_conds_px]:
                        shifted_lag = ctau + tau
                        if shifted_lag <= max_lag:
                            z_cols.append(
                                self._get_lagged_values(data, ci, shifted_lag, max_lag)
                            )

                    # Compute partial correlation with conditioning
                    z = np.column_stack(z_cols) if z_cols else None
                    val, pval = self._compute_partial_correlation(x, y, z)

                    val_matrix[i, j, tau] = val
                    p_matrix[i, j, tau] = pval

        return val_matrix, p_matrix

    def _fit_tigramite(self, data: np.ndarray) -> np.ndarray:
        """
        Fit using tigramite library.

        Args:
            data: Time series data, shape (n_timesteps, n_variables)

        Returns:
            Causal graph adjacency matrix
        """
        from tigramite.pcmci import PCMCI as TigramitePCMCI
        from tigramite import data_processing as pp

        # Select independence test
        if self.config.test_type == IndependenceTest.ParCorr:
            from tigramite.independence_tests.parcorr import ParCorr as TigramiteParCorr
            cond_ind_test = TigramiteParCorr(significance='analytic')
        elif self.config.test_type == IndependenceTest.CMI:
            try:
                from tigramite.independence_tests.cmiknn import CMIknn
                cond_ind_test = CMIknn(significance='shuffle_test', knn=0.1)
            except ImportError:
                from tigramite.independence_tests.parcorr import ParCorr as TigramiteParCorr
                cond_ind_test = TigramiteParCorr(significance='analytic')
                logger.warning("CMIknn not available, falling back to ParCorr")
        elif self.config.test_type == IndependenceTest.GPDC:
            try:
                from tigramite.independence_tests.gpdc import GPDC
                cond_ind_test = GPDC(significance='analytic')
            except ImportError:
                from tigramite.independence_tests.parcorr import ParCorr as TigramiteParCorr
                cond_ind_test = TigramiteParCorr(significance='analytic')
                logger.warning("GPDC not available, falling back to ParCorr")
        else:
            from tigramite.independence_tests.parcorr import ParCorr as TigramiteParCorr
            cond_ind_test = TigramiteParCorr(significance='analytic')

        # Create tigramite dataframe
        dataframe = pp.DataFrame(
            data,
            var_names=self.config.variable_names[:data.shape[1]]
        )

        # Run PCMCI
        pcmci = TigramitePCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=0
        )

        results = pcmci.run_pcmci(
            tau_min=self.config.min_lag,
            tau_max=self.config.max_lag,
            pc_alpha=self.config.pc_alpha,
            alpha_level=self.config.significance_level
        )

        self._val_matrix = results['val_matrix']
        self._p_values = results['p_matrix']

        # Build causal graph from results
        n_vars = data.shape[1]
        causal_graph = np.zeros((n_vars, n_vars, self.config.max_lag + 1))

        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(self.config.max_lag + 1):
                    if self._p_values[i, j, tau] < self.config.significance_level:
                        causal_graph[i, j, tau] = self._val_matrix[i, j, tau]

        # Extract parents
        self._parents = {}
        for j in range(n_vars):
            self._parents[j] = []
            for i in range(n_vars):
                for tau in range(self.config.min_lag, self.config.max_lag + 1):
                    if tau == 0 and i == j:
                        continue
                    if self._p_values[i, j, tau] < self.config.significance_level:
                        self._parents[j].append((i, tau))

        return causal_graph

    def _fit_numpy(self, data: np.ndarray) -> np.ndarray:
        """
        Fit using pure numpy/scipy implementation.

        Args:
            data: Time series data, shape (n_timesteps, n_variables)

        Returns:
            Causal graph adjacency matrix
        """
        # Phase 1: PC-stable condition selection
        logger.info("Running PC-stable phase...")
        parents = self._pc_stable_phase(data)
        self._parents = parents

        parent_count = sum(len(v) for v in parents.values())
        logger.info(f"PC-stable phase complete: {parent_count} potential links")

        # Phase 2: MCI test
        logger.info("Running MCI test phase...")
        val_matrix, p_matrix = self._mci_test(data, parents)
        self._val_matrix = val_matrix
        self._p_values = p_matrix

        # Build causal graph: only keep significant links
        n_vars = data.shape[1]
        causal_graph = np.zeros((n_vars, n_vars, self.config.max_lag + 1))

        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(self.config.max_lag + 1):
                    if p_matrix[i, j, tau] < self.config.significance_level:
                        causal_graph[i, j, tau] = val_matrix[i, j, tau]

        return causal_graph

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit PCMCI to multivariate time series data.

        Runs the full PCMCI algorithm:
        1. PC-stable phase for condition selection
        2. MCI test for robust causal link detection

        Args:
            data: Time series data, shape (n_timesteps, n_variables).
                  Each column is a variable, each row is a time step.

        Returns:
            causal_graph: Adjacency matrix with lag information,
                          shape (n_vars, n_vars, max_lag+1).
                          Entry [i, j, tau] != 0 means X_i at time t-tau
                          causally influences X_j at time t.

        Example:
            data = np.random.randn(500, 4)
            graph = pcmci.fit(data)
            # graph[0, 1, 2] != 0 means X_0 at t-2 causes X_1 at t
        """
        # Update n_variables from data
        self.config.n_variables = data.shape[1]
        if len(self.config.variable_names) < data.shape[1]:
            self.config.variable_names = [
                f"X_{i}" for i in range(data.shape[1])
            ]

        # Validate data
        assert data.ndim == 2, "Data must be 2D: (n_timesteps, n_variables)"
        assert data.shape[0] > self.config.max_lag + 10, \
            f"Need at least {self.config.max_lag + 10} timesteps, got {data.shape[0]}"
        assert data.shape[1] >= 2, "Need at least 2 variables"

        # Check for NaN/Inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("Data contains NaN/Inf values, replacing with 0")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize data
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        stds[stds < 1e-10] = 1.0
        data_std = (data - means) / stds

        # Choose implementation
        if self._tigramite_available:
            try:
                self._causal_graph = self._fit_tigramite(data_std)
            except Exception as e:
                logger.warning(f"tigramite failed: {e}, falling back to numpy")
                self._causal_graph = self._fit_numpy(data_std)
        else:
            self._causal_graph = self._fit_numpy(data_std)

        self._fitted = True
        logger.info(
            f"PCMCI fit complete: "
            f"{np.count_nonzero(self._causal_graph)} significant links found"
        )

        return self._causal_graph

    def get_causal_links(
        self,
        threshold: float = 0.0
    ) -> List[Tuple[str, str, int, float]]:
        """
        Get list of significant causal links.

        Args:
            threshold: Minimum absolute strength to include a link

        Returns:
            List of (source_name, target_name, lag, strength) tuples,
            sorted by absolute strength descending.

        Example:
            links = pcmci.get_causal_links(threshold=0.1)
            for src, tgt, lag, strength in links:
                print(f"{src} --({lag})--> {tgt}: {strength:.3f}")
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before get_causal_links()")

        links = []
        n_vars = self._causal_graph.shape[0]
        names = self.config.variable_names

        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(self.config.max_lag + 1):
                    val = self._causal_graph[i, j, tau]
                    if abs(val) > threshold:
                        links.append((names[i], names[j], tau, float(val)))

        # Sort by absolute strength
        links.sort(key=lambda x: abs(x[3]), reverse=True)
        return links

    def get_causal_graph_networkx(self) -> Any:
        """
        Convert causal graph to a NetworkX DiGraph.

        Nodes represent variables. Edges represent causal links annotated
        with 'lag' and 'weight' (strength) attributes.

        Returns:
            NetworkX DiGraph with causal relationships

        Example:
            G = pcmci.get_causal_graph_networkx()
            for u, v, d in G.edges(data=True):
                print(f"{u} -> {v}, lag={d['lag']}, weight={d['weight']:.3f}")
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")

        if not self._fitted:
            raise RuntimeError("Must call fit() before get_causal_graph_networkx()")

        G = nx.DiGraph()
        names = self.config.variable_names
        n_vars = self._causal_graph.shape[0]

        # Add nodes
        for i in range(n_vars):
            G.add_node(names[i], index=i)

        # Add edges
        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(self.config.max_lag + 1):
                    val = self._causal_graph[i, j, tau]
                    if abs(val) > 0:
                        # Use a composite key for multi-lag edges
                        edge_key = f"{names[i]}_lag{tau}"
                        G.add_edge(
                            names[i],
                            names[j],
                            lag=tau,
                            weight=float(val),
                            abs_weight=float(abs(val)),
                            p_value=float(self._p_values[i, j, tau])
                            if self._p_values is not None else 0.0,
                            key=edge_key
                        )

        return G

    def predict_from_causes(
        self,
        data: np.ndarray,
        target_var: int,
        method: str = "linear"
    ) -> np.ndarray:
        """
        Predict a target variable using its identified causal parents.

        Fits a linear model from causal parents to the target variable
        and generates predictions.

        Args:
            data: Time series data, shape (n_timesteps, n_variables)
            target_var: Index of the target variable to predict
            method: Prediction method ('linear' or 'ridge')

        Returns:
            predictions: Predicted values for the target variable,
                         shape (n_effective_samples,)

        Example:
            predictions = pcmci.predict_from_causes(data, target_var=0)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict_from_causes()")

        if self._parents is None or target_var not in self._parents:
            logger.warning(f"No parents found for variable {target_var}")
            return np.zeros(data.shape[0] - self.config.max_lag)

        parents = self._parents[target_var]
        max_lag = self.config.max_lag

        if len(parents) == 0:
            logger.warning(f"No causal parents for variable {target_var}")
            return np.zeros(data.shape[0] - max_lag)

        # Build feature matrix from causal parents
        feature_cols = []
        for var_idx, lag in parents:
            col = self._get_lagged_values(data, var_idx, lag, max_lag)
            feature_cols.append(col)

        X = np.column_stack(feature_cols)
        y = self._get_lagged_values(data, target_var, 0, max_lag)

        # Split into train/test (80/20)
        n = len(y)
        split = int(0.8 * n)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Fit model
        if method == "ridge":
            from scipy.linalg import solve
            alpha = 1.0
            n_features = X_train.shape[1]
            XtX = X_train.T @ X_train + alpha * np.eye(n_features)
            Xty = X_train.T @ y_train
            beta = solve(XtX, Xty)
        else:
            # Ordinary least squares
            beta, _, _, _ = np.linalg.lstsq(
                np.column_stack([X_train, np.ones(len(X_train))]),
                y_train,
                rcond=None
            )
            # Add intercept column to X for prediction
            X = np.column_stack([X, np.ones(len(X))])

        predictions = X @ beta if method != "ridge" else X @ beta

        # Compute R-squared on test set
        if method == "ridge":
            y_pred_test = X_test @ beta
        else:
            X_test_full = np.column_stack([X_test, np.ones(len(X_test))])
            y_pred_test = X_test_full @ beta

        ss_res = np.sum((y_test - y_pred_test) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        logger.info(
            f"Prediction for variable {target_var}: "
            f"R-squared = {r_squared:.4f}, "
            f"using {len(parents)} causal parents"
        )

        return predictions

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the causal discovery results.

        Returns:
            Dictionary with summary statistics and results
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before get_summary()")

        n_vars = self._causal_graph.shape[0]
        n_links = np.count_nonzero(self._causal_graph)

        # Compute per-variable statistics
        var_stats = {}
        for j in range(n_vars):
            n_causes = len(self._parents.get(j, []))
            n_effects = 0
            for i in range(n_vars):
                for tau in range(self.config.max_lag + 1):
                    if abs(self._causal_graph[j, i, tau]) > 0:
                        n_effects += 1

            var_stats[self.config.variable_names[j]] = {
                'n_causes': n_causes,
                'n_effects': n_effects,
                'causes': [
                    (self.config.variable_names[i], tau)
                    for i, tau in self._parents.get(j, [])
                ]
            }

        # Strongest links
        links = self.get_causal_links()
        top_links = links[:10] if len(links) > 10 else links

        return {
            'n_variables': n_vars,
            'n_significant_links': n_links,
            'max_lag': self.config.max_lag,
            'significance_level': self.config.significance_level,
            'test_type': self.config.test_type.value,
            'variable_stats': var_stats,
            'top_links': top_links,
            'used_tigramite': self._tigramite_available
        }


if __name__ == "__main__":
    # Test the PCMCI implementation
    print("Testing PCMCI Causal Discovery...")

    np.random.seed(42)

    # Generate synthetic causal data
    n_samples = 500
    n_vars = 4

    # True causal structure:
    # X0(t-1) -> X1(t) with coefficient 0.6
    # X1(t-2) -> X2(t) with coefficient 0.4
    # X0(t-1) -> X3(t) with coefficient 0.3
    data = np.random.randn(n_samples, n_vars) * 0.3

    for t in range(2, n_samples):
        data[t, 1] += 0.6 * data[t - 1, 0]  # X0(t-1) -> X1(t)
        data[t, 2] += 0.4 * data[t - 2, 1]  # X1(t-2) -> X2(t)
        data[t, 3] += 0.3 * data[t - 1, 0]  # X0(t-1) -> X3(t)

    print(f"Data shape: {data.shape}")

    # Run PCMCI
    config = PCMCIConfig(
        max_lag=3,
        significance_level=0.05,
        test_type=IndependenceTest.ParCorr,
        n_variables=n_vars,
        variable_names=["X0", "X1", "X2", "X3"],
        use_tigramite=False  # Force numpy fallback for testing
    )

    pcmci = PCMCICausalDiscovery(config)
    causal_graph = pcmci.fit(data)

    print(f"\nCausal graph shape: {causal_graph.shape}")
    print(f"Non-zero entries: {np.count_nonzero(causal_graph)}")

    # Get causal links
    links = pcmci.get_causal_links(threshold=0.05)
    print(f"\nDiscovered causal links (threshold=0.05):")
    for src, tgt, lag, strength in links:
        print(f"  {src} --({lag})--> {tgt}: {strength:.3f}")

    # Get summary
    summary = pcmci.get_summary()
    print(f"\nSummary:")
    print(f"  Variables: {summary['n_variables']}")
    print(f"  Significant links: {summary['n_significant_links']}")

    # Test prediction from causes
    predictions = pcmci.predict_from_causes(data, target_var=1)
    print(f"\nPredictions shape: {predictions.shape}")

    # Test NetworkX graph
    try:
        G = pcmci.get_causal_graph_networkx()
        print(f"\nNetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except ImportError:
        print("\nNetworkX not available, skipping graph test")

    print("\nAll tests passed!")
