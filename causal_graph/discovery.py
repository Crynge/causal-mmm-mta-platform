"""
Causal Graph Discovery Engine

Implements automated DAG discovery using:
- PC Algorithm (Peter-Clark) for constraint-based discovery
- FCI Algorithm for handling latent confounders
- Domain knowledge integration
- Causal effect estimation with DoWhy
- Double Machine Learning (DML) for endogenous variables
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings

try:
    from dowhy import CausalModel
except ImportError:  # pragma: no cover - depends on local environment
    CausalModel = None

try:
    from econml.dml import DMLCateEstimator
except ImportError:  # pragma: no cover - depends on local environment
    DMLCateEstimator = None

warnings.filterwarnings('ignore')


def _require_causal_effect_dependencies() -> None:
    if CausalModel is None:
        raise ImportError(
            "Missing optional causal inference dependency 'dowhy'. "
            "Install the full research stack from requirements.txt to estimate causal effects."
        )


def _require_dml_dependencies() -> None:
    if DMLCateEstimator is None:
        raise ImportError(
            "Missing optional causal inference dependency 'econml'. "
            "Install the full research stack from requirements.txt to run Double Machine Learning."
        )


@dataclass
class CausalConstraint:
    """Represents a domain knowledge constraint."""
    var1: str
    var2: str
    constraint_type: str  # 'must_cause', 'cannot_cause', 'must_be_connected'
    confidence: float = 1.0


@dataclass
class DiscoveredEdge:
    """Represents a discovered causal edge."""
    source: str
    target: str
    edge_type: str  # 'directed', 'bidirected', 'undirected'
    confidence: float
    p_value: Optional[float] = None
    method: str = 'pc'


class PCAlgorithm:
    """
    Peter-Clark algorithm for causal discovery.
    
    Uses conditional independence tests to discover the skeleton
    and orient edges based on v-structures and orientation rules.
    """
    
    def __init__(self, alpha: float = 0.05, max_conditioning_set: int = 3):
        self.alpha = alpha
        self.max_conditioning_set = max_conditioning_set
        self.separation_sets: Dict[Tuple[str, str], Set[str]] = {}
    
    def _conditional_independence_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        method: str = 'partial_correlation'
    ) -> Tuple[float, float]:
        """
        Test conditional independence between X and Y given Z.
        
        Returns
        -------
        test_statistic : float
        p_value : float
        """
        if method == 'partial_correlation':
            return self._partial_correlation_test(x, y, z)
        elif method == 'fisher_z':
            return self._fisher_z_test(x, y, z)
        else:
            raise ValueError(f"Unknown test method: {method}")
    
    def _partial_correlation_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[float, float]:
        """Compute partial correlation and test statistic."""
        n = len(x)
        
        if z.shape[1] == 0:
            # Simple correlation
            corr = np.corrcoef(x.flatten(), y.flatten())[0, 1]
            t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2 + 1e-10)
            from scipy.stats import t
            p_value = 2 * (1 - t.cdf(abs(t_stat), n - 2))
            return corr, p_value
        
        # Partial correlation via regression residuals
        model_x = LinearRegression().fit(z, x)
        model_y = LinearRegression().fit(z, y)
        
        resid_x = x - model_x.predict(z)
        resid_y = y - model_y.predict(z)
        
        partial_corr = np.corrcoef(resid_x.flatten(), resid_y.flatten())[0, 1]
        
        # Fisher's z-transform
        z_score = np.arctanh(partial_corr + 1e-10) * np.sqrt(n - len(z.T) - 3)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return partial_corr, p_value
    
    def _fisher_z_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[float, float]:
        """Fisher's z-test for conditional independence."""
        return self._partial_correlation_test(x, y, z)
    
    def discover_skeleton(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None
    ) -> nx.Graph:
        """
        Discover the undirected skeleton of the causal graph.
        
        Parameters
        ----------
        data : pd.DataFrame
            Observational data
        variables : list, optional
            Variables to include in discovery
            
        Returns
        -------
        nx.Graph
            Undirected graph representing the skeleton
        """
        if variables is None:
            variables = data.columns.tolist()
        
        n_vars = len(variables)
        skeleton = nx.Graph()
        skeleton.add_nodes_from(variables)
        
        # Start with complete graph
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                skeleton.add_edge(variables[i], variables[j])
        
        # Remove edges based on conditional independence
        for cond_size in range(self.max_conditioning_set + 1):
            edges_to_remove = []
            
            for u, v in list(skeleton.edges()):
                if not skeleton.has_edge(u, v):
                    continue
                
                # Get adjacent nodes
                neighbors_u = set(skeleton.neighbors(u)) - {v}
                neighbors_v = set(skeleton.neighbors(v)) - {u}
                
                # Try conditioning sets from u's neighbors
                if len(neighbors_u) >= cond_size:
                    from itertools import combinations
                    for cond_set in combinations(neighbors_u, cond_size):
                        cond_set = list(cond_set)
                        
                        x = data[u].values.reshape(-1, 1)
                        y = data[v].values.reshape(-1, 1)
                        
                        if len(cond_set) > 0:
                            z = data[list(cond_set)].values
                        else:
                            z = np.empty((len(data), 0))
                        
                        _, p_value = self._conditional_independence_test(x, y, z)
                        
                        if p_value > self.alpha:
                            edges_to_remove.append((u, v))
                            self.separation_sets[(u, v)] = set(cond_set)
                            self.separation_sets[(v, u)] = set(cond_set)
                            break
            
            skeleton.remove_edges_from(edges_to_remove)
        
        return skeleton
    
    def orient_edges(
        self,
        skeleton: nx.Graph,
        data: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Orient edges using v-structures and orientation rules.
        
        Implements Meek's rules for edge orientation.
        """
        dag = skeleton.to_directed()
        
        # Rule 1: Orient v-structures (X -> Z <- Y where X and Y are not adjacent)
        for z in dag.nodes():
            predecessors = list(dag.predecessors(z))
            
            for i, x in enumerate(predecessors):
                for y in predecessors[i+1:]:
                    if not skeleton.has_edge(x, y):
                        # Check if Z is in separation set of X and Y
                        sep_set = self.separation_sets.get((x, y), set())
                        
                        if z not in sep_set:
                            # Orient as v-structure: X -> Z <- Y
                            if dag.has_edge(z, x):
                                dag.remove_edge(z, x)
                            if dag.has_edge(z, y):
                                dag.remove_edge(z, y)
        
        # Additional orientation rules (Meek's rules) would go here
        # For brevity, implementing basic rules only
        
        return dag


class FCICAlgorithm(PCAlgorithm):
    """
    Fast Causal Inference (FCI) algorithm.
    
    Extension of PC that handles latent confounders by introducing
    bidirected edges (X <-> Y indicates latent common cause).
    """
    
    def __init__(self, alpha: float = 0.05, max_conditioning_set: int = 3):
        super().__init__(alpha, max_conditioning_set)
        self.potential_latents: Set[Tuple[str, str]] = set()
    
    def discover_with_latents(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None
    ) -> nx.MultiDiGraph:
        """
        Discover causal graph allowing for latent confounders.
        
        Returns a PAG (Partial Ancestral Graph) with:
        - Directed edges (->)
        - Bidirected edges (<->) for latent confounding
        - Partially oriented edges (o->)
        """
        # First run PC algorithm
        skeleton = self.discover_skeleton(data, variables)
        pag = self.orient_edges(skeleton, data)
        
        # Detect potential latent confounders
        # If two variables are dependent but no conditioning set makes them independent
        # and they don't form a v-structure, suspect latent confounder
        
        for u, v in list(skeleton.edges()):
            if (u, v) not in self.separation_sets and (v, u) not in self.separation_sets:
                # Check if this could be due to latent confounding
                self.potential_latents.add((u, v))
        
        # Convert to PAG representation
        # This is simplified; full FCI is more complex
        return pag


class CausalGraphDiscovery:
    """
    Main interface for causal graph discovery.
    
    Combines multiple algorithms with domain knowledge constraints.
    """
    
    def __init__(
        self,
        pc_alpha: float = 0.05,
        fci_max_cond: int = 3,
        expert_constraints: Optional[List[CausalConstraint]] = None
    ):
        self.pc_algo = PCAlgorithm(alpha=pc_alpha, max_conditioning_set=fci_max_cond)
        self.fci_algo = FCICAlgorithm(alpha=pc_alpha, max_conditioning_set=fci_max_cond)
        self.expert_constraints = expert_constraints or []
        self.discovered_graph: Optional[nx.DiGraph] = None
        self.discovery_method: Optional[str] = None
    
    def add_constraint(
        self,
        var1: str,
        var2: str,
        constraint_type: str,
        confidence: float = 1.0
    ):
        """Add a domain knowledge constraint."""
        self.expert_constraints.append(CausalConstraint(
            var1=var1,
            var2=var2,
            constraint_type=constraint_type,
            confidence=confidence
        ))
    
    def load_constraints_from_dict(self, constraints: Dict[str, Any]):
        """Load constraints from dictionary format."""
        if 'must_cause' in constraints:
            for pair in constraints['must_cause']:
                self.add_constraint(pair[0], pair[1], 'must_cause')
        
        if 'cannot_cause' in constraints:
            for pair in constraints['cannot_cause']:
                self.add_constraint(pair[0], pair[1], 'cannot_cause')
    
    def discover(
        self,
        data: pd.DataFrame,
        method: str = 'pc',
        use_expert_constraints: bool = True
    ) -> nx.DiGraph:
        """
        Discover causal graph from data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Observational data
        method : str
            'pc' for PC algorithm, 'fci' for FCI
        use_expert_constraints : bool
            Whether to incorporate domain knowledge
            
        Returns
        -------
        nx.DiGraph
            Discovered causal graph
        """
        if method == 'pc':
            skeleton = self.pc_algo.discover_skeleton(data)
            graph = self.pc_algo.orient_edges(skeleton, data)
        elif method == 'fci':
            graph = self.fci_algo.discover_with_latents(data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply expert constraints
        if use_expert_constraints:
            graph = self._apply_constraints(graph)
        
        self.discovered_graph = graph
        self.discovery_method = method
        
        return graph
    
    def _apply_constraints(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Apply domain knowledge constraints to discovered graph."""
        for constraint in self.expert_constraints:
            if constraint.constraint_type == 'cannot_cause':
                # Remove forbidden edges
                if graph.has_edge(constraint.var1, constraint.var2):
                    graph.remove_edge(constraint.var1, constraint.var2)
                if graph.has_edge(constraint.var2, constraint.var1):
                    graph.remove_edge(constraint.var2, constraint.var1)
            
            elif constraint.constraint_type == 'must_cause':
                # Ensure required edge exists
                if not graph.has_edge(constraint.var1, constraint.var2):
                    graph.add_edge(constraint.var1, constraint.var2)
        
        return graph
    
    def validate_dag(self, graph: Optional[nx.DiGraph] = None) -> bool:
        """Check if the graph is a valid DAG."""
        if graph is None:
            graph = self.discovered_graph
        
        if graph is None:
            return False
        
        return is_directed_acyclic_graph(graph)
    
    def get_causal_ordering(self, graph: Optional[nx.DiGraph] = None) -> List[str]:
        """Get topological ordering of variables."""
        if graph is None:
            graph = self.discovered_graph
        
        if graph is None:
            raise ValueError("No graph discovered yet")
        
        return list(nx.topological_sort(graph))
    
    def estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        graph: Optional[nx.DiGraph] = None,
        method: str = 'backdoor'
    ) -> Dict[str, Any]:
        """
        Estimate causal effect using DoWhy.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data
        treatment : str
            Treatment variable name
        outcome : str
            Outcome variable name
        graph : nx.DiGraph, optional
            Causal graph (uses discovered graph if None)
        method : str
            Identification method ('backdoor', 'frontdoor', 'iv')
        """
        if graph is None:
            graph = self.discovered_graph
        
        if graph is None:
            raise ValueError("No graph available. Run discover() first.")

        _require_causal_effect_dependencies()
        
        # Convert NetworkX graph to DOT format for DoWhy
        graph_str = nx.nx_pydot.to_pydot(graph).to_string()
        
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=graph_str
        )
        
        identified_estimand = model.identify_effect(method_name=method)
        
        # Estimate using matching
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="matching"
        )
        
        return {
            'effect': estimate.value,
            'estimand': identified_estimand,
            'method': method
        }
    
    def compute_e_value(
        self,
        effect_estimate: float,
        outcome_variance: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute E-value for sensitivity analysis.
        
        The E-value quantifies how strong an unmeasured confounder
        would need to be to explain away the observed effect.
        """
        # Simplified E-value calculation
        # For binary outcomes: E-value = RR + sqrt(RR * (RR - 1))
        # where RR is the risk ratio
        
        if effect_estimate > 0:
            rr = np.exp(effect_estimate)
            e_value = rr + np.sqrt(rr * (rr - 1))
        else:
            e_value = 1.0
        
        return {
            'e_value': e_value,
            'interpretation': f"An unmeasured confounder would need to have an association of at least {e_value:.2f} with both treatment and outcome to explain away this effect."
        }
    
    def plot_graph(self, graph: Optional[nx.DiGraph] = None):
        """Plot the causal graph."""
        import matplotlib.pyplot as plt
        
        if graph is None:
            graph = self.discovered_graph
        
        if graph is None:
            raise ValueError("No graph to plot")
        
        plt.figure(figsize=(12, 8))
        
        if isinstance(graph, nx.MultiDiGraph):
            pos = nx.spring_layout(graph)
            nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue',
                           node_size=2000, font_size=10, arrows=True)
        else:
            pos = nx.spring_layout(graph)
            nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue',
                           node_size=2000, font_size=10, arrows=True)
        
        plt.title(f"Causal Graph (Discovered via {self.discovery_method})")
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()


class DoubleMachineLearning:
    """
    Double/Debiased Machine Learning for causal effect estimation.
    
    Handles endogeneity in treatment variables using ML methods
    for nuisance parameter estimation.
    """
    
    def __init__(
        self,
        model_t: Any = None,
        model_y: Any = None,
        model_final: Any = None
    ):
        self.model_t = model_t or GradientBoostingRegressor()
        self.model_y = model_y or GradientBoostingRegressor()
        self.model_final = model_final or LinearRegression()
        self.estimator = None
    
    def fit(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        W: Optional[np.ndarray] = None
    ) -> 'DoubleMachineLearning':
        """
        Fit DML estimator.
        
        Parameters
        ----------
        Y : np.ndarray
            Outcome variable
        T : np.ndarray
            Treatment variable
        X : np.ndarray
            Features (effect modifiers)
        W : np.ndarray, optional
            Confounders
        """
        if W is None:
            W = X

        _require_dml_dependencies()
        
        # Use EconML's DML implementation
        self.estimator = DMLCateEstimator(
            model_t=self.model_t,
            model_y=self.model_y,
            model_final=self.model_final,
            discrete_treatment=False
        )
        
        self.estimator.fit(Y, T, X=X, W=W)
        
        return self
    
    def effect(self, X: np.ndarray) -> np.ndarray:
        """Estimate heterogeneous treatment effects."""
        if self.estimator is None:
            raise ValueError("Must fit estimator first")
        
        return self.estimator.effect(X)
    
    def effect_interval(self, X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Get confidence intervals for treatment effects."""
        if self.estimator is None:
            raise ValueError("Must fit estimator first")
        
        return self.estimator.effect_interval(X, alpha=alpha)
    
    def summary(self) -> str:
        """Get summary of estimated effects."""
        if self.estimator is None:
            raise ValueError("Must fit estimator first")
        
        return "DML estimation completed. Use effect() for predictions."


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate data with known causal structure
    n = 1000
    
    # X -> Y, X -> Z, Y -> Z (Z is collider)
    X = np.random.normal(0, 1, n)
    Y = 0.5 * X + np.random.normal(0, 0.5, n)
    Z = 0.3 * X + 0.4 * Y + np.random.normal(0, 0.5, n)
    W = 0.2 * Y + np.random.normal(0, 0.5, n)  # W is descendant of Y
    
    data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'W': W})
    
    # Initialize discovery
    discovery = CausalGraphDiscovery(pc_alpha=0.05)
    
    # Add domain constraint: weather cannot cause ad spend
    discovery.add_constraint('W', 'X', 'cannot_cause')
    
    # Discover graph
    graph = discovery.discover(data, method='pc')
    
    print("\nDiscovered edges:")
    for edge in graph.edges():
        print(f"  {edge[0]} -> {edge[1]}")
    
    # Validate DAG
    is_valid = discovery.validate_dag()
    print(f"\nIs valid DAG: {is_valid}")
    
    # Estimate causal effect
    effect_result = discovery.estimate_causal_effect(data, 'X', 'Y')
    print(f"\nCausal effect X -> Y: {effect_result['effect']:.4f}")
    
    # E-value
    e_value = discovery.compute_e_value(effect_result['effect'])
    print(f"E-value: {e_value['e_value']:.2f}")
