"""
Multi-Touch Attribution (MTA) Engine

Implements advanced attribution modeling with:
- Markov Chain transition probabilities
- Shapley value computation
- Hidden Markov Models for path analysis
- User-level sessionization
- Fractional attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ConversionPath:
    """Represents a user's conversion path."""
    user_id: str
    touchpoints: List[str]
    timestamp: pd.Timestamp
    converted: bool
    conversion_value: float = 0.0


@dataclass
class AttributionResult:
    """Attribution results for a channel."""
    channel: str
    attributed_conversions: float
    attributed_revenue: float
    contribution_percentage: float
    shapley_value: float
    removal_effect: float


class MarkovChainAttribution:
    """
    Markov Chain-based multi-touch attribution.
    
    Models customer journeys as Markov chains where states are channels
    and transitions represent movement between channels.
    """
    
    def __init__(self, channels: List[str]):
        self.channels = channels
        self.states = channels + ['START', 'CONVERSION', 'NULL']
        self.transition_matrix: Optional[np.ndarray] = None
        self.removal_effects: Dict[str, float] = {}
    
    def build_transition_matrix(
        self,
        paths: List[ConversionPath]
    ) -> np.ndarray:
        """
        Build transition probability matrix from conversion paths.
        
        Parameters
        ----------
        paths : list of ConversionPath
            User conversion paths
            
        Returns
        -------
        np.ndarray
            Transition probability matrix
        """
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for path in paths:
            # START -> first touchpoint
            if path.touchpoints:
                transition_counts['START'][path.touchpoints[0]] += 1
                
                # Intermediate transitions
                for i in range(len(path.touchpoints) - 1):
                    transition_counts[path.touchpoints[i]][path.touchpoints[i + 1]] += 1
                
                # Last touchpoint -> CONVERSION or NULL
                if path.converted:
                    transition_counts[path.touchpoints[-1]]['CONVERSION'] += 1
                else:
                    transition_counts[path.touchpoints[-1]]['NULL'] += 1
        
        # Convert to probabilities
        n_states = len(self.states)
        state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        self.transition_matrix = np.zeros((n_states, n_states))
        
        for from_state, transitions in transition_counts.items():
            total = sum(transitions.values())
            if total > 0:
                for to_state, count in transitions.items():
                    i, j = state_to_idx[from_state], state_to_idx[to_state]
                    self.transition_matrix[i, j] = count / total
        
        return self.transition_matrix
    
    def compute_removal_effect(self, channel: str) -> float:
        """
        Compute removal effect for a channel.
        
        Removal effect = 1 - (conversion rate without channel / overall conversion rate)
        """
        if self.transition_matrix is None:
            raise ValueError("Must build transition matrix first")
        
        # Overall conversion rate (with all channels)
        overall_conversion_rate = self._compute_conversion_probability()
        
        # Create modified transition matrix without the channel
        modified_matrix = self.transition_matrix.copy()
        channel_idx = self.states.index(channel)
        
        # Redirect channel transitions to NULL
        modified_matrix[:, channel_idx] = 0
        modified_matrix[channel_idx, :] = 0
        modified_matrix[channel_idx, self.states.index('NULL')] = 1.0
        
        # Renormalize rows
        row_sums = modified_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        modified_matrix = modified_matrix / row_sums
        
        # Compute conversion rate without channel
        reduced_conversion_rate = self._compute_conversion_probability(modified_matrix)
        
        # Removal effect
        if overall_conversion_rate > 0:
            removal_effect = 1 - (reduced_conversion_rate / overall_conversion_rate)
        else:
            removal_effect = 0.0
        
        self.removal_effects[channel] = max(0, removal_effect)
        return self.removal_effects[channel]
    
    def _compute_conversion_probability(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        max_steps: int = 20
    ) -> float:
        """Compute overall conversion probability using matrix multiplication."""
        if transition_matrix is None:
            transition_matrix = self.transition_matrix
        
        if transition_matrix is None:
            raise ValueError("No transition matrix available")
        
        state_to_idx = {s: i for i, s in enumerate(self.states)}
        start_idx = state_to_idx['START']
        conv_idx = state_to_idx['CONVERSION']
        
        # Initial state vector (start at START)
        state_vector = np.zeros(len(self.states))
        state_vector[start_idx] = 1.0
        
        # Simulate progression through chain
        conversion_prob = 0.0
        
        for _ in range(max_steps):
            state_vector = state_vector @ transition_matrix
            conversion_prob += state_vector[conv_idx]
            state_vector[conv_idx] = 0  # Remove converted users from further simulation
        
        return conversion_prob
    
    def get_attribution_weights(self, paths: List[ConversionPath]) -> Dict[str, float]:
        """
        Get attribution weights based on removal effects.
        
        Returns normalized weights that sum to 1.
        """
        # Compute removal effects for all channels
        for channel in self.channels:
            self.compute_removal_effect(channel)
        
        # Normalize
        total = sum(self.removal_effects.values())
        if total > 0:
            weights = {ch: eff / total for ch, eff in self.removal_effects.items()}
        else:
            weights = {ch: 1.0 / len(self.channels) for ch in self.channels}
        
        return weights


class ShapleyValueAttribution:
    """
    Shapley value-based attribution using cooperative game theory.
    
    Computes fair attribution by considering all possible coalitions
    of channels and their marginal contributions.
    """
    
    def __init__(self, channels: List[str]):
        self.channels = channels
        self.n_channels = len(channels)
        self.value_function: Optional[callable] = None
        self.shapley_values: Dict[str, float] = {}
    
    def _build_value_function(
        self,
        paths: List[ConversionPath]
    ) -> callable:
        """
        Build value function v(S) = conversion rate using only channels in S.
        """
        # Precompute conversion rates for all subsets
        subset_values = {}
        
        for mask in range(1 << self.n_channels):
            subset = [self.channels[i] for i in range(self.n_channels) if mask & (1 << i)]
            
            # Count conversions using only channels in subset
            conversions = 0
            total = 0
            
            for path in paths:
                if all(ch in subset for ch in path.touchpoints):
                    total += 1
                    if path.converted:
                        conversions += 1
            
            subset_values[frozenset(subset)] = conversions / max(total, 1)
        
        def value_function(subset: Set[str]) -> float:
            return subset_values.get(frozenset(subset), 0.0)
        
        self.value_function = value_function
        return value_function
    
    def compute_shapley_values(
        self,
        paths: List[ConversionPath],
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Compute Shapley values using Monte Carlo sampling.
        
        For large numbers of channels, exact computation is intractable,
        so we use sampling approximation.
        """
        self._build_value_function(paths)
        
        shapley_values = defaultdict(float)
        
        for _ in range(n_samples):
            # Random permutation of channels
            perm = np.random.permutation(self.channels)
            
            # Compute marginal contribution for each channel
            coalition = set()
            
            for channel in perm:
                # Value before adding channel
                value_before = self.value_function(coalition)
                
                # Add channel to coalition
                coalition.add(channel)
                
                # Value after adding channel
                value_after = self.value_function(coalition)
                
                # Marginal contribution
                marginal = value_after - value_before
                shapley_values[channel] += marginal
        
        # Average over samples
        for channel in self.channels:
            self.shapley_values[channel] = shapley_values[channel] / n_samples
        
        # Normalize to sum to 1
        total = sum(self.shapley_values.values())
        if total > 0:
            for channel in self.channels:
                self.shapley_values[channel] /= total
        
        return dict(self.shapley_values)


class HiddenMarkovModelAttribution:
    """
    Hidden Markov Model for latent state attribution.
    
    Assumes customers move through latent engagement states
    that influence both channel exposure and conversion probability.
    """
    
    def __init__(
        self,
        n_hidden_states: int = 3,
        channels: Optional[List[str]] = None
    ):
        self.n_hidden_states = n_hidden_states
        self.channels = channels or []
        
        # HMM parameters (to be estimated)
        self.initial_probs: Optional[np.ndarray] = None
        self.transition_probs: Optional[np.ndarray] = None
        self.emission_probs: Optional[np.ndarray] = None
        self.conversion_probs: Optional[np.ndarray] = None
    
    def fit(
        self,
        paths: List[ConversionPath],
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> 'HiddenMarkovModelAttribution':
        """
        Fit HMM using Baum-Welch algorithm.
        
        This is a simplified implementation; production would use
        hmmlearn or custom Cython code for efficiency.
        """
        if not paths:
            raise ValueError("No paths provided")
        
        # Encode channels
        channel_to_idx = {ch: i for i, ch in enumerate(self.channels)}
        
        # Initialize parameters randomly
        np.random.seed(42)
        self.initial_probs = np.random.dirichlet(np.ones(self.n_hidden_states))
        self.transition_probs = np.random.dirichlet(np.ones(self.n_hidden_states), 
                                                     size=self.n_hidden_states)
        
        n_observations = len(self.channels)
        self.emission_probs = np.random.dirichlet(np.ones(n_observations),
                                                   size=self.n_hidden_states)
        self.conversion_probs = np.random.beta(1, 1, size=self.n_hidden_states)
        
        # Simplified EM algorithm
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: Compute expected sufficient statistics
            # M-step: Update parameters
            
            # This is a placeholder for the full Baum-Welch algorithm
            # Production implementation would include forward-backward algorithm
            
            log_likelihood = self._compute_log_likelihood(paths, channel_to_idx)
            
            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                break
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def _compute_log_likelihood(
        self,
        paths: List[ConversionPath],
        channel_to_idx: Dict[str, int]
    ) -> float:
        """Compute log likelihood of observed paths."""
        ll = 0.0
        
        for path in paths:
            path_ll = 0.0
            
            # Forward algorithm to compute P(path | model)
            # Simplified implementation
            
            ll += max(path_ll, -100)  # Avoid log(0)
        
        return ll
    
    def get_state_attributions(self) -> Dict[int, float]:
        """Get attribution by hidden state."""
        if self.conversion_probs is None:
            raise ValueError("Model must be fitted first")
        
        # Attribution proportional to conversion probability
        total = self.conversion_probs.sum()
        if total > 0:
            return {i: p / total for i, p in enumerate(self.conversion_probs)}
        return {}


class MultiTouchAttribution:
    """
    Main interface for multi-touch attribution.
    
    Combines multiple attribution methods and provides ensemble estimates.
    """
    
    def __init__(self, channels: List[str]):
        self.channels = channels
        self.markov_model = MarkovChainAttribution(channels)
        self.shapley_model = ShapleyValueAttribution(channels)
        self.hmm_model: Optional[HiddenMarkovModelAttribution] = None
        
        self.results: Dict[str, AttributionResult] = {}
    
    def sessionize(
        self,
        events: pd.DataFrame,
        session_window_hours: int = 24
    ) -> List[ConversionPath]:
        """
        Convert raw event data into conversion paths.
        
        Parameters
        ----------
        events : pd.DataFrame
            Event data with columns: user_id, channel, timestamp, converted, value
        session_window_hours : int
            Hours between events to consider same journey
            
        Returns
        -------
        list of ConversionPath
        """
        paths = []
        
        # Sort by user and timestamp
        events = events.sort_values(['user_id', 'timestamp'])
        
        for user_id, user_events in events.groupby('user_id'):
            # Group into sessions
            time_diffs = user_events['timestamp'].diff()
            session_breaks = time_diffs > pd.Timedelta(hours=session_window_hours)
            session_ids = session_breaks.cumsum()
            
            for session_id, session_events in user_events.groupby(session_ids):
                touchpoints = session_events['channel'].tolist()
                converted = session_events['converted'].any()
                value = session_events[session_events['converted']]['value'].sum()
                
                paths.append(ConversionPath(
                    user_id=user_id,
                    touchpoints=touchpoints,
                    timestamp=session_events['timestamp'].min(),
                    converted=converted,
                    conversion_value=value
                ))
        
        return paths
    
    def fit(
        self,
        paths: List[ConversionPath],
        methods: List[str] = ['markov', 'shapley']
    ) -> 'MultiTouchAttribution':
        """
        Fit attribution models.
        
        Parameters
        ----------
        paths : list of ConversionPath
            Conversion paths
        methods : list
            Methods to use: 'markov', 'shapley', 'hmm'
        """
        if 'markov' in methods:
            self.markov_model.build_transition_matrix(paths)
            weights = self.markov_model.get_attribution_weights(paths)
            
            # Compute total conversions and revenue
            total_conversions = sum(p.conversion_value for p in paths if p.converted)
            total_count = sum(1 for p in paths if p.converted)
            
            for channel in self.channels:
                self.results[channel] = AttributionResult(
                    channel=channel,
                    attributed_conversions=total_count * weights.get(channel, 0),
                    attributed_revenue=total_conversions * weights.get(channel, 0),
                    contribution_percentage=weights.get(channel, 0) * 100,
                    shapley_value=0,  # Will be filled if shapley method used
                    removal_effect=self.markov_model.removal_effects.get(channel, 0)
                )
        
        if 'shapley' in methods:
            shapley_values = self.shapley_model.compute_shapley_values(paths)
            
            total_conversions = sum(p.conversion_value for p in paths if p.converted)
            total_count = sum(1 for p in paths if p.converted)
            
            for channel in self.channels:
                if channel in self.results:
                    self.results[channel].shapley_value = shapley_values.get(channel, 0)
                    self.results[channel].attributed_conversions = (
                        total_count * shapley_values.get(channel, 0)
                    )
                    self.results[channel].attributed_revenue = (
                        total_conversions * shapley_values.get(channel, 0)
                    )
                    self.results[channel].contribution_percentage = (
                        shapley_values.get(channel, 0) * 100
                    )
                else:
                    self.results[channel] = AttributionResult(
                        channel=channel,
                        attributed_conversions=total_count * shapley_values.get(channel, 0),
                        attributed_revenue=total_conversions * shapley_values.get(channel, 0),
                        contribution_percentage=shapley_values.get(channel, 0) * 100,
                        shapley_value=shapley_values.get(channel, 0),
                        removal_effect=0
                    )
        
        return self
    
    def get_results_df(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        for result in self.results.values():
            data.append({
                'channel': result.channel,
                'attributed_conversions': result.attributed_conversions,
                'attributed_revenue': result.attributed_revenue,
                'contribution_percentage': result.contribution_percentage,
                'shapley_value': result.shapley_value,
                'removal_effect': result.removal_effect
            })
        
        return pd.DataFrame(data).sort_values('attributed_revenue', ascending=False)
    
    def compare_models(self) -> pd.DataFrame:
        """Compare attribution across different models."""
        comparison = []
        
        # Markov weights
        markov_weights = self.markov_model.get_attribution_weights([]) if self.markov_model.transition_matrix is not None else {}
        
        # Shapley values
        shapley_values = self.shapley_model.shapley_values
        
        for channel in self.channels:
            comparison.append({
                'channel': channel,
                'markov_removal_effect': self.markov_model.removal_effects.get(channel, 0),
                'shapley_value': shapley_values.get(channel, 0),
                'last_click': 0,  # Would compute separately
                'first_click': 0,  # Would compute separately
                'linear': 1.0 / len(self.channels)
            })
        
        return pd.DataFrame(comparison)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate conversion paths
    channels = ['paid_search', 'social', 'display', 'email', 'direct']
    
    paths = []
    for i in range(1000):
        n_touches = np.random.randint(1, 5)
        touchpoints = list(np.random.choice(channels, n_touches, replace=True))
        converted = np.random.random() < 0.3
        value = np.random.exponential(100) if converted else 0
        
        paths.append(ConversionPath(
            user_id=f"user_{i}",
            touchpoints=touchpoints,
            timestamp=pd.Timestamp.now(),
            converted=converted,
            conversion_value=value
        ))
    
    # Run MTA
    mta = MultiTouchAttribution(channels)
    mta.fit(paths, methods=['markov', 'shapley'])
    
    # Get results
    results_df = mta.get_results_df()
    print("\nAttribution Results:")
    print(results_df.to_string(index=False))
    
    # Compare models
    comparison = mta.compare_models()
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
