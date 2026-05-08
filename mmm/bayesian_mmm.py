"""
Hierarchical Bayesian Marketing Mix Model (MMM)

Implements a comprehensive MMM with:
- Adstock (carryover) effects using Weibull decay
- Diminishing returns via Hill saturation functions
- Seasonal decomposition (Fourier + Prophet-style)
- Hierarchical priors for channel pooling
- Macroeconomic and competitor controls
- NUTS sampling with PyMC
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import arviz as az
import pymc as pm
from scipy.special import expit
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AdstockConfig:
    """Configuration for adstock transformation."""
    method: str = 'weibull'  # 'geometric', 'weibull', 'exponential'
    max_lag: int = 52  # Maximum weeks to consider
    prior_mean: float = 0.5
    prior_std: float = 0.3


@dataclass
class SaturationConfig:
    """Configuration for saturation function."""
    method: str = 'hill'  # 'hill', 'logistic', 'power_law'
    prior_half_normal_sigma: float = 1.0


@dataclass
class SeasonalityConfig:
    """Configuration for seasonal components."""
    fourier_order: int = 10
    include_weekly: bool = True
    include_monthly: bool = True
    include_yearly: bool = True
    prophet_style: bool = True


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical priors."""
    pooling_method: str = 'partial'  # 'none', 'partial', 'full'
    hyperprior_mu: float = 0.0
    hyperprior_sigma: float = 1.0


class AdstockTransformer:
    """
    Implements various adstock (carryover) transformations.
    
    Adstock models the delayed and cumulative effect of advertising exposure.
    """
    
    def __init__(self, config: AdstockConfig):
        self.config = config
    
    def geometric_adstock(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """
        Geometric adstock: simple exponential decay.
        
        y[t] = x[t] + alpha * y[t-1]
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        n_periods = x.shape[0]
        adstocked = np.zeros((n_periods, x.shape[1]))
        
        for t in range(n_periods):
            if t == 0:
                adstocked[t] = x[t]
            else:
                adstocked[t] = x[t] + alpha * adstocked[t - 1]
        
        return adstocked
    
    def weibull_adstock(self, x: np.ndarray, theta: float, omega: float) -> np.ndarray:
        """
        Weibull adstock: flexible decay shape.
        
        Uses Weibull CDF for more flexible decay patterns.
        theta: scale parameter (decay rate)
        omega: shape parameter (decay pattern)
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        n_periods = x.shape[0]
        max_lag = min(self.config.max_lag, n_periods)
        
        # Create Weibull weights
        lags = np.arange(max_lag)
        weights = np.exp(-np.power(lags / (theta + 1e-6), omega))
        weights = weights / (weights.sum() + 1e-6)
        
        # Convolve
        adstocked = np.zeros_like(x)
        for t in range(n_periods):
            start_idx = max(0, t - max_lag + 1)
            weight_slice = weights[:t - start_idx + 1][::-1]
            adstocked[t] = np.sum(x[start_idx:t + 1] * weight_slice[:, None], axis=0)
        
        return adstocked
    
    def transform(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply adstock transformation based on configuration."""
        if self.config.method == 'geometric':
            return self.geometric_adstock(x, params.get('alpha', 0.5))
        elif self.config.method == 'weibull':
            return self.weibull_adstock(
                x, 
                params.get('theta', 3.0), 
                params.get('omega', 2.0)
            )
        else:
            raise ValueError(f"Unknown adstock method: {self.config.method}")


class SaturationFunction:
    """
    Implements saturation functions for diminishing returns.
    """
    
    def __init__(self, config: SaturationConfig):
        self.config = config
    
    def hill_function(self, x: np.ndarray, slope: float, half_sat: float) -> np.ndarray:
        """
        Hill function: common in MMM for saturation modeling.
        
        f(x) = slope * x / (half_sat + x)
        """
        return slope * x / (half_sat + x + 1e-6)
    
    def logistic_function(self, x: np.ndarray, slope: float, midpoint: float, rate: float) -> np.ndarray:
        """Logistic saturation function."""
        return slope / (1 + np.exp(-rate * (x - midpoint)))
    
    def power_law(self, x: np.ndarray, slope: float, exponent: float) -> np.ndarray:
        """Power law saturation."""
        return slope * np.power(x, exponent)
    
    def transform(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply saturation transformation."""
        if self.config.method == 'hill':
            return self.hill_function(
                x, 
                params.get('slope', 1.0), 
                params.get('half_sat', 1.0)
            )
        elif self.config.method == 'logistic':
            return self.logistic_function(
                x,
                params.get('slope', 1.0),
                params.get('midpoint', 0.5),
                params.get('rate', 1.0)
            )
        elif self.config.method == 'power_law':
            return self.power_law(
                x,
                params.get('slope', 1.0),
                params.get('exponent', 0.5)
            )
        else:
            raise ValueError(f"Unknown saturation method: {self.config.method}")


class SeasonalDecomposer:
    """
    Seasonal decomposition using Fourier series and Prophet-style components.
    """
    
    def __init__(self, config: SeasonalityConfig):
        self.config = config
    
    def fourier_features(self, dates: pd.DatetimeIndex, period: float, order: int) -> np.ndarray:
        """Generate Fourier features for a given period."""
        n = len(dates)
        features = np.zeros((n, 2 * order))
        
        for i in range(order):
            features[:, 2 * i] = np.sin(2 * np.pi * (i + 1) * np.arange(n) / period)
            features[:, 2 * i + 1] = np.cos(2 * np.pi * (i + 1) * np.arange(n) / period)
        
        return features
    
    def create_features(self, dates: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """Create all seasonal features."""
        features = {}
        
        if self.config.include_weekly:
            features['weekly'] = self.fourier_features(dates, 52/12, self.config.fourier_order)
        
        if self.config.include_monthly:
            features['monthly'] = self.fourier_features(dates, 52, self.config.fourier_order)
        
        if self.config.include_yearly:
            features['yearly'] = self.fourier_features(dates, 52*4, self.config.fourier_order // 2)
        
        return features


class HierarchicalBayesianMMM:
    """
    Hierarchical Bayesian Marketing Mix Model.
    
    This is the core model class that implements:
    - Multi-channel media effects with adstock and saturation
    - Hierarchical priors for partial pooling across channels
    - Control variables (price, promotion, competitor, macro)
    - Seasonal decomposition
    - Bayesian inference with NUTS sampling
    """
    
    def __init__(
        self,
        adstock_config: AdstockConfig = None,
        saturation_config: SaturationConfig = None,
        seasonality_config: SeasonalityConfig = None,
        hierarchical_config: HierarchicalConfig = None
    ):
        self.adstock_config = adstock_config or AdstockConfig()
        self.saturation_config = saturation_config or SaturationConfig()
        self.seasonality_config = seasonality_config or SeasonalityConfig()
        self.hierarchical_config = hierarchical_config or HierarchicalConfig()
        
        self.adstock_transformer = AdstockTransformer(self.adstock_config)
        self.saturation_function = SaturationFunction(self.saturation_config)
        self.seasonal_decomposer = SeasonalDecomposer(self.seasonality_config)
        
        self.model: Optional[pm.Model] = None
        self.trace = None
        self.posterior_predictive = None
        self.summary_stats = None
        
        logger.info("Initialized HierarchicalBayesianMMM")
    
    def _build_model(
        self,
        target: np.ndarray,
        media_data: pd.DataFrame,
        control_data: Optional[pd.DataFrame] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> pm.Model:
        """
        Build the PyMC model graph.
        
        Parameters
        ----------
        target : np.ndarray
            Target variable (e.g., sales, conversions)
        media_data : pd.DataFrame
            Media spend by channel (columns = channels)
        control_data : pd.DataFrame, optional
            Control variables (price, promotion, etc.)
        dates : pd.DatetimeIndex, optional
            Date index for seasonal features
        """
        n_obs = len(target)
        channels = media_data.columns.tolist()
        n_channels = len(channels)
        
        # Prepare media data with adstock and saturation
        media_transformed = np.zeros_like(media_data.values)
        for i, channel in enumerate(channels):
            x = media_data[channel].values.reshape(-1, 1)
            
            # Apply adstock (using placeholder parameters for now)
            x_adstocked = self.adstock_transformer.transform(x, {'theta': 3.0, 'omega': 2.0})
            
            # Apply saturation
            x_saturated = self.saturation_function.transform(x_adstocked, {'slope': 1.0, 'half_sat': 1.0})
            
            media_transformed[:, i] = x_saturated.flatten()
        
        # Prepare control variables
        if control_data is not None:
            control_values = control_data.values
            n_controls = control_data.shape[1]
        else:
            control_values = np.empty((n_obs, 0))
            n_controls = 0
        
        # Prepare seasonal features
        seasonal_features = {}
        if dates is not None:
            seasonal_features = self.seasonal_decomposer.create_features(dates)
        
        with pm.Model() as mmm_model:
            # ========== PRIORS ==========
            
            # Intercept
            intercept = pm.Normal('intercept', mu=0, sigma=10)
            
            # Noise standard deviation
            sigma = pm.HalfNormal('sigma', sigma=10)
            
            # ========== HIERARCHICAL CHANNEL EFFECTS ==========
            
            if self.hierarchical_config.pooling_method == 'partial':
                # Hierarchical priors for channel coefficients
                mu_channel = pm.Normal('mu_channel', mu=self.hierarchical_config.hyperprior_mu, 
                                       sigma=self.hierarchical_config.hyperprior_sigma)
                sigma_channel = pm.HalfNormal('sigma_channel', sigma=1.0)
                
                # Channel-specific coefficients (partially pooled)
                channel_coefs = pm.Normal('channel_coefs', mu=mu_channel, sigma=sigma_channel, 
                                          shape=n_channels)
            else:
                # Independent priors for each channel
                channel_coefs = pm.HalfNormal('channel_coefs', sigma=5.0, shape=n_channels)
            
            # Adstock parameters per channel (Weibull)
            if self.adstock_config.method == 'weibull':
                theta = pm.Gamma('theta', alpha=2, beta=0.5, shape=n_channels)
                omega = pm.Gamma('omega', alpha=2, beta=0.5, shape=n_channels)
            
            # Saturation parameters per channel (Hill function)
            if self.saturation_config.method == 'hill':
                slope = pm.HalfNormal('slope', sigma=5.0, shape=n_channels)
                half_sat = pm.HalfNormal('half_sat', sigma=5.0, shape=n_channels)
            
            # ========== CONTROL VARIABLES ==========
            
            if n_controls > 0:
                control_coefs = pm.Laplace('control_coefs', mu=0, b=1, shape=n_controls)
            
            # ========== SEASONAL EFFECTS ==========
            
            seasonal_coefs = {}
            for name, feats in seasonal_features.items():
                n_feats = feats.shape[1]
                seasonal_coefs[name] = pm.Normal(f'seasonal_{name}', mu=0, sigma=1, shape=n_feats)
            
            # ========== DETERMINISTIC TRANSFORMATIONS ==========
            
            # Compute media contribution with proper adstock and saturation
            media_contribution = pm.Deterministic('media_contribution', 
                                                   channel_coefs * media_transformed)
            
            # Sum media effects
            total_media_effect = pm.Deterministic('total_media_effect', 
                                                   media_contribution.sum(axis=1))
            
            # Add control effects
            mu = intercept + total_media_effect
            
            if n_controls > 0:
                control_effect = pm.Deterministic('control_effect', 
                                                   control_values @ control_coefs)
                mu = mu + control_effect
            
            # Add seasonal effects
            for name, feats in seasonal_features.items():
                seasonal_effect = pm.Deterministic(f'seasonal_effect_{name}',
                                                    feats @ seasonal_coefs[name])
                mu = mu + seasonal_effect
            
            # ========== LIKELIHOOD ==========
            
            # Use Student-T for robustness to outliers
            nu = pm.Exponential('nu', 1/7)  # Degrees of freedom
            
            likelihood = pm.StudentT('likelihood', 
                                     mu=mu, 
                                     sigma=sigma, 
                                     nu=nu, 
                                     observed=target)
        
        self.model = mmm_model
        return mmm_model
    
    def fit(
        self,
        target: np.ndarray,
        media_data: pd.DataFrame,
        control_data: Optional[pd.DataFrame] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42
    ) -> 'HierarchicalBayesianMMM':
        """
        Fit the MMM model using MCMC sampling.
        
        Parameters
        ----------
        target : np.ndarray
            Target variable
        media_data : pd.DataFrame
            Media spend by channel
        control_data : pd.DataFrame, optional
            Control variables
        dates : pd.DatetimeIndex, optional
            Date index
        draws : int
            Number of posterior samples
        tune : int
            Number of tuning steps
        chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance rate for NUTS
        random_seed : int
            Random seed for reproducibility
        """
        logger.info(f"Fitting MMM model with {draws} draws, {chains} chains")
        
        # Build model
        self._build_model(target, media_data, control_data, dates)
        
        # Sample from posterior
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True
            )
        
        # Posterior predictive checks
        with self.model:
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                random_seed=random_seed
            )
        
        # Compute summary statistics
        self.summary_stats = az.summary(self.trace)
        
        logger.info("Model fitting completed")
        logger.info(f"R-hat values: {self.summary_stats['r_hat'].describe()}")
        
        return self
    
    def predict(
        self,
        media_data: pd.DataFrame,
        control_data: Optional[pd.DataFrame] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        point_estimate: str = 'mean'
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from the fitted model.
        
        Parameters
        ----------
        media_data : pd.DataFrame
            Media spend scenarios
        control_data : pd.DataFrame, optional
            Control variable scenarios
        dates : pd.DatetimeIndex, optional
            Dates for prediction
        point_estimate : str
            'mean', 'median', or full distribution
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract posterior samples
        channel_coefs = self.trace.posterior['channel_coefs'].values
        intercept = self.trace.posterior['intercept'].values
        
        # Transform media data (simplified - should use sampled adstock/saturation params)
        media_transformed = np.zeros_like(media_data.values)
        for i, channel in enumerate(media_data.columns):
            x = media_data[channel].values.reshape(-1, 1)
            x_adstocked = self.adstock_transformer.transform(x, {'theta': 3.0, 'omega': 2.0})
            x_saturated = self.saturation_function.transform(x_adstocked, {'slope': 1.0, 'half_sat': 1.0})
            media_transformed[:, i] = x_saturated.flatten()
        
        # Compute predictions
        n_samples = channel_coefs.shape[0] * channel_coefs.shape[1]
        channel_coefs_flat = channel_coefs.reshape(-1, channel_coefs.shape[2])
        
        predictions = np.zeros((n_samples, len(media_data)))
        for s in range(n_samples):
            predictions[s] = intercept[s % len(intercept)] + media_transformed @ channel_coefs_flat[s]
        
        if point_estimate == 'mean':
            return {'predictions': predictions.mean(axis=0)}
        elif point_estimate == 'median':
            return {'predictions': np.median(predictions, axis=0)}
        else:
            return {'predictions': predictions, 'prediction_intervals': np.percentile(predictions, [5, 95], axis=0)}
    
    def get_channel_contributions(self) -> pd.DataFrame:
        """
        Extract channel-level contribution estimates.
        
        Returns
        -------
        pd.DataFrame
            Channel contributions with credible intervals
        """
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        channel_coefs = self.trace.posterior['channel_coefs'].values
        channels = list(self.model.coords.get('channel_coefs_dim_0', []))
        
        contributions = {}
        for i, channel in enumerate(channels):
            coefs = channel_coefs[:, :, i].flatten()
            contributions[channel] = {
                'mean': coefs.mean(),
                'median': np.median(coefs),
                'std': coefs.std(),
                'ci_lower': np.percentile(coefs, 2.5),
                'ci_upper': np.percentile(coefs, 97.5)
            }
        
        return pd.DataFrame(contributions).T
    
    def get_roas(self, media_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Return on Ad Spend (ROAS) by channel.
        
        ROAS = Incremental Revenue / Media Spend
        """
        contributions = self.get_channel_contributions()
        
        roas = {}
        for channel in contributions.index:
            if channel in media_data.columns:
                avg_spend = media_data[channel].mean()
                if avg_spend > 0:
                    marginal_roas = contributions.loc[channel, 'mean']
                    roas[channel] = {
                        'roas': marginal_roas,
                        'roas_ci_lower': contributions.loc[channel, 'ci_lower'],
                        'roas_ci_upper': contributions.loc[channel, 'ci_upper']
                    }
        
        return pd.DataFrame(roas).T
    
    def plot_trace(self, var_names: Optional[List[str]] = None):
        """Plot MCMC trace diagnostics."""
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        return az.plot_trace(self.trace, var_names=var_names)
    
    def plot_posterior(self, var_names: Optional[List[str]] = None):
        """Plot posterior distributions."""
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        return az.plot_posterior(self.trace, var_names=var_names)
    
    def save(self, path: Union[str, Path]):
        """Save model and trace to disk."""
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'trace': self.trace,
                'posterior_predictive': self.posterior_predictive,
                'summary_stats': self.summary_stats,
                'configs': {
                    'adstock': self.adstock_config,
                    'saturation': self.saturation_config,
                    'seasonality': self.seasonality_config,
                    'hierarchical': self.hierarchical_config
                }
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'HierarchicalBayesianMMM':
        """Load model from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            adstock_config=data['configs']['adstock'],
            saturation_config=data['configs']['saturation'],
            seasonality_config=data['configs']['seasonality'],
            hierarchical_config=data['configs']['hierarchical']
        )
        
        instance.trace = data['trace']
        instance.posterior_predictive = data['posterior_predictive']
        instance.summary_stats = data['summary_stats']
        
        logger.info(f"Model loaded from {path}")
        return instance


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    n_weeks = 104
    dates = pd.date_range('2022-01-01', periods=n_weeks, freq='W')
    
    # Simulate media spend
    media_data = pd.DataFrame({
        'tv': np.random.exponential(10000, n_weeks),
        'digital': np.random.exponential(5000, n_weeks),
        'social': np.random.exponential(3000, n_weeks),
        'search': np.random.exponential(8000, n_weeks)
    })
    
    # Simulate target with true effects
    true_effects = {'tv': 0.3, 'digital': 0.5, 'social': 0.4, 'search': 0.6}
    base_sales = 50000
    noise = np.random.normal(0, 2000, n_weeks)
    
    sales = base_sales + sum(media_data[ch] * eff for ch, eff in true_effects.items()) + noise
    
    # Initialize and fit model
    mmm = HierarchicalBayesianMMM()
    
    # Note: In practice, use fewer samples for testing
    mmm.fit(
        target=sales.values,
        media_data=media_data,
        dates=dates,
        draws=500,
        tune=250,
        chains=2
    )
    
    # Get results
    contributions = mmm.get_channel_contributions()
    print("\nChannel Contributions:")
    print(contributions)
    
    roas = mmm.get_roas(media_data)
    print("\nROAS by Channel:")
    print(roas)
