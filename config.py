"""
Configuration Management for Causal MMM + MTA Platform

This module provides type-safe configuration loading with validation,
environment variable management, and runtime parameter overrides.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv()


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration."""
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./mmm_mta.db"))
    delta_lake_path: str = field(default_factory=lambda: os.getenv("DELTA_LAKE_PATH", "./data/delta_lake"))
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"))
    timeout: int = 30
    max_retries: int = 3


@dataclass(frozen=True)
class AdPlatformConfig:
    """Ad platform API credentials."""
    # Google Ads
    google_ads_developer_token: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"))
    google_ads_client_id: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_ADS_CLIENT_ID"))
    google_ads_client_secret: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_ADS_CLIENT_SECRET"))
    google_ads_refresh_token: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_ADS_REFRESH_TOKEN"))
    
    # Facebook/Meta
    facebook_app_id: Optional[str] = field(default_factory=lambda: os.getenv("FACEBOOK_APP_ID"))
    facebook_app_secret: Optional[str] = field(default_factory=lambda: os.getenv("FACEBOOK_APP_SECRET"))
    facebook_access_token: Optional[str] = field(default_factory=lambda: os.getenv("FACEBOOK_ACCESS_TOKEN"))
    
    # TikTok
    tiktok_access_token: Optional[str] = field(default_factory=lambda: os.getenv("TIKTOK_ACCESS_TOKEN"))
    
    # Twitter/X
    twitter_consumer_key: Optional[str] = field(default_factory=lambda: os.getenv("TWITTER_CONSUMER_KEY"))
    twitter_consumer_secret: Optional[str] = field(default_factory=lambda: os.getenv("TWITTER_CONSUMER_SECRET"))
    twitter_access_token: Optional[str] = field(default_factory=lambda: os.getenv("TWITTER_ACCESS_TOKEN"))
    twitter_access_token_secret: Optional[str] = field(default_factory=lambda: os.getenv("TWITTER_ACCESS_TOKEN_SECRET"))
    
    # Salesforce
    salesforce_client_id: Optional[str] = field(default_factory=lambda: os.getenv("SALESFORCE_CLIENT_ID"))
    salesforce_client_secret: Optional[str] = field(default_factory=lambda: os.getenv("SALESFORCE_CLIENT_SECRET"))
    salesforce_username: Optional[str] = field(default_factory=lambda: os.getenv("SALESFORCE_USERNAME"))
    salesforce_password: Optional[str] = field(default_factory=lambda: os.getenv("SALESFORCE_PASSWORD"))
    salesforce_security_token: Optional[str] = field(default_factory=lambda: os.getenv("SALESFORCE_SECURITY_TOKEN"))


@dataclass(frozen=True)
class BayesianConfig:
    """Bayesian sampling configuration for PyMC models."""
    num_chains: int = field(default_factory=lambda: int(os.getenv("NUM_CHAINS", "4")))
    num_samples: int = field(default_factory=lambda: int(os.getenv("NUM_SAMPLES", "2000")))
    num_warmup: int = field(default_factory=lambda: int(os.getenv("NUM_WARMUP", "1000")))
    target_accept_rate: float = field(default_factory=lambda: float(os.getenv("TARGET_ACCEPT_RATE", "0.8")))
    max_tree_depth: int = field(default_factory=lambda: int(os.getenv("MAX_TREE_DEPTH", "10")))
    tune: int = field(default_factory=lambda: int(os.getenv("NUM_WARMUP", "1000")))
    random_seed: int = 42


@dataclass(frozen=True)
class CausalDiscoveryConfig:
    """Causal graph discovery configuration."""
    pc_alpha: float = field(default_factory=lambda: float(os.getenv("PC_ALPHA", "0.05")))
    fci_max_conditioning_set: int = field(default_factory=lambda: int(os.getenv("FCI_MAX_CONDITIONING_SET", "3")))
    constraints_path: Optional[str] = field(default_factory=lambda: os.getenv("DOMAIN_KNOWLEDGE_CONSTRAINTS"))
    expert_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.constraints_path and Path(self.constraints_path).exists():
            with open(self.constraints_path, 'r') as f:
                object.__setattr__(self, 'expert_constraints', json.load(f))


@dataclass(frozen=True)
class OptimizationConfig:
    """Budget optimization constraints."""
    min_budget_per_channel: float = field(default_factory=lambda: float(os.getenv("MIN_BUDGET_PER_CHANNEL", "1000")))
    max_budget_per_channel: float = field(default_factory=lambda: float(os.getenv("MAX_BUDGET_PER_CHANNEL", "10000000")))
    roas_floor: float = field(default_factory=lambda: float(os.getenv("ROAS_FLOOR", "2.0")))
    cpa_target: float = field(default_factory=lambda: float(os.getenv("CPA_TARGET", "50.0")))
    budget_smoothness_factor: float = field(default_factory=lambda: float(os.getenv("BUDGET_SMOOTHNESS_FACTOR", "0.1")))


@dataclass(frozen=True)
class DataQualityConfig:
    """Data quality SLA configuration."""
    max_data_latency_hours: int = field(default_factory=lambda: int(os.getenv("MAX_DATA_LATENCY_HOURS", "24")))
    min_data_completeness_pct: float = field(default_factory=lambda: float(os.getenv("MIN_DATA_COMPLETENESS_PCT", "95.0")))
    duplicate_tolerance_pct: float = field(default_factory=lambda: float(os.getenv("DUPLICATE_TOLERANCE_PCT", "0.1")))


@dataclass(frozen=True)
class RayConfig:
    """Ray distributed computing configuration."""
    head_node_host: str = field(default_factory=lambda: os.getenv("RAY_HEAD_NODE_HOST", "localhost"))
    num_cpus: int = field(default_factory=lambda: int(os.getenv("RAY_NUM_CPUS", "16")))
    num_gpus: int = field(default_factory=lambda: int(os.getenv("RAY_NUM_GPUS", "2")))
    object_store_memory: str = field(default_factory=lambda: os.getenv("RAY_OBJECT_STORE_MEMORY", "10GB"))


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoring and logging configuration."""
    sentry_dsn: Optional[str] = field(default_factory=lambda: os.getenv("SENTRY_DSN"))
    prometheus_port: int = field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


@dataclass(frozen=True)
class FeatureStoreConfig:
    """Feast feature store configuration."""
    redis_host: str = field(default_factory=lambda: os.getenv("FEAST_REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("FEAST_REDIS_PORT", "6379")))
    registry_path: str = field(default_factory=lambda: os.getenv("FEAST_REGISTRY_PATH", "./feature_store/registry"))


@dataclass(frozen=True)
class Config:
    """Main configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    ad_platforms: AdPlatformConfig = field(default_factory=AdPlatformConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    causal_discovery: CausalDiscoveryConfig = field(default_factory=CausalDiscoveryConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'Config':
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        return cls()
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate Bayesian parameters
        if self.bayesian.num_samples <= 0:
            errors.append("num_samples must be positive")
        if not 0 < self.bayesian.target_accept_rate <= 1:
            errors.append("target_accept_rate must be in (0, 1]")
        
        # Validate optimization constraints
        if self.optimization.min_budget_per_channel >= self.optimization.max_budget_per_channel:
            errors.append("min_budget_per_channel must be less than max_budget_per_channel")
        
        return errors


def get_config() -> Config:
    """Get global configuration instance."""
    return Config.from_env()


if __name__ == "__main__":
    config = get_config()
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validated successfully")
