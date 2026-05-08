# Causal MMM + MTA Platform

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

## 🚀 Bayesian Marketing Mix Modeling + Multi-Touch Attribution with Causal Inference

Enterprise-grade marketing measurement platform combining hierarchical Bayesian MMM, multi-touch attribution, causal graph discovery, and incrementality testing. Built for organizations spending $10M+ annually on marketing who need a single source of truth for ROI measurement and budget optimization.

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Core Features](#core-features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Documentation](#module-documentation)
- [API Reference](#api-reference)
- [Role-Specific Solutions](#role-specific-solutions)
- [Automation & Integration](#automation--integration)
- [Development](#development)

---

## 🎯 Problem Statement

Marketing leaders waste **>30% of budgets** due to:

1. **Siloed attribution models** - Last-click vs. MMM vs. MTA producing conflicting insights
2. **Inability to measure true incrementality** - No causal framework for lift measurement
3. **Confounding bias** - Media spend correlated with unmeasured factors (seasonality, competitor actions)
4. **Data quality issues** - Ad platform APIs fail, IDs mismatch, latency varies
5. **Slow optimization cycles** - Budget planning takes 3+ months and still fails

This platform provides a **unified causal framework** that automatically detects confounding, handles collider bias, and delivers actionable budget recommendations with confidence intervals.

---

## ✨ Core Features

### 1. Hierarchical Bayesian Time-Series MMM
- **Adstock modeling**: Weibull decay for flexible carryover patterns
- **Saturation functions**: Hill, logistic, and power-law transformations
- **Seasonal decomposition**: Fourier series + Prophet-style components
- **Hierarchical priors**: Partial pooling across channels for improved estimates
- **Control variables**: Price, promotion, competitor spend, macroeconomic indicators
- **NUTS sampling**: Efficient MCMC with PyMC

### 2. Multi-Touch Attribution Engine
- **Markov Chain attribution**: Transition probabilities between channels
- **Shapley value computation**: Fair attribution via cooperative game theory
- **Hidden Markov Models**: Latent state modeling for engagement levels
- **User-level sessionization**: Path reconstruction from clickstream data
- **Fractional attribution**: Continuous credit allocation

### 3. Causal Inference Engine
- **Automated DAG discovery**: PC algorithm + FCI for latent confounders
- **Domain knowledge integration**: Expert constraints on causal structure
- **Causal impact estimation**: Google's Bayesian structural time-series
- **Double Machine Learning**: DML for endogenous media variables
- **E-value reporting**: Sensitivity analysis for hidden confounding

### 4. Incrementality Testing Module
- **Geo-experiment design**: Switchback, synthetic control, cluster-randomized
- **CUPED variance reduction**: ML-enhanced pre-period adjustment
- **Power analysis**: Sample size calculation for lift detection
- **Placebo tests**: Pre-period validation of experimental design

### 5. End-to-End Data Lineage
- **dbt integration**: Transformation pipelines with testing
- **Great Expectations**: Data quality SLAs and validation
- **Delta Lake**: ACID compliance and time-travel queries
- **Feature store**: Feast for consistent feature definitions

### 6. Real-Time Budget Optimization API
- **Markowitz portfolio optimization**: Risk-adjusted budget allocation
- **Convex optimization**: CVXPY with ROAS/CPA constraints
- **Scenario planning**: Counterfactual "what-if" analysis
- **Daily updates**: Automated reallocation based on latest data

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Ingestion Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Meta Ads │ │Google Ads│ │  DV360   │ │Salesforce│           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       └────────────┴────────────┴────────────┘                  │
│                         │                                        │
│                  ┌──────▼──────┐                                │
│                  │   Prefect   │ (Orchestration)                │
│                  └──────┬──────┘                                │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Data Processing Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    PySpark   │  │     dbt      │  │Great Expect. │          │
│  │ (Sessionize) │  │ (Transform)  │  │  (Quality)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                          │                                       │
│                  ┌───────▼───────┐                              │
│                  │  Delta Lake   │ (ACID Storage)               │
│                  └───────┬───────┘                              │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   Analytics Layer                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │Bayesian MMM    │  │  MTA Engine    │  │Causal Discovery│     │
│  │  (PyMC/NUTS)   │  │ (Markov/Shapley)│ │  (PC/FCI/DML)  │     │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘     │
│           │                   │                   │              │
│           └───────────────────┼───────────────────┘              │
│                       ┌───────▼───────┐                         │
│                       │Ensemble Model │                         │
│                       └───────┬───────┘                         │
└───────────────────────────────┼─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                   Optimization Layer                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         CVXPY Budget Optimizer (Markowitz Portfolio)    │    │
│  │  maximize E[revenue] subject to:                        │    │
│  │    - ROAS floor constraints                             │    │
│  │    - CPA targets                                        │    │
│  │    - Budget smoothness                                  │    │
│  │    - Channel caps/floors                                │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                   Presentation Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Streamlit   │  │  Plotly Dash │  │   FastAPI    │          │
│  │  Dashboard   │  │  Visualizations│ │   REST API   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ (for production)
- Redis 7+ (for Feast feature store)
- Ray cluster (optional, for distributed sampling)

### Quick Install
```bash
# Clone repository
git clone https://github.com/Crynge/causal-mmm-mta-platform.git
cd causal-mmm-mta-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys and configuration
```

### Docker Deployment
```bash
# Build image
docker build -t causal-mmm-mta:latest .

# Run with docker-compose
docker-compose up -d

# Access dashboard at http://localhost:8501
```

---

## 🚀 Quick Start

### 1. Basic MMM Example
```python
from mmm.bayesian_mmm import HierarchicalBayesianMMM, AdstockConfig, SaturationConfig
import pandas as pd
import numpy as np

# Load data
media_data = pd.read_csv('media_spend.csv')  # Columns: date, tv, digital, social, search
target = pd.read_csv('sales.csv')['sales'].values
dates = pd.to_datetime(media_data['date'])

# Configure model
adstock_config = AdstockConfig(method='weibull', max_lag=52)
saturation_config = SaturationConfig(method='hill')

# Initialize and fit
mmm = HierarchicalBayesianMMM(
    adstock_config=adstock_config,
    saturation_config=saturation_config
)

mmm.fit(
    target=target,
    media_data=media_data[['tv', 'digital', 'social', 'search']],
    dates=dates,
    draws=2000,
    chains=4
)

# Get results
contributions = mmm.get_channel_contributions()
roas = mmm.get_roas(media_data[['tv', 'digital', 'social', 'search']])

print(contributions)
print(roas)
```

### 2. Multi-Touch Attribution
```python
from mta.attribution import MultiTouchAttribution, ConversionPath
import pandas as pd

# Load user-level event data
events = pd.read_csv('clickstream.csv')
# Columns: user_id, channel, timestamp, converted, conversion_value

# Initialize MTA
channels = ['paid_search', 'social', 'display', 'email', 'direct']
mta = MultiTouchAttribution(channels)

# Sessionize into paths
paths = mta.sessionize(events, session_window_hours=24)

# Fit models
mta.fit(paths, methods=['markov', 'shapley'])

# Get attribution results
results = mta.get_results_df()
print(results)

# Compare models
comparison = mta.compare_models()
print(comparison)
```

### 3. Causal Graph Discovery
```python
from causal_graph.discovery import CausalGraphDiscovery, DoubleMachineLearning
import pandas as pd

# Load aggregated data
data = pd.read_csv('marketing_data.csv')
# Columns: tv_spend, digital_spend, sales, price, competitor_spend, gdp_index

# Initialize discovery with domain constraints
discovery = CausalGraphDiscovery(pc_alpha=0.05)

# Add expert knowledge: competitor spend cannot cause our TV spend
discovery.add_constraint('competitor_spend', 'tv_spend', 'cannot_cause')

# Discover causal graph
graph = discovery.discover(data, method='pc')

# Estimate causal effect of TV on sales
effect = discovery.estimate_causal_effect(
    data=data,
    treatment='tv_spend',
    outcome='sales'
)

print(f"Causal effect: {effect['effect']:.4f}")

# Compute E-value for sensitivity analysis
e_value = discovery.compute_e_value(effect['effect'])
print(f"E-value: {e_value['e_value']:.2f}")
```

### 4. Budget Optimization
```python
from optimizer.budget_allocator import BudgetOptimizer
from config import get_config

# Load configuration
config = get_config()

# Initialize optimizer
optimizer = BudgetOptimizer(config=config)

# Set constraints
optimizer.set_budget_constraint(total_budget=10_000_000)
optimizer.set_roas_floor(2.5)
optimizer.set_channel_bounds(
    min_per_channel=500_000,
    max_per_channel=5_000_000
)

# Optimize
optimal_allocation = optimizer.optimize(
    mmm_model=mmm,
    current_spend=current_budgets
)

print(optimal_allocation)
```

---

## 📚 Module Documentation

### `ingestion/` - Data Connectors
| Connector | Description | Status |
|-----------|-------------|--------|
| `facebook_connector.py` | Meta Ads API integration | ✅ Production |
| `google_ads_connector.py` | Google Ads + DV360 | ✅ Production |
| `tiktok_connector.py` | TikTok Marketing API | ✅ Production |
| `salesforce_connector.py` | CRM data extraction | ✅ Production |
| `plausible_connector.py` | Web analytics | ✅ Production |

### `causal_graph/` - Causal Inference
| Module | Purpose | Algorithms |
|--------|---------|------------|
| `discovery.py` | DAG discovery | PC, FCI, GES |
| `effects.py` | Causal effect estimation | Backdoor, Frontdoor, IV |
| `dml.py` | Double ML | DML, CATE |
| `sensitivity.py` | Robustness checks | E-value, Placebo |

### `mmm/` - Marketing Mix Modeling
| Module | Features |
|--------|----------|
| `bayesian_mmm.py` | Hierarchical Bayesian MMM with NUTS |
| `adstock.py` | Carryover transformations |
| `saturation.py` | Diminishing returns functions |
| `seasonality.py` | Fourier + Prophet decomposition |

### `mta/` - Multi-Touch Attribution
| Module | Method |
|--------|--------|
| `attribution.py` | Markov, Shapley, HMM |
| `sessionization.py` | Path reconstruction |
| `evaluation.py` | Model validation |

### `incrementality/` - Experimentation
| Module | Design |
|--------|--------|
| `geo_experiments.py` | Geo-lift tests |
| `synthetic_control.py` | Synthetic control methods |
| `switchback.py` | Time-based randomization |
| `cuped.py` | Variance reduction |

### `optimizer/` - Budget Allocation
| Module | Optimization |
|--------|--------------|
| `budget_allocator.py` | Markowitz portfolio |
| `constraints.py` | Business rules |
| `scenario_planner.py` | What-if analysis |

### `dashboard/` - Visualization
| App | Framework |
|-----|-----------|
| `executive_view.py` | Streamlit |
| `analyst_dashboard.py` | Plotly Dash |
| `api_server.py` | FastAPI |

---

## 🔌 API Reference

### REST API Endpoints

#### `POST /api/v1/audit`
Run full MMM + MTA analysis.

```json
{
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "channels": ["tv", "digital", "social"],
  "target_metric": "sales",
  "include_causal": true
}
```

Response:
```json
{
  "status": "completed",
  "run_id": "abc123",
  "results": {
    "channel_contributions": {...},
    "roas_by_channel": {...},
    "attribution_weights": {...},
    "recommended_allocation": {...}
  }
}
```

#### `GET /api/v1/budget/optimize`
Get optimal budget allocation.

Query parameters:
- `total_budget`: float
- `roas_floor`: float (optional)
- `cpa_target`: float (optional)

#### `POST /api/v1/experiment/design`
Design geo-lift experiment.

```json
{
  "treatment_regions": ["CA", "TX"],
  "duration_weeks": 8,
  "expected_lift_pct": 5.0,
  "confidence_level": 0.95
}
```

---

## 👥 Role-Specific Solutions

| Role | Problem Solved | How This Repo Delivers |
|------|----------------|------------------------|
| **CEO** | "I don't know which $100M channel drives real growth" | Single dashboard with incremental contribution per channel, causal ROI, risk-adjusted budget recommendations with confidence intervals |
| **CMO** | "Budget planning takes 3 months and still fails" | Automated budget optimizer respecting pacing, seasonality, creative fatigue. Scenario planner with counterfactual plots |
| **Marketer** | "Campaigns look great in last-click but sales dropped" | Daily alerts when attribution conflicts with MMM. SHAP explanations for each channel's contribution |
| **Operations** | "Data is messy: APIs fail, IDs mismatch" | Built-in connectors for 30+ platforms with deduplication, currency conversion, latency monitoring. Data quality SLA enforcement |
| **Analytics** | "Can't trust causal claims without sensitivity analysis" | Full E-value reporting, placebo tests on pre-period outcomes, Jupyter tutorials for custom DAGs. Model cards for each release |

---

## 🔄 Automation & Integration

### Cron Jobs
```bash
# Daily MMM update
0 2 * * * cd /opt/causal-mmm && python scripts/run_mmm.py --output reports/daily

# Weekly budget optimization
0 3 * * 1 cd /opt/causal-mmm && python scripts/optimize_budget.py --email cmo@company.com

# Monthly executive report
0 4 1 * * cd /opt/causal-mmm && python scripts/generate_report.py --format pdf
```

### GitHub Actions
```yaml
name: MMM Pipeline
on:
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:

jobs:
  run-mmm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run MMM
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: python scripts/run_mmm.py
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: mmm-results
          path: reports/
```

### n8n Integration
```json
{
  "nodes": [
    {
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": { "rule": { "interval": [{ "field": "hours", "hoursInterval": 24 }] } }
    },
    {
      "name": "HTTP Request - MMM API",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "https://your-domain.com/api/v1/audit",
        "body": { "start_date": "{{ $now.minus({ days: 30 }).toISOString() }}" }
      }
    },
    {
      "name": "Slack Notification",
      "type": "n8n-nodes-base.slack",
      "parameters": { "text": "Daily MMM report ready: {{ $json.results.summary }}" }
    }
  ]
}
```

---

## 🧪 Development

### Running Tests
```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/integration/ -v

# Coverage
pytest --cov=. --cov-report=html

# Type checking
mypy .

# Linting
black .
flake8 .
```

### Adding New Connectors
1. Create `ingestion/new_platform_connector.py`
2. Implement `BaseConnector` interface
3. Add tests in `tests/test_new_platform.py`
4. Update documentation

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📊 Output Schema

### MMM Results
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "model_diagnostics": {
    "r_hat": {"mean": 1.01, "max": 1.05},
    "effective_samples": 4500,
    "divergences": 0
  },
  "channel_effects": {
    "tv": {
      "coefficient": 0.35,
      "ci_95_lower": 0.28,
      "ci_95_upper": 0.42,
      "roas": 3.2,
      "contribution_pct": 25.5
    }
  },
  "adstock_params": {...},
  "saturation_params": {...}
}
```

### MTA Results
```json
{
  "attribution_method": "ensemble",
  "total_conversions": 15420,
  "total_revenue": 1542000,
  "channel_attribution": {
    "paid_search": {
      "conversions": 4200,
      "revenue": 420000,
      "shapley_value": 0.27,
      "removal_effect": 0.31
    }
  }
}
```

---

## 📄 License

MIT License - Free for commercial use.

## 🤝 Support

- **Documentation**: https://causal-mmm-mta.readthedocs.io
- **Issues**: https://github.com/Crynge/causal-mmm-mta-platform/issues
- **Discussions**: https://github.com/Crynge/causal-mmm-mta-platform/discussions

---

**Built for data-driven marketing teams who demand causal rigor.**

[⭐ Star this repository](https://github.com/Crynge/causal-mmm-mta-platform) if you find it useful!
