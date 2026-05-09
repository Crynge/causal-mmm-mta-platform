# Architecture Notes

## Current module layout

### `mmm/bayesian_mmm.py`

Contains:

- adstock transformations
- saturation functions
- seasonal feature engineering
- hierarchical Bayesian MMM scaffolding

Heavy sampling dependencies such as PyMC and ArviZ are treated as optional until model fitting or plotting is invoked.

### `mta/attribution.py`

Contains:

- Markov-chain attribution
- Shapley-value attribution
- sessionization helpers
- result comparison utilities

This is currently the lightest and easiest part of the repo to run locally.

### `causal_graph/discovery.py`

Contains:

- PC-style skeleton discovery
- FCI-style discovery entrypoint
- expert constraint application
- causal effect and DML hooks

DoWhy and EconML are now treated as optional dependencies and are only required when those advanced methods are used.
