import os

import networkx as nx
import numpy as np
import pandas as pd

from causal_graph.discovery import CausalGraphDiscovery
from config import Config
from mta.attribution import ConversionPath, MarkovChainAttribution
from mmm.bayesian_mmm import AdstockConfig, AdstockTransformer


def test_adstock_transformer_geometric_runs() -> None:
    transformer = AdstockTransformer(AdstockConfig(method="geometric"))
    values = np.array([1.0, 2.0, 0.0])
    result = transformer.transform(values, {"alpha": 0.5})

    assert result.shape == (3, 1)
    assert result[0, 0] == 1.0
    assert result[1, 0] > result[0, 0]


def test_markov_attribution_weights_sum_to_one() -> None:
    model = MarkovChainAttribution(["search", "social"])
    paths = [
        ConversionPath("u1", ["search", "social"], pd.Timestamp("2026-01-01"), True, 100.0),
        ConversionPath("u2", ["social"], pd.Timestamp("2026-01-02"), False, 0.0),
    ]

    model.build_transition_matrix(paths)
    weights = model.get_attribution_weights(paths)

    assert set(weights) == {"search", "social"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_causal_graph_validate_dag() -> None:
    discovery = CausalGraphDiscovery()
    graph = nx.DiGraph()
    graph.add_edges_from([("spend", "sales"), ("seasonality", "sales")])

    assert discovery.validate_dag(graph) is True


def test_config_can_load_without_openai_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DELTA_LAKE_PATH", str(tmp_path / "delta" / "lake"))

    config = Config.from_env()

    assert config.openai.api_key is None
    assert config.validate() == []
