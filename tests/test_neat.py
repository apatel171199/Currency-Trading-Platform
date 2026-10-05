import json
import random

import neat
import numpy as np
import pandas as pd
import pytest

from broker import PaperBroker
from conftest import make_candles, make_pattern_candles
from evolution import Evolution, EvolutionSettings, genome_to_network
from live_trader import LiveTrader, TradingSettings
from ml_strategy import FEATURE_COLUMNS
from neat_strategy import NeatStrategy, run_network
from strategy import STRATEGY_CLASSES

SMALL = dict(population=30, patience=10, workers=1, seed=3)


def test_fast_network_matches_neat_python():
    """Our numpy evaluator must give exactly what neat-python computes."""
    evolution = Evolution(EvolutionSettings(population=30))
    config = evolution._config()
    population = neat.Population(config)
    random.seed(0)

    genomes = list(population.population.values())[:10]
    for genome in genomes:  # grow some hidden neurons and connections
        for _ in range(15):
            genome.mutate(config.genome_config)

    inputs = np.random.default_rng(0).normal(size=(50, len(FEATURE_COLUMNS)))
    zeros = np.zeros(len(FEATURE_COLUMNS))

    for genome in genomes:
        network = genome_to_network(genome, config, zeros, np.ones(len(FEATURE_COLUMNS)),
                                    EvolutionSettings())
        reference = neat.nn.FeedForwardNetwork.create(genome, config)
        expected = [reference.activate(list(row))[0] for row in inputs]
        assert run_network(network, inputs) == pytest.approx(expected, abs=1e-9)

    assert any(len(n.nodes) > 1 for n in genomes)  # hidden neurons were exercised


def _datasets(evolution, make, seeds, n=20000):
    return [evolution.prepare(f"S{seed}", make(seed, n), 1.0) for seed in seeds]


def test_evolution_finds_a_real_pattern(tmp_path):
    evolution = Evolution(EvolutionSettings(generations=4, **SMALL))
    datasets = _datasets(evolution, lambda s, n: make_pattern_candles(n=n, seed=s), [1, 2])
    result = evolution.run(datasets, output_dir=tmp_path, progress=False)

    assert all(selection.approved for selection in result.selections.values())
    assert result.network_file.exists()


def test_evolution_rejected_on_random_data(tmp_path):
    evolution = Evolution(EvolutionSettings(generations=25, **SMALL))
    datasets = _datasets(evolution, lambda s, n: make_candles(n=n, seed=s), [50, 51])
    result = evolution.run(datasets, output_dir=tmp_path, progress=False)

    assert not any(selection.approved for selection in result.selections.values())
    assert all("will not be traded" in s.note.lower() for s in result.selections.values())


def test_test_years_are_never_seen_during_evolution(tmp_path):
    """Changing only the final-exam years must not change the champion."""
    settings = EvolutionSettings(generations=3, **SMALL)
    original = make_pattern_candles(n=20000, seed=4)

    altered = original.copy()
    cut = int(len(altered) * 0.85)  # inside the last 20% (test years)
    altered.iloc[cut:, :] = altered.iloc[cut:, :] * 1.05

    champions = []
    for candles in (original, altered):
        evolution = Evolution(settings)
        data = [evolution.prepare("S", candles, 1.0)]
        result = evolution.run(data, output_dir=tmp_path, progress=False)
        champion = dict(result.network)
        champion.pop("created_at")
        champions.append(json.dumps(champion, sort_keys=True))

    assert champions[0] == champions[1]


def test_neat_strategy_no_look_ahead_and_not_in_grid_search(tmp_path):
    evolution = Evolution(EvolutionSettings(generations=1, **SMALL))
    data = _datasets(evolution, lambda s, n: make_pattern_candles(n=n, seed=s), [5], n=6000)
    result = evolution.run(data, output_dir=tmp_path, progress=False)

    strategy = NeatStrategy(str(result.network_file))
    candles = make_pattern_candles(n=3000, seed=9)
    full = strategy.signals(strategy.prepare(candles))
    past_only = strategy.signals(strategy.prepare(candles.iloc[:2000]))
    pd.testing.assert_series_equal(full.iloc[:2000], past_only, check_names=False)

    assert "NeatStrategy" in STRATEGY_CLASSES
    assert NeatStrategy.parameter_combinations() == []


def test_live_trader_runs_evolved_network(tmp_path):
    evolution = Evolution(EvolutionSettings(generations=2, **SMALL))
    data = _datasets(evolution, lambda s, n: make_pattern_candles(n=n, seed=s), [6])
    result = evolution.run(data, output_dir=tmp_path, progress=False)
    selection = result.selections["S6"]
    selection.approved = True

    candles = make_pattern_candles(n=1500, seed=7)

    def feed(symbol, timeframe, count):
        return candles.iloc[: feed.cursor].tail(count)

    broker = PaperBroker(feed, starting_balance=100_000)
    trader = LiveTrader(broker, {"S6": selection}, TradingSettings(
        max_daily_loss_pct=100, max_drawdown_pct=99, max_spread_pips=5,
        journal_path=str(tmp_path / "journal.csv")))

    for feed.cursor in range(1100, 1300):
        trader.run_once()

    assert broker.closed_trades
