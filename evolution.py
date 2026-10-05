"""Evolves trading "brains" with NEAT (NeuroEvolution of Augmenting Topologies).

How it works
------------
* A population of small neural networks starts out random.
* Every generation each network trades the TRAINING years of every symbol
  through the normal backtester. Its fitness is how consistently it made
  money (a t-statistic of its R per trade), averaged over the symbols,
  minus a small penalty for size so simple networks win ties.
* NEAT keeps the fittest, groups similar networks into species so new
  ideas get time to improve, and breeds/mutates them: weights change and
  new neurons and connections appear.

Keeping it honest
-----------------
Evolution is extremely good at fitting the past, including pure noise.
So the history is split in time:

    | training (60%) | validation (20%) | test (20%) |

* Networks are scored ONLY on the training years.
* Each generation the best few are also tried on the validation years;
  the one that does best there becomes the "champion". Evolution stops
  when the champion has not improved for ``patience`` generations.
* The test years are used ONCE, at the very end, and the champion must
  pass the same significance test as every other strategy (t-stat >= 2
  over >= 30 trades) before the live trader will use it.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import neat
import numpy as np
import pandas as pd

from backtester import Backtester, BacktestResult, BacktestSettings, pip_size_for, price_arrays
from ml_strategy import FEATURE_COLUMNS
from neat_strategy import (
    network_signals, prepare_features, save_network, signals_from_inputs, standardise,
)
from optimizer import Candidate, Metrics, OptimizerSettings, Selection, StrategyOptimizer

CONFIG_TEMPLATE = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1e9
no_fitness_termination = True
pop_size              = {population}
reset_on_extinction   = True
seed                  = {seed}

[DefaultGenome]
num_inputs              = {inputs}
num_hidden              = 0
num_outputs             = 1
initial_connection      = partial_direct 0.2
feed_forward            = True

activation_default      = tanh
activation_mutate_rate  = 0.05
activation_options      = tanh relu sigmoid clamped

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 0.5
bias_max_value          = 5.0
bias_min_value          = -5.0
bias_mutate_power       = 0.3
bias_mutate_rate        = 0.5
bias_replace_rate       = 0.05

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 5.0
response_min_value      = -5.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 5.0
weight_min_value        = -5.0
weight_mutate_power     = 0.4
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.05

enabled_default         = True
enabled_mutate_rate     = 0.02

conn_add_prob           = 0.3
conn_delete_prob        = 0.2
node_add_prob           = 0.1
node_delete_prob        = 0.05

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 2.5

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""


@dataclass(frozen=True)
class EvolutionSettings:
    population: int = 100
    generations: int = 150
    patience: int = 30  # stop after this many generations without a better champion
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    threshold: float = 0.5  # network output needed to trade
    stop_atr_multiple: float = 1.5
    target_atr_multiple: float = 3.0
    min_trades_per_year: float = 15.0
    complexity_penalty: float = 0.01  # fitness lost per connection
    validate_top: int = 5  # best networks per generation tried on validation years
    workers: int = 1
    seed: int = 1


@dataclass
class SymbolData:
    """Everything needed to backtest a network on one symbol."""

    symbol: str
    prices: pd.DataFrame  # Open/High/Low/Close/ATR
    features: pd.DataFrame  # FEATURE_COLUMNS
    train_end: int
    validation_end: int
    spread_pips: float
    arrays: tuple = ()  # Open/High/Low/Close/ATR as numpy arrays
    inputs: np.ndarray | None = None  # standardised features (set by Evolution.run)
    missing: np.ndarray | None = None

    @property
    def years(self) -> float:
        span = self.prices.index[-1] - self.prices.index[0]
        return max(span.days / 365.25, 1 / 365.25)


@dataclass
class GenerationReport:
    generation: int
    best_train_fitness: float
    champion_validation: float
    species: int
    champion_connections: int
    seconds: float


@dataclass
class EvolutionResult:
    network: dict[str, Any]
    network_file: Path
    selections: dict[str, Selection]
    history: list[GenerationReport] = field(default_factory=list)


# --- scoring ----------------------------------------------------------------

def consistency_score(r_values: np.ndarray, min_trades: float) -> float:
    """t-statistic of R per trade (trade count capped at 300). Too few trades
    scores between -1 and -0.5 so evolution is nudged towards trading."""

    if len(r_values) < max(min_trades, 2):
        return -1.0 + 0.5 * len(r_values) / max(min_trades, 1)

    std = r_values.std(ddof=1)
    if std == 0:
        return 0.0

    return float(r_values.mean() / std * math.sqrt(min(len(r_values), 300)))


def _tester(data: SymbolData, settings: EvolutionSettings) -> Backtester:
    return Backtester(None, BacktestSettings(
        stop_atr_multiple=settings.stop_atr_multiple,
        target_atr_multiple=settings.target_atr_multiple,
        spread_pips=data.spread_pips,
        pip_size=pip_size_for(data.symbol),
    ))


def _backtest(data: SymbolData, signals: np.ndarray, start: int, end: int,
              settings: EvolutionSettings) -> BacktestResult:
    """Full backtest with every trade (used for the final report)."""
    return _tester(data, settings).run_signals(data.prices.iloc[start:end], signals[start:end])


def score_network(network: dict[str, Any], datasets: list[SymbolData],
                  settings: EvolutionSettings, part: str) -> float:
    """Average consistency score over all symbols for 'train' or 'validation'."""
    scores = []

    for data in datasets:
        signals = signals_from_inputs(network, data.inputs, data.missing)
        if part == "train":
            start, end = 0, data.train_end
        else:
            start, end = data.train_end, data.validation_end

        fraction = (end - start) / len(data.prices)
        min_trades = settings.min_trades_per_year * data.years * fraction
        arrays = tuple(array[start:end] for array in data.arrays)
        r_values = _tester(data, settings).r_multiples(arrays, signals[start:end])
        scores.append(consistency_score(r_values, min_trades))

    return float(np.mean(scores))


# --- worker processes ---------------------------------------------------------

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(datasets: list[SymbolData], settings: EvolutionSettings) -> None:
    _WORKER_STATE["datasets"] = datasets
    _WORKER_STATE["settings"] = settings


def _score_in_worker(network: dict[str, Any]) -> float:
    return score_network(network, _WORKER_STATE["datasets"], _WORKER_STATE["settings"], "train")


# --- evolution ----------------------------------------------------------------

class Evolution:

    def __init__(self, settings: EvolutionSettings | None = None) -> None:
        self.settings = settings or EvolutionSettings()

        if self.settings.train_ratio + self.settings.validation_ratio >= 0.95:
            raise ValueError("Leave at least 5% of the data for the final test.")

    def prepare(self, symbol: str, candles: pd.DataFrame, spread_pips: float) -> SymbolData:
        if len(candles) < 2000:
            raise ValueError(f"{symbol}: only {len(candles)} candles; evolution needs years of data.")

        full = prepare_features(candles)

        count = len(full)
        return SymbolData(
            symbol=symbol,
            prices=full[["Open", "High", "Low", "Close", "ATR"]],
            features=full[FEATURE_COLUMNS],
            train_end=int(count * self.settings.train_ratio),
            validation_end=int(count * (self.settings.train_ratio + self.settings.validation_ratio)),
            spread_pips=spread_pips,
            arrays=price_arrays(full),
        )

    def run(
        self,
        datasets: list[SymbolData],
        output_dir: str | Path = "reports/neat",
        timeframe: str = "H1",
        progress: bool = True,
    ) -> EvolutionResult:
        settings = self.settings
        random.seed(settings.seed)
        np.random.seed(settings.seed)

        # Feature scaling uses the TRAINING years only (pooled over symbols).
        train_features = np.vstack([
            d.features.iloc[: d.train_end].to_numpy(dtype=float) for d in datasets])
        feature_mean = np.nanmean(train_features, axis=0)
        feature_std = np.nanstd(train_features, axis=0)

        for data in datasets:
            raw = data.features.to_numpy(dtype=float)
            data.missing = ~np.isfinite(raw).all(axis=1)
            data.inputs = standardise(raw, feature_mean, feature_std)

        config = self._config()
        population = neat.Population(config)

        champion: dict[str, Any] | None = None
        champion_score = -math.inf
        champion_generation = 0
        history: list[GenerationReport] = []
        pool = None

        if settings.workers > 1:
            pool = multiprocessing.get_context("spawn").Pool(
                settings.workers, initializer=_init_worker, initargs=(datasets, settings))

        def to_network(genome) -> dict[str, Any]:
            return genome_to_network(genome, config, feature_mean, feature_std, settings)

        def evaluate(genomes, _config) -> None:
            nonlocal champion, champion_score, champion_generation
            started = time.time()

            networks = [to_network(genome) for _, genome in genomes]
            if pool:
                train_scores = pool.map(_score_in_worker, networks)
            else:
                train_scores = [score_network(n, datasets, settings, "train") for n in networks]

            for (_, genome), network, score in zip(genomes, networks, train_scores):
                genome.fitness = score - settings.complexity_penalty * len(_links(network))

            ranked = sorted(zip(genomes, networks), key=lambda item: item[0][1].fitness, reverse=True)
            for (_, genome), network in ranked[: settings.validate_top]:
                if genome.fitness <= 0:
                    continue
                validation = score_network(network, datasets, settings, "validation")
                if validation > champion_score:
                    champion, champion_score = network, validation
                    champion_generation = len(history)

            history.append(GenerationReport(
                generation=len(history),
                best_train_fitness=ranked[0][0][1].fitness,
                champion_validation=champion_score,
                species=len(population.species.species) if population.species else 0,
                champion_connections=len(_links(champion)) if champion else 0,
                seconds=time.time() - started,
            ))

            if progress:
                report = history[-1]
                validation_text = (
                    f"{report.champion_validation:6.2f}" if champion else "  none")
                print(f"  gen {report.generation:3d} | best training fitness "
                      f"{report.best_train_fitness:6.2f} | champion on validation years "
                      f"{validation_text} | species {report.species:2d} | "
                      f"{report.seconds:4.1f}s")

        try:
            for _ in range(settings.generations):
                population.run(evaluate, 1)
                if champion is not None and len(history) - 1 - champion_generation >= settings.patience:
                    if progress:
                        print(f"  No improvement on validation years for {settings.patience} "
                              f"generations - stopping.")
                    break
        finally:
            if pool:
                pool.close()
                pool.join()

        if champion is None:
            # Nothing ever made money in training; keep the last best for the record.
            best = max(population.population.values(), key=lambda g: g.fitness or -math.inf)
            champion = to_network(best)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        network_file = Path(output_dir) / f"champion-{stamp}.json"
        champion["created_at"] = stamp
        champion["symbols"] = [d.symbol for d in datasets]
        save_network(champion, network_file)

        selections = self.final_exam(champion, network_file, datasets, timeframe)
        return EvolutionResult(champion, network_file, selections, history)

    def final_exam(self, network: dict[str, Any], network_file: Path,
                   datasets: list[SymbolData], timeframe: str) -> dict[str, Selection]:
        """Tests the champion ONCE on the untouched test years of each symbol."""
        judge = StrategyOptimizer(settings=OptimizerSettings())
        candidate = Candidate(
            "NeatStrategy", {"network_file": str(network_file)},
            self.settings.stop_atr_multiple, self.settings.target_atr_multiple)

        selections = {}
        for data in datasets:
            signals = network_signals(network, data.features)
            train = _backtest(data, signals, 0, data.train_end, self.settings)
            test = _backtest(data, signals, data.validation_end, len(data.prices), self.settings)

            test_metrics = Metrics.from_result(test)
            approved, reason = judge.judge(test_metrics)

            selections[data.symbol] = Selection(
                symbol=data.symbol,
                timeframe=timeframe,
                approved=approved,
                candidate=candidate,
                in_sample=Metrics.from_result(train),
                out_of_sample=test_metrics,
                candles=len(data.prices),
                data_start=str(data.prices.index[0]),
                data_end=str(data.prices.index[-1]),
                note="" if approved else
                f"Failed the final exam on the test years: {reason}. Will NOT be traded.",
            )
        return selections

    def _config(self) -> neat.Config:
        text = CONFIG_TEMPLATE.format(
            population=self.settings.population,
            seed=self.settings.seed,
            inputs=len(FEATURE_COLUMNS),
        )
        with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as handle:
            handle.write(text)
            path = handle.name
        try:
            return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
        finally:
            os.unlink(path)


def _links(network: dict[str, Any] | None) -> list:
    if not network:
        return []
    return [link for node in network["nodes"] for link in node["links"]]


def genome_to_network(genome, config, feature_mean, feature_std,
                      settings: EvolutionSettings) -> dict[str, Any]:
    """Converts a NEAT genome into the plain JSON network NeatStrategy runs."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    nodes = []

    for node_key, _activation, _aggregation, bias, response, links in net.node_evals:
        nodes.append({
            "key": int(node_key),
            "activation": genome.nodes[node_key].activation,
            "bias": float(bias),
            "response": float(response),
            "links": [[int(source), float(weight)] for source, weight in links],
        })

    return {
        "input_keys": [int(k) for k in config.genome_config.input_keys],
        "output_key": int(config.genome_config.output_keys[0]),
        "nodes": nodes,
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": [float(x) for x in feature_mean],
        "feature_std": [float(x) for x in feature_std],
        "threshold": settings.threshold,
    }
