from optimizer import Selection, StrategyOptimizer, leaderboard, load_selections, save_selections
from conftest import make_candles


def test_random_walk_is_rejected():
    """Pure noise has no edge; the optimizer must not approve anything."""
    data = make_candles(n=20000, seed=104)
    selection, evaluations = StrategyOptimizer().optimize(data, "RW", "H1")
    assert not selection.approved
    assert "will NOT trade" in selection.note
    assert len(evaluations) == len(StrategyOptimizer().candidates())


def test_trending_market_finds_approved_strategy(tmp_path):
    data = make_candles(n=20000, seed=1, drift_period=600)
    selection, evaluations = StrategyOptimizer().optimize(data, "EURUSD", "H1")
    assert selection.approved
    assert selection.out_of_sample.expectancy_r > 0

    table = leaderboard(evaluations, "EURUSD")
    assert table["approved"].sum() >= 1

    path = tmp_path / "best.json"
    save_selections({"EURUSD": selection}, path)
    loaded = load_selections(path)["EURUSD"]
    assert isinstance(loaded, Selection)
    assert loaded.candidate == selection.candidate
    assert loaded.candidate.build().label == selection.candidate.build().label
