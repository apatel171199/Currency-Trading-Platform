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


def test_luck_is_not_enough_to_be_approved():
    from optimizer import Metrics, StrategyOptimizer

    lucky = Metrics(trades=40, return_pct=8.0, win_rate_pct=55.0, profit_factor=1.3,
                    max_drawdown_pct=5.0, expectancy_r=0.15, sqn=0.9, t_stat=0.9)
    approved, reason = StrategyOptimizer()._judge(lucky)
    assert not approved and "luck" in reason

    convincing = Metrics(trades=200, return_pct=40.0, win_rate_pct=50.0, profit_factor=1.5,
                         max_drawdown_pct=8.0, expectancy_r=0.25, sqn=2.0, t_stat=3.0)
    assert StrategyOptimizer()._judge(convincing) == (True, "")
