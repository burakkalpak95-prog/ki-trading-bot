"""Phase 7: Risikomodul (Vorrang vor Signalen) und Paper-Trading-Steuerung."""

import json

import pytest

from kitb import killswitch, models, paper, paths
from kitb.backtest import run_backtest
from kitb.data.synthetic import write_synthetic_dataset
from kitb.ml import build_overlay
from kitb.registry import Registry
from kitb.risk import AccountState, RiskLimits, RiskManager
from kitb.safety import SafetyError

LIMITS = RiskLimits.load()
RM = RiskManager(LIMITS)


def state(**kw):
    base = dict(starting_balance=10000, realized_equity_curve=[], today_realized_pnl=0.0, open_positions=0,
                open_exposure=0.0, open_pairs=set())
    return AccountState(**{**base, **kw})


def test_limits_forbid_leverage_futures_margin_withdrawals():
    for flag in ("leverage_allowed", "futures_allowed", "margin_allowed", "withdrawals_allowed"):
        with pytest.raises(ValueError):
            RiskLimits(**{**LIMITS.__dict__, flag: True})
    with pytest.raises(ValueError):
        RiskLimits(**{**LIMITS.__dict__, "max_total_exposure_fraction": 1.5})


def test_stake_sizing_respects_all_caps():
    s = state()
    # 2 % Risiko bei 10 % Stoploss -> max. 2000; Positionslimit 40 % -> 4000
    assert RM.size_stake(4000, s, -0.10) == pytest.approx(2000)
    assert RM.size_stake(4000, s, -0.02) == pytest.approx(4000)
    assert RM.size_stake(1000, s, -0.02) == pytest.approx(1000)
    full = state(open_exposure=7900)
    assert RM.size_stake(4000, full, -0.02) == pytest.approx(100)  # Gesamtposition max. 80 %


@pytest.mark.parametrize("kw,reason", [
    ({"kill_switch": True}, "kill_switch"),
    ({"state": state(open_pairs={"BTC/USDT"}, open_positions=1)}, "duplicate_position"),
    ({"state": state(open_positions=2)}, "max_open_positions"),
    ({"state": state(today_realized_pnl=-300)}, "max_daily_loss"),
    ({"state": state(realized_equity_curve=[11000, 9000])}, "max_drawdown"),
    ({"data_age_candles": 5}, "stale_data"),
    ({"quote_volume_24h": 50_000}, "insufficient_liquidity"),
    ({"stake": 0.0}, "stake_below_minimum"),
])
def test_entry_blocked(kw, reason):
    args = {"pair": "BTC/USDT", "stake": 1000.0, "state": state(), "kill_switch": False, **kw}
    d = RM.check_entry(**args)
    assert not d.allowed and reason in d.reasons and d.stake == 0


def test_entry_allowed_when_all_ok():
    d = RM.check_entry(pair="BTC/USDT", stake=1000, state=state(), kill_switch=False, data_age_candles=0.5,
                       quote_volume_24h=10_000_000)
    assert d.allowed and d.reasons == []


def test_drawdown_uses_peak():
    s = state(realized_equity_curve=[12000, 10200])
    assert s.drawdown == pytest.approx(0.15)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_risk")
    write_synthetic_dataset(d, ["BTC/USDT", "ETH/USDT"], ["1h"], "2024-01-01", "2024-06-01", seed=4)
    return d


def test_backtest_stakes_within_risk_limits(synth, tmp_path):
    res = run_backtest(datadir=synth, data_source="synthetic", offline=True, results_dir=tmp_path)
    from freqtrade.data.btanalysis import load_backtest_stats

    trades = load_backtest_stats(res["result_file"])["strategy"]["BaselineEmaCross"]["trades"]
    assert trades
    for t in trades:
        # Stake <= 2 % Kapital / 10 % Stoploss bezogen auf das Kapital zum Zeitpunkt (max. Startkapital + Gewinne)
        assert t["stake_amount"] <= LIMITS.max_risk_per_trade_fraction * 20000 / 0.10
        assert t["stake_amount"] <= 0.21 * 20000


def test_kill_switch_blocks_all_entries_in_backtest(synth, tmp_path):
    existed = killswitch.is_active()
    killswitch.activate("Test", "pytest")
    try:
        res = run_backtest(datadir=synth, data_source="synthetic", offline=True, results_dir=tmp_path)
        assert res["metrics"]["total_trades"] == 0
    finally:
        if not existed:
            killswitch.path().unlink(missing_ok=True)


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "r.sqlite")


def _activate(reg, tmp_path, bot="bot_a"):
    overlay = json.loads(build_overlay(bot, "x", tmp_path).read_text())
    good = {"total_trades": 99, "profit_factor": 2, "max_drawdown_account": 0.01, "profit_total": 0.5,
            "excess_vs_buy_and_hold": 0.1, "p_value": 0.001}
    vid = models.register_candidate(reg, bot=bot, identifier="x", freqaimodel="M", overlay=overlay,
                                    data_fingerprints={}, eval_timerange="t", metrics=good, resources={"wall_seconds": 1})
    models.evaluate_criteria(reg, vid, reference_metrics={"profit_total": 0})
    models.approve(reg, vid, by="Burak")
    models.activate(reg, vid, by="Burak")
    return vid


def test_paper_requires_active_approved_model(reg, tmp_path):
    with pytest.raises(SafetyError):
        paper.build_paper_config("bot_a", reg, tmp_path / "ud")


def test_paper_config_is_dry_run_without_keys(reg, tmp_path):
    vid = _activate(reg, tmp_path)
    path, cfg = paper.build_paper_config("bot_a", reg, tmp_path / "ud")
    assert cfg["dry_run"] is True and cfg["trading_mode"] == "spot"
    assert not cfg["exchange"]["key"] and not cfg["exchange"]["secret"]
    assert "dryrun" in cfg["db_url"] and str(tmp_path / "ud" / "paper" / "bot_a") in cfg["db_url"]
    assert cfg["freqai"]["identifier"] == "paper-bot_a-v1" and cfg["kitb_model_version_id"] == vid
    assert "api_server" not in cfg and not cfg["telegram"]["enabled"]
    assert json.loads(path.read_text())["dry_run"] is True
    cmd = paper.command("bot_a", path, tmp_path / "ud")
    assert cmd[3] == "trade" and "--dry-run-wallet" not in cmd


def test_both_bots_get_identical_conditions(reg, tmp_path):
    _activate(reg, tmp_path, "bot_a")
    _activate(reg, tmp_path, "bot_b")
    _, a = paper.build_paper_config("bot_a", reg, tmp_path / "ud")
    _, b = paper.build_paper_config("bot_b", reg, tmp_path / "ud")
    for key in ("dry_run_wallet", "stake_amount", "max_open_trades", "fee", "trading_mode", "exchange"):
        assert a[key] == b[key], key
    assert a["freqai"]["train_period_days"] == b["freqai"]["train_period_days"]


def test_paper_start_refused_with_kill_switch(tmp_path):
    ud = tmp_path / "ud"
    killswitch.activate("Test", "Burak", ud)
    with pytest.raises(SafetyError):
        paper.start("bot_a", foreground=True, user_data_dir=ud)


def test_cli_run_with_kill_switch_exits_cleanly(monkeypatch, tmp_path):
    from kitb import cli

    monkeypatch.setattr(paths, "USER_DATA", tmp_path)
    killswitch.activate("Test", "Burak")
    assert cli.main(["paper", "run", "--bot", "bot_a"]) == 0


def test_paper_report_from_dry_run_db(tmp_path, monkeypatch):
    import sqlite3

    ud = tmp_path / "ud"
    db = ud / "paper" / "bot_a" / "tradesv3.dryrun.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trades (id INTEGER, pair TEXT, is_open INTEGER, open_date TEXT, close_date TEXT, "
                "open_rate REAL, close_rate REAL, amount REAL, stake_amount REAL, close_profit REAL, "
                "close_profit_abs REAL, realized_profit REAL, exit_reason TEXT, fee_open REAL, fee_close REAL)")
    con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, "BTC/USDT", 0, "2026-10-01", "2026-10-02", 100, 110, 10, 1000, 0.1, 98, 98, "exit_signal", 0.001, 0.001),
        (2, "ETH/USDT", 0, "2026-10-03", "2026-10-04", 100, 90, 10, 1000, -0.1, -102, -102, "stop_loss", 0.001, 0.001),
        (3, "BTC/USDT", 1, "2026-10-05", None, 100, None, 10, 1000, None, None, None, None, 0.001, None),
    ])
    con.commit()
    con.close()
    text = paper.report(ud)
    assert "| bot_a | 2 | 1 | -0.04% |" in text and "| bot_b | 0 |" in text
    assert list((ud / "reports").glob("paper-*.md"))


def test_rule_strategy_paper_requires_human_approval(reg, tmp_path):
    from kitb import strategy_approval
    from kitb.models import GovernanceError

    with pytest.raises(SafetyError):
        paper.build_paper_config("trend_d1", reg, tmp_path / "ud")
    with pytest.raises(GovernanceError):
        strategy_approval.approve(reg, "trend_d1", by="bot", reason="automatisch")
    with pytest.raises(GovernanceError):
        strategy_approval.approve(reg, "trend_d1", by="Burak", reason="")
    strategy_approval.approve(reg, "trend_d1", by="Burak", reason="Vorperiode und Test bestanden")
    path, cfg = paper.build_paper_config("trend_d1", reg, tmp_path / "ud")
    assert cfg["dry_run"] is True and "freqai" not in cfg and "timeframe" not in cfg
    assert not cfg["exchange"].get("key") and not cfg["exchange"].get("secret")
    cmd = paper.command("trend_d1", path, tmp_path / "ud")
    assert "DailyTrendSma" in cmd and "--freqaimodel" not in cmd
    strategy_approval.revoke(reg, "trend_d1", by="Burak", reason="Test")
    with pytest.raises(SafetyError):
        paper.build_paper_config("trend_d1", reg, tmp_path / "ud")


def test_paper_ids_include_rule_strategies():
    assert paper.paper_ids() == ["bot_a", "bot_b", "trend_d1"]


def test_paper_status_detects_bot_in_other_container_via_log(tmp_path):
    import os
    import time

    ud = tmp_path / "ud"
    log = paper.paper_dir("trend_d1", ud) / "freqtrade.log"
    log.parent.mkdir(parents=True)
    log.write_text("Bot heartbeat. state='RUNNING'\n")
    assert paper.status("trend_d1", ud)["running"] is True
    old = time.time() - paper.HEARTBEAT_MAX_AGE_S - 60
    os.utime(log, (old, old))
    assert paper.status("trend_d1", ud)["running"] is False
