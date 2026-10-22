import copy

import pytest

from kitb.config import load_freqtrade_config, load_project_config
from kitb.safety import (SafetyError, assert_freqtrade_command_allowed, assert_run_mode_allowed,
                         assert_safe_freqtrade_config, check_freqtrade_config)

PROJECT = load_project_config()
EX, PAIRS = PROJECT["allowed_exchanges"], PROJECT["allowed_pairs"]


def base():
    return copy.deepcopy(load_freqtrade_config())


def test_shipped_config_is_safe():
    assert check_freqtrade_config(base(), EX, PAIRS) == []


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(dry_run=False),
    lambda c: c.pop("dry_run"),
    lambda c: c.update(dry_run="true"),
    lambda c: c.update(trading_mode="futures"),
    lambda c: c.update(margin_mode="isolated"),
    lambda c: c.update(leverage=3),
    lambda c: c.update(initial_state="running"),
    lambda c: c.update(force_entry_enable=True),
    lambda c: c["exchange"].update(key="abc"),
    lambda c: c["exchange"].update(secret="abc"),
    lambda c: c["exchange"].update(name="someexchange"),
    lambda c: c["exchange"].update(pair_whitelist=["BTC/USDT", "DOGE/USDT"]),
    lambda c: c["telegram"].update(enabled=True),
    lambda c: c.update(api_server={"enabled": True, "listen_ip_address": "127.0.0.1"}),
    lambda c: c.update(api_server={"enabled": False, "listen_ip_address": "0.0.0.0"}),
])
def test_unsafe_configs_are_rejected(mutate):
    cfg = base()
    mutate(cfg)
    with pytest.raises(SafetyError):
        assert_safe_freqtrade_config(cfg, EX, PAIRS)


@pytest.mark.parametrize("mode", ["live", "trade", "dry_run", "real", ""])
def test_trading_modes_blocked(mode):
    with pytest.raises(SafetyError):
        assert_run_mode_allowed(mode)


@pytest.mark.parametrize("mode", ["data", "backtest", "paper"])
def test_research_modes_allowed(mode):
    assert_run_mode_allowed(mode)


@pytest.mark.parametrize("cmd", ["trade", "webserver"])
def test_freqtrade_trade_commands_blocked(cmd):
    with pytest.raises(SafetyError):
        assert_freqtrade_command_allowed(cmd)
    with pytest.raises(SafetyError):
        assert_freqtrade_command_allowed(cmd, mode="live")


def test_trade_only_in_paper_mode():
    assert_freqtrade_command_allowed("trade", mode="paper")
    with pytest.raises(SafetyError):
        assert_freqtrade_command_allowed("webserver", mode="paper")


def test_cli_has_no_trading_command():
    from kitb.cli import build_parser

    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert not {"trade", "live", "dry-run"} & set(sub.choices)
