import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    close = 100 + np.arange(48, dtype=float)
    return pd.DataFrame({"date": dates, "open": close - 0.5, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.full(48, 10.0)})


@pytest.fixture(autouse=True)
def _isolated_user_data(tmp_path_factory, monkeypatch):
    """Tests dürfen NIE das echte user_data berühren (Not-Stopp, Freigaben, Paper-DBs der laufenden Bots!).

    Jeder Test bekommt ein eigenes user_data; Code-Ordner (Strategien, FreqAI-Modelle) werden nur verlinkt.
    """
    from kitb import paths

    real = paths.USER_DATA
    ud = tmp_path_factory.mktemp("user_data")
    for sub in ("strategies", "freqaimodels"):
        if (real / sub).exists():
            (ud / sub).symlink_to((real / sub).resolve(), target_is_directory=True)
    monkeypatch.setattr(paths, "USER_DATA", ud)
    monkeypatch.setattr(paths, "REGISTRY_DB", ud / "kitb.sqlite")
    monkeypatch.setattr(paths, "RESULTS_DIR", ud / "backtest_results")
    yield ud


@pytest.fixture(autouse=True, scope="session")
def _guard_real_kill_switch():
    """Sicherheitsnetz: schlägt fehl, falls ein Test trotzdem den echten Not-Stopp angelegt hat."""
    from kitb import paths

    real = paths.USER_DATA / "KILL_SWITCH"
    existed = real.exists()
    yield
    assert existed or not real.exists(), "Ein Test hat den ECHTEN Not-Stopp gesetzt – Tests müssen user_data isolieren."
