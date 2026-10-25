"""Simulierte Trades als Lerndaten (Burak, 24.10.2026: „Probiere es einmal aus, wenn du siehst, dass es hilft, bau es ein“).

Auf den echten Stundenkerzen seit 2022 wird für jede Stunde und jeden Lern-Coin durchgerechnet, wie ein Einstieg
ausgegangen wäre – für jede Haltedauer (1/2/4/8 h) und jede Stoploss-Stufe (−2/−4/−6 %), nach realistischen Kosten
(Gebühr + Slippage + halber Spread je Seite). Ohne Blick in die Zukunft: Merkmale aus der Kerze VOR dem Einstieg,
Einstieg zum Eröffnungskurs der nächsten Kerze. Stoploss vorsichtig gerechnet: berührt das Tief einer Kerze die Stufe,
gilt der Trade als ausgestoppt (auch wenn das Hoch derselben Kerze vorher ein Plus gezeigt hätte).

Die Simulationen werden IMMER verwertet (Burak 24.10.: „die gewonnenen Daten auf jeden Fall verwerten“): im
Gewinnchance-Modell und bei der Wahl von Haltedauer und Stoploss. ``evaluate`` wählt an den ECHTEN Paper-Trades nur,
wie stark sie zählen (``sim_weight.json``).
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kitb import paths

log = logging.getLogger(__name__)

HOLD_HOURS = [1, 2, 4, 8]
STOPS = [2, 4, 6]
RECENT_15M_DAYS = 60
SAMPLE = 60_000  # so viele Simulationen nimmt das Lernen höchstens auf (die jüngsten), damit das Neuanpassen schnell bleibt


def sim_dir(user_data: Path | None = None) -> Path:
    return (user_data or paths.USER_DATA) / "ai_memory"


def outcomes(df: pd.DataFrame, cost: float, per_h: int = 1) -> dict[tuple[int, int], np.ndarray]:
    """Ergebnis nach Kosten je (Haltedauer h Stunden, Stoploss %) für einen Einstieg zum Open jeder Kerze (NaN am Ende).
    ``per_h`` = Kerzen je Stunde (4 bei 15-Minuten-Kerzen)."""
    o, low, c = df["open"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    n = len(df)
    out = {}
    for hh in HOLD_HOURS:
        h = hh * per_h
        # Tiefster Kurs von Kerze i bis i+h-1 (Einstieg zum Open von i), Ausstieg zum Schluss von i+h-1
        lows = pd.Series(low).rolling(h).min().shift(-(h - 1)).to_numpy()
        exit_close = np.full(n, np.nan)
        if n >= h:
            exit_close[: n - h + 1] = c[h - 1:]
        for s in STOPS:
            stop_px = o * (1 - s / 100)
            hit = lows <= stop_px
            px = np.where(hit, stop_px, exit_close)
            out[(hh, s)] = np.where(np.isfinite(px) & np.isfinite(lows), px / o - 1 - cost, np.nan)
    return out


def table_pair(df: pd.DataFrame, pair: str, cost: float, user_data: Path | None = None,
               flow: bool = True, minutes: int = 60) -> pd.DataFrame:
    """Je Kerze: Marktlage der Vorkerze + Ergebnis für JEDE Kombination aus Haltedauer und Stoploss (Spalten r_h_s)."""
    from kitb import shared_memory

    df = df.sort_values("date").reset_index(drop=True)
    feats = shared_memory.market_snapshot(df, minutes, pair if flow else None, user_data).shift(1)  # Merkmale der Vorkerze
    res = outcomes(df, cost, max(1, 60 // minutes))
    out = feats.assign(pair=pair, date=df["date"], **{f"r_{h}_{s}": v for (h, s), v in res.items()})
    return out.iloc[200 * max(1, 60 // minutes):].dropna(subset=[f"r_{HOLD_HOURS[0]}_{STOPS[0]}"])  # Vorlauf EMA200


def _trades_from_table(tab: pd.DataFrame, hs: np.ndarray, ss: np.ndarray, label: str,
                       hour_factor: float = 1.0) -> pd.DataFrame:
    """Simulierte Trades aus der Tabelle: gewählte Haltedauer/Stufe je Zeile -> Ergebnis, Grund, Schlusszeit."""
    from kitb.shared_memory import SNAPSHOT_FEATURES

    cols = {(h, s): tab[f"r_{h}_{s}"].to_numpy() for h in HOLD_HOURS for s in STOPS}
    profit = np.full(len(tab), np.nan)
    for (h, s), v in cols.items():
        m = (hs == h) & (ss == s)
        profit[m] = v[m]
    hours = hs.astype(float) * hour_factor
    feats = tab.reindex(columns=SNAPSHOT_FEATURES)
    out = feats.assign(ki="simulation" if hour_factor == 1 else "simulation_aktie", own=False, pair=tab["pair"].values,
                       kind="sim", open_date=tab["date"].astype(str).values,
                       close_date=(tab["date"] + pd.to_timedelta(hours, unit="h")).astype(str).values,
                       hold_hours=hours, profit_pct=profit,
                       reason=[f"{label} · halten_h:{h} · stop:{s}" for h, s in zip(hs, ss)])
    return out.dropna(subset=["profit_pct"]).reset_index(drop=True)


def simulate_pair(df: pd.DataFrame, pair: str, cost: float, rng: np.random.Generator,
                  user_data: Path | None = None, flow: bool = True) -> pd.DataFrame:
    """Je Kerze ein simulierter Trade mit zufällig gewählter Haltedauer/Stufe (Runde 0: reines Ausprobieren)."""
    tab = table_pair(df, pair, cost, user_data, flow)
    return _trades_from_table(tab, rng.choice(HOLD_HOURS, len(tab)), rng.choice(STOPS, len(tab)), "sim")


STOCKS = {"SP500": "S&P 500", "NASDAQ100": "Nasdaq 100"}  # FRED, Tagesschlusskurse ~10 Jahre, ohne Schlüssel
STOCK_COST = 0.001  # Unsicher: grobe Kosten je Runde für Index-Handel (ETF), nur für die Simulation


def stock_candles(series: str, get_text=None) -> pd.DataFrame:
    """Tageskerzen eines Aktienindex aus FRED (nur Schlusskurs: Open = Vortagesschluss, Tief/Hoch daraus)."""
    import io

    from kitb import market_flow

    get_text = get_text or market_flow._get_text
    raw = pd.read_csv(io.StringIO(get_text(market_flow.FRED_URL, {"id": series})))
    raw.columns = ["date", "close"]
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw = raw.dropna()
    c = raw["close"].to_numpy(float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"date": pd.to_datetime(raw["date"], utc=True), "open": o, "high": np.maximum(o, c),
                         "low": np.minimum(o, c), "close": c, "volume": np.nan})


def stock_table(get_text=None) -> pd.DataFrame:
    parts = []
    for series in STOCKS:
        try:
            df = stock_candles(series, get_text)
        except Exception:  # noqa: BLE001 – ohne Internet nur Krypto
            continue
        if len(df) >= 300:
            parts.append(table_pair(df, series, STOCK_COST, None, flow=False))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def simulate_stocks(rng: np.random.Generator, get_text=None) -> pd.DataFrame:
    """Aktien-„Was wäre wenn“ (Burak 24.10.): je Handelstag ein Trade, Haltedauer 1/2/4/8 TAGE (Tageskerzen) und
    Stoploss −2/−4/−6 % auf Schlusskursbasis. Unsicher: andere Zeitskala als Krypto; ohne Trader-/Gesamtmarkt-Merkmale
    (fehlen für Aktien -> beim Lernen Durchschnitt)."""
    tab = stock_table(get_text)
    if tab.empty:
        return tab
    return _trades_from_table(tab, rng.choice(HOLD_HOURS, len(tab)), rng.choice(STOPS, len(tab)), "sim_aktie", 24.0)


def build(*, pairs: list[str] | None = None, datadir: Path | None = None, user_data: Path | None = None,
          until: str | None = None, seed: int = 7, stocks: bool = True, get_text=None,
          get=None) -> dict[str, Any]:
    from kitb.config import load_project_config
    from kitb.costs import CostModel
    from kitb.data.storage import load_ohlcv_raw

    project = load_project_config()
    pairs = pairs or project.get("learning_pairs") or project["allowed_pairs"]
    datadir = datadir or paths.DATA_DIR / "binance"
    cost = CostModel.from_project_config(project).round_trip_cost
    rng = np.random.default_rng(seed)
    parts = []
    for pair in pairs:
        df = load_ohlcv_raw(datadir, pair, "1h")
        if get is not False:  # die letzten ~41 Tage frisch von Binance (lokale Kerzen sind oft nicht aktuell)
            try:
                from kitb import shared_memory

                recent = shared_memory._klines_1h(pair, get or __import__("kitb.market_flow").market_flow._get)
                recent = recent[recent["date"] < pd.Timestamp.now(tz="UTC").floor("h")]  # nur abgeschlossene Kerzen
                df = pd.concat([df[df["date"] < recent["date"].min()] if not df.empty else df, recent],
                               ignore_index=True) if not recent.empty else df
            except Exception:  # noqa: BLE001
                pass
        if df.empty:
            continue
        if until:
            df = df[df["date"] < pd.Timestamp(until, tz="UTC")]
        parts.append(table_pair(df, pair, cost, user_data).assign(asset="krypto"))
        # Punkt 2 (Burak 25.10.): KI 1/2 handeln auf 15 Minuten -> zusätzlich 15-Minuten-Einstiege der letzten
        # RECENT_15M_DAYS Tage (nicht seit 2022: der Arbeitsspeicher reicht dafür nicht)
        try:
            q = load_ohlcv_raw(datadir, pair, "15m")
            if get is not False:  # die letzten ~10 Tage frisch von Binance: lokale 15m-Dateien lädt kein Dienst nach
                try:
                    r = _klines_15m(pair, get or __import__("kitb.market_flow").market_flow._get)
                    r = r[r["date"] < pd.Timestamp.now(tz="UTC").floor("15min")]
                    if not r.empty:
                        q = pd.concat([q[q["date"] < r["date"].min()] if not q.empty else q, r], ignore_index=True)
                except Exception:  # noqa: BLE001
                    pass
            q = q[q["date"] >= q["date"].max() - pd.Timedelta(days=RECENT_15M_DAYS + 3)] if not q.empty else q
            if until:
                q = q[q["date"] < pd.Timestamp(until, tz="UTC")]
            if len(q) > 1000:
                parts.append(table_pair(q, pair, cost, user_data, minutes=15).assign(asset="krypto15"))
        except Exception:  # noqa: BLE001 – ohne 15-Minuten-Daten bleibt es bei Stundenkerzen
            log.exception("15-Minuten-Simulation %s", pair)
    crypto = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    stock = stock_table(get_text).assign(asset="aktie") if stocks else pd.DataFrame()
    table = pd.concat([x for x in (crypto, stock) if not x.empty], ignore_index=True) if len(crypto) + len(stock) \
        else pd.DataFrame()
    d = sim_dir(user_data)
    d.mkdir(parents=True, exist_ok=True)
    if table.empty:
        return {"trades": 0}
    num = table.select_dtypes("float64").columns
    table[num] = table[num].astype("float32")  # halber Speicher
    _atomic_parquet(table, d / "sim_table.parquet")
    # Bereits gespielte Runden bleiben erhalten (Gedächtnis und Rundenzähler, Prüfbericht 25.10., Punkt 5a). Nur ohne
    # Gedächtnis gibt es Runde 0: zufällig ausprobieren (Haltedauer/Stufe gleichverteilt), damit sofort Lerndaten da sind
    mem_file = d / "simulated_trades.parquet"
    played = pd.DataFrame()
    if mem_file.exists():
        try:
            old = pd.read_parquet(mem_file)
            if "round" in old:
                played = old[old["round"].notna()]
        except Exception:  # noqa: BLE001 – unlesbare Datei: neu anfangen
            log.exception("Sim-Gedächtnis nicht lesbar")
    if not played.empty:
        sim = played.tail(MEMORY_CAP).reset_index(drop=True)
    else:
        sim = _round_trades(table, rng.choice(HOLD_HOURS, len(table)), rng.choice(STOPS, len(table)), "sim")
        _atomic_parquet(sim, mem_file)
    return {"trades": len(sim), "krypto": int((table["asset"] == "krypto").sum()),
            "aktien": int((table["asset"] == "aktie").sum()), "krypto_15m": int((table["asset"] == "krypto15").sum()),
            "cost_round_trip": cost,
            "win_rate": float((sim["profit_pct"] > 0).mean())}


def _klines_15m(pair: str, get) -> pd.DataFrame:
    from kitb import market_flow

    rows = get(f"{market_flow.SPOT}/api/v3/klines", {"symbol": market_flow.symbol(pair), "interval": "15m", "limit": 1000})
    return pd.DataFrame({"date": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
                         **{c: [float(r[i]) for r in rows] for c, i in
                            (("open", 1), ("high", 2), ("low", 3), ("close", 4), ("volume", 5))}})


def _round_trades(table: pd.DataFrame, hs: np.ndarray, ss: np.ndarray, label: str) -> pd.DataFrame:
    stock = (table["asset"] == "aktie").to_numpy() if "asset" in table else np.zeros(len(table), bool)
    parts = []
    if (~stock).any():
        parts.append(_trades_from_table(table[~stock], hs[~stock], ss[~stock], label))
    if stock.any():
        parts.append(_trades_from_table(table[stock], hs[stock], ss[stock], label.replace("sim", "sim_aktie", 1),
                                        24.0))
    return pd.concat(parts, ignore_index=True)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    """Erst in eine Hilfsdatei schreiben, dann umbenennen: die Bots lesen nie eine halb geschriebene Datei."""
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


_LOADED: dict[tuple, pd.DataFrame] = {}


def load(user_data: Path | None = None, n: int = SAMPLE) -> pd.DataFrame:
    """Jüngste ``n`` simulierte Trades (neueste Runden zuerst behalten), zwischengespeichert bis zur nächsten Runde."""
    p = sim_dir(user_data) / "simulated_trades.parquet"
    if not p.exists():
        return pd.DataFrame()
    key = (str(p), p.stat().st_mtime, n)
    if key not in _LOADED:
        _LOADED.clear()
        df = pd.read_parquet(p)
        if "round" in df:
            df = df.sort_values("round", kind="stable")
        stock = df["ki"] == "simulation_aktie" if "ki" in df else pd.Series(False, index=df.index)
        # Krypto: die jüngsten Runden; Aktien: höchstens ein Fünftel der Menge (andere Zeitskala -> zählen weniger)
        crypto = df[~stock].tail(n - n // 5)
        stocks = df[stock].tail(n // 5)
        _LOADED[key] = pd.concat([stocks, crypto]).reset_index(drop=True)
    return _LOADED[key]


# --- Endlos-Simulation: die KI spielt mit ihrem aktuellen Wissen, lernt aus dem Ergebnis und spielt wieder ----------

ROUND_SIZE = 20_000  # Einstiegsgelegenheiten je Runde (zufällig aus der ganzen Kursgeschichte)
EXPLORE = 0.15  # Anteil Lern-Trades je Runde (auch gegen die eigene Einschätzung einsteigen, wie im Paper-Betrieb)
MEMORY_CAP = 3 * SAMPLE  # so viele gespielte Trades bleiben in der Datei (älteste Runden fallen heraus)


def _thompson(stats: pd.DataFrame, key: str, size: int, rng: np.random.Generator) -> np.ndarray:
    sd = stats["std"].clip(lower=0.005).to_numpy() / np.sqrt(stats["n"].to_numpy() + 1)
    mean = (stats["mean"] * stats["n"] / (stats["n"] + 1)).to_numpy()
    draw = rng.normal(mean, sd, (size, len(stats)))
    return stats[key].to_numpy()[draw.argmax(axis=1)]


def play_round(table: pd.DataFrame, memory: pd.DataFrame, real: pd.DataFrame, r: int,
               rng: np.random.Generator, size: int = ROUND_SIZE) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Eine Runde: Modell aus echten Trades + bisherigen Simulationen -> einsteigen, wo es sich lohnt (plus Lern-Trades),
    Haltedauer/Stoploss nach bisherigen Ergebnissen (Thompson) -> Ergebnis aus der Tabelle -> zurück ins Gedächtnis."""
    from kitb import self_learning as sl

    mem = memory.tail(SAMPLE)
    if not mem.empty and not real.empty:
        mem = mem.assign(relevance=sl.relevance(mem, real.dropna(subset=["profit_pct"])))
    st = sl.train(pd.concat([mem, real], ignore_index=True)) if not real.empty or not mem.empty else {}
    rows = table.iloc[rng.choice(len(table), min(size, len(table)), replace=False)].reset_index(drop=True)
    if st.get("model") is not None:
        p = st["model"].predict_proba(sl._design(rows, st["columns"], st["fill"])[0])[:, 1]
        take = (p >= st["threshold"]) | (rng.random(len(rows)) < EXPLORE)
    else:
        p, take = np.full(len(rows), np.nan), np.ones(len(rows), bool)
    learn = pd.concat([mem, real], ignore_index=True)
    crypto_hist = learn[learn.get("ki", pd.Series("", index=learn.index)) != "simulation_aktie"]
    hold = sl.hold_stats(crypto_hist, HOLD_HOURS, minutes=60) if not crypto_hist.empty else \
        pd.DataFrame({"halten": HOLD_HOURS, "n": 0.0, "mean": 0.0, "std": 0.02})
    stops = []
    for o in STOPS:
        part = crypto_hist[crypto_hist["reason"].map(sl.stop_from_tag) == o] if not crypto_hist.empty else crypto_hist
        r_ = part["profit_pct"].to_numpy() if len(part) else np.array([])
        w = sl._weights(part) if len(part) else np.array([])
        n = float(w.sum())
        mean = float((r_ * w).sum() / n) if n else 0.0
        stops.append({"stop": o, "n": n, "mean": mean,
                      "std": float(np.sqrt((w * (r_ - mean) ** 2).sum() / n)) if len(r_) > 1 else 0.02})
    rows = rows[take].reset_index(drop=True)
    hs = _thompson(hold, "halten", len(rows), rng)
    ss = _thompson(pd.DataFrame(stops), "stop", len(rows), rng)
    trades = _round_trades(rows, hs, ss, f"sim · runde {r}").assign(round=r)
    all_combos = rows[[f"r_{h}_{s}" for h in HOLD_HOURS for s in STOPS]].to_numpy(float)
    stats = {"runde": r, "zeit": pd.Timestamp.now(tz="UTC").isoformat(), "gelegenheiten": int(len(take)),
             "genommen": int(take.sum()), "Ø genommen": float(trades["profit_pct"].mean()) if len(trades) else None,
             "Ø zufällig": float(np.nanmean(all_combos)) if all_combos.size else None,
             "Treffer genommen": float((trades["profit_pct"] > 0).mean()) if len(trades) else None,
             "Filter aktiv": bool(st.get("active")), "Haltedauer bevorzugt (h)": int(hold.sort_values("mean")["halten"].iloc[-1]),
             "Stoploss bevorzugt (%)": int(pd.DataFrame(stops).sort_values("mean")["stop"].iloc[-1]),
             # wie viele der genutzten Simulationen die KI effektiv als brauchbar einstuft (Rest zählt kaum)
             "relevant (effektiv)": sl.effective(mem["relevance"].to_numpy()) if "relevance" in mem else len(mem),
             "genutzt": int(len(mem))}
    return trades, stats


def _write_status(d: Path, memory: pd.DataFrame, stats: dict[str, Any], table_rows: int) -> None:
    """Kurzer Stand für die Anzeige (Desktop-App, Reiter „Simulation“): was ins Gedächtnis fließt."""
    rounds_file = d / "sim_rounds.jsonl"
    total = 0
    try:
        hist = [json.loads(x) for x in rounds_file.read_text(encoding="utf-8").splitlines()[-2000:]]
        total = sum(int(h.get("genommen") or 0) for h in hist)
        last_hour = [h for h in hist if pd.Timestamp(h["zeit"]) > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)]
    except Exception:  # noqa: BLE001
        hist, last_hour = [], []
    status = {
        "stand": pd.Timestamp.now(tz="UTC").isoformat(),
        "runden": stats["runde"],
        "runden_letzte_stunde": len(last_hour),
        "sim_trades_gespielt_gesamt": total,
        "sim_trades_letzte_stunde": sum(int(h.get("genommen") or 0) for h in last_hour),
        "im_gedaechtnis": int(len(memory)),
        "davon_aktien": int((memory["ki"] == "simulation_aktie").sum()) if "ki" in memory else 0,
        "von_den_bots_genutzt": SAMPLE,
        "gewicht_je_sim_trade": weight(d.parent if d.name == "ai_memory" else None),
        "kurs_gelegenheiten": table_rows,
        "letzte_runde": stats,
        "pause": _pause_state(d),  # Feedback der KI: gestoppt, weil nichts Brauchbares mehr kommt (mit Grund)
    }
    (d / "sim_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def loop(user_data: Path | None = None, rounds: int | None = None, pause: float = 2.0) -> None:
    """Endlos (Burak 24.10.: „wenn ein Durchlauf fertig ist, soll der nächste starten, die KI soll das Gewonnene direkt
    verwerten“). Jede Runde schreibt das Gedächtnis neu; die laufenden KIs lesen es beim nächsten Neuanpassen."""
    import time

    from kitb import shared_memory

    d = sim_dir(user_data)
    rng = np.random.default_rng()
    done = 0
    while rounds is None or done < rounds:
        paused = _pause_state(d)
        if paused:  # die KI hat gemeldet: aus der Simulation ist nichts Brauchbares mehr zu holen
            if _new_kinds_while_paused(d, paused, user_data):
                (d / "sim_pause.json").unlink(missing_ok=True)
                print("Simulation: Probelauf – neue Simulationsart in der Kurstabelle, die KI prüft sie", flush=True)
                continue
            n_real = len(shared_memory.load(user_data))
            if n_real >= paused["echte_trades"] + RESUME_AFTER_REAL:
                res = evaluate(user_data)
                if res.get("hilft"):
                    (d / "sim_pause.json").unlink(missing_ok=True)
                    print("Simulation: läuft wieder – mit den neuen echten Trades helfen Simulationen", flush=True)
                    continue
                _set_pause(d, n_real, "Neu geprüft: Simulationen helfen weiterhin nicht. " + paused["grund"])
            done += 1
            time.sleep(PAUSE_CHECK_S if rounds is None else 0)
            continue
        tab_file = d / "sim_table.parquet"
        if not tab_file.exists() or time.time() - tab_file.stat().st_mtime > TABLE_REFRESH_H * 3600:
            print(f"Simulation: Kurstabelle neu {build(user_data=user_data)}", flush=True)
            evaluate(user_data)
        table = pd.read_parquet(tab_file)
        # Ergebnis bekannt: Aktien halten bis 8 Tage, Krypto bis 8 h. Früher galt 8 Tage für alle – damit fehlte gerade
        # die jüngste Krypto-Woche, also die aktuelle Marktlage (25.10.: „Schattenhandel“ auf den letzten Stunden)
        stock = (table["asset"] == "aktie") if "asset" in table else pd.Series(False, index=table.index)
        horizon = np.where(stock, pd.Timedelta(days=8), pd.Timedelta(hours=max(HOLD_HOURS) + 1))
        table = table[table["date"] + horizon < pd.Timestamp.now(tz="UTC")]
        memory = pd.read_parquet(d / "simulated_trades.parquet") if (d / "simulated_trades.parquet").exists() \
            else pd.DataFrame()
        r = int(memory["round"].max()) + 1 if "round" in memory and memory["round"].notna().any() else 1
        real = shared_memory.load(user_data)
        if not real.empty:
            real = real.assign(own=False)
        trades, stats = play_round(table, memory, real, r, rng)
        memory = pd.concat([memory, trades], ignore_index=True).tail(MEMORY_CAP)
        _atomic_parquet(memory, d / "simulated_trades.parquet")
        with open(d / "sim_rounds.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(stats, default=str) + "\n")
        _write_status(d, memory, stats, len(table))
        print(f"Simulation: {stats}", flush=True)
        if r % 25 == 0:
            evaluate(user_data)
            stop, why = saturated(d)
            if stop:
                _set_pause(d, len(real), why)
                print(f"Simulation: pausiert – {why}", flush=True)
        done += 1
        time.sleep(pause)


TABLE_REFRESH_H = 2      # Kurstabelle öfter neu: die jüngsten Stunden (aktuelle Marktlage) kommen schneller dazu
RESUME_AFTER_REAL = 50   # nach so vielen neuen echten Trades prüft die pausierte Simulation, ob sie wieder hilft
PAUSE_CHECK_S = 1800
SATURATION_CHECKS = 3    # so viele Prüfungen in Folge (je 25 Runden) ohne Nutzen -> Pause


def saturated(d: Path) -> tuple[bool, str]:
    """Gewinnt die KI noch brauchbare Daten aus der Simulation? (Burak 25.10.: „wenn sie meint, dass sie keine brauchbaren
    Daten mehr aus der Simulation gewinnen kann, soll die Simulation gestoppt werden“)

    Nein, wenn (1) die letzten Prüfungen an ECHTEN Trades alle sagen, dass Simulationen die Vorhersage nicht verbessern,
    und (2) die KI in der Simulation selbst nicht mehr besser wird (Vorsprung „Ø genommen − Ø zufällig“ der letzten 100
    Runden nicht größer als der 100 davor)."""
    try:
        evals = [json.loads(x) for x in (d / "sim_evals.jsonl").read_text(encoding="utf-8").splitlines()]
        rounds = [json.loads(x) for x in (d / "sim_rounds.jsonl").read_text(encoding="utf-8").splitlines()[-200:]]
    except Exception:  # noqa: BLE001
        return False, ""
    last = [e for e in evals if "hilft" in e][-SATURATION_CHECKS:]
    if len(last) < SATURATION_CHECKS or any(e["hilft"] for e in last):
        return False, ""
    gap = [(r.get("Ø genommen") or 0) - (r.get("Ø zufällig") or 0) for r in rounds]
    if len(gap) >= 200 and np.mean(gap[100:]) > np.mean(gap[:100]) + 0.0002:
        return False, ""
    return True, (f"Die letzten {SATURATION_CHECKS} Prüfungen an echten Trades: Simulationen verbessern die Vorhersage "
                  f"nicht (Trennschärfe ohne {last[-1].get('auc_ohne')}, mit {last[-1].get('auc_mit')}), und in der "
                  f"Simulation selbst wird die KI nicht mehr besser. Bereits Gelerntes bleibt im Gedächtnis.")


def _set_pause(d: Path, n_real: int, why: str) -> None:
    prev = _pause_state(d) or {}
    (d / "sim_pause.json").write_text(json.dumps({
        "pausiert": True, "seit": prev.get("seit") or pd.Timestamp.now(tz="UTC").isoformat(),
        "geprueft": pd.Timestamp.now(tz="UTC").isoformat(), "echte_trades": n_real, "grund": why,
        "weiter_ab_echten_trades": n_real + RESUME_AFTER_REAL,
        "arten": _kinds(d / "sim_table.parquet") or prev.get("arten")}, indent=2), encoding="utf-8")
    try:  # Anzeige sofort aktualisieren, auch wenn keine Runde mehr läuft
        st = json.loads((d / "sim_status.json").read_text(encoding="utf-8"))
        (d / "sim_status.json").write_text(json.dumps({**st, "pause": _pause_state(d)}, indent=2, default=str),
                                           encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _kinds(tab_file: Path) -> list[str]:
    try:
        return sorted(pd.read_parquet(tab_file, columns=["asset"])["asset"].dropna().unique())
    except Exception:  # noqa: BLE001
        return []


def _new_kinds_while_paused(d: Path, paused: dict[str, Any], user_data: Path | None) -> bool:
    """Auch in der Pause bleibt die Kurstabelle aktuell. Kommt eine Art hinzu, die bei der Pause noch fehlte (25.10.:
    15-Minuten-Kerzen, pausiert schon vor dem Update), darf die KI sie erst ausprobieren – sonst würde sie nie geprüft.
    Bringt sie nichts, pausiert die nächste Prüfung (nach 25 Runden) wieder."""
    import time

    tab_file = d / "sim_table.parquet"
    if not tab_file.exists() or time.time() - tab_file.stat().st_mtime <= TABLE_REFRESH_H * 3600:
        return False
    print(f"Simulation (pausiert): Kurstabelle neu {build(user_data=user_data)}", flush=True)
    return bool(set(_kinds(tab_file)) - set(paused.get("arten") or ["aktie", "krypto"]))


def _pause_state(d: Path) -> dict[str, Any] | None:
    try:
        return json.loads((d / "sim_pause.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# Kandidaten: Anteil ALLER Simulationen zusammen am Gewicht aller echten Trades (0,1 = Simulationen zählen wie ein Zehntel
# der echten Trades). Nie 0: Simulationen werden immer verwertet (Burak 24.10.).
WEIGHTS = [0.02, 0.1, 0.25, 0.5, 1.0, 2.0]
DEFAULT_WEIGHT = 0.5


def weight(user_data: Path | None = None) -> float:
    """Gewicht der Simulationen (vom Vergleich an echten Trades gewählt). Simulationen zählen IMMER mit (Burak 24.10.)."""
    try:
        return float(json.loads((sim_dir(user_data) / "sim_weight.json").read_text(encoding="utf-8"))["weight"])
    except Exception:  # noqa: BLE001
        return DEFAULT_WEIGHT


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def evaluate(user_data: Path | None = None) -> dict[str, Any]:
    """Wie stark sollen Simulationen zählen? Getestet an den jüngsten 50 % der ECHTEN Paper-Trades.

    Jedes Modell lernt aus den älteren 50 % der echten Trades plus den Simulationen (die vor dem ersten getesteten Trade
    enden) mit einem Kandidaten-Gewicht; zum Vergleich ohne Simulationen. Gewählt wird das Gewicht mit der besten
    Trennschärfe (AUC). Ohne genug echte Trades bleibt das Standardgewicht."""
    from kitb import self_learning, shared_memory

    real = shared_memory.load(user_data)
    if real.empty:
        real = pd.DataFrame(columns=["profit_pct", "close_date", "open_date"])
    real = real.dropna(subset=["profit_pct"]).sort_values("close_date").reset_index(drop=True)
    cols = [c for c in shared_memory.SNAPSHOT_FEATURES if c in real and real[c].notna().mean() > 0.5]
    real = real[real[cols].notna().any(axis=1)].reset_index(drop=True) if cols else real
    out: dict[str, Any] = {"real_trades": len(real), "weight": DEFAULT_WEIGHT}
    sim_all = load(user_data, n=10 * SAMPLE)
    if len(real) < 2 * self_learning.MIN_TRADES or sim_all.empty:
        out["why"] = "zu wenige echte Trades oder keine Simulationen – Standardgewicht"
    else:
        cut = len(real) // 2
        train, test = real.iloc[:cut], real.iloc[cut:]
        first_test = pd.Timestamp(test["open_date"].min())
        first_test = first_test.tz_localize("UTC") if first_test.tzinfo is None else first_test
        sim = sim_all[pd.to_datetime(sim_all["close_date"], utc=True, format="ISO8601") < first_test].tail(SAMPLE)
        y_tr, y_te = (train["profit_pct"] > 0).to_numpy(int), (test["profit_pct"] > 0).to_numpy(int)
        x_tr, fill = self_learning._design(train, cols)
        p0 = self_learning._fit(x_tr, y_tr).predict_proba(self_learning._design(test, cols, fill)[0])[:, 1]
        rows = [{"Gewicht": 0.0, "AUC": _auc(y_te, p0), **_taken(test, p0, y_tr)}]
        rel = self_learning.relevance(sim, train, cols)  # Relevanz nur aus dem Lernteil, der Test bleibt ungesehen
        out["relevante_sims"] = self_learning.effective(rel)
        both = pd.concat([sim, train], ignore_index=True)
        is_sim = np.r_[np.ones(len(sim), bool), np.zeros(len(train), bool)]
        x_b, fill_b = self_learning._design(both, cols)
        x_te = self_learning._design(test, cols, fill_b)[0]
        for w in WEIGHTS:
            m = self_learning._fit(x_b, (both["profit_pct"] > 0).to_numpy(int), sim=is_sim, sim_share=w, rel=rel)
            p = m.predict_proba(x_te)[:, 1]
            rows.append({"Gewicht": w, "AUC": _auc(y_te, p), **_taken(test, p, y_tr)})
        res = pd.DataFrame(rows)
        cand = res[res["Gewicht"] > 0]
        best = cand.sort_values("AUC", ascending=False).iloc[0]
        worse = rows[0]["AUC"] is not None and (best["AUC"] is None or rows[0]["AUC"] > best["AUC"])
        if worse:  # keine Stufe hilft: so wenig wie möglich (immer verwertet, Burak 24.10.)
            best = cand.sort_values("Gewicht").iloc[0]
        out.update(weight=float(best["Gewicht"]), sim_trades=len(sim), avg_all=float(test["profit_pct"].mean()),
                   results=res.to_dict("records"),
                   why=(f"Simulationen mit Anteil {best['Gewicht']} am besten (AUC {best['AUC']:.3f}; ohne "
                        f"Simulationen {rows[0]['AUC']:.3f}" + (", ohne wäre besser – kleinster getesteter Anteil"
                                                                if worse else "") + ")"))
    d = sim_dir(user_data)
    d.mkdir(parents=True, exist_ok=True)
    if "results" in out:
        out["hilft"] = not worse
        with open(d / "sim_evals.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"zeit": pd.Timestamp.now(tz="UTC").isoformat(), "hilft": not worse,
                                "auc_ohne": round(rows[0]["AUC"] or 0, 3), "auc_mit": round(best["AUC"] or 0, 3),
                                "anteil": float(best["Gewicht"]), "echte_trades": len(real)}) + "\n")
    (d / "sim_weight.json").write_text(json.dumps({**out, "checked_at": pd.Timestamp.now(tz="UTC").isoformat()},
                                                  indent=2, default=str), encoding="utf-8")
    return out


def diagnose(user_data: Path | None = None) -> dict[str, Any]:
    """Warum widersprechen Simulationen den echten Trades? (25.10.: schon 2 % Anteil drückt die AUC von 0,76 auf 0,44)

    Prüft: Duplikate unter echten Trades, ob die 0,76 über mehrere Schnitte hält und mit reinen Kursmerkmalen,
    wie gut ein NUR auf Simulationen gelerntes Modell echte Trades trifft, Gewinnquoten, und je Merkmal, ob es bei
    echten Trades und Simulationen in dieselbe Richtung zeigt."""
    from kitb import self_learning as sl
    from kitb import shared_memory

    real = shared_memory.load(user_data).dropna(subset=["profit_pct"]).sort_values("close_date").reset_index(drop=True)
    cols = [c for c in shared_memory.SNAPSHOT_FEATURES if c in real and real[c].notna().mean() > 0.5]
    price = [c for c in shared_memory.PRICE_FEATURES if c in cols]
    sim = load(user_data)
    out: dict[str, Any] = {"echte_trades": len(real), "sims": len(sim), "merkmale": len(cols)}
    if len(real) < 2 * sl.MIN_TRADES or sim.empty:
        return {**out, "why": "zu wenige Daten"}
    key = ["pair", "open_date"] if "pair" in real else ["open_date"]
    out["echte_ohne_duplikate"] = int(len(real.drop_duplicates(key)))
    out["bots"] = real["ki"].value_counts().to_dict() if "ki" in real else {}
    y = (real["profit_pct"] > 0).to_numpy(int)
    ys = (sim["profit_pct"] > 0).to_numpy(int)
    out["gewinnquote_echt"], out["gewinnquote_sim"] = float(y.mean()), float(ys.mean())
    out["ø_echt"], out["ø_sim"] = float(real["profit_pct"].mean()), float(sim["profit_pct"].mean())

    def auc_cut(frac: float, use: list[str]) -> float | None:
        cut = int(len(real) * frac)
        tr, te = real.iloc[:cut], real.iloc[cut:]
        if np.unique(y[:cut]).size < 2:
            return None
        x, fill = sl._design(tr, use)
        return _auc(y[cut:], sl._fit(x, y[:cut]).predict_proba(sl._design(te, use, fill)[0])[:, 1])

    out["auc_ohne_sims_je_schnitt"] = {f: auc_cut(f, cols) for f in (0.4, 0.5, 0.6, 0.7)}
    out["auc_ohne_sims_nur_kurs"] = {f: auc_cut(f, price) for f in (0.4, 0.5, 0.6, 0.7)} if price else {}
    test = real.iloc[len(real) // 2:]
    xs, fill = sl._design(sim, cols)
    m = sl._fit(xs, ys)
    out["auc_nur_sims_auf_echten"] = _auc((test["profit_pct"] > 0).to_numpy(int),
                                          m.predict_proba(sl._design(test, cols, fill)[0])[:, 1])
    xr = sl._design(real, cols, fill)[0]
    rows = []
    for c in cols:
        cr = np.corrcoef(xr[c], y)[0, 1] if xr[c].std() > 0 else np.nan
        cs = np.corrcoef(xs[c], ys)[0, 1] if xs[c].std() > 0 else np.nan
        sd = xs[c].std() or 1.0
        rows.append({"merkmal": c, "korr_echt": cr, "korr_sim": cs, "lage_echt_vs_sim": (xr[c].mean() - xs[c].mean()) / sd,
                     "nan_echt": float(real[c].isna().mean()), "nan_sim": float(sim[c].isna().mean()) if c in sim else 1.0})
    df = pd.DataFrame(rows)
    out["gegenlaeufige_merkmale"] = df[np.sign(df["korr_echt"]) * np.sign(df["korr_sim"]) < 0] \
        .sort_values("korr_echt", key=abs, ascending=False).head(10).round(3).to_dict("records")
    out["staerkste_merkmale_echt"] = df.sort_values("korr_echt", key=abs, ascending=False).head(10).round(3) \
        .to_dict("records")
    out["fremdeste_merkmale"] = df.sort_values("lage_echt_vs_sim", key=abs, ascending=False).head(10).round(3) \
        .to_dict("records")
    d = paths_reports(user_data)
    (d / "sim-diagnose.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def paths_reports(user_data: Path | None) -> Path:
    from kitb import paths

    d = (user_data or paths.USER_DATA) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _taken(test: pd.DataFrame, p: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    taken = test["profit_pct"].to_numpy()[p >= float(y_train.mean())]
    return {"genommen": int(len(taken)), "Ø genommen": float(taken.mean()) if len(taken) else None}
