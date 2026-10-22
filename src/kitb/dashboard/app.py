"""Schlankes lokales Dashboard (Streamlit). Start: ``kitb dashboard`` -> http://127.0.0.1:8501

Für Analysen und Tests nicht erforderlich. Einzige schreibende Aktion: Not-Stopp setzen/zurücksetzen.
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kitb import killswitch
from kitb.config import load_freqtrade_config, load_project_config
from kitb.dashboard import data as D
from kitb.strategies import compute_signals, load_strategy

st.set_page_config(page_title="KI-Trading-Bot", layout="wide")
project = load_project_config()
status = D.system_status()

st.title("KI-Trading-Bot – lokales Dashboard")
st.caption("Keine Anlageberatung. Ergebnisse aus Backtests/Paper-Trading garantieren keine Gewinne.")

ks = status["kill_switch"]
if ks:
    st.error(f"NOT-STOPP AKTIV seit {ks.get('activated_at')} – Grund: {ks.get('reason')} (durch {ks.get('by')})")

overview_tab, *tabs = st.tabs(["Übersicht", "Status", "Kurse & Signale", "Paper-Trading", "Bot A vs. Bot B",
                               "Strategien", "Entscheidungen", "Modelle", "Datenqualität", "Fehler", "Not-Stopp"])


@st.cache_data(ttl=60, show_spinner=False)
def _daily(exchange_id: str, pair: str) -> pd.DataFrame:
    return D.fetch_daily_public(exchange_id, pair)


@st.fragment(run_every=60)
def simple_overview() -> None:
    """Einfache Ansicht für den Alltag: läuft der Bot, was hält er, was hat er vor. Aktualisiert sich jede Minute."""
    ids = list(project.get("paper_strategies", {})) + list(project["bots"])
    bot = st.selectbox("Bot", ids, label_visibility="collapsed") if len(ids) > 1 else ids[0]
    ft = load_freqtrade_config()
    o = D.overview(bot, ft["dry_run_wallet"], fetch=_daily, exchange_id=ft["exchange"]["name"])
    if o["kill_switch"]:
        st.error("🛑 Not-Stopp ist aktiv – der Bot kauft nichts mehr.")
    elif o["running"]:
        st.success(f"🟢 Der Bot läuft (Paper-Trading, nur virtuelles Geld). Letztes Lebenszeichen vor {o['log_age_seconds']} s.")
    elif o["log_age_seconds"] is None:
        st.warning("⚪ Der Bot wurde noch nie gestartet. Doppelklick auf „Trading-Bot starten“.")
    else:
        st.error(f"🔴 Der Bot läuft nicht (letztes Lebenszeichen vor {o['log_age_seconds'] // 60} min). "
                 "Doppelklick auf „Trading-Bot starten“.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Virtuelles Kapital", f"{o['equity']:,.2f} USDT",
              f"{o['equity'] - o['start_balance']:+,.2f} USDT seit Start")
    c2.metric("Offene Positionen", len(o["positions"]))
    c3.metric("Abgeschlossener Gewinn/Verlust", f"{o['realized_pnl']:+,.2f} USDT")
    nd = o["next_decision"]
    c4.metric("Nächste Entscheidung", f"{nd.tz_convert('Europe/Berlin'):%d.%m. %H:%M} Uhr")

    st.subheader("Was der Bot gerade vorhat")
    held = {p["Markt"] for p in o["positions"]}
    for pair, m in o["markets"].items():
        st.write(("✅ " if pair in held else "⏸️ ") + D.explain_market(pair, m, pair in held))
    if o["fetch_error"]:
        st.caption(f"Aktuelle Kurse nicht abrufbar ({o['fetch_error']}).")

    st.subheader("Offene Positionen")
    if o["positions"]:
        st.dataframe(pd.DataFrame(o["positions"]), hide_index=True, width="stretch",
                     column_config={"Gewinn/Verlust (%)": st.column_config.NumberColumn(format="percent")})
    else:
        st.write("Keine – der Bot hält gerade nur USDT.")

    st.subheader("Alle Trades – was er handelt und ob es sich lohnt")
    tt = o["trades_table"]
    if not tt.empty:
        closed = tt[tt["Status"] == "abgeschlossen"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Trades gesamt", len(tt))
        k2.metric("davon mit Gewinn", f"{int((closed['Ergebnis (USDT)'] > 0).sum())} von {len(closed)}" if len(closed) else "–")
        k3.metric("Summe aller Ergebnisse", f"{tt['Ergebnis (USDT)'].fillna(0).sum():+,.2f} USDT")
        st.dataframe(tt.style.map(lambda v: "color: green" if isinstance(v, float) and v > 0 else
                                  ("color: red" if isinstance(v, float) and v < 0 else ""),
                                  subset=["Ergebnis (USDT)", "Ergebnis (%)"]),
                     hide_index=True, width="stretch",
                     column_config={"Ergebnis (%)": st.column_config.NumberColumn(format="percent")})
    else:
        st.write("Noch keine Trades.")

    st.subheader("Wird der Bot besser?")
    perf = o["performance"]
    if perf["curve"].empty:
        st.write("Noch keine Daten – sobald der Bot handelt, erscheint hier der Verlauf.")
    else:
        p1, p2, p3 = st.columns(3)
        p1.metric("Bot seit Start", f"{perf['bot_return']:+.2%}", f"max. Rückgang {perf['bot_drawdown']:.1%}",
                  delta_color="off")
        p2.metric("Einfach halten (BTC/ETH je ½)", f"{perf['hold_return']:+.2%}",
                  f"max. Rückgang {perf['hold_drawdown']:.1%}", delta_color="off")
        p3.metric("Bot minus Halten", f"{perf['bot_return'] - perf['hold_return']:+.2%}")
        st.line_chart(perf["curve"], y_label="USDT")
        st.info(f"Urteil: **{perf['verdict']}**. Der Bot lernt nicht selbst dazu – seine Regel ist fest und wurde von "
                "dir freigegeben. „Besser werden“ heißt hier: Liegt die blaue Bot-Linie über der Halten-Linie und "
                "fällt sie in schwachen Phasen weniger stark? Die Tagesstrategie handelt nur wenige Male im Jahr; "
                "ein belastbares Urteil braucht deshalb viele Monate.")
    st.caption("Regel: Kauf, wenn ein Tag über der 200-Tage-Linie schließt; Verkauf, wenn er darunter schließt "
               "oder 10 % Verlust erreicht sind. Entschieden wird einmal täglich nach Tagesschluss (02:00 bzw. 01:00 Uhr "
               "deutscher Zeit). Zeiten in der Tabelle in UTC. Alles simuliert – kein echtes Geld.")


with overview_tab:
    simple_overview()

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live-Trading", status["live_trading"])
    c2.metric("Erlaubte Modi", ", ".join(status["allowed_run_modes"]))
    c3.metric("Agenten", "an" if status["agents_enabled"] else "aus")
    c4.metric("Not-Stopp", "AKTIV" if ks else "aus")
    st.write("Datenquellen:", ", ".join(status["data_sources"]) or "keine")
    risk = D.risk_state()
    if risk:
        st.subheader("Risikostatus (Paper-Trading)")
        st.json(risk)

with tabs[1]:
    sources = status["data_sources"]
    if not sources:
        st.info("Noch keine Daten. `kitb download` oder `kitb synthetic` ausführen.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        source = c1.selectbox("Quelle", sources)
        pair = c2.selectbox("Markt", project["allowed_pairs"])
        tf = c3.selectbox("Intervall", project["timeframes"])
        strat_name = c4.selectbox("Strategie-Signale", ["(keine)"] + project["evaluation"]["strategies"])
        df = D.candles(source, pair, tf, last_n=1500)
        ps = D.price_status(df, tf)
        if source == "synthetic":
            st.warning("Synthetische Daten – keine echten Kurse.")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{pair} letzter Schluss", f"{ps['last']:,.2f}" if ps["last"] else "–")
        m2.metric("Alter (Kerzen)", ps["age_candles"])
        m3.metric("Aktuell?", "veraltet" if ps["stale"] else "ok")
        view = df.tail(500)
        fig = go.Figure(go.Candlestick(x=view["date"], open=view["open"], high=view["high"], low=view["low"],
                                       close=view["close"], name=pair))
        if strat_name != "(keine)" and tf == "1h" and len(df) > 600:
            sig = compute_signals(load_strategy(strat_name), df, pair).tail(500)
            ent, ex = sig[sig["enter_long"] == 1], sig[sig["exit_long"] == 1]
            fig.add_scatter(x=ent["date"], y=ent["low"] * 0.995, mode="markers", name="Einstieg",
                            marker={"symbol": "triangle-up", "size": 10, "color": "green"})
            fig.add_scatter(x=ex["date"], y=ex["high"] * 1.005, mode="markers", name="Ausstieg",
                            marker={"symbol": "triangle-down", "size": 10, "color": "red"})
        fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    wallet = load_freqtrade_config()["dry_run_wallet"]
    paper_ids = list(project["bots"]) + list(project.get("paper_strategies", {}))
    rows = [D.paper_summary(bot, wallet) for bot in paper_ids]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    for bot in paper_ids:
        t = D.paper_trades(bot)
        if not t.empty:
            st.subheader(f"{bot}: Trades")
            closed = t[t["is_open"] == 0].copy()
            if not closed.empty:
                closed["equity"] = wallet + closed["close_profit_abs"].cumsum()
                st.line_chart(closed.set_index("close_date")["equity"])
            st.dataframe(t, use_container_width=True)
    if all(r["trades"] == 0 for r in rows):
        st.info("Noch keine Paper-Trades. Start mit `kitb paper start` (Phase 7).")

with tabs[3]:
    reps = D.reports("ml")
    if reps:
        choice = st.selectbox("Bericht", reps, format_func=lambda p: p.name)
        st.markdown((choice / "report.md").read_text(encoding="utf-8"))
    else:
        st.info("Noch kein KI-Vergleich. `kitb ml-compare` ausführen.")

with tabs[4]:
    reps = D.reports("compare")
    if reps:
        choice = st.selectbox("Bericht", reps, format_func=lambda p: p.name, key="cmp")
        st.markdown((choice / "report.md").read_text(encoding="utf-8"))
    else:
        st.info("Noch kein Strategievergleich. `kitb compare` ausführen.")

with tabs[5]:
    st.dataframe(D.registry_table("decisions", 500), use_container_width=True)

with tabs[6]:
    mv = D.registry_table("model_versions")
    if not mv.empty:
        st.dataframe(mv[["id", "bot", "version", "status", "approved_by", "eval_timerange", "created_at"]],
                     use_container_width=True)
        sel = st.selectbox("Details", mv["id"].tolist())
        row = mv[mv["id"] == sel].iloc[0]
        st.json({"metrics": json.loads(row["metrics"]), "criteria": json.loads(row["criteria"] or "null")})
    st.subheader("Ereignisse")
    st.dataframe(D.registry_table("model_events"), use_container_width=True)

with tabs[7]:
    st.dataframe(D.latest_data_quality(), use_container_width=True)

with tabs[8]:
    st.dataframe(D.errors(), use_container_width=True)

with tabs[9]:
    st.write("Der Not-Stopp verhindert sofort neue Einstiege; die Paper-Steuerung stoppt laufende Bots.")
    if not ks:
        reason = st.text_input("Grund")
        who = st.text_input("Dein Name")
        if st.button("NOT-STOPP AUSLÖSEN", type="primary", disabled=not (reason and who)):
            killswitch.activate(reason, who)
            st.rerun()
    else:
        who = st.text_input("Dein Name (zum Zurücksetzen)")
        confirm = st.text_input("Zum Zurücksetzen 'ZURÜCKSETZEN' eingeben")
        if st.button("Not-Stopp zurücksetzen", disabled=not (who and confirm == "ZURÜCKSETZEN")):
            killswitch.reset(who)
            st.rerun()
