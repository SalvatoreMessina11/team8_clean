# Team 8 — Clean GLD / IBKR Rolling OOS

Questa è la nuova base pulita del progetto. Non richiede nessun file LSE e non
richiede di copiare il vecchio repository Team 8.

## Struttura

```text
team8_clean_complete/
├── src/
│   ├── ibkr_gld_weekly_data.py
│   ├── ibkr_gld_recent_daily_fetch.py
│   ├── build_daily_60_panel.py
│   ├── BnS.py
│   ├── Heston.py
│   ├── Bates.py
│   ├── Hawkes.py
│   ├── BatesHawkesExact.py
│   ├── calibration_core.py
│   ├── fourier_pricing.py
│   └── model_smoke_test.py
├── data/
│   ├── raw/
│   │   └── daily-treasury-rates2026.csv
│   └── processed/
├── paper/
│   └── rolling_oos_methodology_draft.tex
├── outputs/
├── requirements.txt
└── .gitignore
```

## Strategia corrente

Non usare come primo tentativo il backfill storico da oltre 2.000 contratti.

Il path consigliato ora è:

1. costruire i tassi 2026;
2. scaricare lo storico daily di GLD;
3. verificare i modelli;
4. scaricare un campione mirato di opzioni per le ultime 60 sedute;
5. analizzare `options_GLD_daily_60_coverage.csv`;
6. decidere, sulla base della coverage reale, se il test principale sarà:
   - daily, t -> t+1, ultime 60 sedute; oppure
   - weekly, su un intervallo più lungo.

Il rolling OOS definitivo verrà aggiunto solo dopo questa verifica.

---

# 1. Ambiente virtuale

Apri PowerShell nella cartella del progetto:

```powershell
cd C:\...\team8_clean_complete
```

Crea il virtual environment:

```powershell
py -m venv .venv
```

Attivalo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Poi:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# 2. Tassi Treasury

Non serve TWS.

```powershell
python src\ibkr_gld_weekly_data.py rates --treasury-csv data\raw\daily-treasury-rates2026.csv --output-dir data\processed
```

Output atteso:

```text
data\processed\usd_treasury_history.csv
```

Il CSV originale contiene 169 date e 14 tenor dal 2026-01-02 al 2026-09-02.

---

# 3. Storico GLD

Apri TWS / IB Gateway e abilita API.

Per TWS Paper Trading, normalmente:

```text
port = 7497
```

Esegui:

```powershell
python src\ibkr_gld_weekly_data.py stock --start 2026-01-02 --end 2026-09-02 --output-dir data\processed --port 7497
```

Output atteso:

```text
[OK] stock dates: 2026-01-02 -> 2026-09-02
[OK] weekly buckets: 36
[OK] first weekly observation: 2026-01-02
[OK] last weekly observation: 2026-09-02
```

File:

```text
data\processed\gld_daily_history.csv
data\processed\gld_weekly_dates.csv
```

---

# 4. Test del motore matematico

```powershell
python src\model_smoke_test.py
```

Output finale atteso:

```text
[OK] Black-Scholes price + IV inversion
[OK] Heston pricing
[OK] Bates pricing
[OK] Exact Bates-Hawkes pricing
[OK] Bates limit of exact Hawkes when alpha=0
[OK] model stack ready
```

I modelli inclusi sono:

```text
Black-Scholes
Heston
Bates
Full exact Bates-Hawkes
```

`BatesHawkesExact.py` è il modello Hawkes event-dependent; non viene usato il
proxy stationary-intensity `BatesHawkes.py`.

---

# 5. PATH CONSIGLIATO: ultime 60 sedute giornaliere

Questo è il fetch da usare adesso.

```powershell
python src\ibkr_gld_recent_daily_fetch.py --end 2026-09-02 --sessions 60 --output-dir data\processed --port 7497
```

Il fetch limita fortemente l'universo prima delle richieste IBKR:

```text
massimo 6 expiries x 24 strikes = 144 contratti
```

anziché interrogare circa 2.400 contratti.

Crea:

```text
data\processed\options_GLD_recent_daily_raw.parquet
data\processed\options_GLD_daily_60.parquet
data\processed\options_GLD_daily_60_coverage.csv
data\processed\gld_daily_60_dates.csv
```

Il file da analizzare prima di qualsiasi calibrazione è:

```text
data\processed\options_GLD_daily_60_coverage.csv
```

Con 60 surface utilizzabili avremmo:

```text
60 surface
59 previsioni OOS t -> t+1 per modello
236 forecast model-specifici per 4 modelli
```

---

# 6. Come valutare la coverage

Per ogni data il report contiene, tra le altre cose:

```text
rows
expiries
strikes
rows_volume_ge_1
rows_volume_ge_25
core_rows_90_110
min_dte
max_dte
```

Non basta che `rows > 0`.

Per un rolling OOS serio vogliamo date con una cross-section sufficientemente
ampia e relativamente omogenea.

Prima di scegliere filtri definitivi, confrontare:

- numero di date coperte;
- expiries per data;
- strikes per data;
- opzioni con volume >= 1;
- opzioni con volume >= 25;
- copertura nella fascia 0.90 <= K/S <= 1.10;
- intervallo delle maturity.

---

# 7. Path settimanale

Il file:

```text
src\ibkr_gld_weekly_data.py
```

contiene ancora:

```text
backfill-active
```

ma **non è il percorso consigliato adesso**, perché il tentativo molto ampio
può generare migliaia di richieste e molti contratti storici non sono più
recuperabili.

Se in seguito vorremo testare il weekly, lo faremo con un fetch mirato simile
al daily, non necessariamente con il backfill completo.

---

# 8. build_daily_60_panel.py

Questo file è un'utility opzionale.

Se in futuro possiedi già:

```text
options_GLD_active_contract_intraday.parquet
```

puoi costruire il panel daily senza ricontattare IBKR:

```powershell
python src\build_daily_60_panel.py
```

Nel path consigliato attuale non serve, perché
`ibkr_gld_recent_daily_fetch.py` crea già direttamente il panel daily.

---

# 9. Tassi e no-look-ahead

Per ogni data di calibrazione `t`, il futuro surface builder userà esclusivamente
la curva Treasury più recente con:

```text
curve_date <= t
```

Nessuna osservazione futura dei tassi potrà quindi entrare nella calibrazione o
nel forecast.

---

# 10. Rolling OOS che verrà aggiunto dopo la coverage

La logica finale sarà:

```text
surface(t)
    |
    v
calibrazione modello a t
    |
    v
forecast surface(t+1)
    |
    v
osservazione surface(t+1)
    |
    +--> errore OOS
    |
    v
nuova calibrazione a t+1
```

Benchmark previsti:

```text
expanding prevailing mean
random walk: IV(t+1|t) = IV(t)
```

Metriche:

```text
MAE
RMSE
R2 OOS vs prevailing mean
R2 OOS vs random walk
cumulative loss differential
```

---

# 11. File che NON servono dal vecchio Team 8

Non copiare:

```text
lse_dataset.py
main.py
online_validation.py
historical_validation.py
tools/
Sampling.py
BatesHawkes.py
```

La nuova pipeline verrà costruita direttamente sopra i dati IBKR puliti.

---

# Prossimo checkpoint

Dopo il fetch daily-60, inviare:

```text
data\processed\options_GLD_daily_60_coverage.csv
```

Da quel report si decide la frequenza del test prima di scrivere
`surface_builder.py` e `rolling_oos.py`.
