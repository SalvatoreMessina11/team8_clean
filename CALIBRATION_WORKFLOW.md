# Team 8 — Calibration workflow

## Obiettivo

Questa nota definisce la struttura da usare per le calibrazioni dei modelli GLD:

1. Black--Scholes
2. Heston
3. Bates
4. Full Exact Bates--Hawkes

L'obiettivo è evitare di perdere risultati costosi, permettere il resume automatico dei run interrotti e mantenere una traccia completa dei parametri stimati su ogni data.

---

## 1. Stato attuale del progetto

Le full surface recuperate da IBKR sono salvate in:

```text
data/processed/full_surfaces/
```

Esempi già disponibili:

```text
GLD_2026-09-01_eligible_full_surface.csv
GLD_2026-09-02_eligible_full_surface.csv
```

Le full surface contengono già le quantità necessarie per la calibrazione:

```text
K
T
price
rate
implied_vol
vega
spot
curve_date
expiry
```

Il file attuale:

```text
src/calibrate_one_day.py
```

non deve essere usato come procedura finale sulle full surface, perché costruisce la calibration surface partendo da:

```text
data/processed/options_GLD_daily_60.parquet
```

e applica filtri più stretti rispetto alle full surface.

Per le calibrazioni finali sulle date dense verranno quindi usati due script dedicati:

```text
src/calibrate_surface.py
src/batch_calibrate.py
```

---

## 2. Regola di campionamento prima della calibrazione

La geometria dei 64 nodi deve essere comune alle date dense.

Non si deve usare automaticamente:

```text
CC il 2026-09-01
UU il 2026-09-02
```

solo perché ciascuna strategia vince sulla singola giornata.

La geometria finale deve essere scelta cross-date:

\[
g^\star
=
\arg\min_g
\frac{1}{D}
\sum_{t=1}^{D}
L_{\infty,g,t}.
\]

Le strategie attualmente confrontate sono:

```text
UU = Uniform T + Uniform K
CU = Chebyshev T + Uniform K
UC = Uniform T + Chebyshev K
CC = Chebyshev T + Chebyshev K
EU = Exponential T + Uniform K
EC = Exponential T + Chebyshev K
GU = Gaussian-centered T + Uniform K
GC = Gaussian-centered T + Chebyshev K
```

Parametri fissati per evitare data snooping:

```text
lambda_T  = 1
gaussian_z = 2
```

Il criterio principale è:

\[
L_\infty
=
\max_i
\left|
\widehat{\sigma}^{imp}_i
-
\sigma^{imp}_i
\right|.
\]

Tie-breaker:

```text
1. holdout L_inf
2. holdout RMSE
3. holdout MAE
```

---

## 3. Regola per il numero di osservazioni

Per ogni data \(t\):

\[
\mathcal C_t =
\begin{cases}
\text{64-point structured sample}, & N_t \ge 64,\\
\text{all eligible actual observations}, & N_t < 64.
\end{cases}
\]

Non si devono mai creare artificialmente 64 osservazioni quando il mercato storico recuperabile contiene meno di 64 punti.

L'interpolazione serve per valutare la qualità della geometria di sampling, non per creare contratti sintetici da usare nella calibrazione.

---

## 4. Struttura degli output di calibrazione

La struttura prevista è:

```text
outputs/
└── calibrations/
    ├── calibration_master.csv
    ├── parameters_long.csv
    ├── parameters_wide.csv
    │
    ├── 2026-09-01/
    │   ├── manifest.json
    │   ├── calibration_surface.csv
    │   ├── black_scholes.json
    │   ├── heston.json
    │   ├── bates.json
    │   ├── full_bates_hawkes.json
    │   └── calibration_summary.csv
    │
    ├── 2026-09-02/
    │   ├── manifest.json
    │   ├── calibration_surface.csv
    │   ├── black_scholes.json
    │   ├── heston.json
    │   ├── bates.json
    │   ├── full_bates_hawkes.json
    │   └── calibration_summary.csv
    │
    └── ...
```

---

## 5. Manifest per ogni data

Ogni cartella giornaliera deve contenere un file:

```text
manifest.json
```

con almeno:

```json
{
  "date": "2026-09-02",
  "source_surface": "data/processed/full_surfaces/GLD_2026-09-02_eligible_full_surface.csv",
  "sampling_strategy": "UU",
  "n_full": 1100,
  "n_calibration": 64,
  "spot": null,
  "curve_date": "2026-09-02",
  "seed": 8,
  "profile": "full",
  "objective": "vega_weighted_mse"
}
```

Il manifest serve a ricostruire esattamente:

- quale data è stata calibrata;
- quale surface è stata usata;
- quale geometria di sampling è stata applicata;
- quanti punti erano disponibili;
- quanti punti sono entrati nella calibrazione;
- quale Treasury curve era disponibile senza look-ahead;
- quale seed e profilo numerico sono stati usati.

---

## 6. Salvataggio immediato dopo ogni modello

I risultati non devono essere salvati solo alla fine dei quattro modelli.

La procedura deve essere:

```text
Black-Scholes
    ↓
salva black_scholes.json

Heston
    ↓
salva heston.json

Bates
    ↓
salva bates.json

Full Exact Bates-Hawkes
    ↓
salva full_bates_hawkes.json
```

In questo modo, se il computer si interrompe durante Bates--Hawkes, i risultati precedenti rimangono disponibili.

---

## 7. Resume automatico

Prima di lanciare ogni modello, lo script deve controllare se esiste già un risultato valido.

Schema:

```text
Esiste black_scholes.json con success=True?
    ├── sì  → SKIP
    └── no  → calibra e salva

Esiste heston.json con success=True?
    ├── sì  → SKIP
    └── no  → calibra e salva

Esiste bates.json con success=True?
    ├── sì  → SKIP
    └── no  → calibra e salva

Esiste full_bates_hawkes.json con success=True?
    ├── sì  → SKIP
    └── no  → calibra e salva
```

Un run interrotto deve quindi poter essere rilanciato senza rifare le calibrazioni già completate.

---

## 8. Ordine dei modelli

L'ordine previsto è:

```text
Black-Scholes
      ↓
Heston
      ↓
Bates
      ↓
Full Exact Bates-Hawkes
```

Bates viene calibrato anche quando è necessario come seed per il Full Bates--Hawkes.

---

## 9. Parametri da salvare

### Black--Scholes

```text
sigma
```

### Heston

```text
v0
kappa
theta
xi
rho
```

### Bates

I parametri Heston più i parametri di salto previsti dall'implementazione Bates corrente.

### Full Exact Bates--Hawkes

La calibrazione corrente usa 11 parametri:

```text
v0
kappa
theta
xi
rho
lambda0
lambda_bar
branching_ratio
beta
mu_J
sigma_J
```

e deve essere salvato anche:

\[
\alpha
=
\text{branching ratio}\times\beta.
\]

---

## 10. Tabelle aggregate

### calibration_master.csv

Una riga per data e modello:

```text
date,model,success,objective,elapsed_seconds,n_points,strategy
2026-09-01,Black-Scholes,True,...,...,64,...
2026-09-01,Heston,True,...,...,64,...
2026-09-01,Bates,True,...,...,64,...
2026-09-01,Full Bates-Hawkes,True,...,...,64,...
2026-09-02,Black-Scholes,True,...,...,64,...
...
```

### parameters_long.csv

Formato utile per grafici e analisi temporali:

```text
date,model,parameter,value
2026-09-02,Heston,v0,...
2026-09-02,Heston,kappa,...
2026-09-02,Heston,theta,...
2026-09-02,Full Bates-Hawkes,branching_ratio,...
...
```

### parameters_wide.csv

Formato utile per tabelle e paper:

```text
date,model,v0,kappa,theta,xi,rho,lambda0,lambda_bar,branching_ratio,beta,mu_J,sigma_J,objective
```

---

## 11. Profilo numerico

Le calibrazioni finali devono usare un profilo più accurato del semplice smoke test.

L'attuale `calibrate_one_day.py` prevede:

```text
quick
full
```

Per i risultati finali si userà:

```text
full
```

Il profilo quick rimane utile soltanto per verificare che tutto il pipeline funzioni.

---

## 12. Seed

Per la riproducibilità il seed deve essere salvato nel manifest.

Default attuale:

```text
seed = 8
```

Il seed deve rimanere uguale tra le date, salvo test di robustezza esplicitamente documentati.

---

## 13. Obiettivo di calibrazione

Le implementazioni correnti utilizzano errori di prezzo pesati tramite Vega e una media sui punti della surface.

Quindi il valore dell'obiettivo non cresce meccanicamente soltanto perché una data contiene più osservazioni.

Questo è importante quando si confrontano date con \(N_t\) differente.

---

## 14. Cosa fare adesso

### Fase A — completare alcune full surface dense

Continuare a recuperare alcune date recenti, ad esempio:

```text
2026-09-02
2026-09-01
2026-08-31
2026-08-28
2026-08-24
2026-08-17
```

Non è necessario recuperare decine di date dense se 3--5 date sono già sufficienti a stabilizzare la scelta della geometria.

### Fase B — eseguire lo stesso sampling test su tutte le dense surface

Usare sempre:

```text
src/compare_sampling_all.py
```

con:

```text
n_T = 8
n_K = 8
lambda_T = 1
gaussian_z = 2
```

### Fase C — scegliere una sola geometria comune

Dopo aver confrontato le date dense, determinare \(g^\star\) sulla media cross-date del holdout \(L_\infty\).

### Fase D — iniziare le calibrazioni finali

Una volta fissata la geometria:

```text
Black-Scholes
Heston
Bates
Full Exact Bates-Hawkes
```

su ogni data.

---

## 15. Strategia per non perdere tempo

Poiché Full Bates--Hawkes è il modello più costoso, l'ordine operativo consigliato è:

```text
1. stabilizzare la geometria di sampling
2. calibrare BS
3. calibrare Heston
4. calibrare Bates
5. calibrare Full Bates-Hawkes
```

Se si desidera iniziare prima che la geometria sia definitivamente scelta, eventuali calibrazioni su una strategia provvisoria devono essere etichettate chiaramente come robustness/provisional results e non come risultati finali.

---

# Comandi PowerShell

## A. Entrare nella repository

```powershell
cd C:\Users\salvm\Desktop\GitHub\team8_clean
```

## B. Attivare l'ambiente virtuale

```powershell
.\.venv\Scripts\Activate.ps1
```

Se PowerShell blocca l'attivazione:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## C. Controllare quali full surface sono già disponibili

```powershell
Get-ChildItem "data\processed\full_surfaces\GLD_*_eligible_full_surface.csv"
```

Per ottenere anche righe, expiries e strikes:

```powershell
Get-ChildItem "data\processed\full_surfaces\GLD_*_eligible_full_surface.csv" |
ForEach-Object {
    $d = Import-Csv $_.FullName
    [PSCustomObject]@{
        Date = ($_.BaseName -replace 'GLD_','' -replace '_eligible_full_surface','')
        Rows = $d.Count
        Expiries = ($d.expiry | Sort-Object -Unique).Count
        Strikes = ($d.K | Sort-Object -Unique).Count
        Enough64 = ($d.Count -ge 64)
    }
} | Sort-Object Date | Format-Table -AutoSize
```

---

## D. Sampling test sul 2026-09-01

```powershell
python src\compare_sampling_all.py `
  --surface "data\processed\full_surfaces\GLD_2026-09-01_eligible_full_surface.csv" `
  --output-dir "outputs\sampling_all_2026-09-01" `
  --n-t 8 `
  --n-k 8 `
  --lambda-t 1 `
  --gaussian-z 2
```

## E. Sampling test sul 2026-09-02

```powershell
python src\compare_sampling_all.py `
  --surface "data\processed\full_surfaces\GLD_2026-09-02_eligible_full_surface.csv" `
  --output-dir "outputs\sampling_all_2026-09-02" `
  --n-t 8 `
  --n-k 8 `
  --lambda-t 1 `
  --gaussian-z 2
```

Usare la stessa struttura per le nuove full surface:

```powershell
python src\compare_sampling_all.py `
  --surface "data\processed\full_surfaces\GLD_YYYY-MM-DD_eligible_full_surface.csv" `
  --output-dir "outputs\sampling_all_YYYY-MM-DD" `
  --n-t 8 `
  --n-k 8 `
  --lambda-t 1 `
  --gaussian-z 2
```

---

## F. Creare la cartella generale delle calibrazioni

```powershell
New-Item -ItemType Directory -Force "outputs\calibrations"
```

---

## G. Controllare lo stato Git

```powershell
git status
```

---

## H. Aggiungere questo README alla repository

Dopo aver scaricato questo file e averlo copiato nella root della repository con nome:

```text
CALIBRATION_WORKFLOW.md
```

eseguire:

```powershell
git add CALIBRATION_WORKFLOW.md
git commit -m "Document calibration workflow"
git push origin main
```

---

# Prossimo sviluppo software

Prima di lanciare le calibrazioni finali sulle full surface devono essere creati:

```text
src/calibrate_surface.py
src/batch_calibrate.py
```

`calibrate_surface.py` dovrà:

- leggere direttamente una full surface o un sample da 64 punti;
- non ricostruire la surface dal vecchio `options_GLD_daily_60.parquet`;
- calibrare i quattro modelli;
- salvare ogni risultato appena disponibile;
- supportare `--resume`;
- supportare `--models`;
- supportare `--profile`;
- supportare `--seed`.

`batch_calibrate.py` dovrà:

- ricevere più date;
- scegliere il sample corretto per ogni data;
- saltare automaticamente i risultati già completati;
- aggiornare `calibration_master.csv`;
- aggiornare `parameters_long.csv`;
- aggiornare `parameters_wide.csv`;
- poter essere rilanciato senza perdere il lavoro precedente.

Non lanciare ancora ore di calibrazione sulle full surface tramite il vecchio `calibrate_one_day.py`: prima va completato questo passaggio.
