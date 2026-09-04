TEAM 8 - CLEAN SRC + DENSE-ONLY OOS
===================================

GitHub basis
------------
Repository: SalvatoreMessina11/team8_clean
Branch: main
Reference commit observed during build:
  ac6adab5902169f010f9939b1a0442fba93a07b2

Purpose of this package
-----------------------
1. Replace src/ with a clean set of project scripts.
2. Keep only the two intended IBKR collectors:
     ibkr_gld_today_surface.py
     ibkr_gld_historical_surface.py
   Experimental acquisition scripts are intentionally omitted:
     ibkr_gld_backward_reuse_test.py
     ibkr_gld_conid_backward.py
     ibkr_gld_full_exhaustive.py
3. Correct OOS selection:
   - DTE >= 75;
   - every OOS date must have >=64 valid unique actual observations;
   - every OOS date must have >=3 expiries by default;
   - sparse dates are excluded from this OOS exercise;
   - every origin is sampled with fixed CC 8x8 = 64 actual observations;
   - duplicate-date surface selection chooses the richest valid file after
     DTE filtering rather than adaptive > historical > full priority.
4. Correct the market-data audit so DTE<75 observations are treated as
   outside the official domain, not as a quality failure. The audit also
   chooses the richest valid surface file per date.
5. Add --calibrate-only to oos_validation.py.

Recommended Windows PowerShell replacement
-------------------------------------------
cd C:\Users\salvm\Desktop\GitHub\team8_clean
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Rename-Item .\src "src_backup_$stamp"
Expand-Archive "$env:USERPROFILE\Downloads\team8_SRC_DENSE_OOS_FINAL.zip" `
  -DestinationPath . -Force
.\.venv\Scripts\python.exe -m compileall .\src

Dense audit / date preview
--------------------------
.\.venv\Scripts\python.exe src\audit_market_data.py `
  --start 2026-07-06 `
  --end 2026-09-02 `
  --min-dte 75 `
  --cc-size 64 `
  --min-expiries 3

.\.venv\Scripts\python.exe src\oos_validation.py `
  --start 2026-07-06 `
  --end 2026-09-02 `
  --min-dte 75 `
  --min-surface-points 64 `
  --min-origin-expiries 3 `
  --dry-run

Full rolling-origin calibrations only
-------------------------------------
.\.venv\Scripts\python.exe src\oos_validation.py `
  --start 2026-07-06 `
  --end 2026-09-02 `
  --min-dte 75 `
  --min-surface-points 64 `
  --min-origin-expiries 3 `
  --profile full `
  --calibrate-only

Final rolling OOS scoring
-------------------------
.\.venv\Scripts\python.exe src\oos_validation.py `
  --start 2026-07-06 `
  --end 2026-09-02 `
  --min-dte 75 `
  --min-surface-points 64 `
  --min-origin-expiries 3 `
  --profile full

Notes
-----
- Exactly 64 observations are admitted: 64 is sufficient for fixed CC 8x8.
- The final dense date is target-only in rolling OOS, so it is not calibrated
  by --calibrate-only unless it is also an origin for a later admitted date.
- Calibration resume is automatic: successful JSONs are reused.
- To deliberately rerun all calibrations, add --force-calibration.
- The OOS uses next AVAILABLE DENSE date, not necessarily the next calendar day.
