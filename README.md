# Football-ML-engine
Machine learning system for football match prediction, player analytics, and real-time football intelligence

## Understanding it

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the guide: how the five
stages fit together, what each tab does, why the model predicts goal rates
rather than outcomes, and how a change is measured before it ships.

## Running it

The site is two processes and needs both — the page is served by Vite on
**8443**, and every tab on it reads the API on **8000**. Start them with:

```powershell
.\scripts\start_site.ps1
```

Then open **http://localhost:8443**. The script starts only what is not already
listening, so it is safe to run twice, and it is registered to run at login.
Remove that with:

```powershell
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\FootballML.lnk"
```

Both bind to localhost. Opening the site from another device needs uvicorn on
`--host 0.0.0.0` and `VITE_API_URL` pointing at this machine — deliberately not
the default.

### After changing feature code, restart the API — do not reload it

`POST /admin/reload` re-reads the data files and the model artifact but **not the
code**. Retrain after adding a feature and the running server gets a model
expecting columns its own feature builder cannot produce. It now refuses that
reload and keeps serving the previous model rather than breaking every
prediction, so the failure is loud — but the fix is a restart, not another
retrain.

`pipelines/weekly.py` treats a refused reload as a failed run for the same
reason: every file on disk would be this week's while the site served last
week's model.

## Keeping it current

`pipelines/weekly.py` refreshes results, ratings, the model, predictions and the
projected tables, then tells a running API to pick them up. It is registered as
the Windows scheduled task **FootballML Weekly Refresh**, Mondays at 07:00, and
writes `logs/weekly-<date>.log`.

It is set to run on battery and to catch up a missed run (`StartWhenAvailable`),
because it was skipped in silence on 2026-09-28 for want of exactly those two
settings. Check on it with:

```powershell
Get-ScheduledTask -TaskName "FootballML Weekly Refresh" | Get-ScheduledTaskInfo
```

`LastTaskResult` of `0` is success; `0x800710E0` means Windows refused to start
it.
