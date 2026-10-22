# LLM agents + Probe-Trust: summary

Windows: 40 events + 20 normal (METR-LA, test). Instances per pool: 12, ensemble 10, draws per window 200. Harm = max(RMSE_events, RMSE_normal)/RMSE of the clean ensemble - 1, %, 95 % CI by bootstrap over windows.

Clean ensemble (10 honest, mean): RMSE events 16.92, normal 4.31 mph.

## Attack hide

| share | mean | median | trimmed | Probe-Trust | Probe one-sided | LLM coordinator | Δ Probe−median [CI] | Δ Probe−coordinator [CI] | AUC of weights | coordinator parse errors |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | — | +nan [+nan; +nan] | — | 1.00 | 0.00 |
| 0.2 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | — | +nan [+nan; +nan] | — | 1.00 | 0.00 |
| 0.4 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | 1.00 | 0.00 |
| 0.6 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | 1.00 | 0.00 |

## Attack exagg

| share | mean | median | trimmed | Probe-Trust | Probe one-sided | LLM coordinator | Δ Probe−median [CI] | Δ Probe−coordinator [CI] | AUC of weights | coordinator parse errors |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | — | +nan [+nan; +nan] | — | 1.00 | 0.00 |
| 0.2 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | — | +nan [+nan; +nan] | — | 1.00 | 0.00 |
| 0.4 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | 1.00 | 0.00 |
| 0.6 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | 1.00 | 0.00 |

## Attack hide_aware

| share | mean | median | trimmed | Probe-Trust | Probe one-sided | LLM coordinator | Δ Probe−median [CI] | Δ Probe−coordinator [CI] | AUC of weights | coordinator parse errors |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | — | +nan [+nan; +nan] | — | 0.79 | 0.00 |
| 0.2 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | — | +nan [+nan; +nan] | — | 0.79 | 0.00 |
| 0.4 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | 0.79 | 0.01 |
| 0.6 | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | +nan [+nan; +nan] | 0.79 | 0.01 |

## Check of the criteria (from the plan)

```
{
 "C1_attack_effective(mean>=30 at 0.4)": {
  "hide": NaN,
  "exagg": NaN
 },
 "C2_probe<=10 at 0.4 and 0.6": {
  "hide": [
   NaN,
   NaN
  ],
  "exagg": [
   NaN,
   NaN
  ]
 },
 "C3_probe better than median by >=5 pp at 0.6 (CI<0)": {
  "hide": [
   NaN,
   NaN,
   NaN
  ],
  "exagg": [
   NaN,
   NaN,
   NaN
  ]
 },
 "C4_AUC and clean ensemble": {
  "hide": [
   1.0,
   NaN
  ],
  "exagg": [
   1.0,
   NaN
  ]
 },
 "C5_hide_aware (boundary)": {
  "0.4": {
   "mean": NaN,
   "median": NaN,
   "trimmed": NaN,
   "probe": NaN,
   "probe1": NaN,
   "coord": NaN
  },
  "0.6": {
   "mean": NaN,
   "median": NaN,
   "trimmed": NaN,
   "probe": NaN,
   "probe1": NaN,
   "coord": NaN
  }
 }
}
```