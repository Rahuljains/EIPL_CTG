# CTG Risk Monitor: results summary - run 20261117_2151

Outcome: **pH < 7.15 OR Apgar5 < 7**. 113 adverse labours of 552 (20.5%).
All figures are pooled out-of-fold from a nested 5x3 cross-validation split by labour, with every threshold tuned inside the training folds only.

## Headline

- **Final system: SYS-F**
- **Sensitivity (ever HIGH RISK): 0.478** (95% CI 0.388-0.569) - 54 of 113 adverse labours caught, **59 missed**
- Specificity: 0.793 (21 false alarms per 100 normal labours)
- PPV 0.372, NPV 0.855
- Median warning lead time: 15 min before the end of the recording
- CV-tuned threshold (median across folds): 0.200
- Layer 1 alone caught 43 of 113 (sensitivity 0.381, specificity 0.831)

## All models

| model | sensitivity | specificity | PPV | NPV | adverse caught | false alarms /100 normal | AUROC | median lead time (min) |
|---|---|---|---|---|---|---|---|---|
| L1 | 0.018 | 0.979 | 0.182 | 0.795 | 2/113 | 2.1 |  |  |
| CLIN | 0.0 | 1.0 |  | 0.795 | 0/113 | 0.0 | 0.588 |  |
| CTG-F | 0.416 | 0.811 | 0.362 | 0.844 | 47/113 | 18.9 | 0.634 | 20.0 |
| SEQ-F | 0.442 | 0.809 | 0.373 | 0.849 | 50/113 | 19.1 | 0.638 | 20.0 |
| ZON-F | 0.442 | 0.809 | 0.373 | 0.849 | 50/113 | 19.1 | 0.638 | 20.0 |
| SYS-F | 0.478 | 0.793 | 0.372 | 0.855 | 54/113 | 20.7 | 0.637 | 20.0 |

## Caveats

- Retrospective, Czech and Chinese data, no Indian data. **Not for clinical use.**
- CTU-UHB is an enriched sample, so absolute probabilities will not transfer; ranking should.
- The outcome is a surrogate (cord acidaemia / low Apgar), not injury.
- Few adverse cases, so every interval is wide. n is printed beside every percentage.
- Aggregates only; no participant-level data appears in this file.
