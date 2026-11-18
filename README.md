# CTG Risk Monitor: code submission

Author: Rahul Jain · 18 November 2026

An explainable, two-layer risk monitor for women in high-risk labour, built on open CTG data.

- **Layer 1:** guideline rules (NICE/FIGO) computed from the live CTG every 5 seconds. They raise RED at once and name the rule that fired.
- **Layer 2:** a Naive Bayes risk of adverse neonatal outcome (cord pH < 7.15 or 5-min Apgar < 7). It starts from the pregnancy history and is updated with each 15-minute CTG window.
- **Output:** one zone, GREEN / RED / RED – HIGH RISK, shown in a Streamlit app.

Prototype on retrospective, non-Indian data (CTU-UHB, Czech Republic; JNU-CTG, China). Not yet clinically validated.

## Start here

| If you want to… | Open |
|---|---|
| Understand the methods, results and every design decision | `CODE_SUMMARY.md` |
| See the full evaluation with figures (opens offline in a browser) | `output/reports/results_20261117_2151.html` |
| See the clinical data exploration | `output/reports/eda_clinical_20261117_1556.html` |
| Read the reasoning behind each judgment call | `output/reports/decisions_log.md` |
| Run the demo app | `app/README.md`, or the steps below |
| Read the original plan and justification | `docs/` |

## Headline results (5-fold cross-validation by woman, 552 labours, 113 adverse)

| | Sensitivity | Specificity | AUROC |
|---|---|---|---|
| Layer 1 rules alone | 0.381 (43/113) | 0.831 | - |
| **Final two-layer system** | **0.478 (54/113)** | **0.793** | 0.637 |
| Best Layer 2 ranking (encoder track) | 0.381 | 0.816 | 0.685 |

Adverse rate by final zone: GREEN 12.9%, RED 24.0%, HIGH RISK 37.8%.

## Run the demo app (no dataset needed)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/ctg_monitor_app.py
```

The app loads the shipped model `output/models/engine_bundle_20261117_2221/` and the six de-identified sample cases in `app/sample_data/`.

## Reproduce the full pipeline

1. `pip install -r requirements.txt` (tested on Python 3.9; PyTorch is needed for the encoder only).
2. Download the open datasets into `data/`:
   - `data/ctu-uhb/`: [CTU-UHB Intrapartum CTG Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) (PhysioNet, ODC-By 1.0)
   - `data/jnu-ctg/`: [JNU-CTG](https://zenodo.org/records/21800730) (Zenodo, CC BY 4.0)
3. Run the numbered scripts in the order listed in `CODE_SUMMARY.md` Section 4, then `pytest -q`.

Datasets, intermediate arrays and the per-run logs are not included: they are large, and the datasets are distributed by their owners.

## Folder map

```
codes/
├── README.md, CODE_SUMMARY.md, requirements.txt, clinical_schema.json
├── src/ctg/            library: loaders, rules, features, encoder, Naive Bayes, engine, evaluation
├── scripts/            numbered pipeline 00–09, plus diagnostics x01–x03
├── tests/              45 pytest tests (rules, Naive Bayes, causality, leakage, engine)
├── app/                Streamlit demo app + de-identified sample cases
├── output/models/      final engine bundle, per-fold models of the sealed run, encoder weights
├── output/reports/     EDA and results HTML reports, summaries, decisions log
└── docs/               concept note, modelling plan, EDA and results specifications, methods notes
```
