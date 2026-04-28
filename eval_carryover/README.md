# Carryover Eval

This eval tests whether visible productive context improves downstream work over multiple turns. It is the direct test of the large-model hypothesis: compressed answers can be good final responses but weak future context.

## Shape

Each task runs through four stages:

```text
frame -> generate -> stress_test -> synthesize
```

Each condition controls the style of visible context at each stage:

- `compressed_all`: concise useful output at every stage.
- `exploratory_all`: disciplined exploratory context at every stage.
- `exploratory_then_compressed`: exploratory framing/generation, concise later stages.
- `verbose_unstructured_all`: expansive detail at every stage, without disciplined exploratory structure.

The final artifact from `synthesize` is what gets judged. Intermediate stages are saved for audit, not primary scoring.

## Run A Pilot

```bash
python scripts/run_carryover_eval.py --pilot --out outputs/carryover_pilot_runs.jsonl
python scripts/build_carryover_pairwise_sheet.py --runs outputs/carryover_pilot_runs.jsonl --pairwise-out outputs/carryover_pilot_pairwise_judge_sheet.csv --condition-map-out outputs/carryover_pilot_pairwise_condition_map.csv
python scripts/build_carryover_judge_sheet.py --runs outputs/carryover_pilot_runs.jsonl --judge-out outputs/carryover_pilot_judge_sheet.csv --condition-map-out outputs/carryover_pilot_condition_map.csv
```

The pairwise judge sheet is the primary artifact. The scalar judge sheet is optional diagnostic support. Both are blinded. Do not inspect condition maps until after judging.

## Judge

For the pairwise sheet, compare two final artifacts for the same task and fill:

- `winner`: `a`, `b`, or `tie`
- `confidence`: 1-3
- `notes`: free text

For the optional scalar sheet, score final artifacts only:

- `overall_quality`: 1-5
- `novelty`: 1-5
- `useful_risk_taking`: 1-5
- `assumption_quality`: 1-5
- `uncertainty_quality`: 1-5
- `actionability`: 1-5
- `factual_grounding`: 1-5
- `verbosity_penalty`: 0-3
- `notes`: free text

Then summarize:

```bash
python scripts/summarize_carryover_pairwise.py --judgments outputs/carryover_pilot_pairwise_judge_sheet.csv --condition-map outputs/carryover_pilot_pairwise_condition_map.csv
python scripts/summarize_carryover_judgments.py --judgments outputs/carryover_pilot_judge_sheet.csv --condition-map outputs/carryover_pilot_condition_map.csv
```

## Interpretation

The key outcome is not whether exploratory outputs sound better. The key outcome is whether the final downstream artifact is better after visible exploratory context has had several turns to compound.

The headline metric is paired task win rate, for example:

```text
exploratory_then_compressed beats compressed_all on 15/24 tasks
```
