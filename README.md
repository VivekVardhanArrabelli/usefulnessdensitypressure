# Context-Rich Visible-Context Experiments

Infrastructure for testing whether visible productive context improves downstream work. Current eval defaults use `Qwen/Qwen2.5-7B-Instruct`; the earlier `allenai/OLMo-2-0425-1B-Instruct` runs are retained as a 1B baseline.

## Hypothesis

Current instruction-tuning optimizes something close to "usefulness density per response": maximal task completion, minimal stylistic slack. The narrowed claim being tested is that compressed answers can be good final responses but weak future context. In multi-step work, visible text becomes part of the working substrate for later model turns, humans, agents, and collaborators.

The main experiment asks whether disciplined visible exploration improves downstream artifacts over several stages. The relevant moves are making constraints explicit, considering alternatives before committing, justifying approach selection, preserving uncertainty, and keeping useful adjacent hypotheses available. DPO is a later training path if the carryover eval shows signal. See `eval_carryover/` for the current eval and `docs/productive_moves_rubric.md` for the smaller rubric calibration work.

## Model Choice

- Current default: `Qwen/Qwen2.5-7B-Instruct`, used for the 7B prompt-only ablation after the 1B abstract-prompt result was indistinguishable from seed noise.
- Baseline: `allenai/OLMo-2-0425-1B-Instruct`, retained because it is a cheap Ai2 1B comparison point with public model card and training lineage.
- Qwen2.5-7B is large enough to follow concrete structural prompts while still fitting comfortably on the RTX 6000 Ada eval machine.

## Repo Layout

```text
.
|-- requirements.txt
|-- README.md
|-- configs/
|   `-- decode.json
|-- data/
|   `-- sample_preferences.jsonl
|-- docs/
|   `-- productive_moves_rubric.md
|-- eval/
|   |-- prompts.jsonl
|   `-- system_prompts.json
|-- eval_carryover/
|   |-- README.md
|   |-- tasks.jsonl
|   |-- stage_prompts.json
|   `-- conditions.json
|-- scripts/
|   |-- run_carryover_eval.py
|   |-- build_carryover_pairwise_sheet.py
|   |-- summarize_carryover_pairwise.py
|   |-- build_carryover_judge_sheet.py
|   |-- summarize_carryover_judgments.py
|   |-- train_dpo.py
|   |-- run_eval.py
|   |-- compare_outputs.py
|   `-- log_run.py
`-- outputs/
    `-- .gitkeep
```

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

On Linux or a CUDA box, replace `py` with `python` if that is your launcher.

## Data

Preference data is JSONL, one object per line:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Fields are raw strings, not pre-templated chat transcripts. `scripts/train_dpo.py` converts them into TRL's conversational DPO format and lets the selected model tokenizer apply its chat template.

The real dataset target is 500-2000 high-confidence pairs minimum. Sourcing TODOs:

- Hand-curated pairs for high-precision productive-move scoring.
- Synthetic pairs from a larger model, then manually filtered.
- Mixed hand-curated plus synthetic data.

Pair construction rule: generate pairs only when the scoring rubric shows an unambiguous improvement. For v1 DPO data, filter `factual_short` prompts out of pair generation and keep them in eval as a regression guard. Reverse factual pairs can be added later only after auditing the positive-pair distribution.

`train_dpo.py` validates that every row has non-empty `prompt`, `chosen`, and `rejected` fields, verifies `chosen != rejected`, and prints prompt/response token length stats before model loading. It warns when rows exceed `max_prompt_length` or `max_length`, since silent truncation can invalidate a DPO run.

## Carryover Eval

The primary next experiment is a four-stage visible-context carryover eval:

```text
frame -> generate -> stress_test -> synthesize
```

Run a small pilot first:

```powershell
py scripts/run_carryover_eval.py --pilot --out outputs/carryover_pilot_runs.jsonl
py scripts/build_carryover_pairwise_sheet.py --runs outputs/carryover_pilot_runs.jsonl --pairwise-out outputs/carryover_pilot_pairwise_judge_sheet.csv --condition-map-out outputs/carryover_pilot_pairwise_condition_map.csv
py scripts/build_carryover_judge_sheet.py --runs outputs/carryover_pilot_runs.jsonl --judge-out outputs/carryover_pilot_judge_sheet.csv --condition-map-out outputs/carryover_pilot_condition_map.csv
```

Fill the blinded pairwise judge sheet without opening the condition map. Then summarize:

```powershell
py scripts/summarize_carryover_pairwise.py --judgments outputs/carryover_pilot_pairwise_judge_sheet.csv --condition-map outputs/carryover_pilot_pairwise_condition_map.csv
```

Judge only final `synthesize` artifacts. Intermediate stage outputs are saved for audit.

## Prompt-Only Ablation First

Before training, run the eval battery twice on the base model:

```powershell
py scripts/run_eval.py --system-prompt vanilla --model-label base_vanilla --out outputs/base_vanilla.jsonl
py scripts/run_eval.py --system-prompt wandered --model-label base_wandered --out outputs/base_wandered.jsonl
py scripts/compare_outputs.py --base outputs/base_vanilla.jsonl --tuned outputs/base_wandered.jsonl --out outputs/prompt_ablation_compare.csv
```

Score the CSV manually. If the system prompt captures most of the stylistic delta, the DPO question becomes whether training locks in what prompting already does.

All eval generation reads decoding settings from `configs/decode.json`. Do not pass ad hoc decode settings to one side of a comparison.

## Train

Starter run:

```powershell
py scripts/train_dpo.py --dataset data/sample_preferences.jsonl --beta 0.1 --seed 42 --load-in-4bit
```

Training defaults are tuned for a 1B LoRA DPO run rather than a larger-model survival setting: per-device batch size `4`, gradient accumulation `2`, learning rate `5e-5`, `max_prompt_length=1024`, and `max_length=2048`. Training and generation both use the tokenizer chat template with left padding, matching current TRL DPO expectations for the tokenizer passed as `processing_class`.

Before spending a full GPU run, validate formatting and token lengths:

```powershell
py scripts/train_dpo.py --dataset data/sample_preferences.jsonl --validate-only
```

Then run a short GPU smoke test:

```powershell
py scripts/train_dpo.py --dataset data/sample_preferences.jsonl --beta 0.1 --seed 42 --load-in-4bit --max-steps 2
```

Suggested first sweep on a GPU machine:

```powershell
py scripts/train_dpo.py --dataset data/sample_preferences.jsonl --beta 0.1 --seed 42 --load-in-4bit
py scripts/train_dpo.py --dataset data/sample_preferences.jsonl --beta 0.3 --seed 42 --load-in-4bit
py scripts/train_dpo.py --dataset data/sample_preferences.jsonl --beta 0.5 --seed 42 --load-in-4bit
py scripts/train_dpo.py --dataset data/sample_preferences.jsonl --beta 0.1 --seed 43 --load-in-4bit
```

Adapters are saved under:

```text
outputs/olmo-context-rich-dpo-b{beta}-s{seed}
```

Each run also writes `run_config.json` with model, seed, beta, LoRA settings, dataset hash, decode config hash, and timestamp.

## Evaluate Tuned Adapters

```powershell
py scripts/run_eval.py --system-prompt vanilla --model-label base_vanilla --out outputs/base_vanilla.jsonl
py scripts/run_eval.py --adapter outputs/olmo-context-rich-dpo-b0p1-s42 --system-prompt vanilla --model-label tuned_b0.1_s42 --out outputs/tuned_b0p1_s42.jsonl
py scripts/compare_outputs.py --base outputs/base_vanilla.jsonl --tuned outputs/tuned_b0p1_s42.jsonl --out outputs/tuned_b0p1_s42_compare.csv
```

If decode config hashes differ, `compare_outputs.py` warns and still writes the CSV. Use `--allow-decode-mismatch` to mark that comparison as intentional, or `--strict-decode-match` when you want mismatch to fail.

The comparison CSV columns are:

```text
id, category, prompt, base_output, tuned_output, base_tokens, tuned_tokens,
correctness_base, correctness_tuned, naturalness_base, naturalness_tuned,
context_richness_base, context_richness_tuned, rambling_base, rambling_tuned,
winner, notes
```

`base_tokens` and `tuned_tokens` are required because verbosity is the main week-1 confound.

## Known Failure Modes

- Verbosity confound: if tuned outputs are systematically longer, `context_richness` may favor tuned for trivial reasons. Length-matched comparison in week 2 separates style-matters from more-tokens-helps.
- Surface-token learning: DPO may learn em dashes, hedging phrases, or "wandering" markers without the exploratory move underneath. Score productive lateral moves, not vibes.
- Seed variance at 1B is loud. Use at least 2 seeds per config on real runs.
- Eval contamination: do not use HumanEval-style benchmarks at this scale.
- Chat template, EOS, and pad token mismatches can silently poison base-vs-DPO comparison. `train_dpo.py` prints a template preview and token IDs before training.
- Sycophantic-slop failure: if structurally framed answers are preferred universally, the model becomes verbose on short factual questions. Mitigate this in v1 by filtering `factual_short` prompts from pair construction and keeping them in eval as a regression guard.

## Local Checks

These checks do not require GPU or model download:

```powershell
py -m py_compile scripts/log_run.py scripts/compare_outputs.py scripts/run_eval.py scripts/train_dpo.py
```

Tiny comparison smoke test:

```powershell
py scripts/compare_outputs.py --base outputs/smoke_base.jsonl --tuned outputs/smoke_tuned.jsonl --out outputs/smoke_compare.csv
```

## References

- OLMo model card: https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
- TRL DPO trainer docs: https://huggingface.co/docs/trl/dpo_trainer
- TRL PEFT integration docs: https://huggingface.co/docs/trl/peft_integration
- Transformers bitsandbytes docs: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
