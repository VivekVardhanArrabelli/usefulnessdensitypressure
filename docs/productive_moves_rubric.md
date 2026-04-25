# Productive Moves Scoring Rubric

This rubric scores visible response behavior for prompt-only ablations and DPO pair construction. The target is not a fixed scaffold. The target is specific productive moves that improve the model's working context when a task is ambiguous, multi-step, or load-bearing.

## Scoring Unit

Score one model output for one prompt. Use the prompt's category from `eval/prompts.jsonl`.

Score only the visible answer. Do not infer hidden reasoning. A move earns credit only if it changes the useful working context, not merely because the output is longer or more formal.

## Productive Moves

| move | Present when... | Not present when... |
|---|---|---|
| `constraints_explicit` | The answer names task constraints, givens, assumptions, user goals, or success criteria that shape the solution. | It only paraphrases the prompt or adds generic preamble. |
| `alternatives_considered` | The answer considers two or more genuine approaches, frames, algorithms, tradeoffs, or interpretations. | It lists decorative options that do not affect the answer. |
| `approach_justified` | The answer chooses a method/frame and briefly explains why it fits the prompt. | It proceeds with a method without selection rationale, or just says "best" without grounds. |
| `uncertainty_preserved` | The answer marks ambiguity, assumptions, limits, or uncertainty in a calibrated way. | It hedges known facts, avoids committing to a determinate answer, or adds caveats for ceremony. |

## Category Appropriateness Rules

Use these rules before scoring. A move can be present but inappropriate.

| category | constraints_explicit | alternatives_considered | approach_justified | uncertainty_preserved |
|---|---|---|---|---|
| `factual_short` | No. Direct answer is preferred. | No. | No. | No, unless the fact is unknowable from the prompt. |
| `conversational` | Borderline: appropriate when it gently names the user's stated context or goal. | Usually no, unless the user explicitly asks for options. | Borderline: appropriate for choosing a supportive frame, not for rigid problem solving. | Yes, especially for emotional interpretation or tentative causes. |
| `explanation` | Borderline: appropriate when the prompt is ambiguous or level-of-detail matters. | Usually no. | Borderline: appropriate when choosing an analogy, level, or explanatory route. | Borderline: appropriate for contested topics, assumptions, or simplifications. |
| `advice` | Borderline: appropriate when it extracts decision constraints or user values. | Yes. | Yes. | Yes. |
| `analysis_open` | Yes. | Yes. | Yes. | Yes. |
| `reasoning_multistep` | Yes. | Borderline: appropriate only when there are multiple genuine solution methods. | Yes. | Usually no for determinate math/logic problems; yes only for underspecified inputs. |
| `coding_small` | Borderline: appropriate when the spec, data shape, or edge cases are ambiguous. | Yes when multiple algorithms/APIs are plausible. | Yes. | Rarely; appropriate for stated assumptions, schema uncertainty, version differences, or edge cases. |

Borderline means: mark the move appropriate only if it is materially useful for this specific prompt. Otherwise mark it inappropriate.

## Score Fields

For each output, fill these fields:

```text
prompt_id
category
condition
correctness
constraints_present
constraints_appropriate
alternatives_present
alternatives_appropriate
approach_present
approach_appropriate
uncertainty_present
uncertainty_appropriate
move_score
total_score
overframing_flag
rubric_issue
notes
```

Use binary scoring:

- `correctness = 1` if a reasonable practitioner would find the answer useful and not misleading.
- `correctness = 0` if the answer contains factual errors, misframes the question, fails to engage with what was asked, is unusable, or is materially worse than the prompt requires.
- If `correctness = 0`, write the reason in `notes`.
- Each move has `present = 0/1`.
- If a move is absent, its contribution is `0`.
- If a move is present and appropriate, its contribution is `+1`.
- If a move is present and inappropriate, its contribution is `-1`.

Formula:

```text
move_score = sum(move_contributions)
total_score = (3 * correctness) + move_score
```

Correctness weight is fixed at `3`. A wrong but well-structured answer must not beat a right plain answer.

Set `overframing_flag = 1` when the answer adds visible ceremony that the prompt type does not need, especially for `factual_short`. Overframing is already penalized through inappropriate moves, and it is also a hard veto for pair construction: an overframed candidate cannot be `chosen` regardless of score.

Use `rubric_issue` for free-text diagnostics when the prompt category, move definitions, or appropriateness rules do not fit the case. Review every `rubric_issue` row after scoring before changing the rubric or constructing data.

## Pair Construction Policy

For v1 DPO data, filter `factual_short` prompts out of pair generation. Keep them in eval as a regression guard. Do not add reverse factual pairs until after auditing the positive-pair distribution.

Generate a chosen/rejected pair only when all are true:

```text
candidate_correctness = 1
baseline_correctness = 1 or candidate is clearly more correct
candidate_total_score - baseline_total_score >= 2
candidate_overframing_flag = 0
```

Use the higher-scoring output as `chosen` and the lower-scoring output as `rejected`. Discard ties and ambiguous cases. Do not create a pair merely because the candidate follows a restate-list-pick scaffold.

Count comparison outcomes by score difference only:

- `win`: candidate_total_score - baseline_total_score >= 2
- `loss`: baseline_total_score - candidate_total_score >= 2
- `tie`: everything else

The current 24-prompt eval is a rubric calibration exercise, not a data-generation policy. Do not set category eligibility thresholds from this sample. After scoring, review category patterns qualitatively:

- structural wins consistently in a category: promising for later pair generation
- structural ties consistently in a category: weak signal, likely filter for v1
- structural loses consistently in a category: filter for v1, consider reverse-pair design later

Formal category eligibility thresholds belong to the larger 1000+ prompt run, where percentages are meaningful. Define those thresholds before generating DPO pairs from the larger run. Real data should come from applying this rubric, or a calibrated judge that follows it, to a larger prompt corpus.

## Scoring Workflow

- First score 3-4 calibration prompts that span different categories.
- Adjust the rubric at most once after that calibration pass.
- Then score the remaining outputs in one sitting without mid-scoring rubric changes.
- Publish the completed scores before deciding on DPO data.
- Review all `rubric_issue` rows before any data construction decision.

## Sanity Checks Before DPO

- Score vanilla seed 42, vanilla seed 43, abstract prompt, and structural prompt separately.
- Confirm that structural wins are driven by appropriate moves, not just extra formatting.
- Confirm that factual and other low-complexity prompts are not entering DPO data.
- Inspect categories where structural loses or ties; do not force data from categories without clear signal.
- Keep raw scoring sheets so later reverse-pair decisions can use measured over-application rates.
