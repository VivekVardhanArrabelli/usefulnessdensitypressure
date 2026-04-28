"""Run the multi-stage visible-context carryover eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from log_run import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TASKS = REPO_ROOT / "eval_carryover" / "tasks.jsonl"
DEFAULT_STAGES = REPO_ROOT / "eval_carryover" / "stage_prompts.json"
DEFAULT_CONDITIONS = REPO_ROOT / "eval_carryover" / "conditions.json"
DEFAULT_DECODE_CONFIG = REPO_ROOT / "configs" / "decode_carryover.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "carryover_runs.jsonl"
STRATIFIED_PILOT_TASK_IDS = [
    "science_001",
    "science_002",
    "coding_001",
    "critique_001",
    "strategy_001",
    "weird_001",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a four-stage carryover eval.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--stage-prompts", default=str(DEFAULT_STAGES))
    parser.add_argument("--conditions", default=str(DEFAULT_CONDITIONS))
    parser.add_argument("--decode-config", default=str(DEFAULT_DECODE_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--model-label", default=None)
    parser.add_argument(
        "--condition",
        action="append",
        help="Condition key to run. Repeat for multiple. Defaults to all conditions.",
    )
    parser.add_argument("--task-id", action="append", help="Task id to run. Repeat for multiple.")
    parser.add_argument("--task-limit", type=int, default=None, help="Run only the first N tasks.")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the 6-task stratified pilot across science/coding/critique/strategy/weird-result domains.",
    )
    parser.add_argument("--pilot-first", action="store_true", help="Run the first 6 tasks.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_tasks(path: str | Path) -> list[dict[str, str]]:
    required = {"id", "domain", "title", "seed_task", "final_artifact"}
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contained no tasks")
    return rows


def select_tasks(tasks: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = tasks
    if args.pilot and args.task_id:
        raise ValueError("--pilot cannot be combined with --task-id")
    if args.pilot_first and args.task_id:
        raise ValueError("--pilot-first cannot be combined with --task-id")
    if args.pilot and args.pilot_first:
        raise ValueError("--pilot and --pilot-first are mutually exclusive")
    if args.pilot:
        by_id = {task["id"]: task for task in selected}
        missing = [task_id for task_id in STRATIFIED_PILOT_TASK_IDS if task_id not in by_id]
        if missing:
            raise ValueError(f"Stratified pilot task ids missing from task file: {missing}")
        selected = [by_id[task_id] for task_id in STRATIFIED_PILOT_TASK_IDS]
    elif args.task_id:
        requested = set(args.task_id)
        selected = [task for task in selected if task["id"] in requested]
        found = {task["id"] for task in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Unknown task ids: {missing}")
    limit = 6 if args.pilot_first else args.task_limit
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No tasks selected")
    return selected


def select_conditions(config: dict[str, Any], requested: list[str] | None) -> dict[str, Any]:
    conditions = config.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("conditions config must contain a non-empty 'conditions' object")
    if not requested:
        return conditions
    missing = sorted(set(requested) - set(conditions))
    if missing:
        raise ValueError(f"Unknown conditions: {missing}")
    return {key: conditions[key] for key in requested}


def validate_stage_config(stage_config: dict[str, Any]) -> list[dict[str, str]]:
    stages = stage_config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stage prompt config must contain a non-empty 'stages' list")
    required = {"id", "name", "instruction"}
    for index, stage in enumerate(stages, start=1):
        missing = required - set(stage)
        if missing:
            raise ValueError(f"stage {index} missing fields: {sorted(missing)}")
    return stages


def validate_decode_config(config: dict[str, Any]) -> None:
    required = {"temperature", "top_p", "max_new_tokens", "do_sample", "seed"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Decode config missing keys: {sorted(missing)}")


def configure_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer has no chat_template; carryover eval uses apply_chat_template.")
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no eos_token; refusing ambiguous generation stop handling.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"tokenizer={tokenizer.__class__.__name__}")
    print(f"eos_token={tokenizer.eos_token!r} eos_token_id={tokenizer.eos_token_id}")
    print(f"pad_token={tokenizer.pad_token!r} pad_token_id={tokenizer.pad_token_id}")
    print(f"padding_side={tokenizer.padding_side}")
    return tokenizer


def load_model(model_name: str, adapter: str | None):
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generation_kwargs(decode_config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    do_sample = bool(decode_config["do_sample"])
    kwargs = {
        "max_new_tokens": int(decode_config["max_new_tokens"]),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(decode_config["temperature"])
        kwargs["top_p"] = float(decode_config["top_p"])
    return kwargs


def prior_context(stage_outputs: list[dict[str, Any]]) -> str:
    if not stage_outputs:
        return "No prior stage output yet."
    parts = []
    for stage in stage_outputs:
        parts.append(f"[{stage['stage_id']}]\n{stage['output']}")
    return "\n\n".join(parts)


def build_user_prompt(task: dict[str, str], stage: dict[str, str], prior_outputs: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            "This is a multi-stage work session. Your output will become context for later stages.",
            f"Domain: {task['domain']}",
            f"Title: {task['title']}",
            f"Seed task:\n{task['seed_task']}",
            f"Requested final artifact:\n{task['final_artifact']}",
            f"Stage instruction:\n{stage['instruction']}",
            f"Prior stage outputs:\n{prior_context(prior_outputs)}",
            "Write only the current stage output.",
        ]
    )


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    decode_config: dict[str, Any],
) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model_device(model))
    gen_kwargs = generation_kwargs(decode_config, tokenizer)
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
    prompt_tokens = inputs["input_ids"].shape[-1]
    new_tokens = generated[0, prompt_tokens:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return output, int(new_tokens.shape[-1])


def run_task_condition(
    model: torch.nn.Module,
    tokenizer: Any,
    task: dict[str, str],
    condition_key: str,
    condition: dict[str, Any],
    styles: dict[str, str],
    stages: list[dict[str, str]],
    decode_config: dict[str, Any],
) -> dict[str, Any]:
    stage_styles = condition.get("stage_styles")
    if not isinstance(stage_styles, dict):
        raise ValueError(f"{condition_key} missing stage_styles")

    stage_outputs: list[dict[str, Any]] = []
    for stage in stages:
        stage_id = stage["id"]
        style_key = stage_styles.get(stage_id)
        if style_key not in styles:
            raise ValueError(f"{condition_key}.{stage_id} references unknown style {style_key!r}")
        user_prompt = build_user_prompt(task, stage, stage_outputs)
        output, token_count = generate_one(
            model=model,
            tokenizer=tokenizer,
            system_prompt=styles[style_key],
            user_prompt=user_prompt,
            decode_config=decode_config,
        )
        stage_outputs.append(
            {
                "stage_id": stage_id,
                "stage_name": stage["name"],
                "style": style_key,
                "user_prompt": user_prompt,
                "output": output,
                "output_token_count": token_count,
            }
        )

    final_stage = stage_outputs[-1]
    return {
        "task_id": task["id"],
        "domain": task["domain"],
        "title": task["title"],
        "seed_task": task["seed_task"],
        "final_artifact": task["final_artifact"],
        "condition": condition_key,
        "condition_description": condition.get("description", ""),
        "stages": stage_outputs,
        "final_output": final_stage["output"],
        "final_output_token_count": final_stage["output_token_count"],
    }


def model_label(args: argparse.Namespace) -> str:
    if args.model_label:
        return args.model_label
    if args.adapter:
        return Path(args.adapter).name
    return args.model


def main() -> None:
    args = parse_args()
    tasks = select_tasks(load_tasks(args.tasks), args)
    stage_config = load_json(args.stage_prompts)
    stages = validate_stage_config(stage_config)
    condition_config = load_json(args.conditions)
    styles = condition_config.get("styles")
    if not isinstance(styles, dict) or not styles:
        raise ValueError("conditions config must contain a non-empty 'styles' object")
    conditions = select_conditions(condition_config, args.condition)
    decode_config = load_json(args.decode_config)
    validate_decode_config(decode_config)
    set_seed(int(decode_config["seed"]))

    tokenizer = configure_tokenizer(args.model)
    model = load_model(args.model, args.adapter)
    decode_hash = sha256_file(args.decode_config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        total = len(tasks) * len(conditions)
        with tqdm(total=total, desc="carryover") as progress:
            for task in tasks:
                for condition_key, condition in conditions.items():
                    payload = run_task_condition(
                        model=model,
                        tokenizer=tokenizer,
                        task=task,
                        condition_key=condition_key,
                        condition=condition,
                        styles=styles,
                        stages=stages,
                        decode_config=decode_config,
                    )
                    payload.update(
                        {
                            "model": args.model,
                            "model_label": model_label(args),
                            "decode_config_hash": decode_hash,
                        }
                    )
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    progress.update(1)
    print(f"Wrote carryover generations to {out_path}")


if __name__ == "__main__":
    main()
