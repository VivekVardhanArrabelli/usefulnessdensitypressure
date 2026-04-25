"""Train a small LoRA DPO adapter for context-rich preference tuning."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import DPOConfig, DPOTrainer

from log_run import sha256_file, write_run_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = REPO_ROOT / "data" / "sample_preferences.jsonl"
DEFAULT_DECODE_CONFIG = REPO_ROOT / "configs" / "decode.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA DPO adapter.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--decode-config", default=str(DEFAULT_DECODE_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional trainer max_steps override.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Use QLoRA 4-bit loading.")
    parser.add_argument("--validate-only", action="store_true", help="Validate data/tokenization and exit.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Optional Trainer checkpoint path.")
    return parser.parse_args()


def load_preference_rows(path: str | Path) -> list[dict[str, list[dict[str, str]]]]:
    rows: list[dict[str, list[dict[str, str]]]] = []
    required = {"prompt", "chosen", "rejected"}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            missing = required - set(obj)
            if missing:
                raise ValueError(f"{path}:{line_number} missing required keys: {sorted(missing)}")
            for key in required:
                if not isinstance(obj[key], str) or not obj[key].strip():
                    raise ValueError(f"{path}:{line_number} field {key!r} must be a non-empty string")
            if obj["chosen"].strip() == obj["rejected"].strip():
                raise ValueError(f"{path}:{line_number} chosen and rejected must differ")
            rows.append(
                {
                    "prompt": [{"role": "user", "content": obj["prompt"]}],
                    "chosen": [{"role": "assistant", "content": obj["chosen"]}],
                    "rejected": [{"role": "assistant", "content": obj["rejected"]}],
                }
            )
    if not rows:
        raise ValueError(f"{path} did not contain any preference rows")
    return rows


def configure_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer has no chat_template; OLMo DPO should use apply_chat_template.")
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no eos_token; refusing to train with ambiguous EOS handling.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"eos_token={tokenizer.eos_token!r} eos_token_id={tokenizer.eos_token_id}")
    print(f"pad_token={tokenizer.pad_token!r} pad_token_id={tokenizer.pad_token_id}")
    print(f"padding_side={tokenizer.padding_side}")
    return tokenizer


def chat_token_count(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> int:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(tokens, "keys") and "input_ids" in tokens:
        input_ids = tokens["input_ids"]
        if input_ids is None:
            raise ValueError("apply_chat_template returned a dict without input_ids")
        if hasattr(input_ids, "shape"):
            return int(input_ids.shape[-1])
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)
    if hasattr(tokens, "shape"):
        return int(tokens.shape[-1])
    if tokens and isinstance(tokens[0], list):
        return len(tokens[0])
    return len(tokens)


def summarize_lengths(label: str, values: list[int]) -> None:
    ordered = sorted(values)
    mean = sum(values) / len(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    print(
        f"{label}: n={len(values)} min={ordered[0]} mean={mean:.1f} "
        f"p95={ordered[p95_index]} max={ordered[-1]}"
    )


def validate_preference_lengths(
    tokenizer: Any,
    rows: list[dict[str, list[dict[str, str]]]],
    max_prompt_length: int,
    max_length: int,
) -> None:
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    prompt_truncation_risks: list[int] = []
    response_truncation_risks: list[int] = []

    for index, row in enumerate(rows, start=1):
        prompt_len = chat_token_count(tokenizer, row["prompt"], add_generation_prompt=True)
        chosen_len = chat_token_count(
            tokenizer,
            row["prompt"] + row["chosen"],
            add_generation_prompt=False,
        )
        rejected_len = chat_token_count(
            tokenizer,
            row["prompt"] + row["rejected"],
            add_generation_prompt=False,
        )
        prompt_lengths.append(prompt_len)
        chosen_lengths.append(chosen_len)
        rejected_lengths.append(rejected_len)
        if prompt_len > max_prompt_length:
            prompt_truncation_risks.append(index)
        if chosen_len > max_length or rejected_len > max_length:
            response_truncation_risks.append(index)

    print("\n--- preference token length stats ---")
    summarize_lengths("prompt", prompt_lengths)
    summarize_lengths("prompt+chosen", chosen_lengths)
    summarize_lengths("prompt+rejected", rejected_lengths)
    if prompt_truncation_risks:
        print(
            "WARNING: prompt may be truncated for rows "
            f"{prompt_truncation_risks} with max_prompt_length={max_prompt_length}"
        )
    if response_truncation_risks:
        print(
            "WARNING: prompt+response may be truncated for rows "
            f"{response_truncation_risks} with max_length={max_length}"
        )
    print("--- end length stats ---\n")


def print_chat_template_preview(tokenizer: Any, first_row: dict[str, Any]) -> None:
    prompt_preview = tokenizer.apply_chat_template(
        first_row["prompt"], tokenize=False, add_generation_prompt=True
    )
    chosen_preview = tokenizer.apply_chat_template(
        first_row["prompt"] + first_row["chosen"],
        tokenize=False,
        add_generation_prompt=False,
    )
    print("\n--- chat prompt template preview ---")
    print(prompt_preview[:1200])
    print("--- chat chosen template preview ---")
    print(chosen_preview[:1200])
    print("--- end preview ---\n")


def build_model(model_name: str, load_in_4bit: bool):
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = compute_dtype
    else:
        compute_dtype = torch.float32
        model_kwargs["torch_dtype"] = torch.float32

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    return model


def output_dir_for(output_root: str | Path, beta: float, seed: int) -> Path:
    beta_text = str(beta).replace(".", "p")
    return Path(output_root) / f"olmo-context-rich-dpo-b{beta_text}-s{seed}"


def build_trainer(args: argparse.Namespace, dataset: Dataset, tokenizer: Any):
    output_dir = output_dir_for(args.output_root, args.beta, args.seed)
    model = build_model(args.model, args.load_in_4bit)
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    dpo_config_kwargs = {
        "output_dir": str(output_dir),
        "beta": args.beta,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
    }
    supported_args = set(inspect.signature(DPOConfig).parameters)
    filtered_kwargs = {key: value for key, value in dpo_config_kwargs.items() if key in supported_args}
    skipped_kwargs = sorted(set(dpo_config_kwargs) - set(filtered_kwargs))
    if skipped_kwargs:
        print(f"Skipping unsupported DPOConfig args for installed TRL: {skipped_kwargs}")
    training_args = DPOConfig(**filtered_kwargs)

    direct_trainer_kwargs: dict[str, Any] = {}
    trainer_signature = inspect.signature(DPOTrainer).parameters
    for key in skipped_kwargs:
        if key in trainer_signature:
            direct_trainer_kwargs[key] = dpo_config_kwargs[key]
    if direct_trainer_kwargs:
        print(f"Passing legacy DPOTrainer args directly: {sorted(direct_trainer_kwargs)}")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "peft_config": peft_config,
        **direct_trainer_kwargs,
    }
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        trainer = DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)
    return trainer, output_dir, target_modules


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    rows = load_preference_rows(args.dataset)
    tokenizer = configure_tokenizer(args.model)
    validate_preference_lengths(tokenizer, rows, args.max_prompt_length, args.max_length)
    print_chat_template_preview(tokenizer, rows[0])
    if args.validate_only:
        print("Validation completed; exiting before model load/training.")
        return
    dataset = Dataset.from_list(rows)

    trainer, output_dir, target_modules = build_trainer(args, dataset, tokenizer)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))

    config = {
        "model": args.model,
        "seed": args.seed,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "load_in_4bit": args.load_in_4bit,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "dataset_path": str(Path(args.dataset).resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "decode_config_path": str(Path(args.decode_config).resolve()),
        "decode_config_sha256": sha256_file(args.decode_config),
    }
    run_config_path = write_run_config(output_dir, config)
    print(f"Saved adapter to {output_dir}")
    print(f"Wrote run config to {run_config_path}")


if __name__ == "__main__":
    main()
