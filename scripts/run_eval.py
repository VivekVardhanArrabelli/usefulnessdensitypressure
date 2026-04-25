"""Run base or LoRA-adapted generation on the shared eval prompt set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from log_run import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PROMPTS = REPO_ROOT / "eval" / "prompts.jsonl"
DEFAULT_SYSTEM_PROMPTS = REPO_ROOT / "eval" / "system_prompts.json"
DEFAULT_DECODE_CONFIG = REPO_ROOT / "configs" / "decode.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval prompts through a model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--system-prompts", default=str(DEFAULT_SYSTEM_PROMPTS))
    parser.add_argument("--system-prompt", default="vanilla", help="Key in eval/system_prompts.json.")
    parser.add_argument("--decode-config", default=str(DEFAULT_DECODE_CONFIG))
    parser.add_argument("--out", required=True, help="JSONL file to write.")
    parser.add_argument("--model-label", default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required = {"id", "category", "prompt"}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing required keys: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any prompts")
    return rows


def configure_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer has no chat_template; eval should use apply_chat_template.")
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


def print_chat_template_preview(tokenizer: Any, system_prompt: str, first_prompt: str) -> None:
    rendered = tokenizer.apply_chat_template(
        build_messages(system_prompt, first_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    print("\n--- chat template preview ---")
    print(rendered[:1200])
    print("--- end preview ---\n")


def load_model(model_name: str, adapter: str | None):
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_messages(system_prompt: str, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


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


def validate_decode_config(config: dict[str, Any]) -> None:
    required = {"temperature", "top_p", "max_new_tokens", "do_sample", "seed"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Decode config missing keys: {sorted(missing)}")


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    system_prompt: str,
    decode_config: dict[str, Any],
) -> tuple[str, int]:
    rendered = tokenizer.apply_chat_template(
        build_messages(system_prompt, prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model_device(model))
    gen_kwargs = generation_kwargs(decode_config, tokenizer)
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
    prompt_tokens = inputs["input_ids"].shape[-1]
    new_tokens = generated[0, prompt_tokens:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return output, int(new_tokens.shape[-1])


def default_model_label(adapter: str | None, system_prompt_key: str) -> str:
    if adapter:
        return Path(adapter).name
    return f"base_{system_prompt_key}"


def main() -> None:
    args = parse_args()
    decode_config = load_json(args.decode_config)
    validate_decode_config(decode_config)
    set_seed(int(decode_config["seed"]))

    system_prompts = load_json(args.system_prompts)
    if args.system_prompt not in system_prompts:
        raise ValueError(f"Unknown system prompt key {args.system_prompt!r}")
    system_prompt = str(system_prompts[args.system_prompt])

    prompts = load_prompts(args.prompts)
    tokenizer = configure_tokenizer(args.model)
    print_chat_template_preview(tokenizer, system_prompt, prompts[0]["prompt"])
    model = load_model(args.model, args.adapter)
    model_label = args.model_label or default_model_label(args.adapter, args.system_prompt)
    decode_hash = sha256_file(args.decode_config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(prompts, desc=f"eval:{model_label}"):
            output, token_count = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                system_prompt=system_prompt,
                decode_config=decode_config,
            )
            payload = {
                "id": row["id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "model_label": model_label,
                "output": output,
                "output_token_count": token_count,
                "decode_config_hash": decode_hash,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} generations to {out_path}")


if __name__ == "__main__":
    main()
