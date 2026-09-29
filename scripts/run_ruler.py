"""
RULER runner for KVPruner — reuses our monkeypatched LLaMA + generate pipeline.

Reads RULER-prepared jsonl files ({index, input, outputs, length, ...}),
writes prediction jsonl ({index, input, outputs, pred, length, ...}) per task.

Usage:
    python scripts/run_ruler.py \
        --model_path <path> \
        --method grik \
        --max_capacity_prompts 128 --pruning_ratio 0.4 \
        --data_dir benchmark_root/.../4096/data \
        --save_dir benchmark_root/.../4096/pred \
        --task niah_single_1
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.monkeypatch import replace_model, detect_model_type


METHOD_TO_MODE = {
    'grik': ('grik', 1.0),
}

# RULER's generation config per task category
TOKENS_TO_GENERATE = {
    "niah_single_1": 128, "niah_single_2": 128, "niah_single_3": 128,
    "niah_multikey_1": 128, "niah_multikey_2": 128, "niah_multikey_3": 128,
    "niah_multivalue": 128, "niah_multiquery": 128,
    "vt": 30, "cwe": 120, "fwe": 50,
    "qa_1": 32, "qa_2": 32,
}


def apply_method(model, method, model_type, kv_budget, ratio, recent_size, window_size, kernel_size, pooling, prefill_sdpa):
    mode, tau = METHOD_TO_MODE[method]
    replace_model(method, model_type=model_type)
    for i, layer in enumerate(model.model.layers):
        cfg = layer.self_attn.config
        cfg.window_size = window_size
        cfg.max_capacity_prompt = kv_budget
        cfg.kernel_size = kernel_size
        cfg.pooling = pooling
        cfg.ratio = ratio
        cfg.recent_size = recent_size
        cfg.think_pruner_mode = mode
        cfg.think_pruner_tau = tau
        cfg.prefill_sdpa = bool(prefill_sdpa)
        if hasattr(layer.self_attn, "kv_seq_len"):
            layer.self_attn.kv_seq_len = 0


def reset_attention_state(model):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "kv_seq_len"):
            layer.self_attn.kv_seq_len = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--method", required=True, choices=list(METHOD_TO_MODE))
    parser.add_argument("--max_capacity_prompts", type=int, default=128)
    parser.add_argument("--pruning_ratio", type=float, default=0.4)
    parser.add_argument("--recent_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", default="maxpool")
    parser.add_argument("--data_dir", required=True, help="Directory with <task>/validation.jsonl")
    parser.add_argument("--save_dir", required=True, help="Directory to write <task>.jsonl predictions")
    parser.add_argument("--task", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefill_sdpa", action="store_true",
                        help="Use SDPA for the uncompressed prefill output path while keeping the patched decode/cache path.")
    args = parser.parse_args()

    data_file = Path(args.data_dir) / args.task / "validation.jsonl"
    save_file = Path(args.save_dir) / f"{args.task}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume by line offset: RULER's `index` field is NOT unique across
    # validation lines (same index can map to distinct prompts). Use the
    # number of already-written prediction lines as the offset into
    # validation, so duplicates are not skipped.
    done_lines = 0
    if save_file.exists():
        with open(save_file) as f:
            for line in f:
                if line.strip():
                    done_lines += 1

    with open(data_file) as f:
        data = [json.loads(line) for line in f if line.strip()]
    total = len(data)
    data = data[done_lines:]
    if args.max_samples:
        data = data[: args.max_samples]

    if not data:
        print(f"[{args.task}] nothing to do (all {done_lines}/{total} already predicted)")
        return
    print(f"[{args.task}] resuming from line {done_lines}/{total} ({len(data)} remaining)")

    print(f"[{args.task}] model={args.method} kv={args.max_capacity_prompts} r={args.pruning_ratio} samples={len(data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="auto", attn_implementation="eager",
    )
    model_type = detect_model_type(args.model_path)
    apply_method(model, args.method, model_type, args.max_capacity_prompts,
                 args.pruning_ratio, args.recent_size, args.window_size, args.kernel_size,
                 args.pooling, args.prefill_sdpa)
    model.eval()

    max_new = TOKENS_TO_GENERATE.get(args.task, 128)

    with open(save_file, "a", encoding="utf-8", buffering=1) as fout:
        for sample in tqdm(data, desc=args.task):
            reset_attention_state(model)
            inputs = tokenizer(sample["input"], return_tensors="pt", add_special_tokens=False).to(model.device)
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=max_new,
                        do_sample=False, num_beams=1,
                        temperature=1.0, top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                pred = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                pred = "<OOM>"
                torch.cuda.empty_cache()
                gc.collect()

            fout.write(json.dumps({
                "index": sample["index"],
                "input": sample["input"],
                "outputs": sample["outputs"],
                "pred": pred,
                "length": sample.get("length", -1),
                "others": sample.get("others", {}),
                "truncation": sample.get("truncation", -1),
            }) + "\n")


if __name__ == "__main__":
    main()
