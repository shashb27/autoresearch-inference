"""
submit_run.py — Submit your run to the community leaderboard.

Reads local run data (results.tsv, hardware.json, config.json) and writes
a JSON submission to leaderboard/runs/. Commit and PR it to share your results.

Usage:
    uv run submit_run.py                            # anonymous submission
    uv run submit_run.py --contributor "yourname"   # with attribution
    uv run submit_run.py --branch "autoresearch/mar24"
    uv run submit_run.py --dry-run                  # preview without writing
"""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_DIR = os.path.join(PROJECT_DIR, "leaderboard", "runs")

# Known model parameter counts (params_b) for common models
MODEL_PARAMS = {
    "qwen2.5-0.5b": 0.5, "qwen2.5-1.5b": 1.5, "qwen2.5-3b": 3.0,
    "qwen2.5-7b": 7.0, "qwen2.5-14b": 14.0, "qwen2.5-32b": 32.0,
    "qwen2.5-72b": 72.0,
    "llama-3.1-8b": 8.0, "llama-3.1-70b": 70.0, "llama-3.1-405b": 405.0,
    "llama-3-8b": 8.0, "llama-3-70b": 70.0,
    "mistral-7b": 7.0, "mixtral-8x7b": 46.7, "mixtral-8x22b": 141.0,
    "gemma-2b": 2.0, "gemma-7b": 7.0, "gemma-2-9b": 9.0, "gemma-2-27b": 27.0,
    "phi-3-mini": 3.8, "phi-3-small": 7.0, "phi-3-medium": 14.0,
}

GPU_FAMILIES = {
    "7.0": "Volta", "7.5": "Turing",
    "8.0": "Ampere", "8.6": "Ampere", "8.7": "Ampere",
    "8.9": "Ada Lovelace",
    "9.0": "Hopper",
    "10.0": "Blackwell",
}


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _model_slug(model_id: str) -> str:
    return model_id.lower().replace("/", "-").replace(".", "-").replace("_", "-")


def _gpu_slug(gpu_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", gpu_name.lower()).strip("-")


def _infer_params_b(model_id: str) -> float:
    """Guess parameter count from model ID string."""
    needle = model_id.lower()
    for key, params in MODEL_PARAMS.items():
        if key in needle:
            return params
    # Fallback: look for patterns like "7b", "13b", "70b"
    match = re.search(r"(\d+(?:\.\d+)?)b", needle)
    if match:
        return float(match.group(1))
    return 0.0


def _infer_model_family(model_id: str) -> str:
    """Guess model family from model ID."""
    mid = model_id.lower()
    if "qwen" in mid:
        # Extract version: Qwen2.5 -> "Qwen2.5"
        m = re.search(r"qwen(\d+(?:\.\d+)?)", mid)
        return f"Qwen{m.group(1)}" if m else "Qwen"
    if "llama" in mid:
        m = re.search(r"llama-?(\d+(?:\.\d+)?)", mid)
        return f"Llama {m.group(1)}" if m else "Llama"
    if "mistral" in mid:
        return "Mistral"
    if "mixtral" in mid:
        return "Mixtral"
    if "gemma" in mid:
        m = re.search(r"gemma-?(\d+)", mid)
        return f"Gemma {m.group(1)}" if m else "Gemma"
    if "phi" in mid:
        return "Phi"
    return model_id.split("/")[-1].split("-")[0].title()


def _load_experiments(tsv_path: str) -> list:
    """Parse results.tsv into a list of experiment dicts."""
    if not os.path.exists(tsv_path):
        return []
    experiments = []
    with open(tsv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return []
    for i, line in enumerate(lines[1:]):
        parts = line.strip().split("\t")
        if len(parts) < 6:
            continue
        commit, tok_s_str, ttft_str, vram_str, status, *desc_parts = parts
        desc = "\t".join(desc_parts)
        try:
            tok_s = float(tok_s_str)
            ttft_ms = float(ttft_str)
            vram_gb = float(vram_str)
        except ValueError:
            continue
        experiments.append({
            "n": i,
            "tok_s": tok_s,
            "ttft_ms": ttft_ms,
            "peak_vram_gb": vram_gb,
            "status": status,
            "description": desc,
        })
    return experiments


def _extract_techniques(description: str) -> list:
    """Guess which techniques were used from the best config description."""
    desc = description.lower()
    techniques = []
    if "bf16" in desc or "bfloat16" in desc:
        techniques.append("bf16")
    if "fp8" in desc:
        techniques.append("fp8_quantization")
    if "int8" in desc:
        techniques.append("int8_quantization")
    if "int4" in desc:
        techniques.append("int4_quantization")
    if "flash" in desc or "fa2" in desc:
        techniques.append("flash_attention_2")
    if "custom decode" in desc or "custom_decode" in desc:
        techniques.append("custom_decode_loop")
    if "compile" in desc or "torch.compile" in desc:
        techniques.append("torch_compile")
    if "dynamic" in desc:
        techniques.append("dynamic_shapes")
    if "dynamo" in desc:
        techniques.append("dynamo_cache_tuning")
    if "sdpa" in desc:
        techniques.append("sdpa")
    if "cuda graph" in desc:
        techniques.append("cuda_graphs")
    if "return_dict" in desc:
        techniques.append("return_dict_false")
    if "static cache" in desc:
        techniques.append("static_kv_cache")
    if "tf32" in desc:
        techniques.append("tf32")
    return techniques or ["baseline"]


def build_submission(
    contributor: str,
    branch: str,
    tsv_path: str,
    hardware_path: str,
    config_path: str,
) -> dict:
    """Build a submission dict from local run data."""
    hardware = _load_json(hardware_path)
    config = _load_json(config_path)
    experiments = _load_experiments(tsv_path)

    if not experiments:
        print("ERROR: No experiments found in results.tsv. Run at least one experiment first.")
        sys.exit(1)

    # Find baseline and best
    keeps = [e for e in experiments if e["status"] == "keep"]
    if not keeps:
        print("ERROR: No 'keep' experiments in results.tsv.")
        sys.exit(1)

    baseline_tok_s = keeps[0]["tok_s"]
    best = max(keeps, key=lambda e: e["tok_s"])
    best_tok_s = best["tok_s"]
    gain_pct = round((best_tok_s / baseline_tok_s - 1) * 100, 1) if baseline_tok_s > 0 else 0.0

    # Model info
    model_id = config.get("model_id", "unknown/unknown")
    params_b = _infer_params_b(model_id)
    model_family = _infer_model_family(model_id)

    # Hardware info
    cap = hardware.get("compute_capability", "0.0")
    gpu_family = GPU_FAMILIES.get(cap, f"sm_{cap}")

    # Submission ID
    gpu_slug = _gpu_slug(hardware.get("gpu_name", "unknown-gpu"))
    model_s = _model_slug(model_id).replace("--", "-")
    date_s = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = f"{gpu_slug}-{model_s}-{date_s}-{str(uuid.uuid4())[:6]}"

    submission = {
        "run_id": run_id,
        "submitted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contributor": contributor,
        "branch": branch,
        "hardware": {
            "gpu_name": hardware.get("gpu_name", "Unknown GPU"),
            "gpu_family": gpu_family,
            "compute_capability": cap,
            "vram_total_gb": hardware.get("vram_total_gb", 0.0),
            "cuda_version": hardware.get("cuda_version", ""),
            "bf16_supported": hardware.get("bf16_supported", False),
            "fp8_supported": hardware.get("fp8_supported", False),
        },
        "model": {
            "id": model_id,
            "params_b": params_b,
            "family": model_family,
        },
        "results": {
            "baseline_tok_s": baseline_tok_s,
            "best_tok_s": best_tok_s,
            "gain_pct": gain_pct,
            "best_ttft_ms": best.get("ttft_ms", 0.0),
            "best_vram_gb": best.get("peak_vram_gb", 0.0),
            "total_experiments": len(experiments),
            "keep_count": len([e for e in experiments if e["status"] == "keep"]),
            "discard_count": len([e for e in experiments if e["status"] == "discard"]),
            "crash_count": len([e for e in experiments if e["status"] == "crash"]),
        },
        "best_config": {
            "description": best["description"],
            "techniques": _extract_techniques(best["description"]),
        },
        "experiments": [
            {k: v for k, v in e.items() if k != "peak_vram_gb"}
            for e in experiments
        ],
    }
    return submission


def main():
    parser = argparse.ArgumentParser(
        description="Submit run results to the community leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run submit_run.py
  uv run submit_run.py --contributor "yourname"
  uv run submit_run.py --branch "autoresearch/mar24-h100"
  uv run submit_run.py --dry-run
        """,
    )
    parser.add_argument("--contributor", default="anonymous",
                        help="Your name or handle (default: anonymous)")
    parser.add_argument("--branch",
                        help="Experiment branch name (auto-detected from git if omitted)")
    parser.add_argument("--tsv", default="results.tsv",
                        help="Path to results.tsv (default: results.tsv)")
    parser.add_argument("--hardware", default="hardware.json",
                        help="Path to hardware.json (default: hardware.json)")
    parser.add_argument("--config", default="config.json",
                        help="Path to config.json (default: config.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print submission JSON without writing to disk")
    args = parser.parse_args()

    # Auto-detect branch
    branch = args.branch
    if not branch:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=PROJECT_DIR
            )
            branch = result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            branch = "unknown"

    print("=" * 55)
    print("  autoresearch-inference — Leaderboard Submission")
    print("=" * 55)
    print(f"  Contributor : {args.contributor}")
    print(f"  Branch      : {branch}")
    print(f"  Results     : {args.tsv}")
    print()

    submission = build_submission(
        contributor=args.contributor,
        branch=branch,
        tsv_path=os.path.join(PROJECT_DIR, args.tsv),
        hardware_path=os.path.join(PROJECT_DIR, args.hardware),
        config_path=os.path.join(PROJECT_DIR, args.config),
    )

    r = submission["results"]
    hw = submission["hardware"]
    m = submission["model"]
    print(f"  GPU         : {hw['gpu_name']} ({hw['vram_total_gb']} GB)")
    print(f"  Model       : {m['id']} ({m['params_b']}B params)")
    print(f"  Baseline    : {r['baseline_tok_s']:.2f} tok/s")
    print(f"  Best        : {r['best_tok_s']:.2f} tok/s (+{r['gain_pct']:.1f}%)")
    print(f"  Best TTFT   : {r['best_ttft_ms']:.2f} ms")
    print(f"  Experiments : {r['total_experiments']} total "
          f"({r['keep_count']} keep, {r['discard_count']} discard, {r['crash_count']} crash)")
    print(f"  Best config : {submission['best_config']['description']}")
    print()

    if args.dry_run:
        print("--- DRY RUN — submission JSON (not written) ---")
        print(json.dumps(submission, indent=2))
        return

    # Write submission
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    filename = f"{submission['run_id']}.json"
    out_path = os.path.join(LEADERBOARD_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"  Written to: leaderboard/runs/{filename}")
    print()
    print("  Next steps:")
    print("    git add leaderboard/runs/" + filename)
    print("    git commit -m 'leaderboard: add run " + submission['run_id'] + "'")
    print("    uv run leaderboard.py   # regenerate dashboard")
    print("    git add leaderboard/index.html && git commit -m 'leaderboard: update dashboard'")
    print("    # Then open a PR to share your results!")
    print("=" * 55)


if __name__ == "__main__":
    main()
