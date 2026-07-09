#!/usr/bin/env python3
"""
Local structure/consistency gate for the phase3 ablation pipeline.

Checks:
  1. Required files/dirs exist
  2. All .py files compile (syntax check)
  3. data/phase3 JSON files parse and match expected schema
  4. The VL model name is consistent across src/ablation_run.py and runpod_run.sh
  5. The hardcoded GPU max_memory cap matches the GPU actually present

Run: python3 validate_pipeline.py
"""
import json
import re
import subprocess
import sys
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

errors = []
warnings = []


def error(message):
    errors.append(message)


def warn(message):
    warnings.append(message)


REQUIRED_PATHS = [
    "data/lego/LEGO.tsv",
    "data/lego/images",
    "data/phase3/test_200.json",
    "data/phase3/step4_meta.json",
    "results",
    "src/ablation_run.py",
    "runpod_run.sh",
    "requirements.txt",
]


def check_structure():
    for relative_path in REQUIRED_PATHS:
        if not (ROOT / relative_path).exists():
            error(f"missing required path: {relative_path}")

    images_dir = ROOT / "data/lego/images"
    if images_dir.is_dir() and not any(images_dir.glob("*.png")):
        warn("data/lego/images/ exists but contains no .png files")


def check_python_syntax():
    for py_file in ROOT.rglob("*.py"):
        if ".ipynb_checkpoints" in py_file.parts:
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as compile_error:
            error(f"syntax error in {py_file.relative_to(ROOT)}: {compile_error.msg}")


def check_json():
    test_path = ROOT / "data/phase3/test_200.json"
    if test_path.exists():
        try:
            test_questions = json.loads(test_path.read_text())
            if not isinstance(test_questions, list) or not all("question_id" in question for question in test_questions):
                error("test_200.json: expected a list of objects each with 'question_id'")
        except json.JSONDecodeError as parse_error:
            error(f"test_200.json: invalid JSON ({parse_error})")

    meta_path = ROOT / "data/phase3/step4_meta.json"
    if meta_path.exists():
        try:
            step4_meta = json.loads(meta_path.read_text())
            layers = step4_meta.get("layers")
            if not isinstance(layers, list):
                error("step4_meta.json: expected a 'layers' list")
            else:
                required_keys = ("layer_index", "best_k", "best_silhouette", "bad_expert_indices")
                for layer in layers:
                    for key in required_keys:
                        if key not in layer:
                            error(f"step4_meta.json: layer entry missing '{key}'")
        except json.JSONDecodeError as parse_error:
            error(f"step4_meta.json: invalid JSON ({parse_error})")


MODEL_ID_RE = re.compile(r'MODEL_ID\s*=\s*"([^"]+)"')
REPO_ID_RE = re.compile(r'repo_id\s*=\s*"([^"]+)"')
LOCAL_DIR_RE = re.compile(r'local_dir\s*=\s*"([^"]+)"')
MODEL_NAME_RE = re.compile(r"Qwen3-VL-[\w-]+")
GPU_LAYERS_RE = re.compile(r"gpu_layers=(\d+)")

# Empirically observed GPU footprint per decoder layer placed on GPU 0 during
# from_pretrained() -- covers both the final 4-bit weights and the transient
# bf16 staging tensors that bitsandbytes doesn't free until loading completes
# (see build_device_map()'s docstring in src/ablation_run.py for why layers
# are capped rather than left to device_map="auto").
OBSERVED_GIB_PER_GPU_LAYER = 1.3


def check_model_consistency():
    model_id_refs = {}
    ablation_script_path = ROOT / "src/ablation_run.py"
    if ablation_script_path.exists():
        match = MODEL_ID_RE.search(ablation_script_path.read_text())
        if match:
            model_id_refs["src/ablation_run.py"] = match.group(1)

    runpod_script_path = ROOT / "runpod_run.sh"
    if runpod_script_path.exists():
        text = runpod_script_path.read_text()
        match = REPO_ID_RE.search(text)
        if match:
            model_id_refs["runpod_run.sh:repo_id"] = match.group(1)
        match = LOCAL_DIR_RE.search(text)
        if match:
            model_id_refs["runpod_run.sh:local_dir"] = match.group(1)

    if not model_id_refs:
        warn("no MODEL_ID/repo_id references found to cross-check")
        return

    names = {}
    for source, value in model_id_refs.items():
        match = MODEL_NAME_RE.search(value)
        names[source] = match.group(0) if match else value

    if len(set(names.values())) > 1:
        error(f"VL model name mismatch across files: {names}")


def check_gpu_vram():
    ablation_script_path = ROOT / "src/ablation_run.py"
    if not ablation_script_path.exists():
        return
    match = GPU_LAYERS_RE.search(ablation_script_path.read_text())
    if not match:
        warn("could not find gpu_layers=N in src/ablation_run.py to check")
        return
    gpu_layers = int(match.group(1))
    estimated_gib = gpu_layers * OBSERVED_GIB_PER_GPU_LAYER

    try:
        nvidia_smi_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        warn("nvidia-smi not found; skipping live GPU/VRAM check")
        return

    if nvidia_smi_result.returncode != 0 or not nvidia_smi_result.stdout.strip():
        warn("nvidia-smi returned no GPU; skipping live GPU/VRAM check")
        return

    name, mem_mib = nvidia_smi_result.stdout.strip().splitlines()[0].split(",")
    actual_gib = int(mem_mib.strip()) / 1024
    name = name.strip()

    if estimated_gib > actual_gib * 0.9:
        error(
            f"src/ablation_run.py sets gpu_layers={gpu_layers} (~{estimated_gib:.1f}GiB "
            f"estimated) which leaves little to no margin on the actual GPU ({name}, "
            f"{actual_gib:.1f}GiB) -- lower gpu_layers or expect OOM"
        )
    elif estimated_gib < actual_gib * 0.5:
        warn(
            f"src/ablation_run.py sets gpu_layers={gpu_layers} (~{estimated_gib:.1f}GiB "
            f"estimated), well under half of the actual GPU's {actual_gib:.1f}GiB ({name}) "
            f"-- gpu_layers could likely be raised for faster inference"
        )


def main():
    check_structure()
    check_python_syntax()
    check_json()
    check_model_consistency()
    check_gpu_vram()

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("ERRORS:")
        for error_message in errors:
            print(f"  - {error_message}")
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
        sys.exit(1)

    print(f"\nAll structure/model/GPU checks passed ({len(warnings)} warning(s)).")
    sys.exit(0)


if __name__ == "__main__":
    main()
