"""
Step 5: Ablation runs on RunPod.

Runs the 200 test questions under multiple ablation conditions to test whether
disabling specific layers or experts improves accuracy beyond the 76.5% error baseline.

Three ablation modes:
    full_block   - entire decoder layer is skipped, residual passes straight through
    moe_only     - only the MoE sublayer is zeroed, attention still runs
    expert       - only bad-cluster experts have their gate weights zeroed and
                   renormalized, rest of the layer runs normally

Experiment structure:
    Part A - compare all three modes on the single worst layer (layer 41)
    Part B - apply the best mode from A to 1, 3, and 5 worst layers
    Part C - 10 random ablation trials per layer count (1, 3, 5) as control

Files needed in /workspace/phase3_data/:
    test_200.json    - the 200 test questions (from data/phase3/step1/)
    step4_meta.json  - bad expert indices per layer (from data/phase3/step4/)

Outputs:
    /workspace/phase3_results/ablation_results.json  - all run results
"""

import os
import sys
import gc
import json
import random
from pathlib import Path
from contextlib import contextmanager

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/transformers"

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# --- Config ---

MODEL_ID = "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct"

LEGO_TSV_PATH = "/workspace/lego_data/LEGO.tsv"
IMG_DIR = Path("/workspace/lego_images")

PHASE3_DATA_DIR = Path("/workspace/phase3_data")
OUTPUT_DIR = Path("/workspace/phase3_results")

TEST_QUESTIONS_PATH = PHASE3_DATA_DIR / "test_200.json"
STEP4_META_PATH = PHASE3_DATA_DIR / "step4_meta.json"
RESULTS_PATH = OUTPUT_DIR / "ablation_results.json"

LITE_CATEGORIES = ["height", "position", "rotation", "ordering"]
MCQ_CHOICES = list("ABCDE")

RANDOM_TRIALS_PER_COUNT = 10
ABLATION_LAYER_COUNTS = [1, 3, 5]
RANDOM_SEED = 42


# --- Dependencies ---

def ensure_deps():
    try:
        import bitsandbytes
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "-q"])


# --- Model loading ---

def load_model_and_processor():
    from transformers import (
        Qwen3VLMoeForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    print(f"Loading model [4-bit NF4, attn={attn_impl}]...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    cfg = model.config.text_config
    meta = {
        "num_layers": cfg.num_hidden_layers,
        "num_experts": cfg.num_experts,
        "top_k": cfg.num_experts_per_tok,
    }
    print(f"Model ready. Layers={meta['num_layers']}, Experts={meta['num_experts']}, Top-K={meta['top_k']}")
    return model, processor, meta


# --- Data loading ---

def load_test_questions():
    with open(TEST_QUESTIONS_PATH) as f:
        return json.load(f)


def load_step4_meta():
    with open(STEP4_META_PATH) as f:
        return json.load(f)


def build_question_id_set(test_questions):
    return {q["question_id"] for q in test_questions}


def load_lego_dataframe(test_question_ids):
    df = pd.read_csv(LEGO_TSV_PATH, sep="\t")
    df = df[df["category"].isin(LITE_CATEGORIES)].reset_index(drop=True)
    df = df[df["index"].astype(str).isin(test_question_ids)].reset_index(drop=True)
    print(f"Loaded {len(df)} test questions from LEGO.tsv.")
    return df


def load_image(row):
    p = IMG_DIR / f"{row['index']}.png"
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def build_messages(row):
    question = row["question"]
    question_type = str(row.get("question_type", "")).strip()
    option_cols = [c for c in MCQ_CHOICES if c in row.index and not pd.isna(row.get(c))]

    prompt = f"Question: {question}\n"
    if option_cols:
        prompt += "Options:\n"
        for c in option_cols:
            val = row[c]
            label = "[see image]" if isinstance(val, str) and "<image" in val else str(val)
            prompt += f"{c}. {label}\n"

    prompt += (
        "Please respond with only the letter sequence (e.g. 'BDAC').\n"
        if question_type == "sort"
        else "Please respond with only the letter of the correct answer.\n"
    )

    pil_img = load_image(row)
    content = []
    if pil_img is not None:
        content.append({"type": "image", "image": pil_img})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}], pil_img


# --- Ablation hooks ---

def make_full_block_hook():
    """
    Replaces the entire decoder layer output with its input hidden states.

    This simulates removing the layer completely. The residual stream passes
    through as if the layer never ran, so neither attention nor MoE contributes.
    The hook receives (module, input, output) where input[0] is the hidden
    states before the layer and output[0] is after. Swapping output[0] for
    input[0] achieves the skip.
    """
    def hook(module, inp, output):
        if isinstance(output, tuple):
            return (inp[0],) + output[1:]
        return inp[0]
    return hook


def make_moe_skip_hook():
    """
    Zeros the MoE sublayer output while leaving attention unchanged.

    The parent decoder layer adds the MoE output to the residual stream via:
        hidden_states = residual + mlp_output
    Zeroing mlp_output makes the residual connection a no-op for the MoE
    contribution, isolating whether MoE specifically is responsible for errors
    versus the attention sublayer.
    """
    def hook(module, inp, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook


def make_expert_gate_hook(bad_expert_indices):
    """
    Zeros gate weights for bad-cluster experts and renormalizes the rest.

    Intercepts the gate output after routing weights are computed. Bad experts'
    weights are set to zero so their outputs contribute nothing to the weighted
    sum, then remaining weights are renormalized to sum to 1. This isolates the
    specific experts identified in Step 4 rather than disabling the whole MoE.

    Handles two gate output formats observed in Qwen3-VL:
        - Tensor [tokens, num_experts]: direct routing weight matrix
        - Tuple (raw_logits, routing_weights, selected_indices): sparse format
    """
    bad_set = set(bad_expert_indices)

    def zero_and_renorm(weights_2d):
        weights_2d = weights_2d.clone().float()
        for idx in bad_set:
            if idx < weights_2d.shape[1]:
                weights_2d[:, idx] = 0.0
        row_sums = weights_2d.sum(dim=1, keepdim=True)
        # Avoid division by zero if all remaining experts are also zero
        row_sums = torch.clamp(row_sums, min=1e-8)
        return weights_2d / row_sums

    def hook(module, inp, output):
        if isinstance(output, torch.Tensor) and output.dim() == 2:
            return zero_and_renorm(output)
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            routing_weights = zero_and_renorm(output[1])
            return (output[0], routing_weights) + tuple(output[2:])
        return output

    return hook


# --- Hook registration ---

def find_decoder_layers(model, num_layers):
    for path in ("model.layers", "model.model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) == num_layers:
                return obj
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate decoder layer list in model.")


@contextmanager
def apply_ablation(model, meta, ablation_mode, layer_indices, bad_experts_by_layer=None):
    """
    Context manager that registers ablation hooks and removes them on exit.

    Using a context manager guarantees hook cleanup even if inference raises
    an exception, preventing hooks from leaking into subsequent runs.

    Args:
        ablation_mode: "full_block", "moe_only", or "expert"
        layer_indices: list of layer indices to ablate
        bad_experts_by_layer: dict of layer_index -> list of bad expert indices
                              (only needed for "expert" mode)
    """
    decoder_layers = find_decoder_layers(model, meta["num_layers"])
    hooks = []

    for layer_idx in layer_indices:
        layer = decoder_layers[layer_idx]

        if ablation_mode == "full_block":
            h = layer.register_forward_hook(make_full_block_hook())
            hooks.append(h)

        elif ablation_mode == "moe_only":
            if hasattr(layer, "mlp"):
                h = layer.mlp.register_forward_hook(make_moe_skip_hook())
                hooks.append(h)

        elif ablation_mode == "expert":
            bad_experts = (bad_experts_by_layer or {}).get(layer_idx, [])
            if bad_experts and hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                h = layer.mlp.gate.register_forward_hook(
                    make_expert_gate_hook(bad_experts)
                )
                hooks.append(h)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# --- Inference ---

def detect_image_token_id(processor):
    if hasattr(processor, "image_token_id"):
        return processor.image_token_id
    for name in ("<|image_pad|>", "<image>", "<img>", "[IMG]", "<|vision_pad|>"):
        tid = processor.tokenizer.convert_tokens_to_ids(name)
        if tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError("Could not detect image token ID.")


def run_inference(model, processor, df, meta, run_label):
    """
    Run the 200 test questions through the model and collect per-question results.

    No routing data is captured here since Step 5 only cares about whether
    answers flip relative to baseline. Keeping this lightweight lets us run
    the 30+ ablation configurations without hitting memory limits.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_token_id = detect_image_token_id(processor)

    choice_ids = {
        c: processor.tokenizer.convert_tokens_to_ids(c)
        for c in MCQ_CHOICES
    }

    results = []

    for row_idx in tqdm(range(len(df)), desc=run_label):
        row = df.iloc[row_idx]
        q_id = str(row["index"])
        ground_truth = str(row["answer"]).strip().upper()

        outputs = None
        inputs = None
        try:
            messages, _ = build_messages(row)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)

            next_tok_logits = outputs.logits[0, -1, :]
            scores = {c: next_tok_logits[tid].item() for c, tid in choice_ids.items()}
            pred = max(scores, key=scores.get)
            correct = (pred == ground_truth[0]) if ground_truth else False

        except Exception as exc:
            pred = f"ERROR: {exc}"
            correct = False

        results.append({
            "question_id": q_id,
            "category": row["category"],
            "prediction": pred,
            "correct": correct,
        })

        del outputs, inputs
        torch.cuda.empty_cache()

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    print(f"{run_label}: {correct_count}/{len(results)} correct ({accuracy:.1%})")

    for cat in LITE_CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        cat_acc = sum(1 for r in cat_results if r["correct"]) / len(cat_results)
        print(f"  {cat}: {cat_acc:.1%}")

    return results, accuracy


# --- Experiment parts ---

def run_baseline(model, processor, df, meta):
    print("\n--- Baseline (no ablation) ---")
    results, accuracy = run_inference(model, processor, df, meta, "baseline")
    return {"run": "baseline", "ablation_mode": None, "layers_ablated": [], "accuracy": accuracy, "results": results}


def run_part_a(model, processor, df, meta, bad_experts_by_layer, worst_layer):
    """
    Compare all three ablation modes on the single worst layer.

    The goal is to find which mode produces the biggest accuracy change so
    Parts B and C use the most informative intervention type.
    """
    print(f"\n=== Part A: Ablation type comparison on layer {worst_layer} ===")
    part_a_results = []

    for mode in ["full_block", "moe_only", "expert"]:
        print(f"\n--- Part A: {mode} on layer {worst_layer} ---")
        with apply_ablation(model, meta, mode, [worst_layer], bad_experts_by_layer):
            results, accuracy = run_inference(
                model, processor, df, meta, f"partA_{mode}_layer{worst_layer}"
            )
        part_a_results.append({
            "run": f"partA_{mode}",
            "ablation_mode": mode,
            "layers_ablated": [worst_layer],
            "accuracy": accuracy,
            "results": results,
        })

    return part_a_results


def run_part_b(model, processor, df, meta, bad_experts_by_layer, best_mode, worst_layers_ranked):
    """
    Scale the best ablation mode to 1, 3, and 5 worst layers.

    Tests whether targeting more layers compounds the improvement or degrades
    accuracy by removing too much computation.
    """
    print(f"\n=== Part B: Scaling with {best_mode} ===")
    part_b_results = []

    for count in ABLATION_LAYER_COUNTS:
        layers = worst_layers_ranked[:count]
        print(f"\n--- Part B: {best_mode} on {count} worst layer(s): {layers} ---")
        with apply_ablation(model, meta, best_mode, layers, bad_experts_by_layer):
            results, accuracy = run_inference(
                model, processor, df, meta, f"partB_{best_mode}_{count}layers"
            )
        part_b_results.append({
            "run": f"partB_{best_mode}_{count}layers",
            "ablation_mode": best_mode,
            "layers_ablated": layers,
            "accuracy": accuracy,
            "results": results,
        })

    return part_b_results


def run_part_c(model, processor, df, meta, num_total_layers):
    """
    Random ablation control using MoE-only mode.

    Random layers have no bad-expert clusters, so MoE-only is used regardless
    of which mode won Part A. If targeted ablation outperforms these random
    baselines, it confirms our layer scoring found real signal rather than
    the model just tolerating any layer removal.
    """
    print("\n=== Part C: Random ablation control (MoE-only) ===")
    rng = random.Random(RANDOM_SEED)
    part_c_results = []

    for count in ABLATION_LAYER_COUNTS:
        for trial in range(RANDOM_TRIALS_PER_COUNT):
            random_layers = rng.sample(range(num_total_layers), count)
            print(f"\n--- Part C: {count} random layers, trial {trial + 1} --- layers: {sorted(random_layers)}")
            with apply_ablation(model, meta, "moe_only", random_layers):
                results, accuracy = run_inference(
                    model, processor, df, meta,
                    f"partC_random_{count}layers_trial{trial + 1}"
                )
            part_c_results.append({
                "run": f"partC_random_{count}layers_trial{trial + 1}",
                "ablation_mode": "moe_only",
                "layers_ablated": sorted(random_layers),
                "accuracy": accuracy,
                "results": results,
            })

    return part_c_results


# --- Main ---

def main():
    ensure_deps()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_questions = load_test_questions()
    step4_meta = load_step4_meta()
    test_question_ids = build_question_id_set(test_questions)

    # Layer order in step4_meta matches z-score ranking from Step 2 (worst first).
    worst_layers_ranked = [entry["layer_index"] for entry in step4_meta["layers"]]
    bad_experts_by_layer = {
        entry["layer_index"]: entry["bad_expert_indices"]
        for entry in step4_meta["layers"]
    }

    df = load_lego_dataframe(test_question_ids)
    if len(df) != len(test_questions):
        print(f"WARNING: expected {len(test_questions)} questions, got {len(df)}")

    # Load model once and reuse across all runs
    model, processor, meta = load_model_and_processor()

    worst_layer = worst_layers_ranked[0]
    all_run_results = []

    baseline = run_baseline(model, processor, df, meta)
    all_run_results.append(baseline)
    baseline_accuracy = baseline["accuracy"]

    part_a = run_part_a(model, processor, df, meta, bad_experts_by_layer, worst_layer)
    all_run_results.extend(part_a)

    best_mode = max(part_a, key=lambda r: r["accuracy"])["ablation_mode"]
    print(f"\nBest ablation mode from Part A: {best_mode}")
    print(f"Baseline accuracy: {baseline_accuracy:.1%}")

    part_b = run_part_b(model, processor, df, meta, bad_experts_by_layer, best_mode, worst_layers_ranked)
    all_run_results.extend(part_b)

    part_c = run_part_c(model, processor, df, meta, meta["num_layers"])
    all_run_results.extend(part_c)

    output = {
        "baseline_accuracy": baseline_accuracy,
        "worst_layer_for_part_a": worst_layer,
        "best_mode_from_part_a": best_mode,
        "worst_layers_ranked": worst_layers_ranked,
        "bad_experts_by_layer": bad_experts_by_layer,
        "runs": all_run_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAll results saved to {RESULTS_PATH}")

    print("\n=== SUMMARY ===")
    print(f"Baseline: {baseline_accuracy:.1%}")
    for run in all_run_results[1:]:
        delta = run["accuracy"] - baseline_accuracy
        sign = "+" if delta >= 0 else ""
        print(f"{run['run']}: {run['accuracy']:.1%} ({sign}{delta:.1%})")


if __name__ == "__main__":
    main()
