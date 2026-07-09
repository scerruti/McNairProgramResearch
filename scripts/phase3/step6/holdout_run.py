"""
Step 6 holdout validation.

Runs baseline + best ablation config (moe_only, top 3 worst layers) on the
holdout 200 questions. These questions were never used in layer scoring,
clustering, or the Step 5 ablation experiments, so this is a clean
generalization check.

Best config is read from ablation_results.json so nothing is hardcoded.

Files needed:
    holdout_200.json       - this directory (the 200 holdout questions)
    /workspace/data/phase3/step4_meta.json  - layer order by z-score (worst first)
    /workspace/results/ablation_results.json - Step 5 results (best_mode + layer ranking)

Outputs:
    holdout_results.json - written to this directory
"""

import os
import sys
import gc
import json
import math
from pathlib import Path
from contextlib import contextmanager

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/transformers"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# --- Config ---

BASE_DIR = Path(__file__).resolve().parent

MODEL_ID = "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct"

LEGO_TSV_PATH = "/workspace/data/lego/LEGO.tsv"
IMG_DIR = Path("/workspace/data/lego/images")

PHASE3_DATA_DIR = Path("/workspace/data/phase3")
RESULTS_DIR = Path("/workspace/results")

HOLDOUT_PATH = BASE_DIR / "holdout_200.json"
STEP4_META_PATH = PHASE3_DATA_DIR / "step4_meta.json"
ABLATION_RESULTS_PATH = RESULTS_DIR / "ablation_results.json"
OUTPUT_PATH = BASE_DIR / "holdout_results.json"

LITE_CATEGORIES = ["height", "position", "rotation", "ordering"]
MCQ_CHOICES = list("ABCDE")

# Number of worst layers to ablate, matching the best Part B config from Step 5.
BEST_LAYER_COUNT = 3


# --- Config loading ---

def load_best_config():
    """
    Read best ablation mode and worst layer ranking from Step 5 results.

    Layer ranking comes from step4_meta insertion order, which preserves
    z-score rank (worst first) from Step 2.
    """
    with open(ABLATION_RESULTS_PATH) as file_handle:
        ablation = json.load(file_handle)

    with open(STEP4_META_PATH) as file_handle:
        step4 = json.load(file_handle)

    best_mode = ablation["best_mode_from_part_a"]
    worst_layers_ranked = [entry["layer_index"] for entry in step4["layers"]]
    target_layers = worst_layers_ranked[:BEST_LAYER_COUNT]

    bad_experts_by_layer = {
        entry["layer_index"]: entry["bad_expert_indices"]
        for entry in step4["layers"]
    }

    print(f"Best mode: {best_mode}")
    print(f"Target layers: {target_layers}")
    return best_mode, target_layers, bad_experts_by_layer


# --- Model loading ---

def build_device_map(num_layers, gpu_layers):
    # device_map="auto" places every decoder layer on GPU 0 because the *quantized*
    # footprint (~0.3GiB/layer) comfortably fits -- but bnb's bf16->4bit conversion
    # during from_pretrained() leaks the pre-quantization bf16 staging tensors, so
    # GPU usage during loading tracks the raw bf16 size of shards processed so far
    # (~1.25GiB/layer), not the final quantized size. Capping how many layers ever
    # touch the GPU during loading keeps that transient under the card's ceiling.
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 0,
        "model.language_model.rotary_emb": 0,
        "lm_head": 0,
    }
    for i in range(num_layers):
        device_map[f"model.language_model.layers.{i}"] = 0 if i < gpu_layers else "cpu"
    return device_map


def load_model_and_processor():
    from transformers import (
        AutoConfig,
        Qwen3VLMoeForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    config = AutoConfig.from_pretrained(MODEL_ID)
    num_layers = config.text_config.num_hidden_layers
    device_map = build_device_map(num_layers, gpu_layers=24)

    print(f"Loading model [4-bit NF4, attn={attn_impl}, 24/{num_layers} decoder layers on GPU]...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    text_config = model.config.text_config
    meta = {
        "num_layers": text_config.num_hidden_layers,
        "num_experts": text_config.num_experts,
        "top_k": text_config.num_experts_per_tok,
    }
    print(f"Model ready. Layers={meta['num_layers']}, Experts={meta['num_experts']}, Top-K={meta['top_k']}")
    return model, processor, meta


# --- Data loading ---

def load_holdout_questions():
    with open(HOLDOUT_PATH) as file_handle:
        return json.load(file_handle)


def load_lego_dataframe(question_ids):
    df = pd.read_csv(LEGO_TSV_PATH, sep="\t")
    df = df[df["category"].isin(LITE_CATEGORIES)].reset_index(drop=True)
    df = df[df["index"].astype(str).isin(question_ids)].reset_index(drop=True)
    print(f"Loaded {len(df)} holdout questions from LEGO.tsv.")
    return df


def load_image(row):
    image_path = IMG_DIR / f"{row['index']}.png"
    if not image_path.exists():
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def build_messages(row):
    question = row["question"]
    question_type = str(row.get("question_type", "")).strip()
    option_cols = [option_letter for option_letter in MCQ_CHOICES if option_letter in row.index and not pd.isna(row.get(option_letter))]

    prompt = f"Question: {question}\n"
    if option_cols:
        prompt += "Options:\n"
        for option_letter in option_cols:
            option_value = row[option_letter]
            label = "[see image]" if isinstance(option_value, str) and "<image" in option_value else str(option_value)
            prompt += f"{option_letter}. {label}\n"

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

def make_moe_skip_hook():
    """
    Zeros the MoE sublayer output while leaving attention unchanged.

    The parent decoder layer adds MoE output to the residual stream via:
        hidden_states = residual + mlp_output
    Zeroing mlp_output makes the MoE contribution a no-op for that layer.
    """
    def hook(module, hook_input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook


def find_decoder_layers(model, num_layers):
    for path in ("model.layers", "model.model.layers", "model.language_model.layers"):
        candidate = model
        try:
            for attr in path.split("."):
                candidate = getattr(candidate, attr)
            if hasattr(candidate, "__len__") and len(candidate) == num_layers:
                return candidate
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate decoder layer list in model.")


@contextmanager
def apply_moe_ablation(model, meta, layer_indices):
    """Registers moe_only hooks for the given layers and removes them on exit."""
    decoder_layers = find_decoder_layers(model, meta["num_layers"])
    hooks = []
    for layer_idx in layer_indices:
        layer = decoder_layers[layer_idx]
        if hasattr(layer, "mlp"):
            hooks.append(layer.mlp.register_forward_hook(make_moe_skip_hook()))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


# --- Inference ---

def detect_image_token_id(processor):
    if hasattr(processor, "image_token_id"):
        return processor.image_token_id
    for name in ("<|image_pad|>", "<image>", "<img>", "[IMG]", "<|vision_pad|>"):
        token_id = processor.tokenizer.convert_tokens_to_ids(name)
        if token_id != processor.tokenizer.unk_token_id:
            return token_id
    raise RuntimeError("Could not detect image token ID.")


def run_inference(model, processor, df, meta, run_label):
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

        except Exception as error:
            pred = f"ERROR: {error}"
            correct = False

        results.append({
            "question_id": q_id,
            "category": row["category"],
            "prediction": pred,
            "correct": correct,
        })

        del outputs, inputs
        torch.cuda.empty_cache()

    correct_count = sum(1 for record in results if record["correct"])
    accuracy = correct_count / len(results)
    print(f"{run_label}: {correct_count}/{len(results)} correct ({accuracy:.1%})")

    from collections import defaultdict
    cat_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in results:
        cat_counts[record["category"]]["total"] += 1
        if record["correct"]:
            cat_counts[record["category"]]["correct"] += 1
    for cat, counts in sorted(cat_counts.items()):
        print(f"  {cat}: {counts['correct']}/{counts['total']} ({counts['correct']/counts['total']:.1%})")

    return results, accuracy


# --- Main ---

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    best_mode, target_layers, bad_experts_by_layer = load_best_config()
    holdout_questions = load_holdout_questions()
    question_ids = {str(question["question_id"]) for question in holdout_questions}

    df = load_lego_dataframe(question_ids)
    if len(df) != len(holdout_questions):
        print(f"WARNING: expected {len(holdout_questions)} questions, got {len(df)}")

    model, processor, meta = load_model_and_processor()

    print("\n--- Holdout baseline (no ablation) ---")
    baseline_results, baseline_acc = run_inference(model, processor, df, meta, "holdout_baseline")

    print(f"\n--- Holdout {best_mode} on layers {target_layers} ---")
    with apply_moe_ablation(model, meta, target_layers):
        ablation_results, ablation_acc = run_inference(
            model, processor, df, meta,
            f"holdout_{best_mode}_{len(target_layers)}layers"
        )

    holdout_report = {
        "holdout_baseline_accuracy": baseline_acc,
        "holdout_ablation_accuracy": ablation_acc,
        "ablation_mode": best_mode,
        "layers_ablated": target_layers,
        "n_questions": len(df),
        "runs": [
            {"run": "holdout_baseline", "accuracy": baseline_acc, "results": baseline_results},
            {
                "run": f"holdout_{best_mode}_{len(target_layers)}layers",
                "accuracy": ablation_acc,
                "results": ablation_results,
            },
        ],
    }

    with open(OUTPUT_PATH, "w") as file_handle:
        json.dump(holdout_report, file_handle, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    delta = ablation_acc - baseline_acc
    sign = "+" if delta >= 0 else ""
    print(f"\nHoldout result: baseline={baseline_acc:.1%} ablation={ablation_acc:.1%} delta={sign}{delta:.1%}")


if __name__ == "__main__":
    main()
