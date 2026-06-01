"""
lego_lite_moe_analysis.py
=========================
MoE Expert Routing Analysis - Qwen3-VL-30B-A3B-Instruct on LEGOLite.

Four modular steps:
  1. Inference + visual-token router extraction    (4-bit NF4 quantisation)
  2. Category-level expert heatmaps               (Height/Position/Rotation/Ordering)
  3. Per-expert accuracy correlation              (success-rate when expert fires)
  4. Spatial reasoning expert leaderboard         (ranked by activation × accuracy)

All artefacts land in OUTPUT_DIR (/workspace/lego_lite_analysis/).
Re-running skips Step 1 if results.json already exists.
"""

# ── Environment setup (must come before any HF import) ───────────────────────
import os, sys
os.environ["HF_HOME"]            = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/transformers"

import gc
import json
import string
import base64
import subprocess
from io import BytesIO
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ── edit these paths to match your setup
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID        = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Path to the LEGO.tsv downloaded in the previous script
LEGO_TSV_PATH   = "/workspace/lego_data/LEGO.tsv"

# Directory where per-question images were extracted previously
IMG_DIR         = Path("/workspace/lego_images")

# All outputs go here
OUTPUT_DIR      = Path("/workspace/lego_lite_analysis")

# The four LEGOLite categories of interest
LITE_CATEGORIES = ["height", "position", "rotation", "ordering"]

# Multiple-choice option labels
MCQ_CHOICES     = list("ABCDE")

# Expert leaderboard thresholds (tune if needed)
MIN_ACTIVATION_CORRECT = 0.003   # minimum mean visual-token activation on correct samples
MIN_SUCCESS_DELTA      = 0.001   # minimum (correct_activation - incorrect_activation)
LEADERBOARD_TOP_N      = 30      # how many experts to surface in the leaderboard

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON    = OUTPUT_DIR / "results.json"
EXPERT_RANK_CSV = OUTPUT_DIR / "expert_success_rates.csv"
LEADERBOARD_CSV = OUTPUT_DIR / "spatial_expert_leaderboard.csv"


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK  ── install bitsandbytes if missing
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_deps():
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("Installing bitsandbytes …")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "bitsandbytes", "-q"])

_ensure_deps()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  ── 4-bit NF4 quantisation via bitsandbytes
# ─────────────────────────────────────────────────────────────────────────────
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
        bnb_4bit_use_double_quant=True,   # double quant saves ~0.4 bits/param
    )

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    print(f"Loading {MODEL_ID}  [4-bit NF4, attn={attn_impl}] …")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    cfg = model.config.text_config
    meta = dict(
        num_layers  = cfg.num_hidden_layers,   # 48
        num_experts = cfg.num_experts,          # 128
        top_k       = cfg.num_experts_per_tok,  # 8
    )
    print(f"Ready. Layers={meta['num_layers']}, "
          f"Experts={meta['num_experts']}, Top-K={meta['top_k']}")
    return model, processor, meta


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING  ── filter LEGO.tsv to LEGOLite (400 questions)
# ─────────────────────────────────────────────────────────────────────────────
def load_legolit_dataset():
    df = pd.read_csv(LEGO_TSV_PATH, sep="\t")
    df = df[df["category"].isin(LITE_CATEGORIES)].reset_index(drop=True)
    print(f"LEGOLite: {len(df)} questions | "
          f"categories: {df['category'].value_counts().to_dict()}")
    return df


def _load_image(row) -> Image.Image | None:
    """Load image as PIL object; returns None if missing."""
    p = IMG_DIR / f"{row['index']}.png"
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def build_messages(row) -> tuple[list[dict], Image.Image | None]:
    """Return (messages, pil_image_or_None).

    We pass the PIL Image object directly (not a file:// URL) so the processor
    handles pixel loading itself, avoiding the torchvision decode path.
    """
    question = row["question"]
    qt       = str(row.get("question_type", "")).strip()
    opts     = [c for c in MCQ_CHOICES
                if c in row.index and not pd.isna(row.get(c))]

    prompt = f"Question: {question}\n"
    if opts:
        prompt += "Options:\n"
        for c in opts:
            val = row[c]
            label = "[see image]" if isinstance(val, str) and "<image" in val else str(val)
            prompt += f"{c}. {label}\n"

    prompt += ("Please respond with only the letter sequence (e.g. 'BDAC').\n"
               if qt == "sort"
               else "Please respond with only the letter of the correct answer.\n")

    pil_img = _load_image(row)
    content = []
    if pil_img is not None:
        content.append({"type": "image", "image": pil_img})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}], pil_img


# ─────────────────────────────────────────────────────────────────────────────
# HOOK INFRASTRUCTURE  ── capture router output for every MoE gate
# ─────────────────────────────────────────────────────────────────────────────

# These globals are set/cleared around each forward pass.
_visual_mask: torch.Tensor | None = None   # bool [seq_len], True = image token
_routing_buf: dict[int, np.ndarray] = {}   # layer_idx -> float32 [num_experts]
_num_experts: int = 128


def _make_router_hook(layer_idx: int):
    """
    Register on layer.mlp.gate (Qwen3VLMoeTextTopKRouter).

    Router output is a 3-tuple: (raw_logits, routing_weights, selected_indices)
      raw_logits      : [tokens, num_experts]   – pre-softmax scores (can be negative)
      routing_weights : [tokens, top_k]          – actual weights used (post-softmax, positive)
      selected_indices: [tokens, top_k]           – chosen expert IDs (0..num_experts-1)

    We reconstruct a sparse [tokens, num_experts] weight matrix from routing_weights +
    selected_indices. Non-selected experts get weight 0.  This gives us positive,
    interpretable activation values (each selected expert's true contribution weight).
    Falls back to softmax of raw_logits if the sparse path isn't available.
    """
    def _hook(module, inp, output):
        global _routing_buf
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            return

        rw  = output[1]   # [tokens, top_k]  – routing weights (post-softmax)
        idx = output[2]   # [tokens, top_k]  – selected expert IDs

        if isinstance(rw, torch.Tensor) and isinstance(idx, torch.Tensor) and rw.dim() == 2:
            # Primary path: sparse reconstruction from actual routing weights
            tokens     = idx.shape[0]
            weight_mat = torch.zeros(tokens, _num_experts,
                                     device=rw.device, dtype=torch.float32)
            weight_mat.scatter_(1, idx.long(), rw.float())
        else:
            # Fallback: apply softmax to raw logits so values are positive
            raw = output[0]
            if not (isinstance(raw, torch.Tensor) and raw.dim() == 2
                    and raw.shape[-1] == _num_experts):
                return
            weight_mat = torch.softmax(raw.float(), dim=-1)  # [tokens, num_experts]

        # ── Filter to visual token positions ─────────────────────────────────
        if _visual_mask is not None:
            seq_len = min(weight_mat.shape[0], _visual_mask.shape[0])
            mask    = _visual_mask[:seq_len]
            vis     = weight_mat[:seq_len][mask]        # [num_vis_tokens, num_experts]
        else:
            vis = weight_mat

        avg = vis.mean(dim=0) if vis.numel() > 0 else weight_mat.mean(dim=0)
        _routing_buf[layer_idx] = avg.detach().cpu().numpy()

    return _hook


def _find_decoder_layers(model, num_layers: int):
    for path in ("model.layers", "model.model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) == num_layers:
                return obj
        except AttributeError:
            continue
    for _, mod in model.named_modules():
        if hasattr(mod, "__len__"):
            try:
                if len(mod) == num_layers and hasattr(mod[0], "mlp"):
                    return mod
            except (TypeError, IndexError):
                continue
    raise RuntimeError("Cannot locate decoder layer list in model.")


def register_router_hooks(model, meta: dict) -> list:
    global _num_experts
    _num_experts = meta["num_experts"]
    decoder_layers = _find_decoder_layers(model, meta["num_layers"])
    hooks = []
    for li, layer in enumerate(decoder_layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            h = layer.mlp.gate.register_forward_hook(_make_router_hook(li))
            hooks.append(h)
    print(f"Registered {len(hooks)} router hooks.")
    return hooks


def remove_hooks(hooks: list):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL TOKEN DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _detect_image_token_id(processor) -> int:
    """Return the token ID used for image padding in this processor."""
    # Check model config first (most reliable)
    if hasattr(processor, "image_token_id"):
        return processor.image_token_id
    # Try common token names
    for name in ("<|image_pad|>", "<image>", "<img>", "[IMG]", "<|vision_pad|>"):
        tid = processor.tokenizer.convert_tokens_to_ids(name)
        if tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError(
        "Could not auto-detect image token ID. "
        "Set it manually via IMG_TOKEN_ID at the top of this file."
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 ── Inference + visual-token router extraction
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_and_extract(model, processor, df: pd.DataFrame,
                               meta: dict) -> list[dict]:
    """
    Iterate over LEGOLite, run one forward pass per question.

    For each question we:
      • Build the multimodal prompt (image + text).
      • Run model(**inputs) - a single prefill-only forward pass.
      • Read router activations from hooks, filtered to image-pad token positions.
      • Decode the answer from next-token logits (MCQ: argmax over A–E).
      • Persist everything to results.json.

    Returns a list of result dicts.
    """
    global _visual_mask, _routing_buf

    num_layers  = meta["num_layers"]
    num_experts = meta["num_experts"]

    img_token_id = _detect_image_token_id(processor)
    print(f"Image-pad token ID: {img_token_id}")

    # Pre-compute token IDs for MCQ choices
    choice_ids = {
        c: processor.tokenizer.convert_tokens_to_ids(c)
        for c in MCQ_CHOICES
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hooks   = register_router_hooks(model, meta)
    results = []

    for i in tqdm(range(len(df)), desc="LEGOLite inference"):
        row      = df.iloc[i]
        q_id     = str(row["index"])
        category = row["category"]
        gt       = str(row["answer"]).strip().upper()

        outputs = None
        inputs  = None
        try:
            messages, pil_img = build_messages(row)
            inputs   = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,   # Qwen3: skip <think>…</think> for MCQ scoring
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            input_ids    = inputs["input_ids"]           # [1, seq_len]
            num_vis_toks = int((input_ids[0] == img_token_id).sum())

            # Set visual mask so hooks can filter to image positions
            _visual_mask = (input_ids[0] == img_token_id)   # bool [seq_len]
            _routing_buf = {}

            with torch.no_grad():
                # Single prefill pass - captures router activations via hooks
                # and gives us logits for the next token.
                outputs = model(**inputs, return_dict=True)

            # ── Decode answer from next-token logits ─────────────────────────
            # Qwen3 chat template may prepend <think> before the answer.
            # Score each MCQ letter from the logit distribution; the highest
            # scoring letter among A-E is the model's MCQ answer regardless
            # of whether thinking tokens come first in generation.
            next_tok_logits = outputs.logits[0, -1, :]   # [vocab_size]
            scores  = {c: next_tok_logits[tid].item() for c, tid in choice_ids.items()}
            pred    = max(scores, key=scores.get)
            correct = (pred == gt[0]) if gt else False

            # ── Collect per-layer visual routing ─────────────────────────────
            visual_routing = {
                str(li): _routing_buf.get(li, np.zeros(num_experts)).tolist()
                for li in range(num_layers)
            }

        except Exception as exc:
            pred          = f"ERROR: {exc}"
            correct       = False
            num_vis_toks  = 0
            visual_routing = {str(li): [0.0] * num_experts for li in range(num_layers)}

        results.append({
            "question_id":       q_id,
            "category":          category,
            "prediction":        pred,
            "ground_truth":      gt,
            "correct":           correct,
            "num_visual_tokens": num_vis_toks,
            "visual_routing":    visual_routing,  # layer_str -> [num_experts]
        })

        # ── VRAM hygiene ─────────────────────────────────────────────────────
        del outputs, inputs  # both initialised to None above so always safe
        _visual_mask = None
        _routing_buf = {}
        torch.cuda.empty_cache()

    remove_hooks(hooks)

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} records → {RESULTS_JSON}")

    # ── Accuracy summary ─────────────────────────────────────────────────────
    res_df  = pd.DataFrame([{"category": r["category"], "correct": r["correct"]}
                             for r in results])
    overall = res_df["correct"].mean()
    print(f"Overall accuracy: {overall:.1%}")
    for cat in LITE_CATEGORIES:
        sub = res_df[res_df["category"] == cat]
        if len(sub):
            print(f"  {cat:16s}: {sub['correct'].mean():.1%}  "
                  f"({int(sub['correct'].sum())}/{len(sub)})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 ── Category-level expert heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def compute_category_heatmaps(results: list[dict], meta: dict) -> dict[str, np.ndarray]:
    """
    Step 2: For each category compute a [num_layers × num_experts] matrix of
    average visual-token routing weights.

    Outputs one CSV per category (rows = layers, cols = experts) plus a
    Height−Rotation difference map.

    Returns dict: category -> np.ndarray [num_layers, num_experts]
    """
    L, E = meta["num_layers"], meta["num_experts"]

    sums   = defaultdict(lambda: np.zeros((L, E), dtype=np.float64))
    counts = defaultdict(int)

    for r in results:
        cat = r["category"]
        counts[cat] += 1
        for li_str, weights in r["visual_routing"].items():
            sums[cat][int(li_str)] += np.asarray(weights, dtype=np.float64)

    heatmaps: dict[str, np.ndarray] = {}
    for cat in LITE_CATEGORIES:
        if counts[cat] == 0:
            continue
        hm = sums[cat] / counts[cat]
        heatmaps[cat] = hm

        df_hm = pd.DataFrame(
            hm,
            index   = [f"layer_{li}" for li in range(L)],
            columns = [f"expert_{ei}" for ei in range(E)],
        )
        path = OUTPUT_DIR / f"heatmap_{cat}.csv"
        df_hm.to_csv(path)
        print(f"  Heatmap [{cat}] → {path}")

    # Height vs Rotation difference highlights category-specific experts
    if "height" in heatmaps and "rotation" in heatmaps:
        diff = heatmaps["height"] - heatmaps["rotation"]
        pd.DataFrame(
            diff,
            index   = [f"layer_{li}" for li in range(L)],
            columns = [f"expert_{ei}" for ei in range(E)],
        ).to_csv(OUTPUT_DIR / "heatmap_height_minus_rotation.csv")
        print(f"  Height−Rotation diff → {OUTPUT_DIR / 'heatmap_height_minus_rotation.csv'}")

    return heatmaps


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 ── Per-expert accuracy correlation
# ─────────────────────────────────────────────────────────────────────────────
def rank_experts_by_accuracy(results: list[dict], meta: dict) -> pd.DataFrame:
    """
    Step 3: For every (layer, expert) pair compute:

        mean_activation_correct   – avg routing weight over questions answered correctly
        mean_activation_incorrect – avg routing weight over questions answered incorrectly
        success_delta             – correct − incorrect
                                    positive → expert fires more on correct answers
        delta_<category>          – same but restricted to each category
                                    (critical for diagnosing Ordering failures)

    Saves expert_success_rates.csv sorted by success_delta descending.
    """
    L, E = meta["num_layers"], meta["num_experts"]

    # Global accumulators
    act_c  = np.zeros((L, E), dtype=np.float64)   # correct
    act_i  = np.zeros((L, E), dtype=np.float64)   # incorrect
    cnt_c  = 0
    cnt_i  = 0

    # Per-category accumulators
    cat_act_c   = {cat: np.zeros((L, E), dtype=np.float64) for cat in LITE_CATEGORIES}
    cat_act_i   = {cat: np.zeros((L, E), dtype=np.float64) for cat in LITE_CATEGORIES}
    cat_cnt_c   = defaultdict(int)
    cat_cnt_i   = defaultdict(int)

    for r in results:
        correct = r["correct"]
        cat     = r["category"]
        for li_str, weights in r["visual_routing"].items():
            arr = np.asarray(weights, dtype=np.float64)
            li  = int(li_str)
            if correct:
                act_c[li]          += arr
                cat_act_c[cat][li] += arr
            else:
                act_i[li]          += arr
                cat_act_i[cat][li] += arr
        if correct:
            cnt_c += 1
            cat_cnt_c[cat] += 1
        else:
            cnt_i += 1
            cat_cnt_i[cat] += 1

    # Normalise
    if cnt_c > 0: act_c /= cnt_c
    if cnt_i > 0: act_i /= cnt_i

    records = []
    for li in range(L):
        for ei in range(E):
            rec = {
                "layer":                     li,
                "expert":                    ei,
                "expert_label":              f"L{li}_E{ei}",
                "mean_activation_correct":   act_c[li, ei],
                "mean_activation_incorrect": act_i[li, ei],
                "success_delta":             act_c[li, ei] - act_i[li, ei],
            }
            # Category-level deltas expose where things break (e.g. Ordering)
            for cat in LITE_CATEGORIES:
                nc = cat_cnt_c[cat]
                ni = cat_cnt_i[cat]
                ca = (cat_act_c[cat][li, ei] / nc) if nc > 0 else 0.0
                ia = (cat_act_i[cat][li, ei] / ni) if ni > 0 else 0.0
                rec[f"delta_{cat}"] = ca - ia
            records.append(rec)

    df_rank = pd.DataFrame(records).sort_values("success_delta", ascending=False)
    df_rank.to_csv(EXPERT_RANK_CSV, index=False)
    print(f"\nExpert ranking → {EXPERT_RANK_CSV}")
    print("\nTop-10 by global success_delta:")
    print(df_rank[["expert_label", "mean_activation_correct",
                   "mean_activation_incorrect", "success_delta"]
                  ].head(10).to_string(index=False))

    # Also print ordering-specific ranking (diagnose 0% ordering accuracy)
    print("\nTop-10 by delta_ordering (experts most active on correct ordering answers):")
    ord_top = df_rank.nlargest(10, "delta_ordering") if "delta_ordering" in df_rank.columns else df_rank.head(10)
    print(ord_top[["expert_label", "delta_ordering",
                   "mean_activation_correct"]].to_string(index=False))

    return df_rank


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 ── Spatial reasoning expert leaderboard
# ─────────────────────────────────────────────────────────────────────────────
def find_spatial_experts(df_rank: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Step 4: Surface the expert nodes that are both:
      (a) strongly activated on visual tokens when the model is correct
      (b) show a positive correlation between their activation and accuracy

    Also surfaces per-category specialists so you can see which experts drive
    Height vs Rotation vs Ordering (even if ordering accuracy is near 0).

    Saves spatial_expert_leaderboard.csv.
    """
    # ── Global leaderboard ────────────────────────────────────────────────────
    board = df_rank[
        (df_rank["mean_activation_correct"] >= MIN_ACTIVATION_CORRECT) &
        (df_rank["success_delta"]           >= MIN_SUCCESS_DELTA)
    ].copy().reset_index(drop=True)

    board.index       = range(1, len(board) + 1)
    board.index.name  = "rank"

    top = board.head(LEADERBOARD_TOP_N)
    top.to_csv(LEADERBOARD_CSV)

    print(f"\n{'='*66}")
    print(f"  TOP-{LEADERBOARD_TOP_N} SPATIAL REASONING EXPERTS  "
          f"(activation ≥ {MIN_ACTIVATION_CORRECT}, Δ ≥ {MIN_SUCCESS_DELTA})")
    print(f"{'='*66}")
    print(top[["expert_label", "mean_activation_correct",
               "mean_activation_incorrect", "success_delta"]].to_string())

    # ── Per-category specialists ──────────────────────────────────────────────
    print(f"\n{'─'*66}")
    print("  CATEGORY SPECIALISTS  (top-10 per category by delta_<cat>)")
    print(f"{'─'*66}")
    for cat in LITE_CATEGORIES:
        col = f"delta_{cat}"
        if col not in df_rank.columns:
            continue
        specialists = (
            df_rank[df_rank["mean_activation_correct"] >= MIN_ACTIVATION_CORRECT]
            .nlargest(10, col)[["expert_label", "mean_activation_correct", col]]
        )
        print(f"\n  [{cat.upper()}]")
        print(specialists.to_string(index=False))

    # ── Ordering failure diagnosis ─────────────────────────────────────────────
    if "delta_ordering" in df_rank.columns:
        print(f"\n{'─'*66}")
        print("  ORDERING FAILURE DIAGNOSIS")
        print("  Experts with high activation on ordering tasks but negative delta:")
        print("  (These fire heavily but correlate with WRONG answers - suspect nodes)")
        print(f"{'─'*66}")
        suspects = (
            df_rank[
                (df_rank["mean_activation_correct"] >= MIN_ACTIVATION_CORRECT) &
                (df_rank["delta_ordering"] < -MIN_SUCCESS_DELTA)
            ]
            .nsmallest(10, "delta_ordering")
            [["expert_label", "mean_activation_correct", "delta_ordering"]]
        )
        if len(suspects):
            print(suspects.to_string(index=False))
        else:
            print("  None found at current thresholds.")

    print(f"\nLeaderboard → {LEADERBOARD_CSV}")
    return board


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Step 0: load model + dataset ─────────────────────────────────────────
    model, processor, meta = load_model_and_processor()
    df = load_legolit_dataset()

    # ── Step 1: inference + extraction ───────────────────────────────────────
    print("\n── Step 1: Inference + router extraction ──────────────────────────")
    if RESULTS_JSON.exists():
        print(f"Found existing {RESULTS_JSON} - loading (delete to re-run inference).")
        with open(RESULTS_JSON) as f:
            results = json.load(f)
    else:
        results = run_inference_and_extract(model, processor, df, meta)

    # Free VRAM - Steps 2-4 are CPU/numpy only
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    # ── Step 2: heatmaps ─────────────────────────────────────────────────────
    print("\n── Step 2: Category heatmaps ──────────────────────────────────────")
    heatmaps = compute_category_heatmaps(results, meta)

    # ── Step 3: expert accuracy ranking ──────────────────────────────────────
    print("\n── Step 3: Expert accuracy ranking ────────────────────────────────")
    df_rank = rank_experts_by_accuracy(results, meta)

    # ── Step 4: leaderboard ──────────────────────────────────────────────────
    print("\n── Step 4: Spatial expert leaderboard ─────────────────────────────")
    leaderboard = find_spatial_experts(df_rank, meta)

    print(f"\nDone. All outputs in {OUTPUT_DIR}")
