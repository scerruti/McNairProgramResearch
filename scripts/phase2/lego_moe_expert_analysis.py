"""
Qwen3-VL-30B-A3B MoE Expert Activation Analysis on LEGOLite
============================================================
Runs on NVIDIA A100 80GB. Model weights and data cached under /workspace.

Runs the model locally on the LEGOLite benchmark (height, position, rotation,
ordering) and captures which of the 128 MoE experts activate for each question.
Exports per-question expert frequency maps as JSON grouped by spatial category.

Model architecture (from config.json):
  - 48 decoder layers, all with MoE blocks (decoder_sparse_step=1)
  - 128 experts per layer, 8 active per token (num_experts_per_tok=8)
  - Router: Qwen3VLMoeTextTopKRouter accessed via layer.mlp.gate
"""

import os

# Redirect all HuggingFace downloads to /workspace (/ is full)
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/transformers"

import json
import string
import base64
from io import BytesIO
from collections import Counter, defaultdict

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
LEGO_TSV_URL = "https://opencompass.openxlab.space/utils/VLMEval/LEGO.tsv"
# LITE_CATEGORIES = ["height", "position", "rotation", "ordering"]
DATA_DIR = "/workspace/lego_data"
IMG_DIR = "/workspace/lego_images"

# ── Load model ────────────────────────────────────────────────────────────────

try:
    import flash_attn
    attn_impl = "flash_attention_2"
except ImportError:
    attn_impl = "sdpa"

print(f"Loading {MODEL_ID} (attn: {attn_impl})...")
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    device_map="auto",
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

text_cfg = model.config.text_config
NUM_LAYERS = text_cfg.num_hidden_layers
NUM_EXPERTS = text_cfg.num_experts
TOP_K = text_cfg.num_experts_per_tok
print(f"Loaded. Layers={NUM_LAYERS}, Experts={NUM_EXPERTS}, Top-K={TOP_K}")

# ── Hook infrastructure ──────────────────────────────────────────────────────

expert_log = defaultdict(lambda: Counter())
current_question_id = None
hooks = []


def make_router_hook(layer_idx):
    def hook_fn(module, input, output):
        _, _, router_indices = output
        indices = router_indices.detach().cpu().flatten().tolist()
        key = (current_question_id, layer_idx)
        expert_log[key].update(indices)
    return hook_fn


def find_decoder_layers():
    for attr_path in ["model.layers", "model.model.layers"]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, '__len__') and len(obj) == NUM_LAYERS:
                return obj
        except AttributeError:
            continue
    for name, module in model.named_modules():
        if hasattr(module, '__len__'):
            try:
                if len(module) == NUM_LAYERS and hasattr(module[0], 'mlp'):
                    return module
            except (TypeError, IndexError):
                continue
    raise RuntimeError("Could not find decoder layers.")


decoder_layers = find_decoder_layers()


def register_hooks():
    global hooks
    remove_hooks()
    for layer_idx, layer in enumerate(decoder_layers):
        if hasattr(layer.mlp, 'gate'):
            h = layer.mlp.gate.register_forward_hook(make_router_hook(layer_idx))
            hooks.append(h)
    print(f"Registered {len(hooks)} router hooks.")


def remove_hooks():
    global hooks
    for h in hooks:
        h.remove()
    hooks = []


# ── Load dataset ──────────────────────────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

tsv_path = os.path.join(DATA_DIR, "LEGO.tsv")
if not os.path.exists(tsv_path):
    print("Downloading LEGO.tsv...")
    import urllib.request
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(LEGO_TSV_URL, context=ctx) as r, open(tsv_path, "wb") as f:
        f.write(r.read())

df_full = pd.read_csv(tsv_path, sep="\t")
LITE_CATEGORIES = sorted(df_full["category"].unique())
# df_lite = df_full[df_full["category"].isin(LITE_CATEGORIES)].reset_index(drop=True)
df_lite = df_full.reset_index(drop=True)
print(f"\nLEGO: {len(df_lite)} questions")

full_index_map = {str(row["index"]): row for _, row in df_full.iterrows()}


def save_image_for_row(row):
    idx = str(row["index"])
    img_col = row.get("image", None)
    if pd.isna(img_col) or img_col is None:
        return []

    img_str = str(img_col)
    if len(img_str) < 20:
        try:
            ref_idx = str(int(float(img_str)))
            if ref_idx in full_index_map:
                img_str = str(full_index_map[ref_idx]["image"])
            else:
                return []
        except (ValueError, TypeError):
            return []

    out_path = os.path.join(IMG_DIR, f"{idx}.png")
    if os.path.exists(out_path):
        return [out_path]

    try:
        img_bytes = base64.b64decode(img_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.save(out_path)
        return [out_path]
    except Exception:
        return []


image_path_map = {}
for i in range(len(df_lite)):
    row = df_lite.iloc[i]
    idx = str(row["index"])
    image_path_map[idx] = save_image_for_row(row)


def build_messages(row):
    idx = str(row["index"])
    question = row["question"]
    question_type = row["question_type"]
    option_cols = [c for c in string.ascii_uppercase if c in row.index and pd.notna(row[c])]

    prompt = f"Question: {question}\n"
    if option_cols:
        prompt += "Options:\n"
        for c in option_cols:
            val = row[c]
            if isinstance(val, str) and "<image" in val:
                prompt += f"{c}. [see image]\n"
            else:
                prompt += f"{c}. {val}\n"

    if question_type == "sort":
        prompt += "Please respond with only the sequence of letters (e.g., 'BDAC') that correctly orders the steps.\n"
    else:
        prompt += "Please respond with only the letter of the correct answer.\n"

    content = []
    for p in image_path_map.get(idx, []):
        content.append({"type": "image", "image": f"file://{p}"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


# ── Run inference ─────────────────────────────────────────────────────────────

input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

register_hooks()
results = []
all_expert_data = {}

for idx in tqdm(range(len(df_lite)), desc="LEGO"):
    row = df_lite.iloc[idx]
    q_id = str(row["index"])
    category = row["category"]
    answer_gt = str(row["answer"])
    current_question_id = q_id

    try:
        messages = build_messages(row)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(input_device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64)

        generated_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
        prediction = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    except Exception as e:
        prediction = f"ERROR: {e}"

    question_expert_freq = Counter()
    per_layer = {}
    for layer_idx in range(NUM_LAYERS):
        key = (q_id, layer_idx)
        if key in expert_log:
            question_expert_freq.update(expert_log[key])
            per_layer[str(layer_idx)] = dict(expert_log[key])

    is_correct = prediction.strip().upper().startswith(answer_gt.strip().upper())

    all_expert_data[q_id] = {
        "question_id": q_id,
        "category": category,
        "prediction": prediction,
        "ground_truth": answer_gt,
        "correct": is_correct,
        "expert_frequency": {str(k): v for k, v in question_expert_freq.items()},
        "per_layer_experts": per_layer,
    }
    results.append({
        "question_id": q_id, "category": category,
        "prediction": prediction, "ground_truth": answer_gt, "correct": is_correct,
    })

    for k in [k for k in expert_log if k[0] == q_id]:
        del expert_log[k]
    torch.cuda.empty_cache()

remove_hooks()

# ── Results ───────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results)
overall_acc = results_df["correct"].mean()
print(f"\nOverall accuracy: {overall_acc:.1%}")
for cat in LITE_CATEGORIES:
    cat_df = results_df[results_df["category"] == cat]
    if len(cat_df) > 0:
        print(f"  {cat:12s}: {cat_df['correct'].mean():.1%} ({int(cat_df['correct'].sum())}/{len(cat_df)})")

# Category expert summary
category_expert_freq = defaultdict(Counter)
for q_id, data in all_expert_data.items():
    freq = {int(k): v for k, v in data["expert_frequency"].items()}
    category_expert_freq[data["category"]].update(freq)

category_summary = {}
for cat in LITE_CATEGORIES:
    freq = category_expert_freq[cat]
    if not freq:
        continue
    top_experts = freq.most_common(15)
    category_summary[cat] = {
        "top_15_experts": [{"expert_id": eid, "activation_count": cnt} for eid, cnt in top_experts],
        "total_activations": sum(freq.values()),
        "unique_experts_used": len(freq),
    }

# Export
output = {
    "model": MODEL_ID,
    "benchmark": "LEGO",
    "categories": LITE_CATEGORIES,
    "num_questions": len(df_lite),
    "accuracy": {
        "overall": float(overall_acc),
        **{cat: float(results_df[results_df["category"] == cat]["correct"].mean())
           for cat in LITE_CATEGORIES}
    },
    "category_expert_summary": category_summary,
    "per_question_results": all_expert_data,
}

output_path = "/workspace/lego_moe_expert_analysis.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nExported to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
