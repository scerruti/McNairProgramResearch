"""
Qwen3-VL-30B-A3B MoE Expert Activation Analysis on LEGOLite
============================================================
(Modified to record per-token expert activations and produce image/text-separated counts)
"""
import os
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
DATA_DIR = "/content/lego_data"
IMG_DIR = "/content/lego_images"

try:
    import flash_attn
    attn_impl = "flash_attention_2"
except Exception:
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

text_config = model.config.text_config
NUM_LAYERS = text_config.num_hidden_layers
NUM_EXPERTS = text_config.num_experts
TOP_K = text_config.num_experts_per_tok
print(f"Loaded. Layers={NUM_LAYERS}, Experts={NUM_EXPERTS}, Top-K={TOP_K}")

# ── Hook infrastructure ──────────────────────────────────────────────────────
# expert_log stores per-question+layer data:
# expert_log[(q_id, layer_idx)] = {"per_token": defaultdict(Counter), "global": Counter()}
expert_log = defaultdict(lambda: {"per_token": defaultdict(Counter), "global": Counter()})
current_question_id = None
hooks = []

def make_router_hook(layer_idx):
    def hook_fn(module, input, output):
        """
        The router/gate forward may return a tuple where router indices are the third element,
        or may return the indices directly. We attempt to extract a tensor of selected expert
        indices aligned to token positions. The tensor shape may vary by implementation,
        so this hook handles common shapes (batch, seq, topk) and (batch, topk).
        """
        # Extract router_indices robustly
        router_indices = None
        if isinstance(output, (tuple, list)) and len(output) >= 3:
            router_indices = output[2]
        else:
            router_indices = output

        if router_indices is None:
            return

        try:
            ri = router_indices.detach().cpu()
        except Exception:
            return

        # ri might be e.g. (batch, seq_len, topk)
        if ri.ndim == 3:
            batch, seq_len, topk = ri.shape
            for b in range(batch):
                for t in range(seq_len):
                    vals = ri[b, t]
                    # filter negative/sentinel values if present
                    experts = [int(x) for x in vals.flatten().tolist() if int(x) >= 0]
                    if experts:
                        expert_log[(current_question_id, layer_idx)]["per_token"][t].update(experts)
                        expert_log[(current_question_id, layer_idx)]["global"].update(experts)
        elif ri.ndim == 2:
            # shape (batch, topk) or (batch, seq_len) - fallback to flatten per-batch
            batch, k = ri.shape
            for b in range(batch):
                experts = [int(x) for x in ri[b].flatten().tolist() if int(x) >= 0]
                if experts:
                    # we don't have token positions here; aggregate globally
                    expert_log[(current_question_id, layer_idx)]["global"].update(experts)
        else:
            # unknown shape: fallback to global flatten
            indices = [int(x) for x in ri.flatten().tolist() if int(x) >= 0]
            expert_log[(current_question_id, layer_idx)]["global"].update(indices)
    return hook_fn

def find_decoder_layers():
    for attr_path in ["model.layers", "model.model.layers"]:
        candidate = model
        try:
            for attr in attr_path.split("."):
                candidate = getattr(candidate, attr)
            if hasattr(candidate, '__len__') and len(candidate) == NUM_LAYERS:
                return candidate
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
            hook = layer.mlp.gate.register_forward_hook(make_router_hook(layer_idx))
            hooks.append(hook)
    print(f"Registered {len(hooks)} router hooks.")

def remove_hooks():
    global hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass
    hooks = []

# ── Dataset loading ───────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

tsv_path = os.path.join(DATA_DIR, "LEGO.tsv")
if not os.path.exists(tsv_path):
    print("Downloading LEGO.tsv...")
    import urllib.request
    urllib.request.urlretrieve(LEGO_TSV_URL, tsv_path)

df_full = pd.read_csv(tsv_path, sep="\t")
LITE_CATEGORIES = sorted(df_full["category"].unique())
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
for row_idx in range(len(df_lite)):
    row = df_lite.iloc[row_idx]
    idx = str(row["index"])
    image_path_map[idx] = save_image_for_row(row)

def build_messages(row):
    idx = str(row["index"])
    question = row["question"]
    question_type = row["question_type"]
    option_cols = [letter for letter in string.ascii_uppercase if letter in row.index and pd.notna(row[letter])]

    prompt = f"Question: {question}\n"
    if option_cols:
        prompt += "Options:\n"
        for option_letter in option_cols:
            option_value = row[option_letter]
            if isinstance(option_value, str) and "<image" in option_value:
                prompt += f"{option_letter}. [see image]\n"
            else:
                prompt += f"{option_letter}. {option_value}\n"

    if question_type == "sort":
        prompt += "Please respond with only the sequence of letters (e.g., 'BDAC') that correctly orders the steps.\n"
    else:
        prompt += "Please respond with only the letter of the correct answer.\n"

    content = []
    for image_path in image_path_map.get(idx, []):
        content.append({"type": "image", "image": f"file://{image_path}"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]

# helper: find subsequence
def find_subsequence(full, sub):
    if not sub or not full:
        return -1
    n, m = len(full), len(sub)
    for i in range(n - m + 1):
        if full[i:i+m] == sub:
            return i
    return -1

# ── Run inference ────────────────────────────────────────────────────────────
if hasattr(model, 'device') and model.device.type != 'meta':
    input_device = model.device
else:
    input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

register_hooks()
results = []
all_expert_data = {}

for row_idx in tqdm(range(len(df_lite)), desc="LEGO"):
    row = df_lite.iloc[row_idx]
    q_id = str(row["index"])
    category = row["category"]
    answer_gt = str(row["answer"])
    current_question_id = q_id

    try:
        messages = build_messages(row)
        # full inputs (image + text)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(input_device)

        # text-only inputs for token position alignment
        text_only = [{"role": "user", "content": [{"type": "text", "text": messages[0]["content"][-1]["text"]}]}]
        text_inputs = processor.apply_chat_template(
            text_only,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(input_device)

        # generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64)

        generated_ids = [full_output[len(full_input):] for full_input, full_output in zip(inputs["input_ids"], output_ids)]
        prediction = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    except Exception as error:
        prediction = f"ERROR: {error}"

    # Build token modality mapping: find where text token sequence occurs inside the full input sequence
    image_token_positions = set()
    text_token_positions = set()
    try:
        full_ids = inputs["input_ids"][0].detach().cpu().tolist()
        text_ids = text_inputs["input_ids"][0].detach().cpu().tolist()
        start = find_subsequence(full_ids, text_ids)
        if start != -1:
            text_token_positions = set(range(start, start + len(text_ids)))
            image_token_positions = set(range(len(full_ids))) - text_token_positions
        else:
            # fallback heuristic: if processor gives patch tokens separately, try to use attention_mask or other keys
            # default: mark all tokens as text (safe fallback)
            text_token_positions = set(range(len(full_ids)))
            image_token_positions = set()
    except Exception:
        # if anything goes wrong, fallback to previous behaviour (global-only)
        text_token_positions = set()
        image_token_positions = set()

    # Aggregate expert activations
    question_expert_freq_image = Counter()
    question_expert_freq_text = Counter()
    question_expert_freq = Counter()
    per_layer = {}

    for layer_idx in range(NUM_LAYERS):
        key = (q_id, layer_idx)
        if key in expert_log:
            layer_data = expert_log[key]
            # per_token is dict[token_pos] -> Counter
            per_token = layer_data.get("per_token", {})
            layer_token_map = {}
            for token_pos, c in per_token.items():
                layer_token_map[str(token_pos)] = dict(c)
                if token_pos in text_token_positions:
                    question_expert_freq_text.update(c)
                else:
                    # if we couldn't detect text positions, this will be considered image by default
                    question_expert_freq_image.update(c)
            # also include global counters as fallback
            global_counter = layer_data.get("global", Counter())
            for expert_id, cnt in global_counter.items():
                question_expert_freq.update({expert_id: cnt})
            # store per-layer token map
            per_layer[str(layer_idx)] = layer_token_map

    # combine: if per-token mapping gave counts, prefer those; else use global
    # build final combined counters (prefer token-level counts)
    final_image_freq = question_expert_freq_image if sum(question_expert_freq_image.values()) > 0 else question_expert_freq
    final_text_freq = question_expert_freq_text if sum(question_expert_freq_text.values()) > 0 else Counter()

    is_correct = prediction.strip().upper().startswith(answer_gt.strip().upper())

    # Build all_expert_data entry with explicit modality-separated counts
    all_expert_data[q_id] = {
        "question_id": q_id,
        "category": category,
        "prediction": prediction,
        "ground_truth": answer_gt,
        "correct": is_correct,
        "expert_frequency": {str(expert_id): int(count) for expert_id, count in (question_expert_freq.items() or {}).items()},
        "image_expert_frequency": {str(expert_id): int(count) for expert_id, count in final_image_freq.items()},
        "text_expert_frequency": {str(expert_id): int(count) for expert_id, count in final_text_freq.items()},
        "per_layer_experts": per_layer,
    }
    results.append({
        "question_id": q_id, "category": category,
        "prediction": prediction, "ground_truth": answer_gt, "correct": is_correct,
    })

    # cleanup per-question logs to limit memory
    for log_key in [log_key for log_key in list(expert_log.keys()) if log_key[0] == q_id]:
        del expert_log[log_key]
    torch.cuda.empty_cache()

remove_hooks()

results_df = pd.DataFrame(results)
overall_acc = results_df["correct"].mean()
print(f"\nOverall accuracy: {overall_acc:.1%}")
for cat in LITE_CATEGORIES:
    cat_df = results_df[results_df["category"] == cat]
    if len(cat_df) > 0:
        print(f"  {cat:12s}: {cat_df['correct'].mean():.1%} ({int(cat_df['correct'].sum())}/{len(cat_df)})")

category_expert_freq = defaultdict(Counter)
for q_id, question_data in all_expert_data.items():
    freq = {int(expert_id): count for expert_id, count in question_data["expert_frequency"].items()}
    category_expert_freq[question_data["category"]].update(freq)

category_summary = {}
for cat in LITE_CATEGORIES:
    freq = category_expert_freq[cat]
    if not freq:
        continue
    top_experts = freq.most_common(15)
    category_summary[cat] = {
        "top_15_experts": [{"expert_id": expert_id, "activation_count": count} for expert_id, count in top_experts],
        "total_activations": int(sum(freq.values())),
        "unique_experts_used": int(len(freq)),
    }

analysis_report = {
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

output_path = "/content/lego_moe_expert_analysis.json"
with open(output_path, "w") as output_file:
    json.dump(analysis_report, output_file, indent=2)
print(f"\nExported to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
