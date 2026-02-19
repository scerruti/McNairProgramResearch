"""
LEGO Dataset Adapter for VADAR.

Downloads the LEGO-Puzzles benchmark TSV and converts it into VADAR's
expected annotations.json format so the VADAR pipeline (signature generation,
API generation, program generation, execution) can run on LEGO tasks.

The LEGO benchmark has 1,100 multiple-choice questions across 11 spatial
reasoning categories.  Each question references one or more images that are
base64-encoded inside the TSV.

Usage:
    python -m datasets.lego_dataset              # full benchmark (1100 Qs)
    python -m datasets.lego_dataset --lite        # lite benchmark (220 Qs)
    python -m datasets.lego_dataset --max-questions 20  # small test subset
"""

import argparse
import base64
import io
import json
import os
import string
import sys

import pandas as pd
from PIL import Image

LEGO_TSV_URL = "https://opencompass.openxlab.space/utils/VLMEval/LEGO.tsv"
LEGO_TSV_MD5 = "d595f50e1fb4d4eb12cbc95297893ffc"

LEGO_CATEGORIES = [
    "adjacency", "backwards", "dependency", "height", "multi_view",
    "next_step", "ordering", "outlier", "position", "rotation",
    "rotation_status",
]


def download_lego_tsv(dest_path: str) -> str:
    """Download the LEGO TSV if it doesn't already exist."""
    if os.path.exists(dest_path):
        print(f"LEGO TSV already exists at {dest_path}")
        return dest_path

    print(f"Downloading LEGO TSV from {LEGO_TSV_URL} ...")
    import urllib.request
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(LEGO_TSV_URL, dest_path)
    print(f"Saved to {dest_path}")
    return dest_path


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64-encoded image string into a PIL Image."""
    image_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def build_mcq_question_text(row: pd.Series) -> str:
    """
    Build the full question text including MCQ options.

    For 'sort' type questions, asks the model to output a letter ordering.
    For standard MCQ questions, asks the model to pick A/B/C/D.
    """
    question = row["question"]
    hint = row.get("hint", None)
    question_type = row.get("question_type", "mcq")

    options = {}
    for letter in string.ascii_uppercase:
        if letter in row.index and pd.notna(row[letter]):
            options[letter] = row[letter]

    text = ""
    if pd.notna(hint) if isinstance(hint, str) else hint:
        text += f"Hint: {hint}\n"

    text += f"Question: {question}\n"

    if options:
        text += "Options:\n"
        for key, val in options.items():
            text += f"  {key}. {val}\n"

    if question_type == "sort":
        text += (
            "Answer with only the sequence of letters that correctly "
            "orders the steps (e.g. 'BDAC')."
        )
    else:
        text += (
            "Answer with only the letter of the correct option "
            "(e.g. 'A', 'B', 'C', or 'D')."
        )

    return text


def convert_lego_to_vadar(
    tsv_path: str,
    output_dir: str,
    max_questions: int = -1,
    lite: bool = False,
) -> str:
    """
    Convert the LEGO TSV into VADAR's annotations.json + extracted images.

    Returns the path to the annotations.json file.
    """
    print(f"Loading LEGO TSV from {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(df)} rows")

    if lite:
        if "split" in df.columns:
            df = df[df["split"] == "lite"]
            print(f"Filtered to lite split: {len(df)} rows")
        else:
            df = df.head(220)
            print(f"Using first 220 rows as lite")

    if max_questions > 0:
        df = df.head(max_questions)
        print(f"Limited to {len(df)} questions")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    questions = []
    image_cache = {}

    for idx, row in df.iterrows():
        question_text = build_mcq_question_text(row)

        # Extract answer
        answer = str(row.get("answer", ""))

        # Extract category
        category = str(row.get("category", "unknown"))

        # Extract question type
        question_type = str(row.get("question_type", "mcq"))

        # Determine answer_type for VADAR's evaluation
        answer_type = "str"

        # Handle images - LEGO TSV encodes images as base64 in the 'image' column
        # Some rows may have multiple images separated by special encoding
        image_filename = f"lego_{row['index']}.png"
        image_path = os.path.join(images_dir, image_filename)

        if not os.path.exists(image_path):
            try:
                if "image" in row.index and pd.notna(row["image"]):
                    img = decode_base64_image(row["image"])
                    img.save(image_path)
                elif "image_path" in row.index and pd.notna(row["image_path"]):
                    # Image paths might be listed - use the first one
                    img_path_str = str(row["image_path"])
                    if "|" in img_path_str:
                        img_path_str = img_path_str.split("|")[0]
                    if os.path.exists(img_path_str):
                        img = Image.open(img_path_str).convert("RGB")
                        img.save(image_path)
                    else:
                        print(f"Warning: image not found for index {row['index']}")
                        continue
            except Exception as e:
                print(f"Warning: failed to decode image for index {row['index']}: {e}")
                continue

        # Build VADAR-format question dict
        question_dict = {
            "image_index": str(row["index"]),
            "question_index": str(row["index"]),
            "image_filename": image_filename,
            "question": question_text,
            "answer": answer,
            "answer_type": answer_type,
            "category": category,
            "question_type": question_type,
        }
        questions.append(question_dict)

    annotations = {"questions": questions}
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Created {len(questions)} questions in {annotations_path}")
    print(f"Images saved to {images_dir}")
    return annotations_path


def main():
    parser = argparse.ArgumentParser(description="Convert LEGO benchmark to VADAR format")
    parser.add_argument(
        "--tsv-path",
        default=None,
        help="Path to existing LEGO.tsv (will download if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/lego",
        help="Output directory for annotations.json and images",
    )
    parser.add_argument(
        "--max-questions",
        default=-1,
        type=int,
        help="Maximum number of questions to convert (-1 for all)",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Use LEGO-Lite subset (220 questions)",
    )
    args = parser.parse_args()

    if args.tsv_path is None:
        args.tsv_path = os.path.join(args.output_dir, "LEGO.tsv")
        download_lego_tsv(args.tsv_path)

    convert_lego_to_vadar(
        args.tsv_path,
        args.output_dir,
        max_questions=args.max_questions,
        lite=args.lite,
    )


if __name__ == "__main__":
    main()
