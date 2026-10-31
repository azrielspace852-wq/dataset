#!/usr/bin/env python3
"""
Generate a multilingual pretraining dataset (JSONL) from CulturaX.
Target: ~150M tokens across 5 languages.
"""

import json
import os
import sys
from pathlib import Path
from datasets import load_dataset

# ---------- Konfigurasi ----------
LANGUAGES = {
    "ind": "Indonesian",
    "eng": "English",
    "zho": "Chinese",
    "fra": "French",
    "kor": "Korean",
}

# Target token per bahasa (dibagi rata)
TOKENS_PER_LANG = 30_000_000  # 30M × 5 = 150M

# Dataset
DATASET_NAME = "uonlp/CulturaX"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Fungsi bantu ----------
def count_tokens(text: str) -> int:
    """Perkiraan kasar jumlah token (whitespace split)."""
    return len(text.split())

def stream_language(lang_code: str, target_tokens: int):
    """
    Stream dataset CulturaX untuk satu bahasa,
    kumpulkan teks hingga target token tercapai.
    """
    print(f"[INFO] Streaming {lang_code} … target {target_tokens:,} tokens")
    ds = load_dataset(
        DATASET_NAME,
        lang_code,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    total_tokens = 0
    buffer = []

    for example in ds:
        text = example.get("text", "")
        if not text or not text.strip():
            continue

        buffer.append(text)
        total_tokens += count_tokens(text)

        if total_tokens >= target_tokens:
            print(f"[INFO] {lang_code}: reached ~{total_tokens:,} tokens")
            break

    return buffer, total_tokens

# ---------- Main ----------
def main():
    all_data = {}

    for lang_code, lang_name in LANGUAGES.items():
        texts, tok_count = stream_language(lang_code, TOKENS_PER_LANG)
        all_data[lang_code] = {
            "language": lang_name,
            "tokens": tok_count,
            "texts": texts,
        }

    # Tulis JSONL per bahasa
    for lang_code, payload in all_data.items():
        out_path = OUTPUT_DIR / f"{lang_code}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for text in payload["texts"]:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        print(f"[OK] {out_path}  ({payload['tokens']:,} tokens)")

    # Ringkasan
    total = sum(v["tokens"] for v in all_data.values())
    print(f"[DONE] Total estimated tokens: {total:,}")

if __name__ == "__main__":
    main()
