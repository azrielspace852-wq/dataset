#!/usr/bin/env python3
"""
Generate a multilingual pretraining dataset (JSONL) from CulturaX.
Target: ~150M tokens across 5 languages.
"""

import json
from pathlib import Path
from datasets import load_dataset

LANGUAGES = {
    "ind": "Indonesian",
    "eng": "English",
    "zho": "Chinese",
    "fra": "French",
    "kor": "Korean",
}

TOKENS_PER_LANG = 30_000_000          # 30M × 5 = 150M
DATASET_NAME    = "uonlp/CulturaX"
OUTPUT_DIR      = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def count_tokens(text: str) -> int:
    return len(text.split())


def stream_language_to_file(lang_code: str, target_tokens: int, out_path: Path) -> int:
    """Stream dataset CulturaX untuk 1 bahasa, tulis JSONL langsung ke disk."""
    print(f"[INFO] Streaming {lang_code} -> {out_path} (target {target_tokens:,} tokens)", flush=True)

    ds = load_dataset(
        DATASET_NAME,
        lang_code,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    total_tokens = 0
    next_report  = 1_000_000

    with open(out_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if not text or not text.strip():
                continue

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total_tokens += count_tokens(text)

            if total_tokens >= next_report:
                print(f"[INFO] {lang_code}: {total_tokens:,} tokens", flush=True)
                next_report += 1_000_000

            if total_tokens >= target_tokens:
                print(f"[INFO] {lang_code}: reached ~{total_tokens:,} tokens", flush=True)
                break

    return total_tokens


def main() -> None:
    summary = {}

    for lang_code, lang_name in LANGUAGES.items():
        out_path = OUTPUT_DIR / f"{lang_code}.jsonl"
        tok = stream_language_to_file(lang_code, TOKENS_PER_LANG, out_path)
        summary[lang_code] = {"language": lang_name, "tokens": tok, "file": str(out_path)}
        print(f"[OK] {out_path}  ({tok:,} tokens)", flush=True)

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    total = sum(v["tokens"] for v in summary.values())
    print(f"[DONE] Total estimated tokens: {total:,}", flush=True)


if __name__ == "__main__":
    main()
