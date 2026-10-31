#!/usr/bin/env python3
"""
Generate a multilingual pretraining dataset (JSONL) from CulturaX.
Target: ~150M tokens across 5 languages.

Usage:
    python scripts/generate_dataset.py                # semua bahasa
    python scripts/generate_dataset.py --lang id      # satu bahasa saja
"""

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

from datasets import load_dataset

# CulturaX uses ISO 639-1 (2-letter) language codes.
LANGUAGES = {
    "id": "Indonesian",
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "ko": "Korean",
}

TOKENS_PER_LANG = 30_000_000
DATASET_NAME    = "uonlp/CulturaX"
OUTPUT_DIR      = Path("output")


def count_tokens(text: str) -> int:
    return len(text.split())


def stream_language_to_file(lang_code: str, target_tokens: int, out_path: Path) -> int:
    print(f"[INFO] Streaming {lang_code} -> {out_path} (target {target_tokens:,} tokens)",
          flush=True)

    ds = None
    total_tokens = 0
    next_report  = 1_000_000

    try:
        ds = load_dataset(
            DATASET_NAME,
            lang_code,
            split="train",
            streaming=True,
        )

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
                    print(f"[INFO] {lang_code}: reached ~{total_tokens:,} tokens",
                          flush=True)
                    break

            # Paksa data sampai ke disk sebelum file ditutup
            f.flush()
            os.fsync(f.fileno())

    finally:
        # Lepas iterator & koneksi HTTP sebelum interpreter shutdown
        try:
            if ds is not None:
                del ds
        except Exception:
            pass
        gc.collect()

    return total_tokens


def write_summary(summary: dict) -> None:
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"[OK] {summary_path}", flush=True)


def main(langs: list[str]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary = {}
    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    for lang_code in langs:
        if lang_code not in LANGUAGES:
            print(f"[WARN] skip unknown language: {lang_code}", flush=True)
            continue

        out_path = OUTPUT_DIR / f"{lang_code}.jsonl"
        tok = stream_language_to_file(lang_code, TOKENS_PER_LANG, out_path)
        summary[lang_code] = {
            "language": LANGUAGES[lang_code],
            "tokens": tok,
            "file": str(out_path),
        }
        write_summary(summary)
        print(f"[OK] {out_path}  ({tok:,} tokens)", flush=True)

    total = sum(v["tokens"] for v in summary.values())
    print(f"[DONE] Total estimated tokens: {total:,}", flush=True)


def parse_args() -> list[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        action="append",
        help="Kode bahasa (bisa diulang). Default: semua bahasa.",
    )
    args = parser.parse_args()
    if args.lang:
        return args.lang
    return list(LANGUAGES.keys())


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Skip Python shutdown -> hindari PyGILState_Release crash
        os._exit(1)

    sys.stdout.flush()
    sys.stderr.flush()
    # KELUAR PAKSA: lewati shutdown cleanup (tempat crash PyGILState_Release terjadi)
    os._exit(0)
