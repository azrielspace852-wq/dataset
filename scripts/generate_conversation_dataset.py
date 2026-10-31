#!/usr/bin/env python3
"""
Generate multilingual conversation (instruction/chat) dataset from CulturaX.

Target: 200M tokens total (40M per language × 5 languages).

Format (mixed in the same JSONL, chosen per example):
  1. {"question":"…", "answer":"<think>…</think>\n<output>…</output>"}
  2. {"prompt":"…",   "response":"<think>…</think>\n<output>…</output>"}
  3. {"instructions":"…",
      "messages":[
        {"role":"user","content":"…"},
        {"role":"assistant","content":"<think>…</think>\n<output>…</output>"}
      ]}

<think> dan <output> dibuat oleh generator (bukan diambil dari sumber):
  - <think>  = template penalaran per-bahasa + topik yang diambil dari dokumen
  - <output> = ringkasan ekstraktif (beberapa kalimat pertama) dari dokumen

Usage:
  python scripts/generate_conversation_dataset.py
  python scripts/generate_conversation_dataset.py --lang id
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

print("[BOOT] generate_conversation_dataset.py", flush=True)

try:
    from datasets import load_dataset
    import datasets as _ds
    print(f"[BOOT] datasets {_ds.__version__}", flush=True)
except Exception:
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)

# ============== Config ==============
LANGUAGES = {
    "id": "Indonesian",
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "ko": "Korean",
}

TOTAL_TOKENS      = 200_000_000
TOKENS_PER_LANG   = TOTAL_TOKENS // len(LANGUAGES)   # 40M / bahasa
DATASET_NAME      = "uonlp/CulturaX"
OUTPUT_DIR        = Path("output").resolve()
MAX_DOC_CHARS     = 1500
MIN_DOC_CHARS     = 200

# ============== Template per bahasa ==============
TEMPLATES = {
    "id": {
        "qa": [
            "Ringkaslah teks berikut:\n\n{doc}",
            "Apa gagasan utama dari teks berikut?\n\n{doc}",
            "Jelaskan isi teks berikut dengan bahasa Anda sendiri:\n\n{doc}",
            "Sebutkan poin-poin penting dari teks berikut:\n\n{doc}",
        ],
        "prompt": [
            "Berikan ringkasan singkat dari teks berikut.\n\n{doc}",
            "Tolong jelaskan inti dari paragraf berikut.\n\n{doc}",
        ],
        "instruction": [
            "Anda adalah asisten yang membantu. Bacalah konteks berikut lalu jawab pertanyaan pengguna.",
            "Gunakan konteks di bawah ini untuk menjawab pertanyaan pengguna.",
            "Sebagai asisten, jawablah pertanyaan berdasarkan konteks yang diberikan.",
        ],
        "user_ask": [
            "Apa inti dari teks ini?",
            "Jelaskan secara singkat isi teks ini.",
            "Sebutkan tiga poin penting dari teks ini.",
        ],
        "context_prefix": "Konteks:\n{doc}",
        "think": "Untuk menjawab, saya baca konteks dengan cermat. Gagasan utama yang saya tangkap: {topic}. Saya akan menyusun jawaban ringkas berdasarkan poin tersebut.",
        "output": "Berdasarkan konteks, inti teks tersebut adalah: {summary}",
    },
    "en": {
        "qa": [
            "Summarize the following text:\n\n{doc}",
            "What is the main idea of the following text?\n\n{doc}",
            "Explain the following text in your own words:\n\n{doc}",
            "List the key points from the following text:\n\n{doc}",
        ],
        "prompt": [
            "Provide a short summary of the following text.\n\n{doc}",
            "Please explain the core of the following paragraph.\n\n{doc}",
        ],
        "instruction": [
            "You are a helpful assistant. Read the context below and answer the user's question.",
            "Use the context below to answer the user's question.",
            "As an assistant, answer the question based on the provided context.",
        ],
        "user_ask": [
            "What is the main idea of this text?",
            "Briefly explain what this text is about.",
            "List three important points from this text.",
        ],
        "context_prefix": "Context:\n{doc}",
        "think": "To answer, I read the context carefully. The main idea I identified: {topic}. I'll craft a concise answer based on that.",
        "output": "Based on the context, the text is about: {summary}",
    },
    "zh": {
        "qa": [
            "请总结以下文本：\n\n{doc}",
            "以下文本的主要思想是什么？\n\n{doc}",
            "用你自己的话解释以下文本：\n\n{doc}",
            "列出以下文本的要点：\n\n{doc}",
        ],
        "prompt": [
            "请对以下文本做简要总结。\n\n{doc}",
            "请解释以下段落的核心内容。\n\n{doc}",
        ],
        "instruction": [
            "你是一个乐于助人的助手。请阅读以下上下文并回答用户的问题。",
            "使用以下上下文回答用户的问题。",
            "作为助手，请根据提供的上下文回答问题。",
        ],
        "user_ask": [
            "这段文本的主要思想是什么？",
            "简要解释这段文本的内容。",
            "列出这段文本的三个要点。",
        ],
        "context_prefix": "上下文：\n{doc}",
        "think": "为了回答，我先仔细阅读上下文。我识别出的主要思想：{topic}。基于此，我会组织一个简洁的回答。",
        "output": "根据上下文，这段文本主要讲的是：{summary}",
    },
    "fr": {
        "qa": [
            "Résumez le texte suivant :\n\n{doc}",
            "Quelle est l'idée principale du texte suivant ?\n\n{doc}",
            "Expliquez le texte suivant avec vos propres mots :\n\n{doc}",
            "Énumérez les points clés du texte suivant :\n\n{doc}",
        ],
        "prompt": [
            "Donnez un bref résumé du texte suivant.\n\n{doc}",
            "Veuillez expliquer le cœur du paragraphe suivant.\n\n{doc}",
        ],
        "instruction": [
            "Vous êtes un assistant utile. Lisez le contexte ci-dessous puis répondez à la question de l'utilisateur.",
            "Utilisez le contexte ci-dessous pour répondre à la question de l'utilisateur.",
            "En tant qu'assistant, répondez à la question en vous basant sur le contexte fourni.",
        ],
        "user_ask": [
            "Quelle est l'idée principale de ce texte ?",
            "Expliquez brièvement de quoi parle ce texte.",
            "Citez trois points importants de ce texte.",
        ],
        "context_prefix": "Contexte :\n{doc}",
        "think": "Pour répondre, je lis attentivement le contexte. L'idée principale identifiée : {topic}. Je vais formuler une réponse concise sur cette base.",
        "output": "D'après le contexte, le texte porte sur : {summary}",
    },
    "ko": {
        "qa": [
            "다음 텍스트를 요약하세요:\n\n{doc}",
            "다음 텍스트의 주요 아이디어는 무엇인가요?\n\n{doc}",
            "다음 텍스트를 자신의 말로 설명하세요:\n\n{doc}",
            "다음 텍스트의 핵심 사항을 나열하세요:\n\n{doc}",
        ],
        "prompt": [
            "다음 텍스트를 간단히 요약해 주세요.\n\n{doc}",
            "다음 문단의 핵심을 설명해 주세요.\n\n{doc}",
        ],
        "instruction": [
            "당신은 유용한 어시스턴트입니다. 아래 컨텍스트를 읽고 사용자의 질문에 답하세요.",
            "아래 컨텍스트를 사용하여 사용자의 질문에 답하세요.",
            "어시스턴트로서 주어진 컨텍스트를 바탕으로 질문에 답하세요.",
        ],
        "user_ask": [
            "이 텍스트의 주요 아이디어는 무엇인가요?",
            "이 텍스트가 무엇에 관한 것인지 간단히 설명하세요.",
            "이 텍스트의 중요한 점 세 가지를 나열하세요.",
        ],
        "context_prefix": "컨텍스트:\n{doc}",
        "think": "답하기 위해 컨텍스트를 주의 깊게 읽습니다. 파악한 주요 아이디어: {topic}. 이를 바탕으로 간결한 답변을 구성하겠습니다.",
        "output": "컨텍스트에 따르면, 이 텍스트는 다음에 관한 것입니다: {summary}",
    },
}


# ============== Helpers ==============
def count_tokens(text: str) -> int:
    return len(text.split())


def split_sentences(text: str):
    # Terima terminator latin & CJK
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def make_summary(text: str, max_chars: int = 400) -> str:
    out = ""
    for s in split_sentences(text):
        if len(out) + len(s) + 1 > max_chars:
            break
        out = (out + " " + s).strip()
    return out or text[:max_chars]


def make_topic(text: str, max_chars: int = 120) -> str:
    sents = split_sentences(text)
    return (sents[0] if sents else text)[:max_chars]


def clean_doc(text: str):
    if not text:
        return None
    text = text.strip()
    if len(text) < MIN_DOC_CHARS:
        return None
    if text.lower().startswith("http"):
        return None
    return text[:MAX_DOC_CHARS]


def build_example(doc: str, lang: str, rng: random.Random) -> dict:
    t = TEMPLATES[lang]
    summary = make_summary(doc)
    topic   = make_topic(doc)
    think   = t["think"].format(topic=topic)
    output  = t["output"].format(summary=summary)
    answer  = f"<think>{think}</think>\n<output>{output}</output>"

    fmt = rng.randrange(3)
    if fmt == 0:
        q = rng.choice(t["qa"]).format(doc=doc)
        return {"question": q, "answer": answer}
    if fmt == 1:
        p = rng.choice(t["prompt"]).format(doc=doc)
        return {"prompt": p, "response": answer}
    inst = (rng.choice(t["instruction"]).format(doc=doc)
            + "\n\n"
            + t["context_prefix"].format(doc=doc))
    user = rng.choice(t["user_ask"])
    return {
        "instructions": inst,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def example_content_tokens(ex: dict) -> int:
    """Hitung token pada field konten saja (bukan key JSON)."""
    if "question" in ex:
        return count_tokens(ex["question"]) + count_tokens(ex["answer"])
    if "prompt" in ex:
        return count_tokens(ex["prompt"]) + count_tokens(ex["response"])
    total = count_tokens(ex.get("instructions", ""))
    for m in ex["messages"]:
        total += count_tokens(m["content"])
    return total


def write_summary(summary: dict) -> None:
    path = OUTPUT_DIR / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())


def stream_language(lang: str, target_tokens: int, out_path: Path):
    print(f"[INFO] {lang} -> {out_path} (target {target_tokens:,} tokens)", flush=True)
    ds = None
    total = 0
    written = 0
    next_report = 1_000_000
    # Seed deterministik (hash() Python tidak stabil antar run)
    rng = random.Random(42 + sum(ord(c) for c in lang))
    try:
        ds = load_dataset(DATASET_NAME, lang, split="train", streaming=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in ds:
                doc = clean_doc(ex.get("text", ""))
                if not doc:
                    continue
                sample = build_example(doc, lang, rng)
                total += example_content_tokens(sample)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
                if total >= next_report:
                    print(f"[INFO] {lang}: {total:,} tokens / {written:,} examples",
                          flush=True)
                    next_report += 1_000_000
                if total >= target_tokens:
                    break
            f.flush()
            os.fsync(f.fileno())
    finally:
        try:
            if ds is not None:
                del ds
        except Exception:
            pass
        gc.collect()
    return total, written


def main(langs):
    print(f"[INFO] CWD={Path.cwd()}  OUTPUT_DIR={OUTPUT_DIR}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for lang in langs:
        if lang not in LANGUAGES:
            print(f"[WARN] skip unknown lang: {lang}", flush=True)
            continue
        out = OUTPUT_DIR / f"{lang}.jsonl"
        try:
            toks, n = stream_language(lang, TOKENS_PER_LANG, out)
        except Exception:
            print(f"[ERROR] language {lang} failed:", flush=True)
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            continue
        summary[lang] = {
            "language": LANGUAGES[lang],
            "tokens": toks,
            "examples": n,
            "file": str(out),
        }
        write_summary(summary)
        print(f"[OK] {out}  ({toks:,} tokens, {n:,} examples)", flush=True)

    grand = sum(v["tokens"] for v in summary.values())
    print(f"[DONE] Total tokens: {grand:,}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", action="append", help="kode bahasa; boleh diulang")
    a = p.parse_args()
    return a.lang or list(LANGUAGES.keys())


if __name__ == "__main__":
    code = 0
    try:
        main(parse_args())
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)   # bypass shutdown crash (PyGILState_Release)
