#!/usr/bin/env python3
"""Plot byte and BPE-token unigram distributions for enwik8."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from tokenizers import Tokenizer


VOCAB_SIZES = [300, 800, 1000, 2000, 3000, 7000, 10000]
CHUNK_BYTES = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot unigram distributions for bytes and BPE tokens.")
    parser.add_argument("--input", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/enwik8_unigrams"))
    parser.add_argument("--python", type=Path, default=Path("venv/bin/python"))
    parser.add_argument("--force", action="store_true", help="Retrain tokenizers and recount.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def iter_text_chunks(path: Path, chunk_bytes: int):
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            yield chunk.decode("utf-8", errors="replace")


def count_bytes(path: Path) -> tuple[list[int], int]:
    counts = [0] * 256
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            for byte in chunk:
                counts[byte] += 1
    return counts, total


def count_tokens(tokenizer_path: Path, input_path: Path) -> tuple[list[int], int]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    counts = [0] * tokenizer.get_vocab_size()
    total = 0
    for text in iter_text_chunks(input_path, CHUNK_BYTES):
        ids = tokenizer.encode(text).ids
        total += len(ids)
        for token_id in ids:
            counts[token_id] += 1
    return counts, total


def write_distribution_csv(path: Path, label: str, counts: list[int], total: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "symbol_id", "count", "probability"])
        for symbol_id, count in enumerate(counts):
            writer.writerow([label, symbol_id, count, count / total if total else 0.0])


def load_distribution_csv(path: Path) -> tuple[list[int], int]:
    counts: list[int] = []
    total = 0
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count = int(row["count"])
            counts.append(count)
            total += count
    return counts, total


def train_tokenizer(python: Path, input_path: Path, tokenizer_path: Path, vocab_size: int) -> None:
    run(
        [
            str(python),
            "scripts/bpe_tokenize.py",
            "train",
            "--input",
            str(input_path),
            "--tokenizer-output",
            str(tokenizer_path),
            "--vocab-size",
            str(vocab_size),
            "--chunk-bytes",
            str(CHUNK_BYTES),
        ]
    )


def sorted_probabilities(counts: list[int], total: int) -> list[float]:
    return sorted((count / total for count in counts if count), reverse=True)


def plot_distributions(distributions: list[dict], output_dir: Path) -> None:
    cols = 3
    rows = math.ceil(len(distributions) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, 3.7 * rows), sharey=True)
    axes = axes.ravel()
    for ax, item in zip(axes, distributions):
        probs = sorted_probabilities(item["counts"], item["total"])
        ranks = range(1, len(probs) + 1)
        ax.bar(ranks, probs, width=1.0, color=item["color"], alpha=0.9)
        ax.set_title(item["title"])
        ax.set_yscale("log")
        ax.set_xlabel("Symbol rank by frequency")
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for idx in range(len(distributions), len(axes)):
        axes[idx].set_visible(False)
    for row in range(rows):
        axes[row * cols].set_ylabel("Unigram probability")
    fig.suptitle("enwik8 Unigram Distributions: Bytes vs BPE Token Alphabets", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "unigram_distribution_panels.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for item in distributions:
        probs = sorted_probabilities(item["counts"], item["total"])
        ranks = range(1, len(probs) + 1)
        ax.plot(ranks, probs, linewidth=1.9, label=item["title"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Symbol rank by frequency")
    ax.set_ylabel("Unigram probability")
    ax.set_title("enwik8 Rank-Frequency Unigram Distributions")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "unigram_distribution_overlay.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = args.output_dir / "tokenizers"
    csv_dir = args.output_dir / "csv"
    tokenizer_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    colors = [
        "#315c8a",
        "#b24b40",
        "#6f8f3a",
        "#7b5ea7",
        "#b36b2c",
        "#3c8580",
        "#8a4f7d",
        "#526a2e",
    ]
    distributions: list[dict] = []

    byte_csv = csv_dir / "bytes.csv"
    if byte_csv.exists() and not args.force:
        counts, total = load_distribution_csv(byte_csv)
    else:
        started = time.time()
        counts, total = count_bytes(args.input)
        write_distribution_csv(byte_csv, "bytes", counts, total)
        print(json.dumps({"label": "bytes", "symbols": len(counts), "total": total, "seconds": time.time() - started}))
    distributions.append(
        {
            "title": "bytes (256)",
            "counts": counts,
            "total": total,
            "color": colors[0],
        }
    )

    for idx, vocab_size in enumerate(VOCAB_SIZES, start=1):
        tokenizer_path = tokenizer_dir / f"enwik8_bpe_{vocab_size}.json"
        dist_csv = csv_dir / f"bpe_{vocab_size}.csv"
        if args.force or not tokenizer_path.exists():
            train_tokenizer(args.python, args.input, tokenizer_path, vocab_size)

        if dist_csv.exists() and not args.force:
            counts, total = load_distribution_csv(dist_csv)
        else:
            started = time.time()
            counts, total = count_tokens(tokenizer_path, args.input)
            write_distribution_csv(dist_csv, f"bpe_{vocab_size}", counts, total)
            print(
                json.dumps(
                    {
                        "label": f"bpe_{vocab_size}",
                        "symbols": len(counts),
                        "observed_symbols": sum(1 for count in counts if count),
                        "total": total,
                        "seconds": time.time() - started,
                    }
                )
            )
        distributions.append(
            {
                "title": f"BPE {vocab_size}",
                "counts": counts,
                "total": total,
                "color": colors[idx],
            }
        )

    plot_distributions(distributions, args.output_dir)
    print(
        json.dumps(
            {
                "panel_plot": str(args.output_dir / "unigram_distribution_panels.png"),
                "overlay_plot": str(args.output_dir / "unigram_distribution_overlay.png"),
                "csv_dir": str(csv_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
