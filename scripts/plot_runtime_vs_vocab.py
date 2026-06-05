#!/usr/bin/env python3
"""Measure and plot pipeline runtime versus BPE vocabulary size."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt


VOCAB_SIZES = [
    256,
    300,
    350,
    450,
    512,
    864,
    1024,
    1500,
    2048,
    4096,
    8192,
    10000,
    12000,
    16000,
    20000,
    30000,
]
WINDOW_SIZE = 1_048_576
LOOKAHEAD_SIZE = 1024
MIN_MATCH_LENGTH = 3
CHUNK_BYTES = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot runtime vs BPE vocabulary size.")
    parser.add_argument("--input", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/enwik8_vocab_sweep"))
    parser.add_argument("--lztok", type=Path, default=Path("build/lztok"))
    parser.add_argument("--python", type=Path, default=Path("venv/bin/python"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun timings even if runtime CSV already exists.",
    )
    return parser.parse_args()


def timed_run(cmd: list[str]) -> tuple[float, str]:
    print("+", " ".join(cmd), flush=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    seconds = time.perf_counter() - start
    if proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)
    print(f"# seconds: {seconds:.3f}", flush=True)
    return seconds, proc.stdout


def last_json(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def lz_compress(
    lztok: Path,
    input_path: Path,
    output_path: Path,
    mode: str,
) -> tuple[float, dict]:
    seconds, stdout = timed_run(
        [
            str(lztok),
            "compress",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--mode",
            mode,
            "--window-size",
            str(WINDOW_SIZE),
            "--lookahead-size",
            str(LOOKAHEAD_SIZE),
            "--min-match-length",
            str(MIN_MATCH_LENGTH),
        ]
    )
    return seconds, last_json(stdout)


def ensure_tokenizer(
    python: Path,
    input_path: Path,
    tokenizer_path: Path,
    vocab_size: int,
) -> None:
    if tokenizer_path.exists():
        return
    timed_run(
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


def main() -> int:
    args = parse_args()
    runtime_dir = args.output_dir / "runtime"
    tokenizer_dir = args.output_dir / "tokenizers"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "runtime_vs_vocab.csv"
    plot_path = args.output_dir / "runtime_vs_vocab.png"
    cached_rows: dict[int, dict] = {}
    if csv_path.exists() and not args.force:
        with csv_path.open() as handle:
            for row in csv.DictReader(handle):
                cached_rows[int(row["vocab_size"])] = row

    rows: list[dict] = []
    source_bytes = args.input.stat().st_size

    for vocab_size in VOCAB_SIZES:
        if vocab_size in cached_rows:
            rows.append(cached_rows[vocab_size])
            continue

        if vocab_size == 256:
            lz_path = runtime_dir / "runtime.byte.lztok"
            lz_seconds, lz_stats = lz_compress(args.lztok, args.input, lz_path, "byte")
            rows.append(
                {
                    "vocab_size": vocab_size,
                    "train_seconds": 0.0,
                    "encode_seconds": 0.0,
                    "lz_seconds": lz_seconds,
                    "without_training_seconds": lz_seconds,
                    "with_training_seconds": lz_seconds,
                    "tokens": source_bytes,
                    "compressed_file_bytes": lz_path.stat().st_size,
                    "estimated_payload_bits": lz_stats.get("estimated_payload_bits", ""),
                }
            )
            continue

        trained_tokenizer_path = runtime_dir / f"runtime_bpe_{vocab_size}.json"
        token_path = runtime_dir / f"runtime.bpe{vocab_size}.u32"
        lz_path = runtime_dir / f"runtime.bpe{vocab_size}.lztok"

        train_seconds, _ = timed_run(
            [
                str(args.python),
                "scripts/bpe_tokenize.py",
                "train",
                "--input",
                str(args.input),
                "--tokenizer-output",
                str(trained_tokenizer_path),
                "--vocab-size",
                str(vocab_size),
                "--chunk-bytes",
                str(CHUNK_BYTES),
            ]
        )

        encode_seconds, encode_stdout = timed_run(
            [
                str(args.python),
                "scripts/bpe_tokenize.py",
                "encode",
                "--tokenizer",
                str(trained_tokenizer_path),
                "--input",
                str(args.input),
                "--output",
                str(token_path),
                "--chunk-bytes",
                str(CHUNK_BYTES),
            ]
        )
        encode_stats = last_json(encode_stdout)
        lz_seconds, lz_stats = lz_compress(args.lztok, token_path, lz_path, "u32")
        without_training = encode_seconds + lz_seconds
        with_training = train_seconds + without_training
        rows.append(
            {
                "vocab_size": vocab_size,
                "train_seconds": train_seconds,
                "encode_seconds": encode_seconds,
                "lz_seconds": lz_seconds,
                "without_training_seconds": without_training,
                "with_training_seconds": with_training,
                "tokens": encode_stats.get("tokens", token_path.stat().st_size // 4),
                "compressed_file_bytes": lz_path.stat().st_size,
                "estimated_payload_bits": lz_stats.get("estimated_payload_bits", ""),
            }
        )

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = [int(row["vocab_size"]) for row in rows]
    y_without = [float(row["without_training_seconds"]) for row in rows]
    y_with = [float(row["with_training_seconds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, y_without, marker="o", markersize=4.5, linewidth=2.2, label="Without tokenizer training")
    ax.plot(x, y_with, marker="s", markersize=4, linewidth=1.8, linestyle="--", label="With tokenizer training")
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k", "16k", "32k"])
    ax.set_xlim(min(x) * 0.92, max(x) * 1.12)
    ax.set_xlabel("BPE vocabulary size")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("enwik8 Runtime vs BPE Vocabulary")
    ax.grid(True, axis="y", alpha=0.28)
    ax.grid(True, axis="x", which="major", alpha=0.12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)

    print(json.dumps({"csv": str(csv_path), "plot": str(plot_path), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
