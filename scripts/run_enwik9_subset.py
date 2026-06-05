#!/usr/bin/env python3
"""Run the selected enwik9 byte/token LZ77 experiments safely."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt


VOCAB_SIZES = [256, 1024, 2048, 4096, 8192]
WINDOW_SIZE = 1_048_576
LOOKAHEAD_SIZE = 1024
MIN_MATCH_LENGTH = 3
CHUNK_BYTES = 4 * 1024 * 1024
TRAIN_LIMIT_BYTES = 100_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected enwik9 LZ/tokenization experiments.")
    parser.add_argument("--input", type=Path, default=Path("data/enwik9"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/enwik9_subset"))
    parser.add_argument("--lztok", type=Path, default=Path("build/lztok"))
    parser.add_argument("--python", type=Path, default=Path("venv/bin/python"))
    parser.add_argument("--keep-tokens", action="store_true", help="Keep large temporary .u32 token files.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip decompression/cmp verification.")
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


def file_size(path: Path | None) -> int:
    return path.stat().st_size if path and path.exists() else 0


def plot(rows: list[dict], output_dir: Path) -> None:
    x = [int(row["vocab_size"]) for row in rows]
    compressed_mib = [int(row["compressed_file_bytes"]) / (1024 * 1024) for row in rows]
    with_tokenizer_mib = [
        int(row["compressed_plus_tokenizer_bytes"]) / (1024 * 1024) for row in rows
    ]
    without_training = [float(row["without_training_seconds"]) for row in rows]
    with_training = [float(row["with_training_seconds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(x, compressed_mib, marker="o", linewidth=2.2, label="LZ output")
    ax.plot(x, with_tokenizer_mib, marker="s", linestyle="--", linewidth=1.8, label="LZ + tokenizer")
    ax.set_xscale("log", base=2)
    label_map = {256: "256", 1024: "1k", 2048: "2k", 4096: "4k", 8192: "8k"}
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(value, str(value)) for value in x])
    ax.set_xlabel("BPE vocabulary size")
    ax.set_ylabel("Compressed size (MiB)")
    ax.set_title("enwik9 LZ77+Huffman Size")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "compressed_size_vs_vocab.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(x, without_training, marker="o", linewidth=2.2, label="Without tokenizer training")
    ax.plot(x, with_training, marker="s", linestyle="--", linewidth=1.8, label="With tokenizer training")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(value, str(value)) for value in x])
    ax.set_xlabel("BPE vocabulary size")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("enwik9 Runtime")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_vs_vocab.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    tokenizer_dir = output_dir / "tokenizers"
    compressed_dir = output_dir / "compressed"
    temp_dir = output_dir / "tmp"
    for path in (tokenizer_dir, compressed_dir, temp_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_bytes = args.input.stat().st_size
    csv_path = output_dir / "results.csv"
    rows: list[dict] = []
    completed: set[int] = set()
    if csv_path.exists():
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        completed = {int(row["vocab_size"]) for row in rows}

    for vocab_size in VOCAB_SIZES:
        if vocab_size in completed:
            print(f"Skipping completed vocab {vocab_size}", flush=True)
            continue

        tokenizer_path: Path | None = None
        token_path: Path | None = None
        train_seconds = 0.0
        encode_seconds = 0.0
        token_count = source_bytes
        token_u32_bytes = 0
        bytes_per_token = 1.0

        if vocab_size == 256:
            mode = "byte"
            lz_input = args.input
            compressed_path = compressed_dir / "enwik9.byte.w1m.l1024.lztok"
        else:
            mode = "u32"
            tokenizer_path = tokenizer_dir / f"enwik9_bpe_{vocab_size}.json"
            token_path = temp_dir / f"enwik9.bpe{vocab_size}.u32"
            compressed_path = compressed_dir / f"enwik9.bpe{vocab_size}.w1m.l1024.lztok"

            train_seconds, _ = timed_run(
                [
                    str(args.python),
                    "scripts/bpe_tokenize.py",
                    "train",
                    "--input",
                    str(args.input),
                    "--tokenizer-output",
                    str(tokenizer_path),
                    "--vocab-size",
                    str(vocab_size),
                    "--chunk-bytes",
                    str(CHUNK_BYTES),
                    "--limit-bytes",
                    str(TRAIN_LIMIT_BYTES),
                ]
            )
            encode_seconds, encode_stdout = timed_run(
                [
                    str(args.python),
                    "scripts/bpe_tokenize.py",
                    "encode",
                    "--tokenizer",
                    str(tokenizer_path),
                    "--input",
                    str(args.input),
                    "--output",
                    str(token_path),
                    "--chunk-bytes",
                    str(CHUNK_BYTES),
                ]
            )
            encode_stats = last_json(encode_stdout)
            token_count = int(encode_stats["tokens"])
            token_u32_bytes = int(encode_stats["u32_bytes"])
            bytes_per_token = float(encode_stats["bytes_per_token"])
            lz_input = token_path

        lz_seconds, lz_stdout = timed_run(
            [
                str(args.lztok),
                "compress",
                "--input",
                str(lz_input),
                "--output",
                str(compressed_path),
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
        lz_stats = last_json(lz_stdout)

        if not args.skip_verify:
            restored_path = temp_dir / (lz_input.name + ".restored")
            timed_run([str(args.lztok), "decompress", "--input", str(compressed_path), "--output", str(restored_path)])
            timed_run(["cmp", str(lz_input), str(restored_path)])
            restored_path.unlink(missing_ok=True)

        if token_path and not args.keep_tokens:
            token_path.unlink(missing_ok=True)

        row = {
            "vocab_size": vocab_size,
            "mode": mode,
            "source_bytes": source_bytes,
            "tokens": token_count,
            "token_u32_bytes": token_u32_bytes,
            "bytes_per_token": bytes_per_token,
            "tokenizer_json_bytes": file_size(tokenizer_path),
            "compressed_file_bytes": file_size(compressed_path),
            "compressed_plus_tokenizer_bytes": file_size(compressed_path) + file_size(tokenizer_path),
            "train_seconds": train_seconds,
            "encode_seconds": encode_seconds,
            "lz_seconds": lz_seconds,
            "without_training_seconds": encode_seconds + lz_seconds,
            "with_training_seconds": train_seconds + encode_seconds + lz_seconds,
            "estimated_payload_bits": lz_stats.get("estimated_payload_bits", ""),
            "payload_bits_per_source_byte": (
                float(lz_stats["estimated_payload_bits"]) / source_bytes
                if lz_stats.get("estimated_payload_bits")
                else ""
            ),
            "artifact": str(compressed_path),
        }
        rows.append(row)

        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, output_dir)
        print(json.dumps({"completed_vocab": vocab_size, "row": row}, indent=2), flush=True)

    print(json.dumps({"csv": str(output_dir / "results.csv"), "output_dir": str(output_dir), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
