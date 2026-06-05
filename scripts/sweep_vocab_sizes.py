#!/usr/bin/env python3
"""Sweep BPE vocab sizes and plot LZ77 compressed sizes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


VOCAB_SIZES = [
    256,
    257,
    260,
    270,
    280,
    300,
    380,
    512,
    864,
    1024,
    1500,
    1600,
    2048,
    4096,
    8192,
    12000,
    15000,
    17000,
    18000,
    30000,
]
WINDOW_SIZE = 1_048_576
LOOKAHEAD_SIZE = 1024
MIN_MATCH_LENGTH = 3
CHUNK_BYTES = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the enwik8 BPE-vocab LZ77 sweep.")
    parser.add_argument("--input", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/enwik8_vocab_sweep"))
    parser.add_argument("--lztok", type=Path, default=Path("build/lztok"))
    parser.add_argument("--python", type=Path, default=Path("venv/bin/python"))
    parser.add_argument("--force", action="store_true", help="Regenerate all artifacts.")
    parser.add_argument("--verify", action="store_true", help="Decompress and cmp each compressed stream.")
    return parser.parse_args()


def run(cmd: list[str]) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    if proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)
    return proc.stdout


def read_json_line(output: str) -> dict:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def compress(
    lztok: Path,
    input_path: Path,
    output_path: Path,
    mode: str,
    force: bool,
) -> dict:
    if output_path.exists() and not force:
        return {}
    output = run(
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
    return read_json_line(output)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = args.output_dir / "tokenizers"
    token_dir = args.output_dir / "tokens"
    compressed_dir = args.output_dir / "compressed"
    restored_dir = args.output_dir / "restored"
    for path in (tokenizer_dir, token_dir, compressed_dir, restored_dir):
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for vocab_size in VOCAB_SIZES:
        if vocab_size == 256:
            input_path = args.input
            tokenizer_path = None
            token_path = None
            compressed_path = compressed_dir / "enwik8.byte.w1m.l1024.lztok"
            mode = "byte"
            token_count = args.input.stat().st_size
            token_u32_bytes = 0
            bytes_per_token = 1.0
        else:
            tokenizer_path = tokenizer_dir / f"enwik8_bpe_{vocab_size}.json"
            token_path = token_dir / f"enwik8.bpe{vocab_size}.u32"
            compressed_path = compressed_dir / f"enwik8.bpe{vocab_size}.w1m.l1024.lztok"
            mode = "u32"

            if not tokenizer_path.exists() or not token_path.exists() or args.force:
                output = run(
                    [
                        str(args.python),
                        "scripts/bpe_tokenize.py",
                        "train-encode",
                        "--input",
                        str(args.input),
                        "--tokenizer-output",
                        str(tokenizer_path),
                        "--tokens-output",
                        str(token_path),
                        "--vocab-size",
                        str(vocab_size),
                        "--chunk-bytes",
                        str(CHUNK_BYTES),
                    ]
                )
                encode_stats = read_json_line(output)
                token_count = int(encode_stats.get("tokens", 0))
                token_u32_bytes = int(encode_stats.get("u32_bytes", file_size(token_path)))
                bytes_per_token = float(encode_stats.get("bytes_per_token", 0.0))
            else:
                token_u32_bytes = file_size(token_path)
                token_count = token_u32_bytes // 4
                bytes_per_token = args.input.stat().st_size / token_count if token_count else 0.0
            input_path = token_path

        stats = compress(args.lztok, input_path, compressed_path, mode, args.force)

        if args.verify:
            restored_path = restored_dir / (input_path.name + ".restored")
            run([str(args.lztok), "decompress", "--input", str(compressed_path), "--output", str(restored_path)])
            run(["cmp", str(input_path), str(restored_path)])

        row = {
            "vocab_size": vocab_size,
            "mode": mode,
            "source_bytes": args.input.stat().st_size,
            "tokens": token_count,
            "token_u32_bytes": token_u32_bytes,
            "bytes_per_token": bytes_per_token,
            "tokenizer_json_bytes": file_size(tokenizer_path) if tokenizer_path else 0,
            "compressed_file_bytes": file_size(compressed_path),
            "compressed_plus_tokenizer_bytes": file_size(compressed_path)
            + (file_size(tokenizer_path) if tokenizer_path else 0),
            "estimated_payload_bits": stats.get("estimated_payload_bits", ""),
            "payload_bits_per_source_byte": (
                float(stats["estimated_payload_bits"]) / args.input.stat().st_size
                if stats.get("estimated_payload_bits")
                else ""
            ),
            "artifact": str(compressed_path),
        }
        rows.append(row)

    csv_path = args.output_dir / "compressed_sizes.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = [row["vocab_size"] for row in rows]
    y = [row["compressed_file_bytes"] / (1024 * 1024) for row in rows]
    y_with_tok = [row["compressed_plus_tokenizer_bytes"] / (1024 * 1024) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, y, marker="o", markersize=4.5, linewidth=2.2, label="LZ output")
    ax.plot(
        x,
        y_with_tok,
        marker="s",
        markersize=4,
        linewidth=1.8,
        linestyle="--",
        label="LZ output + tokenizer",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k", "16k", "32k"])
    ax.set_xlim(min(x) * 0.92, max(x) * 1.12)
    ax.set_xlabel("BPE vocabulary size")
    ax.set_ylabel("Compressed size (MiB)")
    ax.set_title("enwik8 LZ77+Huffman Size vs BPE Vocabulary")
    ax.grid(True, axis="y", alpha=0.28)
    ax.grid(True, axis="x", which="major", alpha=0.12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plot_path = args.output_dir / "compressed_size_vs_vocab.png"
    plt.savefig(plot_path, dpi=180)

    print(json.dumps({"csv": str(csv_path), "plot": str(plot_path), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
