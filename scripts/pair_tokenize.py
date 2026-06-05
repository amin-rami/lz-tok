#!/usr/bin/env python3
"""Tokenize a byte stream by merging adjacent byte pairs into uint32 symbols."""

from __future__ import annotations

import argparse
import array
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge adjacent bytes (x_i, x_{i+1}) into uint32 symbols x_i * 256 + x_{i+1}."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--odd-policy",
        choices=("error", "pad-zero"),
        default="error",
        help="How to handle odd-length inputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = args.input.stat().st_size
    if source_bytes % 2 and args.odd_policy == "error":
        raise SystemExit("input has odd byte length; use --odd-policy pad-zero to encode the final byte")

    token_count = 0
    pending: int | None = None
    tmp_path = args.output.with_suffix(args.output.suffix + ".part")

    with args.input.open("rb") as inp, tmp_path.open("wb") as out:
        while True:
            chunk = inp.read(args.chunk_bytes)
            if not chunk:
                break
            if pending is not None:
                chunk = bytes([pending]) + chunk
                pending = None
            if len(chunk) % 2:
                pending = chunk[-1]
                chunk = chunk[:-1]
            values = array.array("I")
            values.fromlist(
                [(chunk[i] << 8) | chunk[i + 1] for i in range(0, len(chunk), 2)]
            )
            if sys.byteorder != "little":
                values.byteswap()
            values.tofile(out)
            token_count += len(values)

        if pending is not None:
            if args.odd_policy == "pad-zero":
                values = array.array("I", [pending << 8])
                if sys.byteorder != "little":
                    values.byteswap()
                values.tofile(out)
                token_count += 1
            else:
                raise SystemExit("input has odd byte length")

    tmp_path.replace(args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "source_bytes": source_bytes,
                "tokens": token_count,
                "u32_bytes": token_count * 4,
                "bytes_per_token": source_bytes / token_count if token_count else 0,
                "vocab_size": 65536,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
