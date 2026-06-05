#!/usr/bin/env python3
"""Write per-token source byte lengths for a trained tokenizer."""

from __future__ import annotations

import argparse
import array
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode text and write uint32 byte lengths per token.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output uint32 token byte lengths.")
    parser.add_argument("--starts-output", type=Path, default=None, help="Optional output uint32 token byte starts.")
    parser.add_argument("--ends-output", type=Path, default=None, help="Optional output uint32 token byte ends.")
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    return parser.parse_args()


def iter_text_chunks(path: Path, chunk_bytes: int):
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            yield chunk.decode("utf-8", errors="replace")


def main() -> int:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".part")
    starts_tmp_path = args.starts_output.with_suffix(args.starts_output.suffix + ".part") if args.starts_output else None
    ends_tmp_path = args.ends_output.with_suffix(args.ends_output.suffix + ".part") if args.ends_output else None
    token_count = 0
    total_span_bytes = 0
    chunk_base = 0

    starts_handle = starts_tmp_path.open("wb") if starts_tmp_path else None
    ends_handle = ends_tmp_path.open("wb") if ends_tmp_path else None
    try:
        with tmp_path.open("wb") as out:
            for text in iter_text_chunks(args.input, args.chunk_bytes):
                char_to_byte = [0]
                acc = 0
                for ch in text:
                    acc += len(ch.encode("utf-8"))
                    char_to_byte.append(acc)

                encoding = tokenizer.encode(text)
                lengths = []
                starts = []
                ends = []
                for start, end in encoding.offsets:
                    start_byte = chunk_base + char_to_byte[start]
                    end_byte = chunk_base + char_to_byte[end]
                    lengths.append(end_byte - start_byte)
                    starts.append(start_byte)
                    ends.append(end_byte)

                values = array.array("I", lengths)
                if sys.byteorder != "little":
                    values.byteswap()
                values.tofile(out)

                if starts_handle:
                    start_values = array.array("I", starts)
                    if sys.byteorder != "little":
                        start_values.byteswap()
                    start_values.tofile(starts_handle)
                if ends_handle:
                    end_values = array.array("I", ends)
                    if sys.byteorder != "little":
                        end_values.byteswap()
                    end_values.tofile(ends_handle)

                token_count += len(lengths)
                total_span_bytes += sum(lengths)
                chunk_base += char_to_byte[-1]
    finally:
        if starts_handle:
            starts_handle.close()
        if ends_handle:
            ends_handle.close()

    tmp_path.replace(args.output)
    if args.starts_output and starts_tmp_path:
        starts_tmp_path.replace(args.starts_output)
    if args.ends_output and ends_tmp_path:
        ends_tmp_path.replace(args.ends_output)
    print(
        json.dumps(
            {
                "tokenizer": str(args.tokenizer),
                "input": str(args.input),
                "output": str(args.output),
                "tokens": token_count,
                "total_span_bytes": total_span_bytes,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
