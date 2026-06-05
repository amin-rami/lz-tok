#!/usr/bin/env python3
"""Train byte-level BPE tokenizers and encode datasets as raw uint32 token IDs."""

from __future__ import annotations

import argparse
import array
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train byte-level BPE on benchmark data and emit raw little-endian uint32 token IDs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a byte-level BPE tokenizer.")
    add_common_train_args(train)

    encode = subparsers.add_parser("encode", help="Encode a dataset with an existing tokenizer.")
    encode.add_argument("--tokenizer", type=Path, required=True, help="Tokenizer JSON produced by train.")
    encode.add_argument("--input", type=Path, required=True, help="Input dataset file.")
    encode.add_argument("--output", type=Path, required=True, help="Output .u32 token-id file.")
    encode.add_argument(
        "--chunk-bytes",
        type=int,
        default=DEFAULT_CHUNK_BYTES,
        help="Binary chunk size for streaming text into the tokenizer.",
    )

    both = subparsers.add_parser("train-encode", help="Train a tokenizer and encode the same input.")
    add_common_train_args(both)
    both.add_argument("--tokens-output", type=Path, required=True, help="Output .u32 token-id file.")

    return parser.parse_args()


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Training dataset file.")
    parser.add_argument("--tokenizer-output", type=Path, required=True, help="Output tokenizer JSON path.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Target BPE vocabulary size.")
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency for BPE merges.",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=DEFAULT_CHUNK_BYTES,
        help="Binary chunk size for streaming text into the tokenizer.",
    )
    parser.add_argument(
        "--limit-bytes",
        type=int,
        default=None,
        help="Train on only the first N bytes, useful for quick experiments.",
    )


def iter_text_chunks(path: Path, chunk_bytes: int, limit_bytes: int | None = None) -> Iterable[str]:
    remaining = limit_bytes
    with path.open("rb") as handle:
        while True:
            size = chunk_bytes if remaining is None else min(chunk_bytes, remaining)
            if size <= 0:
                break
            chunk = handle.read(size)
            if not chunk:
                break
            if remaining is not None:
                remaining -= len(chunk)
            yield chunk.decode("utf-8", errors="replace")


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    return tokenizer


def train_tokenizer(
    input_path: Path,
    output_path: Path,
    vocab_size: int,
    min_frequency: int,
    chunk_bytes: int,
    limit_bytes: int | None,
) -> Tokenizer:
    if vocab_size < 256:
        raise SystemExit("BPE vocab size must be at least 256 for byte-level alphabet coverage.")
    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    started = time.time()
    tokenizer.train_from_iterator(
        iter_text_chunks(input_path, chunk_bytes, limit_bytes),
        trainer=trainer,
        length=None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "command": "train",
                "input": str(input_path),
                "tokenizer": str(output_path),
                "vocab_size": tokenizer.get_vocab_size(),
                "target_vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "limit_bytes": limit_bytes,
                "seconds": elapsed,
            }
        )
    )
    return tokenizer


def encode_file(tokenizer: Tokenizer, input_path: Path, output_path: Path, chunk_bytes: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    token_count = 0
    source_bytes = input_path.stat().st_size
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    with tmp_path.open("wb") as out:
        for text in iter_text_chunks(input_path, chunk_bytes):
            ids = tokenizer.encode(text).ids
            token_count += len(ids)
            values = array.array("I", ids)
            if sys.byteorder != "little":
                values.byteswap()
            values.tofile(out)
    tmp_path.replace(output_path)

    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "command": "encode",
                "input": str(input_path),
                "output": str(output_path),
                "source_bytes": source_bytes,
                "tokens": token_count,
                "u32_bytes": token_count * 4,
                "bytes_per_token": source_bytes / token_count if token_count else 0,
                "seconds": elapsed,
            }
        )
    )


def main() -> int:
    args = parse_args()
    if args.command == "train":
        train_tokenizer(
            args.input,
            args.tokenizer_output,
            args.vocab_size,
            args.min_frequency,
            args.chunk_bytes,
            args.limit_bytes,
        )
    elif args.command == "encode":
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        encode_file(tokenizer, args.input, args.output, args.chunk_bytes)
    elif args.command == "train-encode":
        tokenizer = train_tokenizer(
            args.input,
            args.tokenizer_output,
            args.vocab_size,
            args.min_frequency,
            args.chunk_bytes,
            args.limit_bytes,
        )
        encode_file(tokenizer, args.input, args.tokens_output, args.chunk_bytes)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
