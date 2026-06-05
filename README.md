# LZ_Tok

Experiments for comparing LZ77 compression over raw bytes and tokenizer token IDs.

## Build

```sh
venv/bin/python -m pip install -r requirements.txt
make
```

This builds `build/lztok`.

## LZ77 + Huffman CLI

```sh
build/lztok compress \
  --input data/enwik8 \
  --output /tmp/enwik8.lztok \
  --mode byte \
  --window-size 32768 \
  --lookahead-size 258 \
  --min-match-length 3

build/lztok decompress \
  --input /tmp/enwik8.lztok \
  --output /tmp/enwik8.restored

cmp data/enwik8 /tmp/enwik8.restored
```

For tokenized input, use `--mode u32`. Token files are raw little-endian
`uint32_t` token IDs:

```sh
build/lztok compress --input tokens.u32 --output tokens.lztok --mode u32
build/lztok decompress --input tokens.lztok --output tokens.restored.u32
```

## Training BPE Tokens

The BPE tokenizer is learned from the benchmark data and writes token IDs in the
raw `uint32_t` format expected by `lztok --mode u32`.

```sh
venv/bin/python scripts/bpe_tokenize.py train-encode \
  --input data/enwik8 \
  --tokenizer-output tokenizers/enwik8_bpe_8192.json \
  --tokens-output data/enwik8.bpe8192.u32 \
  --vocab-size 8192
```

Then run LZ77 over the token sequence:

```sh
build/lztok compress \
  --input data/enwik8.bpe8192.u32 \
  --output /tmp/enwik8_bpe8192.lztok \
  --mode u32 \
  --window-size 1048576 \
  --lookahead-size 1024 \
  --min-match-length 3
```

The tokenizer uses byte-level BPE with no normalization and no special tokens.
Input is decoded as UTF-8 with replacement for invalid byte sequences, which is
appropriate for the Wikipedia benchmark text but should be kept in mind for
arbitrary binary data.

## Tunables

- `--window-size`: LZ77 backward search window in symbols. Maximum is `1048576`.
- `--lookahead-size`: maximum emitted match length in symbols. Maximum is `1024`.
- `--min-match-length`: shortest match to emit. Minimum is `3`.

The fixed choices are:

- longest-match parsing
- hash-chain match search
- search capped at `256` candidates per position
- DEFLATE-style length and distance buckets, extended to support a 1 MiB window
  and 1024-symbol matches
- canonical Huffman coding for literals/lengths and distances

`analyze` runs the same LZ77 parse and Huffman accounting without writing a
compressed file:

```sh
build/lztok analyze --input data/enwik8 --mode byte
```
