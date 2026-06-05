#!/usr/bin/env python3
"""Plot LZ77 match-length distributions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot byte vs token LZ match-length distributions.")
    parser.add_argument("--byte-csv", type=Path, default=Path("results/enwik8_match_lengths/byte_w1m_l1024.csv"))
    parser.add_argument("--token-csv", type=Path, default=Path("results/enwik8_match_lengths/bpe864_w1m_l1024.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/enwik8_match_lengths/match_length_byte_vs_bpe864.png"))
    parser.add_argument("--zoom-output", type=Path, default=Path("results/enwik8_match_lengths/match_length_byte_vs_bpe864_x0_30.png"))
    parser.add_argument(
        "--stacked-zoom-output",
        type=Path,
        default=Path("results/enwik8_match_lengths/match_length_histograms_stacked_x0_30.png"),
    )
    parser.add_argument("--cdf-output", type=Path, default=Path("results/enwik8_match_lengths/match_length_cdf_byte_vs_bpe864.png"))
    return parser.parse_args()


def load_distribution(path: Path, kind: str) -> tuple[list[int], list[float], int]:
    lengths: list[int] = []
    probs: list[float] = []
    total = 0
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["kind"] != kind:
                continue
            lengths.append(int(row["length"]))
            count = int(row["count"])
            total += count
            probs.append(float(row["probability"]))
    return lengths, probs, total


def stats(lengths: list[int], probs: list[float]) -> dict:
    mean = sum(length * prob for length, prob in zip(lengths, probs))
    cdf = 0.0
    quantiles = {}
    for length, prob in zip(lengths, probs):
        cdf += prob
        for q in (0.5, 0.9, 0.99):
            if q not in quantiles and cdf >= q:
                quantiles[q] = length
    mode_idx = max(range(len(probs)), key=probs.__getitem__)
    return {
        "mean": mean,
        "mode": lengths[mode_idx],
        "mode_probability": probs[mode_idx],
        "p50": quantiles.get(0.5),
        "p90": quantiles.get(0.9),
        "p99": quantiles.get(0.99),
        "max": max(lengths),
    }


def main() -> int:
    args = parse_args()
    byte_lengths, byte_probs, byte_total = load_distribution(args.byte_csv, "symbols")
    token_lengths, token_probs, token_total = load_distribution(args.token_csv, "symbols")
    effective_lengths, effective_probs, effective_total = load_distribution(args.token_csv, "effective")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(byte_lengths, byte_probs, width=0.9, alpha=0.58, label="bytes: L in bytes")
    ax.bar(token_lengths, token_probs, width=0.9, alpha=0.58, label="BPE 864: L in tokens")
    ax.set_xlabel("LZ77 match length L")
    ax.set_ylabel("Probability")
    ax.set_title("enwik8 LZ77 Match-Length Histogram")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(byte_lengths, byte_probs, width=0.9, alpha=0.58, label="bytes: L in bytes")
    ax.bar(token_lengths, token_probs, width=0.9, alpha=0.58, label="BPE 864: L in tokens")
    ax.bar(
        effective_lengths,
        effective_probs,
        width=0.9,
        alpha=0.42,
        label="BPE 864: effective L in bytes",
    )
    ax.set_xlim(0, 30)
    ax.set_xlabel("LZ77 match length L")
    ax.set_ylabel("Probability")
    ax.set_title("enwik8 LZ77 Match-Length Histogram, L <= 30")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.zoom_output, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
    stacked = [
        (axes[0], byte_lengths, byte_probs, "bytes: L in bytes", "#315c8a"),
        (axes[1], token_lengths, token_probs, "BPE 864: L in tokens", "#b24b40"),
        (axes[2], effective_lengths, effective_probs, "BPE 864: effective L in bytes", "#6f8f3a"),
    ]
    for ax, lengths, probs, title, color in stacked:
        ax.bar(lengths, probs, width=0.9, alpha=0.86, color=color)
        ax.set_xlim(0, 30)
        ax.set_ylabel("Probability")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("LZ77 match length L")
    fig.suptitle("enwik8 LZ77 Match-Length Histograms, L <= 30", y=0.995)
    fig.tight_layout()
    fig.savefig(args.stacked_zoom_output, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    byte_cdf = []
    total = 0.0
    for prob in byte_probs:
        total += prob
        byte_cdf.append(total)
    token_cdf = []
    total = 0.0
    for prob in token_probs:
        total += prob
        token_cdf.append(total)
    ax.plot(byte_lengths, byte_cdf, linewidth=2.1, label="bytes: L in bytes")
    ax.plot(token_lengths, token_cdf, linewidth=2.1, label="BPE 864: L in tokens")
    ax.set_xlabel("LZ77 match length L")
    ax.set_ylabel("CDF: P(match length <= L)")
    ax.set_title("enwik8 LZ77 Match-Length CDF")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.cdf_output, dpi=180)

    summary = {
        "plot": str(args.output),
        "zoom_plot": str(args.zoom_output),
        "stacked_zoom_plot": str(args.stacked_zoom_output),
        "cdf_plot": str(args.cdf_output),
        "byte_matches": byte_total,
        "token_matches": token_total,
        "effective_matches": effective_total,
        "byte": stats(byte_lengths, byte_probs),
        "bpe864_tokens": stats(token_lengths, token_probs),
        "bpe864_effective_bytes": stats(effective_lengths, effective_probs) if effective_lengths else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
