#!/usr/bin/env python3
"""Download and extract the enwik8/enwik9 benchmark datasets."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    "enwik8": {
        "url": "https://www.mattmahoney.net/dc/enwik8.zip",
        "zip_size": 36_445_475,
        "raw_size": 100_000_000,
    },
    "enwik9": {
        "url": "https://www.mattmahoney.net/dc/enwik9.zip",
        "zip_size": 322_592_222,
        "raw_size": 1_000_000_000,
    },
}

CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract enwik8/enwik9 into a local data directory."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(DATASETS),
        default=None,
        help="Datasets to download. Defaults to both.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where archives and extracted files are stored.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded .zip archives after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract even when files already exist.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="Override a dataset URL, e.g. --url enwik8=https://mirror/enwik8.zip.",
    )
    return parser.parse_args()


def url_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --url value {value!r}; expected NAME=URL.")
        name, url = value.split("=", 1)
        if name not in DATASETS:
            raise SystemExit(f"Unknown dataset in --url: {name!r}.")
        if not url:
            raise SystemExit(f"Missing URL for dataset {name!r}.")
        overrides[name] = url
    return overrides


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{size} B"


def progress_line(name: str, downloaded: int, total: int | None) -> str:
    if total:
        pct = downloaded / total * 100
        return f"\r{name}: {format_bytes(downloaded)} / {format_bytes(total)} ({pct:5.1f}%)"
    return f"\r{name}: {format_bytes(downloaded)}"


def existing_file_is_valid(path: Path, expected_size: int | None) -> bool:
    return path.exists() and (expected_size is None or path.stat().st_size == expected_size)


def download(url: str, destination: Path, expected_size: int | None, force: bool) -> None:
    if existing_file_is_valid(destination, expected_size) and not force:
        print(f"{destination} already exists with expected size.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if force:
        tmp_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)

    headers = {}
    downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request) as response:
            status = getattr(response, "status", None)
            if downloaded and status != 206:
                tmp_path.unlink(missing_ok=True)
                downloaded = 0

            total_header = response.headers.get("Content-Length")
            total = int(total_header) + downloaded if total_header else expected_size
            mode = "ab" if downloaded else "wb"

            with tmp_path.open(mode) as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    print(progress_line(destination.name, downloaded, total), end="")
            print()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not download {url}: {exc}") from exc

    actual_size = tmp_path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise SystemExit(
            f"{tmp_path} has size {actual_size}, expected {expected_size}. "
            "Delete it or rerun with --force."
        )

    tmp_path.replace(destination)


def extract_single_file(zip_path: Path, output_path: Path, expected_size: int, force: bool) -> None:
    if existing_file_is_valid(output_path, expected_size) and not force:
        print(f"{output_path} already exists with expected size.")
        return

    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise SystemExit(f"{zip_path} failed ZIP validation at member {bad_member!r}.")

        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) != 1:
            raise SystemExit(f"{zip_path} should contain exactly one file, found {len(members)}.")

        member = members[0]
        if member.file_size != expected_size:
            raise SystemExit(
                f"{zip_path}:{member.filename} has size {member.file_size}, "
                f"expected {expected_size}."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, tempfile.NamedTemporaryFile(
            dir=output_path.parent, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(source, tmp, length=CHUNK_SIZE)

    tmp_path.replace(output_path)
    print(f"Extracted {output_path} ({format_bytes(output_path.stat().st_size)}).")


def main() -> int:
    args = parse_args()
    selected_datasets = args.datasets or sorted(DATASETS)
    data_dir = args.data_dir
    archives_dir = data_dir / "archives"
    overrides = url_overrides(args.url)

    for name in selected_datasets:
        metadata = DATASETS[name]
        url = overrides.get(name, metadata["url"])
        zip_path = archives_dir / f"{name}.zip"
        raw_path = data_dir / name

        print(f"\n{name}")
        download(url, zip_path, metadata["zip_size"], args.force)
        extract_single_file(zip_path, raw_path, metadata["raw_size"], args.force)

        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)
            print(f"Removed {zip_path}.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
