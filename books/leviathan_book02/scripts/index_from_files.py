#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from openai import OpenAI


DEFAULT_VS_NAME = "hobbes-leviathan-book02"
DEFAULT_EXTS = (".txt", ".md", ".pdf")
POLL_INTERVAL_S = 2.0


def parse_args() -> argparse.Namespace:
    book_root = Path(__file__).resolve().parent.parent
    default_input = book_root / "output"

    p = argparse.ArgumentParser(description="Upload Book II files and index them in an OpenAI vector store.")
    p.add_argument(
        "--input-dir",
        default=str(default_input),
        help=f"Directory containing chunk files (default: {default_input})",
    )
    p.add_argument(
        "--vector-store-id",
        default=None,
        help="Existing vector store id (vs_...). If omitted, a new vector store is created.",
    )
    p.add_argument(
        "--vector-store-name",
        default=DEFAULT_VS_NAME,
        help=f"Name used when creating a new vector store (default: {DEFAULT_VS_NAME})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to upload (for quick tests).",
    )
    return p.parse_args()


def collect_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in DEFAULT_EXTS]
    if limit is not None:
        files = files[:limit]
    return files


def upload_files(client: OpenAI, paths: list[Path]) -> list[str]:
    file_ids: list[str] = []
    total = len(paths)
    for idx, path in enumerate(paths, 1):
        with path.open("rb") as fh:
            obj = client.files.create(file=fh, purpose="assistants")
        file_ids.append(obj.id)
        print(f"[{idx}/{total}] uploaded {path.name} -> {obj.id}")
    return file_ids


def wait_batch(client: OpenAI, vector_store_id: str, batch_id: str) -> None:
    while True:
        batch = client.vector_stores.file_batches.retrieve(
            vector_store_id=vector_store_id,
            batch_id=batch_id,
        )
        print(f"batch status={batch.status} counts={batch.file_counts}")
        if batch.status in ("completed", "failed", "cancelled"):
            if batch.status != "completed":
                raise RuntimeError(f"Indexing ended with status={batch.status}")
            return
        time.sleep(POLL_INTERVAL_S)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = collect_files(input_dir, limit=args.limit)
    if not files:
        raise SystemExit(f"No files found in {input_dir} with extensions: {', '.join(DEFAULT_EXTS)}")

    print(f"Found {len(files)} files in {input_dir}")

    client = OpenAI()
    file_ids = upload_files(client, files)

    if args.vector_store_id:
        vector_store_id = args.vector_store_id
        print(f"Using existing vector store: {vector_store_id}")
    else:
        vs = client.vector_stores.create(name=args.vector_store_name)
        vector_store_id = vs.id
        print(f"Created vector store: {vector_store_id} ({args.vector_store_name})")

    batch = client.vector_stores.file_batches.create(
        vector_store_id=vector_store_id,
        file_ids=file_ids,
    )
    print(f"Created file batch: {batch.id}")
    wait_batch(client, vector_store_id, batch.id)

    print("\n✅ Indexing completed.")
    print(f"vector_store_id: {vector_store_id}")
    print("Set this env var to enable Book II in Tender:")
    print(f"export TENDER_VS_HOBBES_LEVIATHAN_BOOK02={vector_store_id}")


if __name__ == "__main__":
    main()
