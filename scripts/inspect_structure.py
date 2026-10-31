#!/usr/bin/env python3
"""Extract and print a document's section tree and chunk-to-section mapping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structure.ingest import ingest_document, section_tree_text  # noqa: E402
from structure.store import StructureStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--db", default=str(ROOT / "data" / "sah_rag.db"))
    parser.add_argument("--json-dir", default=str(ROOT / "data" / "structure"))
    parser.add_argument("--show-chunks", action="store_true")
    args = parser.parse_args()

    store = StructureStore(args.db, args.json_dir)
    info = ingest_document(args.path, store)
    print(f"\nDocument: {info['document'].filename}")
    print(f"Strategy: {info['strategy']} | Sections: {info['sections']} | Chunks: {info['chunks']}\n")
    print(section_tree_text(store, info["doc_id"], include_chunks=True))
    print("\nChunk-to-section mapping")
    for section in store.get_sections(info["doc_id"]):
        for chunk in store.get_chunks_for_section(section["section_id"]):
            print(f"  {chunk['chunk_id']} -> {section['section_id']} ({section['title']}, page {chunk['page_start']})")
            if args.show_chunks:
                print(f"      {chunk['text'][:160].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
