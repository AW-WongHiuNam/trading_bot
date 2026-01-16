#!/usr/bin/env python3
"""列出 Qdrant 向量資料庫內容的小工具

用法例子：
  python view_vectors.py --path qdrant_db --collection api_calls --limit 200
  python view_vectors.py --filter-key type --filter-value api_response

此腳本會印出每筆向量的 id、metadata 摘要與文件預覽（前 400 字元）。
"""
from __future__ import annotations

import argparse
import json
from typing import List, Tuple

from qdrant_client import QdrantClient


def main() -> None:
    parser = argparse.ArgumentParser(description="View Qdrant vector DB contents")
    parser.add_argument("--path", default="qdrant_db", help="Qdrant local path (folder)")
    parser.add_argument("--collection", default="api_calls", help="Collection name")
    parser.add_argument("--limit", type=int, default=1000, help="Max items to fetch")
    parser.add_argument("--filter-key", help="Metadata key to filter on (optional)")
    parser.add_argument("--filter-value", help="Metadata value to match (optional)")
    args = parser.parse_args()

    client = QdrantClient(path=args.path)

    items: List[Tuple[str, str, dict]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            limit=args.limit,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            if args.filter_key and args.filter_value is not None:
                if str(payload.get(args.filter_key)) != args.filter_value:
                    continue
            doc = payload.get("document", "")
            items.append((str(p.id), doc, payload))
            if len(items) >= args.limit:
                break
        if not offset or len(items) >= args.limit:
            break

    print(f"Found {len(items)} items (showing up to {args.limit})")
    for idx, (pid, doc, payload) in enumerate(items, start=1):
        print("----")
        print(f"#{idx} id: {pid}")
        try:
            meta_str = json.dumps(payload, ensure_ascii=False)
        except Exception:
            meta_str = str(payload)
        print("meta:", meta_str)
        preview = (doc or "")[:400].replace("\n", " ")
        print("doc preview:", preview)


if __name__ == "__main__":
    main()
