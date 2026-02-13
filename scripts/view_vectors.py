#!/usr/bin/env python3
"""列出 SQLite + hnswlib 向量庫內容的小工具

用法例子：
    python view_vectors.py --sqlite ./vector_store.sqlite --table api_calls --limit 200
    python view_vectors.py --sqlite ./vector_store.sqlite --filter-key type --filter-value api_response

此腳本會印出每筆向量的 id、metadata 摘要與文件預覽（前 400 字元）。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from typing import List, Tuple


def main() -> None:
    parser = argparse.ArgumentParser(description="View SQLite vector contents")
    parser.add_argument("--sqlite", default="vector_store.sqlite", help="Path to SQLite file")
    parser.add_argument("--table", default="api_calls", help="Table name")
    parser.add_argument("--limit", type=int, default=1000, help="Max items to fetch")
    parser.add_argument("--filter-key", help="Metadata key to filter on (optional)")
    parser.add_argument("--filter-value", help="Metadata value to match (optional)")
    args = parser.parse_args()

    items: List[Tuple[str, str, dict]] = []
    where_clause = ""
    params: List[str] = []
    if args.filter_key and args.filter_value is not None:
        where_clause = f"WHERE json_extract(metadata, '$.{args.filter_key}') = ?"
        params = [args.filter_value]

    query = f"SELECT id, document, metadata FROM {args.table} {where_clause} ORDER BY created_at DESC LIMIT ?"
    params.append(args.limit)

    with sqlite3.connect(args.sqlite) as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        for pid, doc, payload in cur.fetchall():
            try:
                payload_obj = json.loads(payload) if payload else {}
            except Exception:
                payload_obj = {}
            items.append((str(pid), doc or "", payload_obj or {}))

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
