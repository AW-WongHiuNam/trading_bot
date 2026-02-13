"""Compatibility stub.

The implementation lives in scripts/alpha_fetch.py.
Keep this file so existing imports and commands still work:
  - `python alpha_fetch.py ...`
  - `from alpha_fetch import fetch_av`
"""

from scripts.alpha_fetch import API_URL, fetch_av, main, save_json, save_jsonl

__all__ = ["API_URL", "fetch_av", "save_json", "save_jsonl", "main"]


if __name__ == "__main__":
    main()
