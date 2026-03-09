from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AlphaKeySelection:
    api_key: str
    key_id: str
    index: int
    used: int


class AlphaKeyRotator:
    def __init__(self, keys: List[str], per_key_limit: int = 25):
        cleaned = [k.strip() for k in keys if isinstance(k, str) and k.strip()]
        if not cleaned:
            raise ValueError("No Alpha Vantage API keys configured")

        self.keys = cleaned
        self.per_key_limit = max(int(per_key_limit), 1)
        self._lock = threading.Lock()
        self._active_index = 0
        self._counts: Dict[str, int] = {}

    @staticmethod
    def _key_id(api_key: str) -> str:
        return hashlib.sha1(api_key.encode("utf-8")).hexdigest()[:12]

    def _pick_index(self) -> int:
        start = int(self._active_index) % len(self.keys)
        for step in range(len(self.keys)):
            idx = (start + step) % len(self.keys)
            key = self.keys[idx]
            kid = self._key_id(key)
            used = int(self._counts.get(kid, 0))
            if used < self.per_key_limit:
                return idx
        return start

    def acquire(self) -> AlphaKeySelection:
        with self._lock:
            idx = self._pick_index()
            self._active_index = idx

            key = self.keys[idx]
            kid = self._key_id(key)
            used = int(self._counts.get(kid, 0))
            return AlphaKeySelection(api_key=key, key_id=kid, index=idx, used=used)

    def mark_request(self, selected: AlphaKeySelection, exhausted: bool = False) -> AlphaKeySelection:
        with self._lock:
            used = int(self._counts.get(selected.key_id, 0))
            next_used = max(used + 1, self.per_key_limit) if exhausted else (used + 1)
            self._counts[selected.key_id] = next_used

            if exhausted or next_used >= self.per_key_limit:
                self._active_index = (selected.index + 1) % len(self.keys)
            else:
                self._active_index = selected.index

            return AlphaKeySelection(
                api_key=selected.api_key,
                key_id=selected.key_id,
                index=selected.index,
                used=next_used,
            )


def parse_alpha_keys(single_key: str = "", multi_keys_csv: str = "") -> List[str]:
    keys: List[str] = []
    if multi_keys_csv:
        keys.extend([part.strip() for part in multi_keys_csv.split(",") if part.strip()])
    if single_key and single_key.strip():
        single = single_key.strip()
        if single not in keys:
            keys.append(single)
    return keys


def is_alpha_quota_error(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("Information", "Note"):
        msg = payload.get(key)
        if isinstance(msg, str):
            low = msg.lower()
            if "premium" in low or "rate limit" in low or "requests per day" in low:
                return True
    return False


_ROTATOR_CACHE: Dict[str, AlphaKeyRotator] = {}


def build_rotator(single_key: str, multi_keys_csv: str, per_key_limit: int) -> AlphaKeyRotator | None:
    keys = parse_alpha_keys(single_key=single_key, multi_keys_csv=multi_keys_csv)
    if not keys:
        return None
    cache_key = "|".join(keys) + f"::limit={max(int(per_key_limit), 1)}"
    existing = _ROTATOR_CACHE.get(cache_key)
    if existing is not None:
        return existing
    created = AlphaKeyRotator(keys=keys, per_key_limit=per_key_limit)
    _ROTATOR_CACHE[cache_key] = created
    return created
