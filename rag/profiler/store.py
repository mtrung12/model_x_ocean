"""Persistent JSONL store for generated psychological profiles.

One file per training corpus. Each line is a JSON object:
    {
        "user_id": "user_42",
        "trait_labels": {"cOPN": "high", ..., "cNEU": "low"},
        "facets": { "N1": {"signal": "high", "evidence": "..."}, ... },
        "linguistic": { "pronouns": "...", "emotion": "...", ... },
        "raw": "<full profiler output>",
        "valid": true,
        "model": "gpt-4o-2024-08-06"
    }

The store is append-safe (re-runs skip already-profiled user_ids) and
checkpoints every N records.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional


class ProfileStore:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self.entries: Dict[str, Dict] = {}

    def load(self) -> None:
        self.entries = {}
        if not self.store_path or not os.path.exists(self.store_path):
            return
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = item.get("user_id")
                if uid:
                    self.entries[uid] = item

    def save(self) -> None:
        if not self.store_path:
            return
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        tmp = self.store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for entry in self.entries.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp, self.store_path)

    def add(
        self,
        user_id: str,
        trait_labels: Dict[str, str],
        profile: Dict,
        model: str,
    ) -> None:
        self.entries[user_id] = {
            "user_id": user_id,
            "trait_labels": trait_labels,
            "facets": profile.get("facets", {}),
            "linguistic": profile.get("linguistic", {}),
            "raw": profile.get("raw", ""),
            "valid": bool(profile.get("valid", False)),
            "model": model,
        }

    def get(self, user_id: str) -> Optional[Dict]:
        return self.entries.get(user_id)

    def has(self, user_id: str) -> bool:
        return user_id in self.entries

    def get_all(self) -> List[Dict]:
        return list(self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, user_id: str) -> bool:
        return user_id in self.entries

    def __iter__(self) -> Iterable[Dict]:
        return iter(self.entries.values())
