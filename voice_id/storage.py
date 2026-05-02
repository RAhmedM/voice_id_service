"""Voiceprint store.

Pickle-backed for V1 with an interface designed to swap for Redis / SQLite /
a vector DB later by replacing this class while keeping the signatures the same.
"""
import pickle
from pathlib import Path
from threading import Lock
from typing import Dict, List

import numpy as np


class VoiceprintStore:
    """Maps speaker name -> 192-dim L2-normalized centroid."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = Lock()
        self._db: Dict[str, np.ndarray] = self._load()

    def _load(self) -> Dict[str, np.ndarray]:
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self._db, f)
        tmp.replace(self.path)

    def add(self, name: str, centroid: np.ndarray) -> None:
        with self._lock:
            self._db[name] = centroid.astype(np.float32)
            self._save()

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self._db:
                del self._db[name]
                self._save()
                return True
            return False

    def list_names(self) -> List[str]:
        with self._lock:
            return sorted(self._db.keys())

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._db) == 0

    def identify(self, embedding: np.ndarray, threshold: float) -> Dict:
        """Return `{speaker, confidence, scores}`.

        `speaker` is the best-match name if the top score >= threshold, else None.
        `scores` is a full ranking of all enrolled speakers, highest first.
        """
        with self._lock:
            snapshot = dict(self._db)

        if not snapshot:
            return {"speaker": None, "confidence": None, "scores": {}}

        scores = {
            name: float(np.dot(embedding, centroid))
            for name, centroid in snapshot.items()
        }
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_name, best_score = ranked[0]
        matched = best_name if best_score >= threshold else None
        return {
            "speaker": matched,
            "confidence": best_score,
            "scores": dict(ranked),
        }
