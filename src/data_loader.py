import json
import logging
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)


def detect_format(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return "json_array"
    elif suffix == ".jsonl":
        return "jsonl"
    elif suffix == ".gz":
        stem = p.stem.lower()
        if stem.endswith(".jsonl"):
            return "jsonl_gz"
        if stem.endswith(".json"):
            return "json_gz"
        return "jsonl_gz"
    else:
        return "jsonl"


def stream_candidates(path: str) -> Generator[dict, None, None]:
    fmt = detect_format(path)
    p = Path(path)

    if fmt == "json_array":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            else:
                yield data
        return

    if fmt in ("jsonl_gz",):
        import gzip
        f = gzip.open(p, "rt", encoding="utf-8")
    else:
        f = open(p, "r", encoding="utf-8")

    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON line (first 80 chars): {line[:80]}")
                continue


def load_sample(path: str, n: int = 50) -> list[dict]:
    candidates = []
    for i, c in enumerate(stream_candidates(path)):
        if i >= n:
            break
        candidates.append(c)
    return candidates


def count_candidates(path: str) -> int:
    count = 0
    for _ in stream_candidates(path):
        count += 1
    return count
