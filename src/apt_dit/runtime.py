"""Local artifact paths, provenance, and a conservative serial GPU-time ledger."""
import fcntl
import hashlib
import json
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone

from .upstream import ROOT


def local_path(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All artifacts must stay inside the project")
    return path


def write_json(path, data):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GPUBudget(AbstractContextManager):
    """Charge wall time of a GPU process, including I/O, to a durable 8h cap.

    Interrupted records remain reserved at their full allowance until reviewed.
    Concurrent project runs are rejected to keep timing and accounting clear.
    """
    def __init__(self, stage, hours):
        self.path = ROOT / "artifacts/gpu_ledger.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage, self.allowance = stage, hours * 3600

    def __enter__(self):
        self.lock = (self.path.parent / "gpu.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.records = json.loads(self.path.read_text()) if self.path.exists() else []
        self.used = sum(r.get("seconds", r["reserved_seconds"]) for r in self.records)
        self.allowance = min(self.allowance, 8 * 3600 - self.used)
        if self.allowance <= 0:
            self.lock.close()
            raise RuntimeError("Aggregate GPU budget exhausted")
        self.record = dict(stage=self.stage, started=datetime.now(timezone.utc).isoformat(),
                           reserved_seconds=self.allowance, status="running")
        self.records.append(self.record)
        write_json(self.path, self.records)
        self.start = time.monotonic()
        return self

    def check(self):
        if time.monotonic() - self.start >= self.allowance:
            raise TimeoutError(f"GPU allowance reached for {self.stage}")

    def __exit__(self, exc_type, exc, tb):
        self.record.update(seconds=time.monotonic() - self.start,
                           status="complete" if exc_type is None else "failed",
                           error=None if exc is None else str(exc))
        write_json(self.path, self.records)
        self.lock.close()


def manifest():
    import random
    path = ROOT / "artifacts/manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(6220)
    data = {}
    for split, size, offset in (("train", 128, 10000), ("validation", 32, 20000), ("final", 64, 30000)):
        data[split] = [dict(seed=offset + i, class_id=rng.randrange(1000)) for i in range(size)]
    write_json(path, data)
    return data
