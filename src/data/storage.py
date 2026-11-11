"""Save run artifacts."""

import json
import pickle
from pathlib import Path


def atomic_json(path, value):
    """Atomic json."""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def read_pickle(path):
    """Read pickle."""
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


class Stages:
    def __init__(self, directory, resume=False):
        """Initialize model state."""
        self.resume = resume
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def build(self, name, build):
        """Build reusable method artifacts."""
        target = self.directory / f"{name}.pkl"
        if self.resume and target.exists():
            print(f"[resume] {name}", flush=True)
            return read_pickle(target)
        print(f"[offline] {name}", flush=True)
        value = build()
        temp = target.with_suffix(".tmp")
        with temp.open("wb") as stream:
            pickle.dump(value, stream, protocol=4)
        temp.replace(target)
        return value
