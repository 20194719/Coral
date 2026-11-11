"""Persist and replay judgments within one query and seed."""

import json
from pathlib import Path
from src.data.storage import atomic_json


class CheckpointProbe:
    """Replay completed adaptive batches before requesting new labels."""

    def __init__(self, probe, directory):
        """Initialize model state."""
        self.probe, self.directory, self.index = probe, Path(directory), 0
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, pairs, requested):
        """Return verified labels for the current adaptive batch."""
        path = self.directory / f"batch-{self.index:04}.json"
        payload = dict(
            pairs=[list(p) for p in pairs], requested=[list(r) for r in requested]
        )
        if path.exists():
            record = json.loads(path.read_text())
            if any(record[k] != v for k, v in payload.items()):
                raise ValueError("Adaptive continuation differs from its saved batch")
            answers = record["answers"]
        else:
            answers = self.probe(pairs, requested)
            if len(answers) != len(pairs) or any(
                len(a) != 3 or any(flag and y not in (0, 1) for flag, y in zip(need, a))
                for need, a in zip(requested, answers)
            ):
                raise ValueError("Invalid online labels; checkpoint was not committed")
            atomic_json(path, dict(payload, answers=answers))
        self.index += 1
        return answers
