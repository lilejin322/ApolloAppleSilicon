"""
One search's directory: its settings, the log of every evaluation, the
evaluated scenarios, the Pareto front and the checkpoint it resumes from.

    runs/<run_id>/config.json          the search's settings
    runs/<run_id>/evaluations.jsonl    one line per evaluation
    runs/<run_id>/scenarios/<name>.pkl every evaluated scenario
    runs/<run_id>/pareto.json          the Pareto front after the last generation
    runs/<run_id>/checkpoint.pkl       after every evaluation

Files other than the log are replaced atomically. A search killed after
logging an evaluation but before its checkpoint evaluates that individual
again on resuming, so its name can appear twice in the log; the last line
counts.
"""

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..scenario.scenario_chromosome import Scenario

__all__ = ["Checkpoint", "Member", "RunStore", "new_run_id"]

Member = Tuple[Scenario, Tuple[float, ...] | None, str | None, Tuple[str, ...], Tuple[str, ...]]
"""
An individual as stored: its scenario, its fitness values (None until evaluated), its record name, its
findings' keys and Planning's changes of state in its run; older checkpoints hold the first three or four.
"""


@dataclass(frozen=True)
class Checkpoint:
    """
    Everything a search needs to go on where it stopped.

    :param int generation: The generation being evaluated; 0 is the random one.
    :param population: The population selected after the previous generation; empty in generation 0.
    :param offspring: This generation's individuals, evaluated or not.
    :param rng_state: ``Random.getstate()`` of the search's random source.
    :param pareto: The Pareto front so far.
    :param float seconds: Search seconds used so far.
    """

    generation: int
    population: Tuple[Member, ...]
    offspring: Tuple[Member, ...]
    rng_state: Any
    pareto: Tuple[Member, ...]
    seconds: float


def new_run_id(now: datetime | None = None) -> str:
    """
    A search's name: ``nsga2_<UTC time>``, e.g. ``nsga2_20261101T123005Z``.

    :param datetime | None now: The time; now by default.
    :rtype: str
    """
    return "nsga2_" + (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


class RunStore:
    """
    An existing search directory; ``create`` makes a new one.

    :param directory: The directory, holding ``config.json``.
    :raises FileNotFoundError: The directory holds no ``config.json``.
    """

    directory: Path

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not (self.directory / "config.json").is_file():
            raise FileNotFoundError(f"{self.directory} is not a search directory: it has no config.json")

    @classmethod
    def create(cls, runs_dir: str | Path, config: Mapping[str, Any], run_id: str | None = None) -> "RunStore":
        """
        Make ``runs_dir/<run_id>`` with ``config.json``.

        :param runs_dir: Where searches live, e.g. ``runs``.
        :param config: The search's settings, JSON-serialisable.
        :param str | None run_id: The directory's name; ``new_run_id()`` by default.
        :rtype: RunStore
        :raises FileExistsError: The directory exists.
        """
        directory = Path(runs_dir) / (run_id or new_run_id())
        (directory / "scenarios").mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n")
        return cls(directory)

    @property
    def run_id(self) -> str:
        """The directory's name. :rtype: str"""
        return self.directory.name

    @property
    def config(self) -> Dict[str, Any]:
        """The search's settings. :rtype: Dict[str, Any]"""
        return json.loads((self.directory / "config.json").read_text())

    def save_config(self, config: Mapping[str, Any]) -> None:
        """Replace ``config.json``, as when a resumed search gets more hours."""
        text = json.dumps(dict(config), indent=2, sort_keys=True) + "\n"
        self._write(self.directory / "config.json", text.encode())

    def log(self, entry: Mapping[str, Any]) -> None:
        """
        Append one evaluation to ``evaluations.jsonl`` and flush it to disk.

        :param entry: JSON-serialisable, e.g. ``Evaluation.to_json()``.
        """
        with open(self.directory / "evaluations.jsonl", "a") as log:
            log.write(json.dumps(dict(entry), sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())

    def evaluations(self) -> List[Dict[str, Any]]:
        """
        Every logged evaluation, in order.

        :rtype: List[Dict[str, Any]]
        """
        path = self.directory / "evaluations.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []

    def save_scenario(self, name: str, scenario: Scenario) -> None:
        """Keep ``scenario`` as ``scenarios/<name>.pkl``."""
        self._write(self.directory / "scenarios" / f"{name}.pkl", pickle.dumps(scenario))

    def load_scenario(self, name: str) -> Scenario:
        """The scenario kept as ``name``. :rtype: Scenario"""
        return pickle.loads((self.directory / "scenarios" / f"{name}.pkl").read_bytes())

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Replace ``checkpoint.pkl``."""
        self._write(self.directory / "checkpoint.pkl", pickle.dumps(checkpoint))

    def load_checkpoint(self) -> Checkpoint | None:
        """The last checkpoint, or None before the first. :rtype: Checkpoint | None"""
        path = self.directory / "checkpoint.pkl"
        return pickle.loads(path.read_bytes()) if path.exists() else None

    def save_findings(self, entries: Sequence[Mapping[str, Any]]) -> None:
        """Replace ``findings.json``: each finding's key with the best evaluation that found it."""
        text = json.dumps([dict(e) for e in entries], indent=2) + "\n"
        self._write(self.directory / "findings.json", text.encode())

    def save_pareto(self, entries: Sequence[Mapping[str, Any]]) -> None:
        """Replace ``pareto.json``: one entry per member of the front."""
        self._write(self.directory / "pareto.json", (json.dumps([dict(e) for e in entries], indent=2) + "\n").encode())

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
