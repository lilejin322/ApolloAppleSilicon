"""
Evaluate one scenario on Apollo for the search: run it in the container,
judge its record and score it.

A run the environment broke (the container, the bridge, the HMI) is tried
once more; a second failure stops the search rather than become a fitness. A
run that tested nothing (the scenario could not be built, or Apollo rejected
the route so the ego never drove) scores ``WORST``. Records take 35-85 MB a
minute, so only those with a finding, or with a NaN prediction to look into, are kept.
"""

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Tuple

from websocket import WebSocketException

from ..analyze import RecordAnalysis, RecordAnalyzer, Violation
from ..manager.apollo_manager import record_name
from ..manager.container import RECORD_DIR, ApolloContainer, ContainerError
from ..manager.scenario_runner import ScenarioRunner
from ..map import MapService
from ..scenario.scenario_chromosome import Scenario, ScenarioContext, ScenarioRules
from ..scenario.sections import InvalidSection
from ..utils import ApolloVehicleConfig
from .objectives import EGO_KINDS, WORST, Objectives, score
from .signature import Places, signature

__all__ = ["ApolloEvaluator", "Evaluation", "EvaluationError", "INFRASTRUCTURE_ERRORS", "Judge", "RunnerFactory"]

INFRASTRUCTURE_ERRORS = (ContainerError, OSError, subprocess.TimeoutExpired, WebSocketException)
"""
Failures of the environment rather than of the scenario: Docker and container
commands, and the bridge's and the HMI's connections (``OSError`` covers
connection errors and timeouts).
"""

RunnerFactory = Callable[..., ScenarioRunner]
"""``ScenarioRunner.from_scenario`` or a stand-in, called ``(scenario, context, container, *, rules, record)``."""
Judge = Callable[[Path], Tuple[RecordAnalysis, Tuple[str, ...]]]
"""A record to what it shows (``RecordAnalyzer.analyze``) and its findings' keys (``signature``)."""


class EvaluationError(RuntimeError):
    """The environment failed twice running one scenario; the search stops rather than score it."""


@dataclass(frozen=True)
class Evaluation:
    """
    One scenario evaluated.

    :param int generation: Its generation.
    :param int individual: Its index among the generation's offspring.
    :param str name: Its record name, ``<run_id>_gen_<g>_ind_<i>``.
    :param Objectives objectives: Its fitness; ``WORST`` when invalid.
    :param bool valid: Whether the run tested something: the scenario was built and the ego drove.
    :param str reason: How the run ended (``RunResult.reason``), or why it is invalid.
    :param violations: Everything the oracles found.
    :param str | None worst: What happened at the least blame margin.
    :param signature: The keys of what it found (``search.signature``): equal ones are duplicates for selection.
    :param transitions: Each change of Planning's state (``RecordAnalysis.transitions``), for its rarity.
    :param Path | None record: The record on the host, when kept.
    :param float seconds: Wall seconds the evaluation took.
    """

    generation: int
    individual: int
    name: str
    objectives: Objectives
    valid: bool
    reason: str
    violations: Tuple[Violation, ...] = ()
    worst: str | None = None
    signature: Tuple[str, ...] = ()
    transitions: Tuple[str, ...] = ()
    record: Path | None = None
    seconds: float = 0.0

    def summary(self) -> str:
        """
        One line for the log: the name, how the run ended, the objectives and the ego's kinds.

        :rtype: str
        """
        kinds = sorted({violation.kind for violation in self.violations} & EGO_KINDS)
        return (f"{self.name}: {self.reason}, margin {self.objectives.margin:.2f} m, {self.objectives.kinds} kinds"
                + (f" ({', '.join(kinds)})" if kinds else "") + ("" if self.valid else ", invalid"))

    def to_json(self) -> Dict[str, Any]:
        """
        The evaluation as one line of ``evaluations.jsonl``.

        :rtype: Dict[str, Any]
        """
        return {"generation": self.generation, "individual": self.individual, "name": self.name,
                "margin": self.objectives.margin, "kinds": self.objectives.kinds, "valid": self.valid,
                "reason": self.reason, "worst": self.worst, "signature": list(self.signature),
                "transitions": list(self.transitions), "seconds": round(self.seconds, 1),
                "record": None if self.record is None else str(self.record),
                "violations": [{"kind": v.kind, "t": v.t, "subject": v.subject, "detail": v.detail}
                               for v in self.violations]}


class ApolloEvaluator:
    """
    Run one scenario in the container and score its record:
    ``evaluate(scenario, generation, individual)`` for ``run_nsga2``.

    :param ApolloContainer container: The container, running the HMI on the context's map, the bridge and the modules.
    :param ScenarioContext context: The map context the scenarios are valid on.
    :param MapService map_service: The map Apollo drives, for the oracles.
    :param ApolloVehicleConfig vehicle: The ego's dimensions.
    :param descriptor_path: The protobuf descriptor of the container's Apollo.
    :param str run_id: The search's name, the records' prefix.
    :param ScenarioRules rules: The rules the scenarios are valid under.
    :param str keep_records: ``"interesting"`` keeps a record with a violation of the ego's, a negative margin
        or a NaN prediction (to look into); ``"all"`` keeps every record.
    :param RunnerFactory runner: Builds the runner of a scenario.
    :param Judge | None judge: Judges a record; ``judge_record`` by default.
    :param float min_free_gb: Gigabytes the records' disk must have free before a run starts.
    :param Path | None records: Where the records land on the host; the container's ``data/bag`` by default.
    :raises ValueError: An unknown ``keep_records``.
    """

    logger: Logger

    def __init__(self, container: ApolloContainer, context: ScenarioContext, *, map_service: MapService,
                 vehicle: ApolloVehicleConfig, descriptor_path: str | Path, run_id: str,
                 rules: ScenarioRules = ScenarioRules(), keep_records: Literal["interesting", "all"] = "interesting",
                 runner: RunnerFactory = ScenarioRunner.from_scenario, judge: Judge | None = None,
                 min_free_gb: float = 10.0, records: Path | None = None) -> None:
        if keep_records not in ("interesting", "all"):
            raise ValueError(f"keep_records must be 'interesting' or 'all', not {keep_records!r}")
        self.logger = logging.getLogger("ApolloEvaluator")
        self.container = container
        self.context = context
        self.map_service = map_service
        self.vehicle = vehicle
        self.descriptor_path = descriptor_path
        self.run_id = run_id
        self.rules = rules
        self.keep_records = keep_records
        self.runner = runner
        self.judge = judge if judge is not None else self.judge_record
        self.min_free_gb = min_free_gb
        self._analyzer: RecordAnalyzer | None = None
        self._places: Places | None = None
        self.records = records if records is not None or container is None else container.host_path(RECORD_DIR)

    def __call__(self, scenario: Scenario, generation: int, individual: int) -> Evaluation:
        """
        Evaluate ``scenario`` as individual ``individual`` of generation ``generation``.

        :rtype: Evaluation
        :raises EvaluationError: The environment failed both times, or the records' disk is nearly full.
        """
        started = time.monotonic()
        name = f"{self.run_id}_{record_name(generation, individual)}"
        if self.records is not None and Path(self.records).exists():
            free = shutil.disk_usage(self.records).free / 1e9
            if free < self.min_free_gb:
                # Before the run, so the checkpoint has every evaluation done: --resume goes on with this one.
                raise EvaluationError(f"only {free:.1f} GB free under {self.records}, below {self.min_free_gb:g} GB; "
                                      f"free some and --resume")

        def evaluation(objectives: Objectives, valid: bool, reason: str, **found: Any) -> Evaluation:
            return Evaluation(generation, individual, name, objectives, valid, reason,
                              seconds=time.monotonic() - started, **found)

        for attempt in (1, 2):
            try:
                runner = self.runner(scenario, self.context, self.container, rules=self.rules, record=name)
            except InvalidSection as error:
                return evaluation(WORST, False, f"invalid: {error}")
            try:
                result = runner.run()
                break
            except INFRASTRUCTURE_ERRORS as error:
                if attempt == 2:
                    raise EvaluationError(f"{name} failed twice: {type(error).__name__}: {error}") from error
                self.logger.warning("%s failed (%s: %s); running it once more", name, type(error).__name__, error)
        record = runner.apollo.record_file
        if result.reason == "routing_failed":
            _discard(record)
            return evaluation(WORST, False, f"routing_failed: {runner.apollo.routing_error}")
        analysis, keys = self.judge(record)
        violations = analysis.violations
        blind = sorted({violation.subject for violation in violations if violation.kind == "prediction_nan"})
        if blind:
            self.logger.warning("%s: Prediction predicted NaN trajectories for %s", name, ", ".join(blind))
        objectives = score(violations, analysis.margin)
        keep = self.keep_records == "all" or objectives.kinds > 0 or objectives.margin < 0 or bool(blind)
        if not keep:
            _discard(record)
        return evaluation(objectives, True, result.reason, violations=tuple(violations), worst=analysis.worst,
                          signature=keys, transitions=analysis.transitions, record=record if keep else None)

    def judge_record(self, record: Path) -> Tuple[RecordAnalysis, Tuple[str, ...]]:
        """
        Analyze ``record`` (``RecordAnalyzer``) and key what it found.

        :param Path record: The record on the host.
        :returns: What it shows and the findings' keys.
        """
        if self._analyzer is None:
            self._analyzer = RecordAnalyzer(self.map_service, self.vehicle, self.descriptor_path)
            self._places = Places(self.map_service)
        analysis = self._analyzer.analyze(record)
        return analysis, signature(analysis.violations, analysis.verdicts, analysis.worst_at, self._places)


def _discard(record: Path | None) -> None:
    if record is not None:
        Path(record).unlink(missing_ok=True)
