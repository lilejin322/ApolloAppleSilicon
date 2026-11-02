"""
Search for scenarios the ego Apollo is to blame for, with NSGA-II, on this
checkout's container.

    .venv/bin/python -m apollo_container.search --map borregas_ave --population 20 --generations 30
    .venv/bin/python -m apollo_container.search --resume runs/nsga2_20261101T120000Z

Every evaluation plays one scenario in real time, about a minute and a half
with the reset and the analysis. See docs/search.md.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from ..genetic.scenario_chromosome import ScenarioKinds, ScenarioOperatorConfig
from ..manager.container import ApolloContainer
from ..map import MapService
from ..scenario import ADAgent, BicycleAgent, NPCVehicleAgent, RouteSpace, VehicleType, walk_network
from ..scenario.scenario_chromosome import ScenarioContext
from ..scenario.traffic_signal import signal_timings
from ..utils import get_vehicle_params
from .evaluation import ApolloEvaluator, EvaluationError
from .nsga2 import NSGA2Config, run_nsga2
from .store import RunStore

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "message" / "common_msgs_v9.desc"
VEHICLE_PARAMS = ROOT / "apollo" / "modules" / "common" / "data" / "vehicle_param.pb.txt"
EGO_TURN_RADIUS = 6.0
"""Metres; the ego's routes keep to it, as in ``integration/analyze_validation.py``."""


def hmi_map_name(name: str) -> str:
    """
    The HMI's name of a map directory, e.g. ``borregas_ave`` -> ``Borregas Ave``.

    :rtype: str
    """
    return " ".join(word.capitalize() for word in name.split("_"))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """
    The command line.

    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(prog="python -m apollo_container.search", description=__doc__.split("\n\n")[0])
    parser.add_argument("--map", default="borregas_ave", help="map directory under map/data (default borregas_ave)")
    parser.add_argument("--population", type=int, default=20, help="individuals per generation, a multiple of 4")
    parser.add_argument("--generations", type=int, default=30, help="generations after the random one")
    parser.add_argument("--hours", type=float,
                        help="stop starting evaluations after this many hours in all; with --resume, a new total")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cxpb", type=float, default=0.8, help="crossover chance per pair")
    parser.add_argument("--mutpb", type=float, default=0.3, help="mutation chance per child")
    parser.add_argument("--fixed-signals", action="store_true", help="keep the default signal plans")
    parser.add_argument("--keep-all-records", action="store_true", help="keep records without findings too")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help="where searches live (default runs/)")
    parser.add_argument("--resume", type=Path, metavar="DIR", help="go on with the search in DIR, with its settings")
    return parser.parse_args(argv)


def settings(args: argparse.Namespace) -> Dict[str, Any]:
    """
    The search's settings as ``config.json`` keeps them.

    :rtype: Dict[str, Any]
    """
    return {"map": args.map, "population": args.population, "generations": args.generations, "hours": args.hours,
            "seed": args.seed, "cxpb": args.cxpb, "mutpb": args.mutpb, "evolve_signals": not args.fixed_signals,
            "keep_records": "all" if args.keep_all_records else "interesting"}


def resumed(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    A resumed search's settings: its own, with the total ``--hours`` given on
    the command line, which lets a search that ran out of time go on.

    :rtype: Dict[str, Any]
    """
    return config if args.hours is None else {**config, "hours": args.hours}


def nsga2_config(config: Dict[str, Any]) -> NSGA2Config:
    """
    The loop's settings from ``config.json``.

    :rtype: NSGA2Config
    """
    return NSGA2Config(population=config["population"], generations=config["generations"], hours=config["hours"],
                       cxpb=config["cxpb"], mutpb=config["mutpb"], seed=config["seed"])


def scenario_context(service: MapService) -> ScenarioContext:
    """
    The map context the scenarios are generated and validated on.

    :rtype: ScenarioContext
    """
    return ScenarioContext(RouteSpace(service, ADAgent.rules(min_turn_radius=EGO_TURN_RADIUS)),
                           {vt: RouteSpace(service, NPCVehicleAgent.rules(vt)) for vt in VehicleType},
                           BicycleAgent.space(service), walk_network(service), signal_timings(service))


def prepare(container: ApolloContainer, map_name: str) -> None:
    """
    Start the container and the HMI, select ``map_name`` unless it is current
    (selecting stops the mode's modules, so it comes first), then start the
    bridge and the ADS modules.

    :param ApolloContainer container: The container.
    :param str map_name: The HMI's name of the map, e.g. ``Borregas Ave``.
    """
    container.start_container()
    container.start_hmi()
    with container.open_hmi() as hmi:
        if hmi.get_current_map() != map_name:
            hmi.select_map(map_name)
    container.start_bridge()
    container.start_ads_modules()


def _log_to(directory: Path) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file = logging.FileHandler(directory / "search.log")
    console = logging.StreamHandler()
    console.addFilter(lambda record: record.levelno >= logging.WARNING or record.name == "NSGA2")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (file, console):
        handler.setFormatter(formatter)
        root.addHandler(handler)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run or resume a search; returns the exit status.

    :rtype: int
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.resume:
        store = RunStore(args.resume)
        config = resumed(store.config, args)
        if config != store.config:
            nsga2_config(config)  # Bad settings fail before they are kept.
            store.save_config(config)
    else:
        nsga2_config(settings(args))  # Bad settings fail before a directory is made.
        store = RunStore.create(args.runs_dir, settings(args))
    config = store.config
    _log_to(store.directory)
    service = MapService(DESCRIPTOR, map_path=ROOT / "map" / "data" / config["map"] / "base_map.bin")
    context = scenario_context(service)
    container = ApolloContainer(ROOT / "apollo", descriptor_path=DESCRIPTOR)
    prepare(container, hmi_map_name(config["map"]))
    evaluator = ApolloEvaluator(container, context, map_service=service,
                                vehicle=get_vehicle_params(VEHICLE_PARAMS, descriptor_path=DESCRIPTOR),
                                descriptor_path=DESCRIPTOR, run_id=store.run_id, keep_records=config["keep_records"])
    try:
        result = run_nsga2(ScenarioKinds(context), evaluator, store, nsga2_config(config),
                           operators=ScenarioOperatorConfig(evolve_signals=config["evolve_signals"]))
    except EvaluationError as error:
        print(f"{store.directory}: stopped: {error}", file=sys.stderr)
        return 1
    print(f"{store.directory}: generation {result.generation} {'finished' if result.finished else 'out of time'}, "
          f"Pareto front of {len(result.pareto)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
