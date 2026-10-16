"""
Abstract public interface for legacy Dreamview and Dreamview Plus clients.

Author: Lejin Li, Kyushu University & National Institute of Informatics, Japan
Contact: li.lejin.147@s.kyushu-u.ac.jp
"""

from abc import ABC, abstractmethod
from types import TracebackType
from shapely.geometry import Point
from typing import Dict, List, Optional, Tuple, Type, Self


class DreamviewInterface(ABC):
    """
    Declare the public operations supported by both Dreamview clients.

    Command methods submit backend requests. Completion guarantees and backend
    error feedback depend on the concrete client and its protocol. Available
    mode, map, vehicle and module names also depend on the backend configuration.
    Each implementation owns its connection initialization and lifecycle.
    """

    @abstractmethod
    def close(self) -> None:
        """
        Close the client's connections and release its background resources.
        """
        ...

    @abstractmethod
    def __enter__(self) -> Self:
        """
        Enter the client context.

        :returns: This client instance.
        :rtype: DreamviewInterface
        """
        ...

    @abstractmethod
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """
        Close the client when leaving its context, allowing exceptions to propagate.

        :param Optional[Type[BaseException]] exc_type: Exception type, or None.
        :param Optional[BaseException] exc: Exception instance, or None.
        :param Optional[TracebackType] tb: Exception traceback, or None.
        """
        ...

    @abstractmethod
    def select_mode(self, mode: str) -> None:
        """
        Request selection of an HMI mode.

        :param str mode: Backend mode display name.
        """
        ...

    @abstractmethod
    def get_current_mode(self) -> str:
        """
        Read the currently selected HMI mode.

        :returns: Mode display name.
        :rtype: str
        """
        ...

    @abstractmethod
    def select_map(self, map_name: str) -> None:
        """
        Request selection and loading of a map.

        :param str map_name: Backend map display name, such as Borregas Ave.
        """
        ...

    @abstractmethod
    def get_current_map(self) -> str:
        """
        Read the currently selected map.

        :returns: Map display name.
        :rtype: str
        """
        ...

    @abstractmethod
    def select_vehicle(self, vehicle: str) -> None:
        """
        Request selection of a vehicle and its configuration.

        :param str vehicle: Backend vehicle display name.
        """
        ...

    @abstractmethod
    def get_current_vehicle(self) -> str:
        """
        Read the currently selected vehicle.

        :returns: Vehicle display name.
        :rtype: str
        """
        ...

    @abstractmethod
    def start_sim_control(self,
                          dynamic_model: str = "Simulation Perfect Control") -> None:
        """
        Request Sim Control activation and dynamic-model selection.

        Readiness checks and waiting behavior depend on the concrete client.

        :param str dynamic_model: Dynamic-model display name.
        """
        ...

    @abstractmethod
    def stop_sim_control(self) -> None:
        """
        Request Sim Control shutdown.

        Legacy Dreamview disables Sim Control; Plus switches operation to Record.
        """
        ...

    @abstractmethod
    def get_sim_control(self) -> bool:
        """
        Read the simulation status exposed by the concrete backend.

        Legacy Dreamview reports whether the Sim Control manager is enabled.
        Plus reports whether HMI has a current dynamic model. These observations
        do not establish that localization or chassis messages are being published.

        :returns: The concrete backend's simulation-status flag.
        :rtype: bool
        """
        ...

    @abstractmethod
    def start_module(self, module: str) -> None:
        """
        Request startup of an individual module.

        :param str module: Module name in the current mode.
        """
        ...

    @abstractmethod
    def stop_module(self, module: str) -> None:
        """
        Request shutdown of an individual module.

        :param str module: Module name in the current mode.
        """
        ...

    @abstractmethod
    def get_modules(self) -> Dict[str, bool]:
        """
        Read HMI's process-state observations for the current mode's modules.

        States depend on Monitor and do not establish module readiness.

        :returns: Module names mapped to their observed running states.
        :rtype: Dict[str, bool]
        """
        ...

    @abstractmethod
    def send_routing_request(self, points: List[Point],
                             heading: Optional[float] = None) -> None:
        """
        Request a route through ordered waypoints to a destination.

        Set the initial vehicle pose separately with send_initial_point().

        :param List[Point] points: Non-empty list of map-coordinate points.
                                   The last is the destination; preceding points
                                   are intermediate waypoints. Only x and y are used.
        :param Optional[float] heading: Destination heading in radians, or None.
        """
        ...

    @abstractmethod
    def get_route_path(self) -> Tuple[List[List[Point]], float]:
        """
        Read cached route segments and their routing timestamp.

        This does not establish that the route belongs to the latest request.

        :returns: (paths, routing_time), with ordered Shapely Points per segment
                  and a timestamp in seconds. paths is [] when no route is cached.
        :rtype: Tuple[List[List[Point]], float]
        """
        ...

    @abstractmethod
    def send_initial_point(self, point: Point, heading: float = 0.0) -> None:
        """
        Request a new initial pose for the active simulation model.

        The backend may also asynchronously clear Planning's current command.

        :param Point point: Non-empty point in map coordinates; only x and y are used.
        :param float heading: Initial heading in radians, default 0.0.
        """
        ...

    @abstractmethod
    def get_current_pose(self) -> Tuple[Point, Optional[float]]:
        """
        Read the backend's cached vehicle position and heading.

        :returns: A map-coordinate Point and heading in radians, or None if
                  heading is absent.
        :rtype: Tuple[Point, Optional[float]]
        """
        ...

    @abstractmethod
    def setup_mode(self) -> None:
        """
        Request startup of all modules in the current mode, without an ACK.
        """
        ...

    @abstractmethod
    def reset_mode(self) -> None:
        """
        Request shutdown of all modules in the current mode, without an ACK.
        """
        ...

    @abstractmethod
    def enter_auto_mode(self) -> None:
        """
        Request COMPLETE_AUTO_DRIVE, without a success acknowledgement.

        The backend first attempts to enter manual mode as a prerequisite.
        """
        ...

    @abstractmethod
    def disengage(self) -> None:
        """
        Request COMPLETE_MANUAL, without a success acknowledgement.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Request world-cache and active simulation-model reset.

        Perfect Control retains pose and does not explicitly zero speed or
        acceleration. ADS modules continue running and world clearing is
        asynchronous. Legacy Dreamview sends without ACK; Plus checks its ACK
        and raises RuntimeError on backend failure.
        """
        ...
