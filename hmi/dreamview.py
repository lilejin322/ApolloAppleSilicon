"""
Client for the legacy Dreamview backend in the Apollo 9.0 checkout.

Author: Lejin Li, Kyushu University & National Institute of Informatics, Japan
Contact: li.lejin.147@s.kyushu-u.ac.jp

Legacy commands use flat JSON and generally have no acknowledgement. Returning
from a command means it was sent, not that Apollo completed it. Query replies
are matched by type, since this protocol has no request IDs.
"""

import json
import select
import time
import websocket
from websocket import WebSocket
from math import isfinite
from pathlib import Path
from logging import getLogger
from shapely.geometry import Point
from threading import Event, RLock, Thread
from google.protobuf.message import Message
from types import TracebackType
from typing import Any, List, Optional, Tuple, Dict, Union, Callable, Type, Self
if __package__:
    from .interface import DreamviewInterface
    from ..message.descriptor import DescriptorBundle
else:
    from interface import DreamviewInterface
    from message.descriptor import DescriptorBundle

logger = getLogger(__name__)

# The backend broadcasts HMIStatus about once a second (~1.35 KB) to every
# connection. Unread, it fills the socket buffers of a long-held instance, and
# the backend then blocks on every broadcast to it.
_IDLE_DRAIN_S = 1.0


class Dreamview(DreamviewInterface):
    """
    Legacy counterpart to DreamviewPlus, defaulting to port 8899.

    Keep one instance per scenario. Calls on an instance are serialized.
    Unrelated broadcasts are discarded; this is not a streaming subscriber.
    A query first discards frames that arrived before it, so its reply is
    no older than the query. A background thread also discards them every
    ``_IDLE_DRAIN_S`` while no call runs, so an instance held idle for long,
    e.g. across a genetic algorithm run, does not stall the backend. After a
    query times out, reconnect before querying again: a reply arriving late,
    after the next query is sent, cannot be distinguished from that query's
    reply.

    :param str host: the host of the legacy Dreamview websocket server,
                     in apple mac os Docker Desktop, we don't have Docker bridge
                     network, so we have to use localhost.
    :param int port: the port of the legacy Dreamview websocket server (new has been moved to 8888)
                     TODO: multi-ctn start should assign other ports when launch legacy Dreamview.
    :param float timeout: the timeout for the websocket connection.
    :param str | Path descriptor_path: Required protobuf descriptor bundle used by this client.
    """
    descriptor_path: Path
    """The protobuf message descriptor path"""
    host: str
    """The host ip address"""
    port: int
    """The port of Dreamview, default to 8899 in version 9.0+"""
    timeout: float
    """The websocket timeout"""
    ws: WebSocket
    """The websocket connection for Dreamview to its backend"""

    def __init__(self, host="localhost", port: int = 8899,
                 timeout: float = 20.0, *, descriptor_path: str | Path) -> None:
        """
        Constructor

        :param str host: the host of the legacy Dreamview websocket server,
                         in apple mac os Docker Desktop, we don't have Docker bridge
                         network, so we have to use localhost.
        :param int port: the port of the legacy Dreamview websocket server (new has been moved to 8888)
                         TODO: multi-ctn start should assign other ports when launch legacy Dreamview.
        :param float timeout: the timeout for the websocket connection.
        :param str | Path descriptor_path: Required protobuf descriptor bundle used by this client.
        """
        if isinstance(timeout, bool) or not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.descriptor_path = DescriptorBundle(descriptor_path).descriptor_path
        self.host = host
        self.port = port
        self.timeout = timeout
        self._lock = RLock()
        self.ws = websocket.create_connection(
            f"ws://{host}:{port}/websocket", timeout=timeout
        )
        self._closed = Event()
        self._drainer = Thread(target=self._drain_while_idle,
                               name="dreamview-drain", daemon=True)
        self._drainer.start()

    def close(self) -> None:
        """
        Close the websocket conn.
        """
        self._closed.set()
        with self._lock:
            try:
                self.ws.close()
            except Exception as e:
                logger.exception(f"Error closing Dreamview websocket: {e}")
        self._drainer.join()

    def __enter__(self) -> Self:
        """
        Usage: with Dreamview(descriptor_path="/schemas/apollo.desc") as dv:
                   dv.selectmode('Pnc')
                   dv.start_sim_control()
                   dv.start_module("Planning")
               # leave with will call __exit__()

        :returns: This dreamview instance
        :rtype: Dreamview
        """
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """
        Close the client when leaving its context.

        :param Optional[Type[BaseException]] exc_type: Exception type, or None.
        :param Optional[BaseException] exc_value: Exception instance, or None.
        :param Optional[TracebackType] traceback: Exception traceback, or None.
        """
        self.close()

    def select_mode(self, mode: str) -> None:
        """
        Send CHANGE_MODE using a name from the backend's mode configuration.
        
        :param str mode: the mode to select
        """
        self._hmi_action("CHANGE_MODE", mode)

    def get_current_mode(self) -> str:
        """
        Get the current selected Dreamview mode.

        :returns: Mode name, such as "Pnc" or "Default".
        :rtype: str
        """
        return self._status_name("currentMode")

    def select_map(self, map_name: str) -> None:
        """
        Select the map, same as modify the global flag txt file, stop current mode's modules
        reload the map.
        No suggestion for use but an API, I think it's better to modify the tag using script directly
        modify target: /apollo/modules/common/data/global_flagfile.txt
        
        :param str map_name: the map name, note that should convert the name to pass in CAPITAL & SPACE.
        E.g.                 borregas_ave -> 'Borregas Ave'
                             san_mateo    -> 'San Mateo'
                             shalun       -> 'Shalun'
        """
        self._hmi_action("CHANGE_MAP", map_name)

    def get_current_map(self) -> str:
        """
        Get the currently selected map.

        :returns: Map name, such as "Borregas Ave".
        :rtype: str
        """
        return self._status_name("currentMap")

    def select_vehicle(self, vehicle: str) -> None:
        """
        Select the vehicle, note that this will also copy the corresponding vehicle's sensor configs

        No suggestion for use but an API, contains wrong legacy sensor's configurations that don't match 9.0
        If you use carla for real-sensing simulation, should modify and fix it.
        modify target: /apollo/modules/common/data/vehicle_param.pb.txt

        :param str vehicle: the vehicle name, note that should convert the same to pass in CAPITAL & SPACE
        E.g.                kitti_140   -> 'Kitti 140'
                            mkz_example -> 'Mkz Example'
        """
        self._hmi_action("CHANGE_VEHICLE", vehicle)

    def get_current_vehicle(self) -> str:
        """
        Get the currently selected vehicle.

        :returns: Vehicle name, such as "Mkz Example".
        :rtype: str
        """
        return self._status_name("currentVehicle")

    def start_sim_control(self, dynamic_model="Simulation Perfect Control") -> None:
        """
        Switch the operation to Sim Control and select the dynamic model.

        Wait for HMI to report the model list and selection, within timeout.
        This does not verify localization/chassis publication. A timeout does
        not undo commands already sent.

        :param str dynamic model: currently only "Simulation Perfect Control" has been implemented
                                  This means that users can define their only vehicle 'dynamic model'
                                  for PnC testing.
        """
        with self._lock:
            deadline = time.monotonic() + self.timeout
            self._send("ToggleSimControl", enable=True)
            self._hmi_action("LOAD_DYNAMIC_MODELS")
            self._wait_status(
                lambda status: dynamic_model in status.get("dynamicModels", []),
                deadline,
            )
            self._hmi_action("CHANGE_DYNAMIC_MODEL", dynamic_model)
            self._wait_status(
                lambda status: status.get("currentDynamicModel") == dynamic_model,
                deadline,
            )

    def stop_sim_control(self) -> None:
        """
        Send ToggleSimControl(false); no acknowledgement is available.
        """
        self._send("ToggleSimControl", enable=False)

    def get_sim_control(self) -> bool:
        """
        Get whether Sim Control is enabled.

        No request reports it: the backend pushes SimControlStatus only to a
        newly joined client, so this reads it from a short-lived connection.
        The backend asks the sim_control process, so this is its actual state,
        unlike currentDynamicModel, which stays set after Sim Control stops.

        :returns: True if Sim Control is enabled.
        :rtype: bool
        :raises RuntimeError: SimControlStatus has no boolean ``enabled``.
        :raises websocket.WebSocketException: No SimControlStatus arrives
            within ``timeout``, or connecting or receiving fails.
        """
        deadline = time.monotonic() + self.timeout
        probe = websocket.create_connection(
            f"ws://{self.host}:{self.port}/websocket", timeout=self.timeout
        )
        try:
            status = self._recv_type(probe, "SimControlStatus", deadline)
        finally:
            probe.close()
        enabled = status.get("enabled")
        if not isinstance(enabled, bool):
            raise RuntimeError("SimControlStatus returned invalid enabled")
        return enabled

    def start_module(self, module: str) -> None:
        """
        Turn on the distinct module, same effect as click on the panel in your browser.
        
        :param str module: the module you want to call up, same as 'source cyber_setup.sh & cyber_launch start'
                           command in the ctn.
        cyber_launch E.g.
            cyber_launch start /apollo/modules/prediction/launch/prediction.launch
        """
        self._hmi_action("START_MODULE", module)

    def stop_module(self, module: str) -> None:
        """
        Stop the module, same effect as click on the panel in your browser.
        
        :param str module: the module you want to stop, same as 'source cyber_setup.bash & cyber_launch stop'
                           command in the ctn.
        
        cyber_launch E.g.
            cyber_launch stop /apollo/modules/planning/planning_component/launch/planning.launch
        """
        self._hmi_action("STOP_MODULE", module)

    def get_modules(self) -> Dict[str, bool]:
        """
        Get the running state of the current mode's modules.

        These are HMI's observations of module processes, updated by Monitor.
        They may be stale when Monitor is not running and do not establish
        that a module is ready to process data.

        :returns: {module_name: running}, for the current mode's modules only.
                  An empty dict if the mode has no modules or the status omits
                  the modules field.
        :rtype: Dict[str, bool]
        :raises RuntimeError: HMIStatus data is invalid, modules is not a dict,
            or a module's running state is not a bool.
        :raises websocket.WebSocketTimeoutException: No matching HMIStatus
            arrives within timeout.
        :raises websocket.WebSocketException: Sending or receiving fails,
            including a closed connection.
        """
        modules = self._get_hmi_status().get("modules", {})
        if not isinstance(modules, dict) or any(
            not isinstance(value, bool) for value in modules.values()
        ):
            raise RuntimeError("HMIStatus returned invalid modules")
        return modules

    def send_routing_request(self, points: List[Point],
                             heading: Optional[float] = None) -> None:
        """
        Send intermediate waypoints followed by a destination, without ACK.

        Like DreamviewPlus, start pose is set separately. Apollo 9 constructs
        a LaneFollowCommand; an explicit legacy 'start' field is ignored.

        :param List[Point] points: Non-empty list of Shapely Points in map coordinates.
                                   The last point is the destination; earlier points are
                                   intermediate waypoints, not an explicit start pose.
                                   Only x and y are used.
        :param Optional[float] heading: Optional destination heading in radians, default None.
        """
        if not points:
            raise ValueError("points must contain at least one Point")
        poses = [self._point_xy(point) for point in points]
        end = poses[-1]
        if heading is not None:
            end["heading"] = self._number(heading)
        payload = {"end": end}
        if len(poses) > 1:
            payload["waypoint"] = poses[:-1]
        self._send("SendRoutingRequest", **payload)

    def get_route_path(self) -> Tuple[List[List[Point]], float]:
        """
        Read cached route segments and routingTime; [] means no route.

        Segment order and optional z are preserved. The backend downsamples
        points by angle for display, so a straight stretch keeps only its
        endpoints. This does not establish that the cached route belongs to
        the latest routing request.

        :returns: (paths, routing_time), where each path is an ordered list of
                  Shapely Points in map coordinates, preserving optional z.
                  paths is [] when no route is cached. routing_time is the
                  backend's routingTime timestamp in seconds.
        :rtype: Tuple[List[List[Point]], float]
        """
        data = self._request("RequestRoutePath", "RoutePath")
        try:
            paths = data["routePath"]
            if not isinstance(paths, list):
                raise ValueError("routePath must be a list")
            result = []
            for path in paths:
                if not isinstance(path["point"], list):
                    raise ValueError("point must be a list")
                result.append([
                    Point(*(self._number(point[key]) for key in
                            (("x", "y", "z") if "z" in point else ("x", "y"))))
                    for point in path["point"]
                ])
            return result, self._number(data["routingTime"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("RoutePath returned invalid route data") from exc

    def send_initial_point(self, point: Point, heading: float = 0.0) -> None:
        """
        Send SetStartPoint; an active Sim Control model is required.

        The backend also requests CLEAR_PLANNING if planning.dag is running.
        Backend rejection is logged server-side, not returned to this client.

        :param Point point: Non-empty Shapely Point in map coordinates.
                            Only x and y are used.
        :param float heading: Initial vehicle heading in radians, default 0.0.
                              this value is mandatory in the old dv backend.
        """
        pose = self._point_xy(point)
        pose["heading"] = self._number(heading)
        self._send("SetStartPoint", point=pose)

    def get_current_pose(self) -> Tuple[Point, Optional[float]]:
        """
        Read the cached car pose from SimulationWorld (no GetStartPoint API).
        
        :returns: A Shapely Point in map coordinates and heading in radians.
                  Heading is None if the backend does not provide it.
        :rtype: Tuple[Point, Optional[float]]
        """
        world = self._get_simulation_world()
        car = world.auto_driving_car
        if not car.HasField("position_x") or not car.HasField("position_y"):
            raise RuntimeError("SimulationWorld has no vehicle position")
        try:
            point = Point(self._number(car.position_x), self._number(car.position_y))
            heading = self._number(car.heading) if car.HasField("heading") else None
        except ValueError as exc:
            raise RuntimeError("SimulationWorld returned invalid pose") from exc
        return point, heading

    def setup_mode(self) -> None:
        """
        Ask HMI to start all modules in the current mode.

        Sends SETUP_MODE without waiting for a backend acknowledgement or
        module readiness. Use get_modules() to observe process status through
        Monitor; a running process does not establish that the module is ready.

        :raises websocket.WebSocketException: Sending fails, including a
            socket timeout or a closed connection.
        """
        self._hmi_action("SETUP_MODE")

    def reset_mode(self) -> None:
        """
        Ask HMI to stop all modules in the current mode.

        Sends RESET_MODE without waiting for a backend acknowledgement or
        process shutdown. Use get_modules() to observe process status through
        Monitor. This resets the mode's modules; reset() instead resets the
        world cache and active simulation model.

        :raises websocket.WebSocketException: Sending fails, including a
            socket timeout or a closed connection.
        """
        self._hmi_action("RESET_MODE")

    def enter_auto_mode(self) -> None:
        """
        Request a transition to COMPLETE_AUTO_DRIVE.

        The backend first requests COMPLETE_MANUAL, then requests auto mode
        only if that transition succeeds. Neither result is returned to this
        client. Verify the actual driving mode on the chassis channel.
        Perfect Control always publishes auto mode and cannot complete the
        manual prerequisite.

        :raises websocket.WebSocketException: Sending fails, including a
            socket timeout or a closed connection.
        """
        self._hmi_action("ENTER_AUTO_MODE")

    def disengage(self) -> None:
        """
        Request a transition to COMPLETE_MANUAL.

        Sends DISENGAGE without a backend acknowledgement. Verify the actual
        driving mode on the chassis channel. Perfect Control always publishes
        auto mode, so this request cannot switch that model to manual mode.

        :raises websocket.WebSocketException: Sending fails, including a
            socket timeout or a closed connection.
        """
        self._hmi_action("DISENGAGE")

    def reset(self) -> None:
        """
        Request world-cache and active simulation-model reset, without ACK.

        Perfect Control clears its route header/trajectory but retains pose
        and does not explicitly zero speed/acceleration. Timers and ADS modules
        keep running. This does not send CLEAR_PLANNING to Planning. The world
        cache clears asynchronously, even if the model reset fails.

        An enabled Sim Control manager with an active model is required for
        the model reset. Backend failure is logged server-side and is not
        returned to this client. Returning does not establish that either
        the model reset or the world-cache update has completed.

        :raises websocket.WebSocketException: Sending fails, including a
            socket timeout or a closed connection.
        """
        self._send("Reset")

    def _hmi_action(self, action, value=None) -> None:
        """
        Send a legacy HMI action without waiting for a backend acknowledgement.

        Returning means the request was sent, not that the action succeeded.
        Backend rejection is logged server-side and is not reported here.

        :param str action: Backend HMI action name, such as START_MODULE or
                           CHANGE_MODE.
        :param Optional[str] value: Action value, such as a module or mode name.
                                    Omitted from the payload when None.
        :raises TypeError: The payload contains a value that cannot be JSON encoded.
        :raises ValueError: The payload contains non-finite numbers or a
            circular reference.
        :raises websocket.WebSocketException: Sending fails, including a
            socket timeout or a closed connection.
        """
        fields = {"action": action}
        if value is not None:
            fields["value"] = value
        self._send("HMIAction", **fields)

    def _get_hmi_status(self) -> Dict[str, Any]:
        """
        Read HMI status (modes/maps/vehicles/modules/dynamic models).

        The reply may be a status-change broadcast sent just before the
        reply, never one queued before this call. It is an observation, not
        acknowledgement of a preceding command.

        :returns: HMIStatus JSON fields mapped to their values, for example
                  currentMode (str), modes (List[str]), modules (Dict[str, bool])
                  and dynamicModels (List[str]). Values may also contain
                  numbers, booleans, nested dicts or None.
        :rtype: Dict[str, Any]
        :raises RuntimeError: HMIStatus data is missing or is not a dict.
        :raises websocket.WebSocketTimeoutException: No matching HMIStatus
            arrives within timeout.
        :raises websocket.WebSocketException: Sending or receiving fails,
            including a closed connection.
        """
        response = self._request("HMIStatus", "HMIStatus")
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("HMIStatus returned invalid data")
        return data

    def _status_name(self, field: str) -> str:
        """
        Read a non-empty name field from the latest HMIStatus observation.

        :param str field: HMIStatus JSON field name, such as currentMode,
                          currentMap or currentVehicle.
        :returns: The field's original string value, without stripping whitespace.
        :rtype: str
        :raises RuntimeError: HMIStatus data is invalid, or the field is missing,
            is not a string, or contains only whitespace.
        :raises websocket.WebSocketTimeoutException: No matching HMIStatus
            arrives within timeout.
        :raises websocket.WebSocketException: Sending or receiving fails,
            including a closed connection.
        """
        value = self._get_hmi_status().get(field)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"HMIStatus returned missing or invalid {field}")
        return value

    def _request(self, msg_type, response_type, deadline=None,
                 **fields) -> Union[Dict[str, Any], bytes]:
        """
        Send a legacy request and wait for a response of the expected type.

        Serializes access to the connection, discards queued frames before
        sending, and skips unrelated replies. The legacy protocol has no
        request IDs, so a matching type does not prove request correlation.
        Restores the socket timeout to self.timeout when the call finishes.

        :param str msg_type: Backend request type, such as HMIStatus or
                             RequestRoutePath.
        :param Optional[str] response_type: Expected JSON type, such as HMIStatus
                                            or RoutePath. None selects the first
                                            binary frame instead.
        :param Optional[float] deadline: Absolute deadline on the time.monotonic()
                                         clock. None uses the current monotonic
                                         time plus self.timeout.
        :param Any fields: Additional top-level JSON request fields, supplied
                           as keyword arguments.
        :returns: The complete matching JSON response dict, or raw binary bytes
                  when response_type is None. Backend error fields are not checked.
        :rtype: Union[Dict[str, Any], bytes]
        :raises TypeError: The request contains a value that cannot be JSON encoded.
        :raises ValueError: The request contains non-finite numbers or a
            circular reference.
        :raises websocket.WebSocketTimeoutException: The deadline expires or
            a socket operation times out before a matching response arrives.
        :raises websocket.WebSocketException: Sending or receiving fails,
            including a closed connection.
        """
        with self._lock:
            if deadline is None:
                deadline = time.monotonic() + self.timeout
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise websocket.WebSocketTimeoutException("Dreamview request timed out")
                self.ws.settimeout(remaining)
                self._drain(deadline)
                self._send(msg_type, **fields)
                return self._recv_type(self.ws, response_type, deadline)
            finally:
                self.ws.settimeout(self.timeout)

    def _send(self, msg_type, **fields) -> None:
        """
        Send a flat JSON message over the legacy Dreamview connection.

        Serializes access to the connection and sends without waiting for a
        response. Returning does not establish that the backend accepted or
        completed the requested operation.

        :param str msg_type: Backend message type, such as HMIAction or Reset.
        :param Dict[str, Any] fields: Additional top-level JSON fields, supplied
                                      as keyword arguments. A type field overrides
                                      msg_type in the serialized payload.
        :raises TypeError: The payload contains a value that cannot be JSON encoded.
        :raises ValueError: The payload contains non-finite numbers or a
            circular reference.
        :raises websocket.WebSocketException: Sending fails, including a
            socket timeout or a closed connection.
        """
        with self._lock:
            self.ws.send(json.dumps({"type": msg_type, **fields}, allow_nan=False))

    def _get_simulation_world(self, planning: bool = False) -> Message:
        """
        Decode a binary SimulationWorld protobuf from this client's schema.

        Reads cached world data. Requests sent before the world is ready are
        dropped without a delayed response, so the call can time out.

        :param bool planning: Include detailed PnC monitor data in planning_data
                              when True, default False. Planning trajectories
                              may still be present when False.
        :returns: The decoded apollo.dreamview.SimulationWorld protobuf containing
                  the backend's cached vehicle, object and planning state.
        :rtype: Message
        :raises KeyError: SimulationWorld is not defined in the selected schema.
        :raises google.protobuf.message.DecodeError: The binary response cannot
            be parsed as a SimulationWorld protobuf.
        :raises websocket.WebSocketTimeoutException: No binary response arrives
            within timeout, including when the world is not ready or the
            backend suppresses an oversized response.
        :raises websocket.WebSocketException: Sending or receiving fails,
            including a closed connection.
        """
        world_cls = DescriptorBundle(self.descriptor_path).message("apollo.dreamview.SimulationWorld")
        raw = self._request("RequestSimulationWorld", None, planning=planning)
        world = world_cls()
        world.ParseFromString(raw)
        return world

    @staticmethod
    def _number(value: int | float) -> float:
        """
        Validate a finite int or float and convert it to float.

        :param int | float value: Numeric value to validate. Booleans are
                                       rejected even though bool subclasses int.
        :returns: The validated value converted to float.
        :rtype: float
        :raises ValueError: The value is a bool, is not an int or float, or is
            NaN or infinite.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("coordinate, heading or timestamp must be finite")
        return float(value)

    @classmethod
    def _point_xy(cls, point: Point) -> Dict[str, float]:
        """
        Extract finite x and y coordinates from a non-empty Shapely Point.

        :param Point point: Point in map coordinates. Only x and y are used;
                            any z coordinate is ignored.
        :returns: {"x": x, "y": y}, with both coordinates converted to float.
        :rtype: Dict[str, float]
        :raises TypeError: point is not a Shapely Point.
        :raises ValueError: point is empty, or x or y is NaN or infinite.
        """
        if not isinstance(point, Point):
            raise TypeError("point must be a Shapely Point")
        if point.is_empty:
            raise ValueError("point must not be empty")
        return {"x": cls._number(point.x), "y": cls._number(point.y)}

    def _wait_status(self, predicate: Callable[[Dict[str, Any]], bool], deadline: float) -> None:
        """
        Poll HMIStatus until the predicate is satisfied or the deadline expires.

        Passes each status data dict to predicate and returns when its result
        is truthy. Queries share the same deadline; unsuccessful checks pause
        for up to 0.05 seconds before retrying. Predicate exceptions propagate.

        :param Callable[[Dict[str, Any]], bool] predicate: Function that checks
            whether the observed HMI status satisfies the required condition.
        :param float deadline: Absolute deadline on the time.monotonic() clock,
                               shared by all status requests and retry delays.
        :raises RuntimeError: HMIStatus data is missing or is not a dict.
        :raises websocket.WebSocketTimeoutException: The deadline expires or
            a status request times out before the predicate is satisfied.
        :raises websocket.WebSocketException: Sending or receiving fails,
            including a closed connection.
        """
        while True:
            status = self._request("HMIStatus", "HMIStatus", deadline=deadline).get("data")
            if not isinstance(status, dict):
                raise RuntimeError("HMIStatus returned invalid data")
            if predicate(status):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise websocket.WebSocketTimeoutException("Timed out waiting for HMI status")
            time.sleep(min(0.05, remaining))

    def _drain(self, deadline: float) -> None:
        """
        Discard buffered frames while the connection's socket is readable.

        The backend queues HMIStatus at connection time and broadcasts it on
        every status change. Draining before a request prevents those queued
        frames from being mistaken for its reply. Stops when no data is
        immediately readable or the connection has no socket; reading an
        incomplete frame may still block until the socket timeout.

        :param float deadline: Absolute deadline on the time.monotonic() clock,
                               checked before reading each frame.
        :raises websocket.WebSocketTimeoutException: The deadline expires while
            draining, or reading a frame reaches the socket timeout.
        :raises websocket.WebSocketConnectionClosedException: A close frame is
            received or the connection closes while reading.
        :raises websocket.WebSocketException: Reading or processing a frame fails.
        :raises OSError: Checking socket readiness or accessing the socket fails.
        """
        sock = self.ws.sock
        while sock is not None and select.select([sock], [], [], 0)[0]:
            if time.monotonic() >= deadline:
                raise websocket.WebSocketTimeoutException("Dreamview request timed out")
            opcode, _ = self.ws.recv_data_frame(control_frame=True)
            if opcode == websocket.ABNF.OPCODE_CLOSE:
                raise websocket.WebSocketConnectionClosedException("Dreamview disconnected")

    def _drain_while_idle(self) -> None:
        """
        Periodically discard queued broadcasts in the background drain thread.

        Waits on the close event for _IDLE_DRAIN_S between attempts, then
        acquires the connection lock before draining. This prevents the thread
        from consuming responses while another call holds the lock. Each drain
        uses the current monotonic time plus self.timeout as its deadline.

        Exits when close() sets the event, including if it was set while the
        thread waited for the lock. WebSocket and OS errors are caught, logged
        as warnings, and stop the thread without closing the connection or
        retrying. The error is not stored or forwarded to the caller; later
        operations detect any remaining connection problem themselves.
        """
        while not self._closed.wait(_IDLE_DRAIN_S):
            with self._lock:
                if self._closed.is_set():
                    return
                try:
                    self._drain(time.monotonic() + self.timeout)
                except (websocket.WebSocketException, OSError) as exc:
                    logger.warning("Stopped discarding Dreamview broadcasts: %s", exc)
                    return

    @staticmethod
    def _recv_type(ws, response_type: Optional[str], deadline: float):
        """
        Read until a matching JSON response or the first binary frame arrives.

        Invalid JSON, non-dict JSON values and unrelated messages are discarded.
        Matching uses only the top-level type field; the legacy protocol has
        no request IDs. Each receive uses the remaining time until deadline,
        so skipped messages do not extend it. The socket timeout is updated
        before each receive and is not restored by this helper.

        :param WebSocket ws: Connection from which to receive messages.
        :param Optional[str] response_type: Expected top-level JSON type value.
                                            None selects the first binary frame
                                            and skips all text messages.
        :param float deadline: Absolute deadline on the time.monotonic() clock.
        :returns: The complete matching JSON response dict, or raw binary bytes
                  when response_type is None. Backend error fields are not checked.
        :rtype: Union[Dict[str, Any], bytes]
        :raises websocket.WebSocketTimeoutException: The deadline expires or
            receiving times out before a matching message arrives.
        :raises websocket.WebSocketConnectionClosedException: An empty message
            is received or the connection closes while receiving.
        :raises websocket.WebSocketException: Receiving or processing a
            WebSocket frame fails.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise websocket.WebSocketTimeoutException("Dreamview request timed out")
            ws.settimeout(remaining)
            raw = ws.recv()
            if raw == "" or raw == b"":
                raise websocket.WebSocketConnectionClosedException("Dreamview disconnected")
            if isinstance(raw, bytes):
                if response_type is None:
                    return raw
                continue
            try:
                response = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if (response_type is not None and isinstance(response, dict)
                    and response.get("type") == response_type):
                return response
