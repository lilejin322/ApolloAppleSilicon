"""
Dreamview Plus websocket client.

Author: Lejin Li, Kyushu University & National Institute of Informatics, Japan
Contact: li.lejin.147@s.kyushu-u.ac.jp

Talks to a Dreamview Plus that is already running at ws://host:port/websocket.
Note that the format of the wb backend of Dreamview Plus is different from the 
old Dreamview 1.0, so the websocket client should be rewritten as follow.
"""

import json
import time
import uuid
import websocket
from websocket import WebSocket
from functools import lru_cache
from math import isfinite
from shapely import Point
from typing import Dict, List, Optional, Tuple, Any
from logging import getLogger
from google.protobuf.descriptor_pb2 import FieldDescriptorProto, FileDescriptorProto
from google.protobuf.message import DecodeError, Message
from google.protobuf.message_factory import GetMessages
if __package__:
    from .interface import DreamviewInterface
    from ..message.msg_cls_v9 import message
else:
    from interface import DreamviewInterface
    from message.msg_cls_v9 import message

# These HMI actions are the ones whose handler sends a response frame.
# Waiting on any other action hangs, because Dreamview Plus never replies.
_ACK_ACTIONS = {"CHANGE_MAP", "CHANGE_OPERATION", "CHANGE_MODE"}

# The backend adds a connection to its broadcast list only after the handshake,
# so a one-shot HMIStatus broadcast can miss a just-opened /hmi connection.
_HMI_RESUBSCRIBE_S = 1.0


logger = getLogger(__name__)


@lru_cache(maxsize=None)
def _stream_data_cls() -> type:
    """
    Build the StreamData envelope used by Dreamview Plus binary streams.

    It is defined in modules/dreamview_plus/proto/data_handler.proto, which is
    not part of the bundled common_msgs descriptor.

    :returns: The StreamData message class.
    :rtype: type
    """
    file_proto = FileDescriptorProto(
        name="modules/dreamview_plus/proto/data_handler.proto",
        package="apollo.dreamview", syntax="proto2",
    )
    stream_data = file_proto.message_type.add(name="StreamData")
    for number, name, field_type in (
        (1, "type", FieldDescriptorProto.TYPE_STRING),
        (2, "action", FieldDescriptorProto.TYPE_STRING),
        (3, "data_name", FieldDescriptorProto.TYPE_STRING),
        (4, "channel_name", FieldDescriptorProto.TYPE_STRING),
        (5, "data", FieldDescriptorProto.TYPE_BYTES),
    ):
        stream_data.field.add(name=name, number=number, type=field_type,
                              label=FieldDescriptorProto.LABEL_OPTIONAL)
    return GetMessages([file_proto])["apollo.dreamview.StreamData"]


def _decode_hmi_status(raw: bytes) -> Optional[Message]:
    """
    Decode the HMIStatus carried by a /hmi StreamData frame.

    :param raw: A frame received from /hmi; text frames are str.
    :returns: The HMIStatus, or None if the frame does not carry one
    :rtype: Optional[Message]
    :raises RuntimeError: The frame cannot be decoded.
    """
    if not isinstance(raw, bytes):
        return None
    try:
        stream = _stream_data_cls().FromString(raw)
        if stream.data_name != "hmistatus":
            return None
        return message("apollo.dreamview.HMIStatus").FromString(stream.data)
    except DecodeError as exc:
        raise RuntimeError("/hmi returned an undecodable HMIStatus frame") from exc


class DreamviewPlus(DreamviewInterface):
    """
    Demo representation of a Dreamview Plus websocket client.
    Note that the new version of Dreamvie_plus has a welcome page,
    we've removed the welcome page and rebuilt the Dreamview_plus frontend using Node.js.
    see: https://github.com/lilejin322/BaiduApollo/tree/r9.0.1_applesilicon

    memo: I think this class is multi-thread safe from outside, i.e., each thread create their
    instance, call the public method simultaneously.

    :param str host: the host of the Dreamview Plus websocket server,
                     in apple mac os Docker Desktop, we don't have Docker bridge
                     network, so we have to use localhost.
    :param int port: the port of the Dreamview Plus websocket server (old has been moved to 8899)
                     TODO: multi-ctn start should assign other ports when launch Dreamview Plus.
    :param float timeout: the timeout for the websocket connection.
    """
    host: str
    """The host ip address"""
    port: int
    """The port of DreamviewPlus, default to 8888 in version 9.0+"""
    timeout: float
    """The websocket timeout, also the overall deadline of get_modules()"""
    ws: WebSocket
    """The websocket connection for Dreamview to its backend"""

    def __init__(self, host: str = "localhost", port: int = 8888,
                 timeout: float = 20.0) -> None:
        """
        Constructor

        :param str host: the host of the Dreamview Plus websocket server,
                         in apple mac os Docker Desktop, we don't have Docker bridge
                         network, so we have to use localhost.
        :param int port: the port of the Dreamview Plus websocket server (old has been moved to 8899)
                         TODO: multi-ctn start should assign other ports when launch Dreamview Plus.
        :param float timeout: the timeout for the websocket connection.
        """
        if isinstance(timeout, bool) or not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.host = host
        self.port = port
        self.timeout = timeout
        url = f"ws://{self.host}:{self.port}/websocket"
        self.ws = websocket.create_connection(url, timeout=timeout)

    def close(self) -> None:
        """
        Close the websocket conn.
        """
        try:
            self.ws.close()
        except Exception as e:
            logger.error(f"Error occurred while closing websocket: {e}")

    def __enter__(self):
        """
        Usage: with DreamviewPlus() as dvp:
                   dvp.selectmode('Pnc')
                   dvp.start_sim_control()
                   dv.start_module("Planning")
               # leave with will call __exit__()
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def select_mode(self, mode: str) -> None:
        """
        Select the mode, just to control the HMI layout

        :param str mode: the mode to select, candidates=['Default', 'Pnc', 'Perception', 'Vehicle Test'].
        """
        self._hmi_action("CHANGE_MODE", mode)

    def get_current_mode(self) -> str:
        """
        Get the currently selected Dreamview Plus mode.

        :returns: Mode name, such as "Pnc" or "Default".
        :rtype: str
        """
        info = self._request("GetInitData", {})
        if info.get("code") != 0:
            raise RuntimeError(
                f"GetInitData failed: {info.get('message', info)}"
            )

        data = info.get("data")
        mode = data.get("currentMode") if isinstance(data, dict) else None
        if not isinstance(mode, str) or not mode.strip():
            raise RuntimeError("GetInitData returned a missing or invalid currentMode")
        return mode

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

        No request returns it, so it is read from one HMIStatus frame on /hmi,
        the same way as get_modules().

        :returns: Map name, such as "Borregas Ave".
        :rtype: str
        :raises RuntimeError: No map is selected, or the frame cannot be decoded.
        :raises websocket.WebSocketException: No HMIStatus frame arrives in time,
            or connecting, sending or receiving fails.
        """
        return self._status_name("current_map")

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

        No request returns it, so it is read from one HMIStatus frame on /hmi,
        the same way as get_modules().

        :returns: Vehicle name, such as "Mkz Example".
        :rtype: str
        :raises RuntimeError: No vehicle is selected, or the frame cannot be decoded.
        :raises websocket.WebSocketException: No HMIStatus frame arrives in time,
            or connecting, sending or receiving fails.
        """
        return self._status_name("current_vehicle")

    def start_sim_control(self, dynamic_model="Simulation Perfect Control") -> None:
        """
        Switch the operation to Sim Control and select the dynamic model.

        The model list is filled by LOAD_DYNAMIC_MODELS, so the selection
        has to follow that call. The model, not the operation flag, is what
        publishes localization and chassis.

        :param str dynamic_model: currently only "Simulation Perfect Control" has been implemented
                                  This means that users can define their only vehicle 'dynamic model'
                                  for PnC testing.
        """
        self._hmi_action("CHANGE_OPERATION", "Sim_Control")
        self._hmi_action("LOAD_DYNAMIC_MODELS")
        time.sleep(1)
        self._hmi_action("CHANGE_DYNAMIC_MODEL", dynamic_model)

    def stop_sim_control(self) -> None:
        """
        Switch to Record operation to stop Sim Control.

        The backend stops the simulation and clears the selected dynamic
        model and its available model list. This does not start recording.
        """
        self._hmi_action("CHANGE_OPERATION", "Record")

    def get_sim_control(self) -> bool:
        """
        Get whether Sim Control is running a dynamic model.

        The model, not the operation flag, publishes localization and chassis,
        so this is whether HMIStatus has a current dynamic model. Leaving the
        Sim Control operation clears it at once, and the backend clears it
        within about a second once the sim_control process reports no model.
        If that process becomes unreachable, the last model is kept.

        :returns: True if a dynamic model is running.
        :rtype: bool
        :raises RuntimeError: An HMIStatus frame cannot be decoded.
        :raises websocket.WebSocketException: No HMIStatus frame arrives in time,
            or connecting, sending or receiving fails.
        """
        return bool(self._get_hmi_status().current_dynamic_model.strip())

    def start_module(self, module: str) -> None:
        """
        Turn on the distinct module, same effect as click on the panel in your browser.

        :param str module: the module you want to call up, same as 'source cyber_setup.bash & cyber_launch start'
                           command in the ctn.
        cyber_launch E.g.
            cyber_launch  start  /apollo/modules/prediction/launch/prediction.launch

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

        Dreamview Plus pushes HMIStatus only as a stream on /hmi, so this opens a
        short-lived /hmi connection and subscribes with dataFrequencyMs=0: the
        backend broadcasts one frame without replacing the shared stream timer
        that the browser uses. It never unsubscribes, since that stops the
        browser's stream. No backend state is left behind, so instances in
        other threads may call it at the same time.

        The states are Monitor's observations of each module's processes; they
        stop updating while Monitor is not running.

        NOTE: Currently this function should not be used, the Dreamview plus backend
              has bugs that will read the legacy Dreamview 1.0's config txt, which
              will return all modules [false] in 'Default' mode

        :returns: {module_name: running}, for the current mode's modules only.
        :rtype: Dict[str, bool]
        :raises RuntimeError: An HMIStatus frame cannot be decoded.
        :raises websocket.WebSocketTimeoutException: No HMIStatus frame arrives
            within ``timeout``.
        :raises websocket.WebSocketException: Connecting, sending or receiving fails.
        """
        return dict(self._get_hmi_status().modules)

    def send_routing_request(self, points: List[Point],
                             heading: Optional[float] = None) -> None:
        """
        Send ordered waypoints and a destination from the current vehicle pose.

        :param List[Point] points: Non-empty list of Shapely Points in map coordinates.
                                   The last point is the destination; earlier points are
                                   intermediate waypoints, not an explicit start pose.
                                   Only x and y are used.
        :param Optional[float] heading: Optional destination heading in radians, default None.
        """
        if not points:
            raise ValueError("points must contain at least one Point")
        for point in points:
            if not isinstance(point, Point):
                raise TypeError("points must contain only Shapely Points")
            if point.is_empty:
                raise ValueError("points must not contain empty Points")

        end = {"x": points[-1].x, "y": points[-1].y}
        if heading is not None:
            end["heading"] = heading
        request = {"end": end}
        if len(points) > 1:
            request["waypoint"] = [
                {"x": point.x, "y": point.y} for point in points[:-1]
            ]
        info = self._request("SendRoutingRequest", request)
        code = info.get("code")
        if code not in (None, 0):
            raise RuntimeError(
                f"SendRoutingRequest failed: {info.get('message', info)}"
            )

    def get_route_path(self) -> Tuple[List[List[Point]], float]:
        """
        Get the current route paths and routing timestamp from Dreamview Plus.

        :returns: (paths, routing_time), with each path represented by an
                  ordered list of Shapely Points. Path boundaries and any z
                  coordinates are preserved. No route is represented by [].
                  routing_time is the backend's routingTime, in seconds.
        :raises RuntimeError: If the request fails or route data is invalid.

        This reads cached route data from backend; it does not wait for a new route or
        establish that the route belongs to the latest routing request.
        """
        info = self._request("RequestRoutePath", {})
        if info.get("code") != 0:
            raise RuntimeError(
                f"RequestRoutePath failed: {info.get('message', info)}"
            )

        data = info.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("routePath"), list):
            raise RuntimeError("RequestRoutePath returned invalid routePath")
        routing_time = data.get("routingTime")
        if (
            isinstance(routing_time, bool)
            or not isinstance(routing_time, (int, float))
            or not isfinite(routing_time)
        ):
            raise RuntimeError("RequestRoutePath returned invalid routingTime")

        paths = []
        for path in data["routePath"]:
            if not isinstance(path, dict) or not isinstance(path.get("point"), list):
                raise RuntimeError("RequestRoutePath returned invalid path points")
            points = []
            for point in path["point"]:
                if not isinstance(point, dict):
                    raise RuntimeError("RequestRoutePath returned invalid point")
                coordinates = [point.get("x"), point.get("y")]
                if "z" in point:
                    coordinates.append(point["z"])
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                    for value in coordinates
                ):
                    raise RuntimeError("RequestRoutePath returned invalid coordinates")
                points.append(Point(*coordinates))
            paths.append(points)

        return paths, float(routing_time)

    def send_initial_point(self, point: Point, heading: float = 0.0) -> None:
        """
        Reset the simulated vehicle's initial position and heading.
        Sim Control must be enabled before calling this method.

        :param Point point: Non-empty Shapely Point in map coordinates.
                            Only x and y are used.
        :param float heading: Initial heading in radians, default 0.0
                              this value is mandatory in the dvp backend.
        """
        if not isinstance(point, Point):
            raise TypeError("point must be a Shapely Point")
        if point.is_empty:
            raise ValueError("point must not be empty")

        info = self._request(
            "SetStartPoint",
            {"point": {"x": point.x, "y": point.y, "heading": heading}},
        )
        if info.get("code") != 0:
            raise RuntimeError(
                f"SetStartPoint failed: {info.get('message', info)}"
            )

    def get_current_pose(self) -> Tuple[Point, Optional[float]]:
        """
        Get the latest vehicle pose held by Dreamview Plus.

        :returns: A Shapely Point in map coordinates and heading in radians.
                  Heading is None if the backend does not provide it.
        :rtype: Tuple[Point, Optional[float]]
        """
        info = self._request("GetStartPoint", {})
        if info.get("code") != 0:
            raise RuntimeError(
                f"GetStartPoint failed: {info.get('message', info)}"
            )

        pose = info.get("data")
        if not isinstance(pose, dict):
            raise RuntimeError("GetStartPoint returned invalid pose data")
        for field in ("x", "y", "heading"):
            if field == "heading" and field not in pose:
                continue
            value = pose.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise RuntimeError(f"GetStartPoint returned invalid {field}")

        heading = float(pose["heading"]) if "heading" in pose else None
        return Point(pose["x"], pose["y"]), heading

    def setup_mode(self) -> None:
        """
        Ask HMI to start all modules in the current mode, without ACK.

        Returning means the request was sent. Use get_modules() to observe
        module status; this does not wait for modules to become ready.
        """
        self._hmi_action("SETUP_MODE")

    def reset_mode(self) -> None:
        """
        Ask HMI to stop current mode's modules, without ACK.

        This sets the backend shutdown flag and requests module shutdown.
        Use get_modules() to observe status. Simulation reset is handled
        separately by reset().
        """
        self._hmi_action("RESET_MODE")

    def enter_auto_mode(self) -> None:
        """
        Request COMPLETE_AUTO_DRIVE, without a success acknowledgement.

        The backend first switches to manual mode, then requests auto mode.
        Returning does not establish that either transition succeeded.
        """
        self._hmi_action("ENTER_AUTO_MODE")

    def disengage(self) -> None:
        """
        Request COMPLETE_MANUAL, without a success acknowledgement.

        Returning means the request was sent, not that the chassis switched
        to manual mode. Perfect Control always publishes auto mode.
        """
        self._hmi_action("DISENGAGE")

    def reset(self) -> None:
        """
        Request world-cache and active simulation-model reset and check its ACK.

        Perfect Control clears its route header/trajectory but retains pose
        and does not explicitly zero speed/acceleration. Timers and ADS modules
        keep running. This does not send CLEAR_PLANNING to Planning. The world
        cache clears asynchronously, even if the model reset fails; success
        does not establish that the world update has completed. Clearing is
        applied on a simworld update, which requires a stream subscription.

        :raises RuntimeError: Reset fails or its response has no success code.
        :raises websocket.WebSocketException: Sending or receiving fails.
        """
        info = self._request("Reset", {})
        if info.get("code") != 0:
            raise RuntimeError(f"Reset failed: {info.get('message', info)}")

    def _hmi_action(self, action: str, value: Any = None):
        """
        Send an HMI action and wait only for actions in ``_ACK_ACTIONS``.

        Actions outside ``_ACK_ACTIONS`` return after sending, without checking
        whether the backend completed the action. For acknowledged actions,
        a zero or missing response code is accepted.

        :param str action: Backend HMI action name, such as ``CHANGE_MAP``.
        :param value: Optional JSON-serializable action value; omitted if None.
        :returns: None
        :raises RuntimeError: The response contains a nonzero error code.
        :raises websocket.WebSocketException: Sending or receiving fails,
            including a socket timeout.
        """
        info = {"action": action}
        if value is not None:
            info["value"] = value
        if action not in _ACK_ACTIONS:
            self._send("HMIAction", info)
            return
        response = self._request("HMIAction", info)
        code = response.get("code")
        if code not in (None, 0):
            raise RuntimeError(f"{action} failed: {response.get('message', response)}")

    def _get_hmi_status(self) -> Message:
        """
        Read one HMIStatus over a short-lived /hmi connection.

        /hmi is opened first, then one-shot broadcasts are triggered with
        dataFrequencyMs=0. The subscribe is resent every ``_HMI_RESUBSCRIBE_S``
        in case the backend had not registered the connection yet. Its reply on
        ``ws`` has no request ID and is discarded by a later ``_recv_response``.
        Frames pushed by the browser's stream are accepted too, as each one is
        a full snapshot.

        :returns: The HMIStatus message.
        :rtype: Message
        :raises RuntimeError: An HMIStatus frame cannot be decoded.
        :raises websocket.WebSocketTimeoutException: No HMIStatus frame arrives
            within ``timeout``.
        :raises websocket.WebSocketException: Connecting, sending or receiving fails.
        """
        hmi_ws: WebSocket = websocket.create_connection(
            f"ws://{self.host}:{self.port}/hmi", timeout=self.timeout)
        try:
            now = time.monotonic()
            deadline = now + self.timeout
            resend_at = now
            while now < deadline:
                if now >= resend_at:
                    self._send("subscribe", {"websocketName": "hmistatus",
                                             "dataFrequencyMs": 0})
                    resend_at = now + _HMI_RESUBSCRIBE_S
                hmi_ws.settimeout(min(resend_at, deadline) - now)
                try:
                    status = _decode_hmi_status(hmi_ws.recv())
                except websocket.WebSocketTimeoutException:
                    status = None
                if status is not None:
                    return status
                now = time.monotonic()
            raise websocket.WebSocketTimeoutException("No HMIStatus frame received on /hmi")
        finally:
            hmi_ws.close()

    def _status_name(self, field: str) -> str:
        """
        Read a name field of HMIStatus, such as ``current_map``.

        :param str field: The HMIStatus field name.
        :returns: The non-empty field value.
        :rtype: str
        :raises RuntimeError: The field is empty, i.e. nothing is selected.
        """
        value = getattr(self._get_hmi_status(), field)
        if not value.strip():
            raise RuntimeError(f"HMIStatus returned a missing or invalid {field}")
        return value

    def _request(self, msg_type: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request and wait for the response with its generated request ID.

        Use this only for message types whose backend handler sends a response.
        The returned payload is not checked for application-level errors.

        :param str msg_type: Backend message type, such as ``GetInitData``.
        :param Dict[str, Any] info: JSON-serializable request payload.
        :returns: The matching response's ``data.info``, or an empty dict if absent.
        :raises websocket.WebSocketException: Sending or receiving fails,
            including a socket timeout.
        """
        request_id = self._send(msg_type, info)
        return self._recv_response(request_id)

    def _send(self, msg_type: str, info: Dict[str, Any]) -> str:
        """
        Send a JSON message with a new UUID request ID without waiting for a reply.

        The envelope contains ``type`` and ``data``, with ``requestId`` and
        ``info`` nested inside ``data``.

        :param str msg_type: Backend message type, such as ``HMIAction``.
        :param dict info: JSON-serializable request payload.
        :returns: The generated request ID used to correlate a response.
        :rtype: str
        :raises TypeError: The payload contains a value that cannot be JSON encoded.
        :raises ValueError: The payload contains a circular reference.
        :raises websocket.WebSocketException: The message cannot be sent.
        """
        request_id = str(uuid.uuid4())
        payload = {
            "type": msg_type,
            "data": {"requestId": request_id, "info": info},
        }
        self.ws.send(json.dumps(payload))
        return request_id

    def _recv_response(self, request_id: str) -> Dict[str, Any]:
        """
        Read messages until ``data.requestId`` matches the requested ID.

        Messages that cannot be JSON decoded and responses with other request
        IDs are discarded, not buffered. This method must not be used by
        concurrent requests sharing the same connection. Each receive uses the
        socket timeout; there is no overall deadline across received messages.

        :param str request_id: Request ID returned by ``_send``.
        :returns: The matching response's ``data.info``, or an empty dict if absent.
        :raises websocket.WebSocketException: Receiving fails, including a
            socket timeout or connection closure.
        """
        while True:
            raw = self.ws.recv()
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            data = msg.get("data", {})
            if data.get("requestId") != request_id:
                continue
            return data.get("info", {})
