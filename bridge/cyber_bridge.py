"""
Channel that communicates with the CyberRT framework via cyber_bridge.

Adapted from the CyberRT bridge client in DoppelTest:

    Y. Huai et al., "Doppelganger Test Generation for Revealing Bugs in
    Autonomous Driving Software," ICSE 2023.

We thank the authors for their contributions.
"""

import select
import socket
import time
from pathlib import Path
from math import isfinite
from logging import getLogger
from collections import defaultdict
from typing import Any, DefaultDict, List, Set, Union
from threading import Lock, RLock, Thread, current_thread
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from .channel import Channel, Channels
from .operation import BridgeOp
from .service import Service, ServiceCallError, Services

if "." in __package__:
    from ..message.descriptor import DescriptorBundle
else:
    from message.descriptor import DescriptorBundle


logger = getLogger(__name__)


def to_bytes(s: str) -> bytes:
    """
    Converts string to bytes using ascii

    :param str s: string to be converted
    :returns: bytes in ascii
    :rtype: bytes
    """
    return bytes(s, "ascii")


class CyberBridge:
    """
    TCP client for Apollo's publish/subscribe and service bridge.

    Keep one instance for a whole scenario. ``call_service`` starts reception
    automatically. Subscriber callbacks run on the receive thread; call
    synchronous services from another thread instead. ``stop`` pauses
    reception; ``close`` is final, and a new instance must register its readers
    and writers again.
    """

    MAX_FIELD_SIZE = 64 * 1024 * 1024

    def __init__(self, host: str, port=9090, timeout=5.0, *,
                 descriptor_path: Union[str, Path]) -> None:
        """
        Constructor

        :param str host: the host ip address
        :param int port: the port
        :param descriptor_path: Descriptor bundle for registration and typed definitions.
        :param time
        """
        if timeout <= 0 or not isfinite(timeout):
            raise ValueError("timeout must be finite and positive")
        self.descriptor_path = DescriptorBundle(descriptor_path).descriptor_path
        self.channels = Channels(self.descriptor_path)
        self.services = Services(self.descriptor_path)
        self.conn = socket.create_connection((host, port), timeout=timeout)
        self.conn.setblocking(False)
        self.timeout = timeout
        self.subscribers: DefaultDict[str, List] = defaultdict(list)
        self.publishable_channel: Set[str] = set()
        self._spinning = False
        self.t = None
        self._closed = False
        self._buffer = bytearray()
        self._send_lock = Lock()
        self._publisher_lock = Lock()
        self._descriptors_registered = False
        self._state_lock = RLock()
        self._pending = {}
        self._next_request_id = 1

    @property
    def connected(self) -> bool:
        """Whether the connection is open; False after close or disconnect."""
        return not self._closed

    @property
    def spinning(self) -> bool:
        """Whether the receive thread is dispatching; ``stop`` keeps the connection."""
        return self._spinning

    @staticmethod
    def __prepare_bytes(data: bytes) -> bytes:
        return len(data).to_bytes(4, "little") + data

    def _send(self, data: bytes):
        # A frame may require several writes. Keep concurrent callers' frames
        # together, and never continue a connection after an incomplete write.
        with self._send_lock:
            if self._closed:
                raise ConnectionError("CyberBridge connection is closed")
            deadline = time.monotonic() + self.timeout
            remaining = memoryview(data)
            try:
                while remaining:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        raise TimeoutError("Timed out sending to CyberBridge")
                    _, writable, _ = select.select([], [self.conn], [], wait)
                    if not writable:
                        raise TimeoutError("Timed out sending to CyberBridge")
                    try:
                        sent = self.conn.send(remaining)
                    except BlockingIOError:
                        continue
                    if sent == 0:
                        raise ConnectionError("CyberBridge connection closed during send")
                    remaining = remaining[sent:]
            except (OSError, ValueError) as exc:
                self._terminate(ConnectionError(f"CyberBridge send failed: {exc}"))
                raise

    def add_subscriber(self, channel: Channel, cb):
        def cb_wrapper(data):
            parsed_msg = channel.msg_cls()
            parsed_msg.ParseFromString(data)
            cb(parsed_msg)

        with self._state_lock:
            self.subscribers[channel.name].append(cb_wrapper)
        try:
            self._send(BridgeOp.AddReader
                       + self.__prepare_bytes(to_bytes(channel.name))
                       + self.__prepare_bytes(to_bytes(channel.msg_cls.DESCRIPTOR.full_name)))
        except Exception:
            with self._state_lock:
                self.subscribers[channel.name].remove(cb_wrapper)
            raise

    def add_publisher(self, channel: Channel):
        """Send the selected descriptors before this connection's first writer.

        Op 1 and op 3 have no acknowledgement. Local state records successful
        transmission, not server acknowledgement of writer creation.
        """
        with self._publisher_lock:
            if channel.name in self.publishable_channel:
                return
            data = b""
            if not self._descriptors_registered:
                descriptors = DescriptorBundle(self.descriptor_path).serialized_file_descriptors()
                # [1][uint32 count][length + FileDescriptorProto] ...
                data = (BridgeOp.RegisterDesc
                        + len(descriptors).to_bytes(4, "little")
                        + b"".join(self.__prepare_bytes(d) for d in descriptors))
            data += (BridgeOp.AddWriter
                     + self.__prepare_bytes(to_bytes(channel.name))
                     + self.__prepare_bytes(to_bytes(channel.msg_cls.DESCRIPTOR.full_name)))
            self._send(data)
            self._descriptors_registered = True
            self.publishable_channel.add(channel.name)

    def publish(self, channel: Channel, data: bytes):
        """Publish serialized protobuf bytes after registering the publisher."""
        if not isinstance(data, bytes):
            raise TypeError("data must be serialized protobuf bytes")
        self._send(BridgeOp.Publish
                   + self.__prepare_bytes(to_bytes(channel.name))
                   + self.__prepare_bytes(data))

    def call_service(self, service: Union[str, Service], request: Any,
                     response_cls=None, timeout=5.0, *, request_type=None):
        """Call an Apollo Cyber service and return its protobuf response.

        ``service`` is a ``Service`` (e.g. ``bridge.services.LaneFollow``), which
        supplies the name, request type and response class, or a service name.
        ``request`` may be a protobuf message, or bytes accompanied by the full
        protobuf ``request_type`` name. Omit ``response_cls`` with a name to
        return bytes. Timeout raises ``TimeoutError``; op 6 with ok=0 raises
        ``ServiceCallError``; disconnect/stop raises ``ConnectionError``.
        A local timeout does not cancel work already submitted to Apollo.
        ``timeout`` limits response waiting; writes use the connection timeout.
        """
        if timeout <= 0 or not isfinite(timeout):
            raise ValueError("timeout must be finite and positive")
        if isinstance(service, Service):
            if request_type is None:
                request_type = service.request_cls.DESCRIPTOR.full_name
            if response_cls is None:
                response_cls = service.response_cls
            service = service.name
        if current_thread() is self.t:
            raise RuntimeError("call_service cannot block the subscriber thread")
        if isinstance(request, bytes):
            if not request_type:
                raise ValueError("request_type is required for serialized requests")
            payload = request
        else:
            actual_type = request.DESCRIPTOR.full_name
            if request_type is not None and request_type != actual_type:
                raise ValueError("request_type does not match the protobuf request")
            request_type = actual_type
            payload = request.SerializeToString()
        fields = (self.__prepare_bytes(to_bytes(service))
                  + self.__prepare_bytes(to_bytes(request_type))
                  + self.__prepare_bytes(payload))
        self.spin()
        future = Future()
        with self._state_lock:
            if self._closed or not self._spinning:
                raise ConnectionError("CyberBridge is not receiving")
            if self._next_request_id > 0xffffffff:
                raise OverflowError("Request IDs exhausted; create a new connection")
            request_id = self._next_request_id
            self._next_request_id += 1
            self._pending[request_id] = future
        try:
            # [5][uint32 request id][service][request type][request bytes]
            self._send(BridgeOp.CallService + request_id.to_bytes(4, "little") + fields)
            try:
                response = future.result(timeout=timeout)
            except FutureTimeoutError as exc:
                raise TimeoutError(f"CyberBridge service {service!r} timed out") from exc
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)
        if response_cls is None:
            return response
        message = response_cls()
        message.ParseFromString(response)
        return message

    def _read_field(self, offset):
        if len(self._buffer) < offset + 4:
            return None
        size = int.from_bytes(self._buffer[offset:offset + 4], "little")
        if size > self.MAX_FIELD_SIZE:
            raise ValueError(f"CyberBridge field exceeds {self.MAX_FIELD_SIZE} bytes")
        end = offset + 4 + size
        if len(self._buffer) < end:
            return None
        return bytes(self._buffer[offset + 4:end]), end

    def on_read(self, data: bytes):
        """Consume a TCP chunk, retaining partial frames and dispatching all frames."""
        self._buffer.extend(data)
        while self._buffer:
            op = self._buffer[0]
            if op == 4:
                field = self._read_field(1)
                if field is None:
                    return
                _, offset = field
                field = self._read_field(offset)
                if field is None:
                    return
                _, end = field
                frame = bytes(self._buffer[:end])
                del self._buffer[:end]
                self.receive_publish(frame)
            elif op == 6:
                if len(self._buffer) < 6:
                    return
                request_id = int.from_bytes(self._buffer[1:5], "little")
                ok = self._buffer[5]
                if ok not in (0, 1):
                    raise ValueError(f"Invalid service response status: {ok}")
                field = self._read_field(6)
                if field is None:
                    return
                response, end = field
                del self._buffer[:end]
                with self._state_lock:
                    future = self._pending.pop(request_id, None)
                    if future is not None:
                        if ok:
                            future.set_result(response)
                        else:
                            future.set_exception(ServiceCallError(
                                f"CyberBridge service request {request_id} failed"))
                # Late responses to timed-out calls have no pending request.
            else:
                raise ValueError(f"Unknown CyberBridge operation: {op}")

    def receive_publish(self, data: bytes):
        """Dispatch one complete op-4 frame (normally called by on_read)."""
        if not self._spinning:
            return
        topic_length = int.from_bytes(data[1:5], "little")
        offset = 5 + topic_length
        topic = data[5:offset].decode("ascii")
        size = int.from_bytes(data[offset:offset + 4], "little")
        payload = data[offset + 4:offset + 4 + size]
        with self._state_lock:
            subscribers = tuple(self.subscribers.get(topic, ()))
        for subscriber in subscribers:
            try:
                subscriber(payload)
            except Exception:
                logger.exception("CyberBridge subscriber failed on %s", topic)

    def _fail_pending(self, error):
        with self._state_lock:
            for future in self._pending.values():
                future.set_exception(error)
            self._pending.clear()

    def _terminate(self, error):
        with self._state_lock:
            self._closed = True
            self._spinning = False
            self._fail_pending(error)
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()

    def _spin(self):
        while self._spinning:
            try:
                readable, _, _ = select.select([self.conn], [], [], 0.1)
                if not readable:
                    continue
                try:
                    data = self.conn.recv(65536)
                except BlockingIOError:
                    continue
                if not data:
                    raise ConnectionError("CyberBridge peer disconnected")
                self.on_read(data)
            except Exception as exc:
                self._terminate(ConnectionError(f"CyberBridge receive failed: {exc}"))
                return

    def spin(self):
        with self._state_lock:
            if self._closed:
                raise ConnectionError("CyberBridge connection is closed")
            if self._spinning:
                return
            if self.t is not None and self.t.is_alive():
                raise RuntimeError("CyberBridge receive thread is still stopping")
            self._spinning = True
            self.t = Thread(target=self._spin, name="CyberBridge", daemon=True)
            self.t.start()

    def stop(self):
        """Pause reception and fail pending calls; safe before spin or in a callback."""
        with self._state_lock:
            self._spinning = False
            thread = self.t
            self._fail_pending(ConnectionError("CyberBridge reception stopped"))
        if thread is not None and thread is not current_thread():
            thread.join()

    def close(self):
        self._terminate(ConnectionError("CyberBridge connection closed"))
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
