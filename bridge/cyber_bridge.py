"""
Artifact that publish-subscribe with the CyberRT framework via cyber_bridge.

Adapted from the CyberRT bridge client in DoppelTest:

    Y. Huai et al., "Doppelganger Test Generation for Revealing Bugs in
    Autonomous Driving Software," ICSE 2023.

We thank the authors for their contributions.

New Version NOTE: Compatible with version 1.0 from op (1) to (4),
                  but the publish() need to adjust to publish(channel, msg)
                  rather than old API publish(channel, msg.SerializeToString())
                  op(5) & (6) must re-compile the new cpp for the apollo in ctn
"""

import select
import socket
import time
from pathlib import Path
from math import isfinite
from logging import getLogger
from collections import defaultdict
from types import TracebackType
from typing import DefaultDict, Dict, List, Type, Callable, Optional, Tuple, Self
from google.protobuf.message import Message
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
    host: str
    """The host ip address"""
    port: int
    """The port of cyber_bridge, default to 9090"""
    timeout: float
    """The socket timeout max time"""
    descriptor_path: Path
    """The protobuf message descriptor path"""
    channels: Channels
    """The candidate channels"""
    services: Services
    """The candidate services"""
    conn: socket.socket
    """The socket connection"""
    subscribers: DefaultDict[str, List]
    """The registered subscribers and their callback functions"""
    publishable_channel: Dict[str, Type[Message]]
    """The registered channel names and their protobuf message classes"""

    MAX_FIELD_SIZE = 64 * 1024 * 1024

    def __init__(self, host: str, port=9090, timeout=5.0, *,
                 descriptor_path: str | Path) -> None:
        """
        Constructor

        :param str host: The host ip address
        :param int port: The port of cyber_bridge, default to 9090
        :param float timeout: The socket timeout max time
        :param str | Path descriptor_path: Descriptor bundle for registration and typed definitions
        """
        if timeout <= 0 or not isfinite(timeout):
            raise ValueError("timeout must be finite and positive")
        self.host: str = host
        self.port: int = port
        self.timeout: float = timeout
        self.descriptor_path: Path = DescriptorBundle(descriptor_path).descriptor_path
        self.channels = Channels(self.descriptor_path)
        self.services = Services(self.descriptor_path)
        # Create the TCP connection using socket
        self.conn = socket.create_connection((host, port), timeout=timeout)
        self.conn.setblocking(False)
        self.subscribers: DefaultDict[str, List] = defaultdict(list)
        self.publishable_channel: Dict[str, Type[Message]] = {}
        self._spinning = False
        self._receive_thread = None
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
        """
        Whether the connection is open; False after close or disconnect.
        
        :returns: The connected status
        :rtype: bool
        """
        return not self._closed

    @property
    def spinning(self) -> bool:
        """
        Whether the receive thread is dispatching; ``stop`` keeps the connection.
        
        :returns: The spinning status
        :rtype: bool
        """
        return self._spinning

    @staticmethod
    def __prepare_bytes(data: bytes) -> bytes:
        """
        Prefix a byte field with its length for the CyberBridge wire protocol.

        :param bytes data: The raw bytes to encode as a length-prefixed field.
        :returns: A four-byte unsigned little-endian length followed by the data.
        :rtype: bytes
        :raises OverflowError: If the data length exceeds the uint32 range.
        """
        return len(data).to_bytes(4, "little") + data

    def _send(self, data: bytes) -> None:
        """
        Send all bytes without interleaving concurrent callers' data.

        Retry partial writes until all data is sent or ``self.timeout`` expires.
        The timeout starts after acquiring the send lock. A socket error,
        invalid socket state or timeout closes the connection and fails pending
        service calls; the original exception is re-raised.

        :param bytes data: The complete encoded CyberBridge packet to send.
        :raises ConnectionError: If the connection is closed or a send returns zero.
        :raises TimeoutError: If sending does not finish within ``self.timeout``.
        :raises OSError: If waiting for or writing to the socket fails.
        :raises ValueError: If the socket is invalid for ``select``.
        """
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

    def add_subscriber(self, channel: Channel, cb: Callable[[Message], None]) -> None:
        """
        Register a local callback and request a reader for the channel.

        Incoming bytes are parsed with ``channel.msg_cls`` before calling
        ``cb(message)`` on the receive thread. Multiple callbacks may be
        registered for the same channel. Call ``spin()`` to start reception.
        Keep callbacks short and call synchronous services from another thread.

        Op 2 has no acknowledgement: returning confirms transmission of the
        reader request, not server-side reader creation. If sending fails,
        this callback registration is removed and the exception is re-raised.

        :param Channel channel: The channel name and protobuf message class.
        :param Callable[[Message], None] cb: A callback accepting one decoded protobuf message.
        :raises TypeError: If ``channel`` is not ``Channel``.
        :raises ConnectionError: If the connection is closed or breaks during send.
        :raises TimeoutError: If sending exceeds the connection timeout.
        :raises OSError: If waiting for or writing to the socket fails.
        :raises ValueError: If the socket is invalid for ``select``.
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be a Channel instance")
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

    def add_publisher(self, channel: Channel) -> None:
        """
        Request a writer for the channel, registering descriptors if needed.

        Send the selected descriptor bundle before this connection's first
        writer request. Later writer requests reuse that registration. A
        channel name already registered with the same message class is skipped,
        including when multiple threads request it concurrently. Registering
        the same name with a different message class is rejected. Call this
        method before publishing messages on the channel.

        Op 1 and op 3 have no acknowledgement. Local state records successful
        transmission, not server acknowledgement of writer creation. If
        sending fails, the channel is not added to the local publisher registry.

        :param Channel channel: The channel name and protobuf message class.
        :raises TypeError: If ``channel`` is not ``Channel``.
        :raises ConnectionError: If the connection is closed or breaks during send.
        :raises TimeoutError: If sending exceeds the connection timeout.
        :raises OSError: If waiting for or writing to the socket fails.
        :raises ValueError: If the channel name is already registered with a
            different message class, or the socket is invalid for ``select``.
        :raises OverflowError: If a field length or descriptor count exceeds uint32.
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be a Channel instance")
        with self._publisher_lock:
            if channel.name in self.publishable_channel:
                if self.publishable_channel[channel.name] is not channel.msg_cls:
                    raise ValueError(f"Publisher already registered with a different message class: {channel.name!r}")
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
            self.publishable_channel[channel.name] = channel.msg_cls

    def publish(self, channel: Channel, msg: Message) -> None:
        """
        Serialize and publish a protobuf message on a registered channel.

        Call ``add_publisher(channel)`` first. The channel's message class must
        match its registered class, and ``msg`` must be an instance of that
        class from the same schema. Serialization is handled internally.

        Op 4 has no acknowledgement. Returning confirms transmission to the
        socket, not delivery to Cyber readers.

        :param Channel channel: The channel on which to publish the message.
        :param Message msg: An instance of the registered protobuf message class.
        :raises TypeError: If ``channel`` is not a ``Channel`` or ``msg`` is not
            an instance of the registered message class.
        :raises RuntimeError: If no publisher is registered for the channel.
        :raises ConnectionError: If the connection is closed or breaks during send.
        :raises TimeoutError: If sending exceeds the connection timeout.
        :raises OSError: If waiting for or writing to the socket fails.
        :raises ValueError: If the channel's message class differs from the
            registered class, or the socket is invalid for ``select``.
        :raises OverflowError: If a field length exceeds the uint32 range.
        :raises google.protobuf.message.EncodeError: If the message cannot be
            serialized, such as when required fields are missing.
        """
        if not isinstance(channel, Channel):
            raise TypeError("channel must be a Channel instance")
        with self._publisher_lock:
            if channel.name not in self.publishable_channel:
                raise RuntimeError(f"Publisher is not registered for channel {channel.name!r}")
            msg_cls = self.publishable_channel[channel.name]
            if channel.msg_cls is not msg_cls:
                raise ValueError(f"Channel message class differs from registered publisher: {channel.name!r}")
            if not isinstance(msg, msg_cls):
                raise TypeError("msg must be an instance of the registered message class")
            self._send(BridgeOp.Publish
                       + self.__prepare_bytes(to_bytes(channel.name))
                       + self.__prepare_bytes(msg.SerializeToString()))

    def call_service(self, service: Service, request: Message,
                     timeout: float = 5.0) -> Message:
        """
        Call an Apollo Cyber service and wait for its matching response.

        Reception starts automatically. Concurrent calls are matched to their
        responses by request ID. This method must be called outside the receive
        thread so that subscriber callbacks do not block response processing.

        Invalid usage: calling a service synchronously from a subscriber
        callback on the same bridge raises ``RuntimeError``::

            service = bridge.services.QueryStatus

            def on_localization(msg):
                request = service.request_cls(command_id=123)
                bridge.call_service(service, request)  # Raises RuntimeError.

            bridge.add_subscriber(bridge.channels.Localization, on_localization)
            bridge.spin()

        The callback runs on the receive thread. Waiting for a service response
        there would prevent that same thread from processing the response.
        Instead, have the callback enqueue work for the main thread or another
        worker thread, and call ``call_service()`` from that thread.

        The ``Service`` defines the endpoint, request class and response class.
        The request must be an instance of that class from the same schema.
        Serialization and response parsing are handled internally.

        ``timeout`` limits response waiting after sending; writes use
        ``self.timeout``. A response timeout leaves the connection open and
        does not cancel work already submitted to Apollo. Late responses to
        timed-out requests are discarded.

        :param Service service: The endpoint and its protobuf message classes.
        :param Message request: An instance of ``service.request_cls``.
        :param float timeout: A finite, positive response timeout in seconds,
            defaulting to 5.0.
        :returns: The response decoded as an instance of ``service.response_cls``.
        :rtype: Message
        :raises TypeError: If ``service`` is not a ``Service`` or the request
            is not an instance of its declared request class.
        :raises ValueError: If the timeout or socket is invalid.
        :raises RuntimeError: If called from the receive thread or while the
            previous receive thread is still stopping.
        :raises ConnectionError: If the connection is closed, reception stops,
            or the peer disconnects before the response arrives.
        :raises TimeoutError: If sending or waiting for the response times out.
        :raises ServiceCallError: If the bridge returns op 6 with ``ok=0``.
        :raises OSError: If waiting for or writing to the socket fails.
        :raises OverflowError: If request IDs are exhausted or a field length
            exceeds the uint32 range.
        :raises google.protobuf.message.EncodeError: If the request cannot be
            serialized, such as when required fields are missing.
        :raises google.protobuf.message.DecodeError: If the response cannot be
            parsed by the selected response class.
        """
        if timeout <= 0 or not isfinite(timeout):
            raise ValueError("timeout must be finite and positive")
        if not isinstance(service, Service):
            raise TypeError("service must be a Service definition")
        if not isinstance(request, service.request_cls):
            raise TypeError("request must be an instance of service.request_cls")
        if current_thread() is self._receive_thread:
            raise RuntimeError("call_service cannot block the subscriber thread")
        fields = (self.__prepare_bytes(to_bytes(service.name))
                  + self.__prepare_bytes(to_bytes(service.request_cls.DESCRIPTOR.full_name))
                  + self.__prepare_bytes(request.SerializeToString()))
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
                raise TimeoutError(f"CyberBridge service {service.name!r} timed out") from exc
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)
        message = service.response_cls()
        message.ParseFromString(response)
        return message

    def _read_field(self, offset: int) -> Optional[Tuple[bytes, int]]:
        """
        Read a length-prefixed byte field from the receive buffer.

        The field starts with a four-byte unsigned little-endian length.
        Return None if the length prefix or payload is incomplete, allowing
        the caller to wait for more TCP data. The buffer is not modified.

        :param int offset: The nonnegative byte offset of the length prefix
            within ``self._buffer``.
        :returns: A tuple of the payload bytes and the offset immediately after
            the field, or None if the complete field has not arrived.
        :rtype: Optional[Tuple[bytes, int]]
        :raises ValueError: If the declared payload length exceeds
            ``MAX_FIELD_SIZE``.
        """
        if len(self._buffer) < offset + 4:
            return None
        size = int.from_bytes(self._buffer[offset:offset + 4], "little")
        if size > self.MAX_FIELD_SIZE:
            raise ValueError(f"CyberBridge field exceeds {self.MAX_FIELD_SIZE} bytes")
        end = offset + 4 + size
        if len(self._buffer) < end:
            return None
        return bytes(self._buffer[offset + 4:end]), end

    def on_read(self, data: bytes) -> None:
        """
        Buffer a TCP chunk and dispatch every complete frame in arrival order.

        A chunk may contain part of a frame or multiple frames. Incomplete
        frames remain in ``self._buffer`` until more data arrives; complete
        frames are removed before dispatch.

        Op 4 frames are passed to ``receive_publish()`` for channel-based
        callback dispatch. Op 6 frames resolve the pending service call with
        the matching request ID, supplying response bytes on success or a
        ``ServiceCallError`` on bridge failure. Responses with no matching
        pending call, including late responses after a timeout, are discarded.

        Normally called by the receive thread. Calls must not overlap because
        access to the receive buffer is not protected by a lock. Protocol
        errors propagate to the receive loop, which terminates the connection.

        :param bytes data: The next raw chunk read from the TCP connection.
        :raises ValueError: If an operation is unsupported, a service response
            status is not 0 or 1, or a field exceeds ``MAX_FIELD_SIZE``.
        :raises UnicodeDecodeError: If a published channel name is not ASCII.
        """
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

    def receive_publish(self, data: bytes) -> None:
        """
        Dispatch one complete publish frame to the channel's subscribers.

        Extract the ASCII channel name and serialized payload from an op 4
        frame. Pass the payload to a snapshot of the registered callbacks in
        registration order. Wrappers created by ``add_subscriber()`` decode
        the payload with the channel's message class before calling user code.

        Callbacks run synchronously on the calling thread, normally the receive
        thread. Payload decoding and callback exceptions are logged separately
        for each callback so remaining subscribers can still run. Callback
        return values are ignored.

        If reception is stopped on entry, or the channel has no subscribers,
        no callbacks are invoked. This method assumes ``on_read()`` has already
        assembled a complete frame and checked its field lengths.

        :param bytes data: A complete frame containing op 4, a length-prefixed
            channel name, and a length-prefixed serialized message payload.
        :raises UnicodeDecodeError: If the channel name is not ASCII.
        """
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

    def _fail_pending(self, error: Exception) -> None:
        """
        Fail all pending service calls and clear their request records.

        Set the supplied exception on each pending Future so calls waiting in
        ``future.result()`` wake and raise it. The exception is delivered to
        those waiting threads rather than raised directly by this method.
        The state lock protects the operation against concurrent request
        registration and response handling. An empty registry requires no action.

        Used when reception stops or the connection terminates, so callers do
        not have to wait for their individual response timeouts.

        :param Exception error: The exception to deliver to every pending call.
        """
        with self._state_lock:
            for future in self._pending.values():
                future.set_exception(error)
            self._pending.clear()

    def _terminate(self, error: Exception) -> None:
        """
        Mark the connection closed, fail pending calls, and close the socket.

        Update the closed and spinning flags under the state lock, then deliver
        ``error`` to all pending service calls through ``_fail_pending()``.
        Shut down both socket directions and release the socket. Shutdown
        errors are ignored so socket closure is still attempted.

        This operation is final; further communication requires a new bridge
        instance. It requests the receive loop to exit but does not join its
        thread, so it can also be called by the receive thread itself.

        :param Exception error: The exception delivered to pending service calls.
        :raises OSError: If closing the socket fails. Errors from ``shutdown()``
            are suppressed.
        """
        with self._state_lock:
            self._closed = True
            self._spinning = False
            self._fail_pending(error)
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()

    def _spin(self) -> None:
        """
        Receive and dispatch TCP data while ``self._spinning`` is true.

        Used as the background thread target by ``spin()``. Wait for socket
        readability with a 0.1-second maximum idle wait, then read up to 65,536
        bytes and pass them to ``on_read()``. Data arrival wakes the wait early;
        this is not a fixed 10 Hz processing loop. Retry ``BlockingIOError``
        because another read attempt may be needed on the nonblocking socket.

        An empty read indicates peer disconnection. Exceptions from waiting,
        reading or frame processing terminate the connection and fail pending
        service calls with a ``ConnectionError`` containing the failure reason.
        Exceptions caught here are handled internally rather than re-raised.

        :raises OSError: If socket closure inside ``_terminate()`` fails.
        """
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

    def spin(self) -> None:
        """
        Start a background daemon thread to receive and dispatch bridge messages.

        Return immediately after starting the thread. Repeated calls while
        reception is active have no effect. The state lock ensures concurrent
        callers do not start multiple receive threads.

        Call after registering subscribers to begin receiving their messages.
        ``call_service()`` also starts reception automatically. After ``stop()``
        completes, this method can resume reception on the same connection.
        A closed or disconnected bridge requires a new instance.

        :raises ConnectionError: If the connection is closed.
        :raises RuntimeError: If the previous receive thread is still stopping
            or a new thread cannot be started.
        """
        with self._state_lock:
            if self._closed:
                raise ConnectionError("CyberBridge connection is closed")
            if self._spinning:
                return
            if self._receive_thread is not None and self._receive_thread.is_alive():
                raise RuntimeError("CyberBridge receive thread is still stopping")
            self._spinning = True
            self._receive_thread = Thread(target=self._spin, name="CyberBridge", daemon=True)
            self._receive_thread.start()

    def stop(self) -> None:
        """
        Pause reception and fail pending service calls while keeping TCP open.

        Clear the spinning flag and deliver a ``ConnectionError`` with the
        message "CyberBridge reception stopped" to every pending service call.
        Existing subscriber callbacks and publisher registrations are retained.
        Use ``spin()`` to resume reception on the same connection;
        ``call_service()`` also resumes reception automatically.

        When called outside the receive thread, wait for that thread to exit.
        A callback already executing is not interrupted, so a blocked callback
        can delay this method indefinitely. When called from a subscriber
        callback, skip joining the current thread; the receive loop exits after
        its current processing completes.

        Safe to call before ``spin()`` or repeatedly. Pending service calls fail
        locally; work already submitted to Apollo is not cancelled.
        """
        with self._state_lock:
            self._spinning = False
            thread = self._receive_thread
            self._fail_pending(ConnectionError("CyberBridge reception stopped"))
        if thread is not None and thread is not current_thread():
            thread.join()

    def close(self) -> None:
        """
        Permanently close the connection and stop reception.

        Fail pending service calls with a ``ConnectionError`` carrying the
        message "CyberBridge connection closed", shut down and close the socket,
        then wait for the receive thread to exit. When called from a subscriber
        callback, skip joining the current thread; it exits after its current
        processing completes. An executing callback is not interrupted.

        Unlike ``stop()``, this method releases the connection. Further
        communication requires a new bridge instance and registration of its
        readers and writers. Calling ``close()`` again is safe after a successful
        close. This method is also called when leaving a ``with`` block.

        :raises OSError: If closing the socket fails. Errors from ``shutdown()``
            are suppressed by ``_terminate()``.
        """
        self._terminate(ConnectionError("CyberBridge connection closed"))
        self.stop()

    def reconnect(self) -> None:
        """
        Close the current TCP connection and open a new one to the same server.

        Stop reception, close the socket (failing pending service calls), then
        connect again to ``host:port``. All subscriber callbacks, publisher
        registrations and the descriptor-registration flag are cleared, because
        the cyber_bridge server drops readers and writers with the old
        connection. The instance stays usable and keeps its identity, so
        holders of this object do not need a new reference.

        After reconnecting, callers must add their publishers and subscribers
        again and call ``spin()`` to resume reception. The cyber_bridge server
        process itself is not affected.

        :raises OSError: If the new connection cannot be opened.
        """
        if not self._closed:
            self.close()
        else:
            self.stop()
        conn = socket.create_connection((self.host, self.port), timeout=self.timeout)
        conn.setblocking(False)
        with self._state_lock, self._publisher_lock:
            self.conn = conn
            self.subscribers = defaultdict(list)
            self.publishable_channel = {}
            self._descriptors_registered = False
            self._buffer = bytearray()
            self._receive_thread = None
            self._spinning = False
            self._closed = False

    def __enter__(self) -> Self:
        """
        Return this bridge as the context managed by a ``with`` block.

        Keep one block open for the whole scenario, registering publishers and
        subscribers before using them. For example::

            with CyberBridge(host, descriptor_path=desc) as bridge:
                bridge.add_publisher(channel)
                bridge.add_subscriber(channel, callback)
                bridge.spin()
                # Publish messages and perform scenario operations here.

        Leaving the block calls ``close()``, including when an exception occurs.
        The instance cannot be reused afterward. Further communication requires
        a new bridge instance and re-registering every publisher and subscriber;
        registrations are not inherited by the new connection.

        :returns: This bridge instance.
        :rtype: CyberBridge
        """
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """
        Close the bridge when leaving the ``with`` block.

        Call ``close()`` on both normal and exceptional exits, closing the TCP
        connection and failing pending service calls. To communicate again,
        create a new bridge and register its publishers and subscribers again.
        An exception raised inside the block propagates after cleanup.

        :param Optional[Type[BaseException]] exc_type: Exception type, or None.
        :param Optional[BaseException] exc_value: Exception instance, or None.
        :param Optional[TracebackType] traceback: Exception traceback, or None.
        :raises OSError: If closing the socket fails.
        """
        self.close()
