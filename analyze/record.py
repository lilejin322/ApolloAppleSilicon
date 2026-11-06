"""
Read Apollo Cyber record files in plain Python, without Cyber.

A record is a sequence of sections. Each starts with 16 bytes, the
``SectionType`` as an int32, 4 bytes of padding and the payload size as an
int64, little-endian, followed by that many bytes of one message of
``cyber/proto/record.proto``:

- the file opens with a ``HEADER`` section whose message is padded to 2048 bytes;
- ``CHANNEL`` sections give each channel's name and message type;
- ``CHUNK_HEADER`` / ``CHUNK_BODY`` pairs carry the messages, each a
  ``SingleMessage``: channel name, receive time in nanoseconds, serialized content;
- an ``INDEX`` section closes a complete record.

The container format is decoded with the bundled ``cyber_record.desc``,
compiled from Apollo's ``record.proto``. The message contents are decoded with
the caller's descriptor, which must be the schema of the Apollo that recorded
the file.
"""

import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Collection, Dict, Iterator, Set, Tuple, Type
from google.protobuf.message import DecodeError, Message

from ..message.descriptor import DescriptorBundle

__all__ = ["RecordError", "RecordMessage", "RecordReader"]

RECORD_DESCRIPTOR = Path(__file__).resolve().parents[1] / "message" / "cyber_record.desc"
"""``cyber/proto/record.proto`` compiled into a descriptor set; the format is the same in every Apollo release."""

_SECTION = struct.Struct("<iiq")
"""``struct Section { SectionType type; int64_t size; }``: int32, padding to 8 bytes, int64."""
_HEADER_LENGTH = 2048
"""Bytes reserved for the header message after the first section."""
_HEADER, _CHUNK_HEADER, _CHUNK_BODY, _INDEX, _CHANNEL = range(5)
"""``SectionType`` values."""


class RecordError(ValueError):
    """The file is not a record this reader can decode."""


@dataclass(frozen=True)
class RecordMessage:
    """
    One recorded message.

    :param str channel: Its channel, e.g. ``/apollo/localization/pose``.
    :param int time: When the recorder received it, in nanoseconds since the epoch.
    :param Message message: The decoded message.
    """
    channel: str
    """Its channel, e.g. ``/apollo/localization/pose``."""
    time: int
    """When the recorder received it, in nanoseconds since the epoch."""
    message: Message
    """The decoded message."""


class RecordReader:
    """
    One record file and the channels it holds.

    :param str | Path path: The record file, e.g. ``ApolloManager.record_file``.
    :param str | Path descriptor_path: The protobuf descriptor of the Apollo that recorded it.
    :raises RecordError: The file does not start with a record header, or is compressed.
    """
    path: Path
    """The record file."""
    header: Message
    """The record's ``apollo.cyber.proto.Header``; ``is_complete`` is False when recording did not finish."""
    channels: Dict[str, str]
    """Message type of every channel, by channel name."""
    _bundle: DescriptorBundle
    """The descriptor of the Apollo that recorded the file: decodes the messages' contents."""
    _header_cls: Type[Message]
    """``apollo.cyber.proto.Header``, from the bundled record descriptor."""
    _channel_cls: Type[Message]
    """``apollo.cyber.proto.Channel``, from the bundled record descriptor."""
    _body_cls: Type[Message]
    """``apollo.cyber.proto.ChunkBody``, from the bundled record descriptor."""

    def __init__(self, path: str | Path, *, descriptor_path: str | Path) -> None:
        """
        Read the header and channel table, rejecting unsupported compression or an invalid header.

        :param str | Path path: The record file, e.g. ``ApolloManager.record_file``.
        :param str | Path descriptor_path: The protobuf descriptor of the Apollo that recorded it.
        :raises RecordError: The file does not start with a record header, or is compressed.
        :raises FileNotFoundError: If the record or the descriptor does not exist.
        """
        self.path = Path(path)
        self._bundle = DescriptorBundle(descriptor_path)
        record_format = DescriptorBundle(RECORD_DESCRIPTOR)
        self._header_cls = record_format.message("apollo.cyber.proto.Header")
        self._channel_cls = record_format.message("apollo.cyber.proto.Channel")
        self._body_cls = record_format.message("apollo.cyber.proto.ChunkBody")
        with self.path.open("rb") as stream:
            kind, size = self._section(stream)
            if kind != _HEADER or not 0 < size <= _HEADER_LENGTH:
                raise RecordError(f"{self.path} does not start with a record header")
            self.header = self._header_cls.FromString(stream.read(size))
            if self.header.compress:
                raise RecordError(f"{self.path} is compressed, which this reader does not support")
            self.channels = {}
            for kind, payload in self._sections(stream, wanted={_CHANNEL}):
                channel = self._channel_cls.FromString(payload)
                self.channels[channel.name] = channel.message_type

    def validate(self, *, required_channels: Collection[str] = ()) -> None:
        """
        Require a complete, structurally intact record before formal scoring.

        Ordinary message iteration remains available for partial-file inspection.

        :param Collection[str] required_channels: Channels that must each hold at least one message.
        :raises RecordError: If recording did not finish, a section is cut off or cannot be decoded, the index is
            missing, the message or channel count differs from the header's, or the record holds no message at
            all or none on a required channel.
        """
        if not self.header.is_complete:
            raise RecordError(f"{self.path}: recording is incomplete")
        counts = Counter()
        indexed = False
        try:
            with self.path.open("rb") as stream:
                stream.seek(0, 2)
                end = stream.tell()
                stream.seek(_SECTION.size + _HEADER_LENGTH)
                while stream.tell() < end:
                    kind, size = self._section(stream)
                    if stream.tell() + size > end:
                        raise RecordError(f"{self.path}: truncated section")
                    if kind == _CHUNK_BODY:
                        body = self._body_cls.FromString(stream.read(size))
                        counts.update(m.channel_name for m in body.messages)
                    elif kind == _INDEX:
                        indexed = True
                        stream.seek(size, 1)
                    else:
                        stream.seek(size, 1)
        except (EOFError, DecodeError) as error:
            raise RecordError(f"{self.path}: truncated or invalid record") from error
        if not indexed or sum(counts.values()) != self.header.message_number:
            raise RecordError(f"{self.path}: missing index or message count mismatch")
        if len(self.channels) != self.header.channel_number:
            raise RecordError(f"{self.path}: channel count mismatch")
        if not counts or any(not counts[c] for c in required_channels):
            raise RecordError(f"{self.path}: no messages on required channels")

    def messages(self, channels: Collection[str] | None = None) -> Iterator[RecordMessage]:
        """
        Decode the messages of ``channels``, in file order (chunks in recording order).

        :param Collection[str] | None channels: Channel names; None decodes every channel whose message type the
            descriptor knows.
        :returns: The messages; other channels are skipped without decoding.
        :rtype: Iterator[RecordMessage]
        :raises RecordError: A requested channel's message type is not in the descriptor.
        """
        classes = self._classes(channels)
        with self.path.open("rb") as stream:
            self._section(stream)
            for _, payload in self._sections(stream, wanted={_CHUNK_BODY}):
                for single in self._body_cls.FromString(payload).messages:
                    cls = classes.get(single.channel_name)
                    if cls is not None:
                        yield RecordMessage(single.channel_name, single.time, cls.FromString(single.content))

    def _classes(self, channels: Collection[str] | None) -> Dict[str, Type[Message]]:
        """
        Resolve selected channel types; reject unknown explicit selections and skip unknown implicit ones.

        :param Collection[str] | None channels: Channel names; None selects every channel the descriptor can decode.
        :returns: The message class of each selected channel the record holds, by channel name.
        :rtype: Dict[str, Type[Message]]
        :raises RecordError: A channel named in ``channels`` has a message type the descriptor does not define.
        """
        if channels is None:
            names = [name for name, kind in self.channels.items() if kind in self._bundle]
        else:
            names = [name for name in channels if name in self.channels]
            unknown = sorted(f"{name} ({self.channels[name]})" for name in names
                             if self.channels[name] not in self._bundle)
            if unknown:
                raise RecordError(f"the descriptor does not define the message types of {', '.join(unknown)}")
        return {name: self._bundle.message(self.channels[name]) for name in names}

    def _sections(self, stream: BinaryIO, wanted: Set[int]) -> Iterator[Tuple[int, bytes]]:
        """
        Payloads of the ``wanted`` section types after the header; other payloads are skipped.

        :param BinaryIO stream: The open record file; it is moved to just after the header first.
        :param Set[int] wanted: ``SectionType`` values to yield.
        :returns: ``(section type, payload)`` in file order; iteration ends at the end of the file, or at a
            section cut off while the record was written.
        :rtype: Iterator[Tuple[int, bytes]]
        """
        stream.seek(_SECTION.size + _HEADER_LENGTH)
        while True:
            try:
                kind, size = self._section(stream)
            except EOFError:
                return
            if kind in wanted:
                payload = stream.read(size)
                if len(payload) < size:
                    return  # A record cut off while it was written.
                yield kind, payload
            else:
                stream.seek(size, 1)

    def _section(self, stream: BinaryIO) -> Tuple[int, int]:
        """
        Read one section header and return its type and payload byte count.

        :param BinaryIO stream: The open record file, at the start of a section.
        :returns: ``(section type, payload size in bytes)``; the stream is left at the payload.
        :rtype: Tuple[int, int]
        :raises EOFError: If fewer than a complete section header's bytes remain.
        :raises RecordError: If the declared payload size is negative.
        """
        raw = stream.read(_SECTION.size)
        if len(raw) < _SECTION.size:
            raise EOFError
        kind, _, size = _SECTION.unpack(raw)
        if size < 0:
            raise RecordError(f"{self.path}: negative section size")
        return kind, size
