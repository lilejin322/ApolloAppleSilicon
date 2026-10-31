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
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Collection, Dict, Iterator, Tuple

from google.protobuf.message import Message

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
    time: int
    message: Message


class RecordReader:
    """
    One record file and the channels it holds.

    :param path: The record file, e.g. ``ApolloManager.record_file``.
    :param descriptor_path: The protobuf descriptor of the Apollo that recorded it.
    :raises RecordError: The file does not start with a record header, or is compressed.
    """

    path: Path
    header: Message
    """The record's ``apollo.cyber.proto.Header``; ``is_complete`` is False when recording did not finish."""
    channels: Dict[str, str]
    """Message type of every channel, by channel name."""

    def __init__(self, path: str | Path, *, descriptor_path: str | Path) -> None:
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

    def messages(self, channels: Collection[str] | None = None) -> Iterator[RecordMessage]:
        """
        Decode the messages of ``channels``, in file order (chunks in recording order).

        :param channels: Channel names; None decodes every channel whose message type the descriptor knows.
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

    def _classes(self, channels: Collection[str] | None) -> Dict[str, type]:
        if channels is None:
            names = [name for name, kind in self.channels.items() if self._known(kind)]
        else:
            names = [name for name in channels if name in self.channels]
            unknown = sorted(f"{name} ({self.channels[name]})" for name in names if not self._known(self.channels[name]))
            if unknown:
                raise RecordError(f"the descriptor does not define the message types of {', '.join(unknown)}")
        return {name: self._bundle.message(self.channels[name]) for name in names}

    def _known(self, message_type: str) -> bool:
        try:
            self._bundle.message(message_type)
        except KeyError:
            return False
        return True

    def _sections(self, stream: BinaryIO, wanted: set) -> Iterator[Tuple[int, bytes]]:
        """Payloads of the ``wanted`` section types after the header; other payloads are skipped."""
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
        raw = stream.read(_SECTION.size)
        if len(raw) < _SECTION.size:
            raise EOFError
        kind, _, size = _SECTION.unpack(raw)
        if size < 0:
            raise RecordError(f"{self.path}: negative section size")
        return kind, size
