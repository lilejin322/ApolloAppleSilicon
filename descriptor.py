"""
Message classes built from Apollo's common_msgs FileDescriptorSet.

Author: Lejin Li, Kyushu University & National Institute of Informatics, Japan
Contact: li.lejin.147@s.kyushu-u.ac.jp

For more information about the serialization of protobuf messages, see:
https://buf.build/docs/reference/compilation-and-descriptors/#descriptor-production
https://googleapis.dev/python/protobuf/latest/google/protobuf/descriptor.html
"""

from functools import lru_cache
from pathlib import Path
from typing import Tuple, Set, Iterator

from google.protobuf.descriptor_pb2 import FileDescriptorSet
from google.protobuf.message import Message
from google.protobuf.message_factory import GetMessages

class DescriptorBundle:
    """
    Registry of protobuf message classes loaded from a FileDescriptorSet.

    A DescriptorBundle represents one serialized protobuf descriptor bundle.
    Different descriptor files can be loaded simultaneously, which makes this
    class suitable for comparing protobuf definitions across different Apollo
    versions.

    :param str | Path descriptor_path: Path to the serialized FileDescriptorSet
    """

    def __init__(self, descriptor_path: str | Path) -> None:
        """
        Constructor

        :param str | Path descriptor_path: Path to the serialized FileDescriptorSet
        :raises FileNotFoundError: If the descriptor file does not exist
        :raises ValueError: If the descriptor file is not a valid FileDescriptorSet
        :raises protobuf.error.DecodeError: If the descriptor file is not a valid FileDescriptorSet
        """
        self._descriptor_path = Path(descriptor_path).expanduser().resolve()

        if not self._descriptor_path.is_file():
            raise FileNotFoundError(
                f"Descriptor file does not exist: {self._descriptor_path}"
            )

        self._descriptor_set = self._load_descriptor_set()
        self._messages: dict[str, type[Message]] = GetMessages(
            list(self._descriptor_set.file)
        )

    def _load_descriptor_set(self) -> FileDescriptorSet:
        """
        Load the FileDescriptorSet from disk.

        :returns: The parsed FileDescriptorSet
        :rtype: FileDescriptorSet
        """
        descriptor_set = FileDescriptorSet()
        descriptor_set.ParseFromString(self._descriptor_path.read_bytes())
        return descriptor_set

    @property
    def descriptor_path(self) -> Path:
        """
        Return the path to the descriptor bundle.

        :returns: Path to the serialized FileDescriptorSet
        :rtype: Path
        """
        return self._descriptor_path

    @property
    def descriptor_set(self) -> FileDescriptorSet:
        """
        Return the loaded FileDescriptorSet.

        :returns: The parsed FileDescriptorSet
        :rtype: FileDescriptorSet
        """
        return self._descriptor_set

    def serialized_file_descriptors(self) -> Tuple[bytes, ...]:
        """
        Return serialized FileDescriptorProto messages in bundle order.

        :returns: A tuple containing one serialized FileDescriptorProto for
                  each file in the descriptor set
        :rtype: Tuple[bytes, ...]
        """
        return tuple(
            file_descriptor.SerializeToString()
            for file_descriptor in self._descriptor_set.file
        )

    def messages(self) -> dict[str, type[Message]]:
        """
        Return all protobuf message classes by fully qualified name.

        :returns: A mapping from fully qualified protobuf message names to
                  their dynamically generated Python classes
        :rtype: dict[str, type[Message]]
        """
        return self._messages

    def message(self, full_name: str) -> type[Message]:
        """
        Return a protobuf message class by fully qualified name.

        :param str full_name: Fully qualified protobuf message name
        :returns: The corresponding protobuf message class
        :rtype: type[Message]
        :raises KeyError: If the message is not defined in this bundle
        """
        try:
            return self._messages[full_name]
        except KeyError as exc:
            raise KeyError(
                f"{full_name!r} is not defined in "
                f"{self._descriptor_path.name!r}"
            ) from exc

    def message_names(self) -> Set[str]:
        """
        Return all fully qualified message names in the bundle.

        :returns: A set of fully qualified protobuf message names
        :rtype: Set[str]
        """
        return set(self._messages)

    def has_message(self, full_name: str) -> bool:
        """
        Check whether a message exists in the bundle.

        :param str full_name: Fully qualified protobuf message name
        :returns: True if the message exists, otherwise False
        :rtype: bool
        """
        return full_name in self._messages

    def __getitem__(self, full_name: str) -> type[Message]:
        """
        Return a message class using dictionary-style access.

        Example::

            Header = bundle["apollo.common.Header"]
        """
        return self.message(full_name)

    def __contains__(self, full_name: object) -> bool:
        """
        Return whether a message name exists in the bundle.
        """
        return full_name in self._messages

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over fully qualified message names.
        """
        return iter(self._messages)

    def __len__(self) -> int:
        """
        Return the number of message classes in the bundle.
        """
        return len(self._messages)

    def __repr__(self) -> str:
        """
        String representation of the class.
        """
        return (
            f"{self.__class__.__name__}("
            f"path={str(self._descriptor_path)!r}, "
            f"messages={len(self)})"
        )


@lru_cache(maxsize=None)
def _load_descriptor_bundle(
    descriptor_path: Path,
) -> DescriptorBundle:
    """Load and cache a DescriptorBundle for a normalized path."""
    return DescriptorBundle(descriptor_path)


def load_descriptor_bundle(
    descriptor_path: str | Path,
) -> DescriptorBundle:
    """
    Load a descriptor bundle.

    Repeated calls referring to the same resolved path return the same
    DescriptorBundle instance. Different paths are loaded independently.

    :param str | Path descriptor_path: Path to the FileDescriptorSet
    :returns: Descriptor bundle associated with the path
    :rtype: DescriptorBundle
    """
    path = Path(descriptor_path).expanduser().resolve()
    return _load_descriptor_bundle(path)


# ---------------------------------------------------------------------------
# Default Apollo common_msgs bundle
# ---------------------------------------------------------------------------

_DEFAULT_DESCRIPTOR_PATH = Path(__file__).with_name("common_msgs.desc")
default_bundle = load_descriptor_bundle(_DEFAULT_DESCRIPTOR_PATH)


def message(full_name: str) -> type[Message]:
    """
    Return a message class from the default Apollo descriptor bundle.

    :param str full_name: Fully qualified protobuf message name
    :rtype: type[Message]
    """
    return default_bundle.message(full_name)


# ---------------------------------------------------------------------------
# Common Apollo message classes
# ---------------------------------------------------------------------------

PointENU = message("apollo.common.PointENU")
Point3D = message("apollo.common.Point3D")
Header = message("apollo.common.Header")
VehicleConfig = message("apollo.common.VehicleConfig")

PerceptionObstacle = message("apollo.perception.PerceptionObstacle")
PerceptionObstacles = message("apollo.perception.PerceptionObstacles")
TrafficLight = message("apollo.perception.TrafficLight")
TrafficLightDetection = message("apollo.perception.TrafficLightDetection")

LaneWaypoint = message("apollo.routing.LaneWaypoint")
RoutingRequest = message("apollo.routing.RoutingRequest")

ExternalCommandPose = message("apollo.external_command.Pose")
LaneFollowCommand = message("apollo.external_command.LaneFollowCommand")
ValetParkingCommand = message("apollo.external_command.ValetParkingCommand")
ActionCommand = message("apollo.external_command.ActionCommand")
CommandStatusRequest = message("apollo.external_command.CommandStatusRequest")
CommandStatus = message("apollo.external_command.CommandStatus")

LocalizationEstimate = message("apollo.localization.LocalizationEstimate")
ADCTrajectory = message("apollo.planning.ADCTrajectory")
Chassis = message("apollo.canbus.Chassis")

Crosswalk = message("apollo.hdmap.Crosswalk")
Junction = message("apollo.hdmap.Junction")
Lane = message("apollo.hdmap.Lane")
LaneBoundary = message("apollo.hdmap.LaneBoundary")
Overlap = message("apollo.hdmap.Overlap")
Map = message("apollo.hdmap.Map")
Signal = message("apollo.hdmap.Signal")
StopSign = message("apollo.hdmap.StopSign")
