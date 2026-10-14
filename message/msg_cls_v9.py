from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .descriptor import DescriptorBundle
from google.protobuf.message import Message

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


# TODO: Need an Factory method to generate the message, hence will not limited to one version
# User only have to prepare the .desc file and protobuf python will parse

# ---------------------------------------------------------------------------
# Default Apollo common_msgs bundle
# ---------------------------------------------------------------------------

_DEFAULT_DESCRIPTOR_PATH = Path(__file__).with_name("common_msgs_v9.desc")
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
