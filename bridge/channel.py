"""
Channel that communicate with CyberRT framework in the Docker container.
"""

from pathlib import Path
from typing import Type
from dataclasses import dataclass
from google.protobuf.message import Message
if "." in __package__:
    from ..message.descriptor import DescriptorBundle
else:
    from message.descriptor import DescriptorBundle


@dataclass
class Channel:
    """
    Class representing information regarding cyber bridge channels.

    :param str name: The name of the channel
    :param Type[Message] msg_cls: The protobuf message class, not an instance
    """
    name: str
    """The name of the channel"""
    msg_cls: Type[Message]
    """The protobuf message class, not an instance"""


class Channels:
    """
    Channel definitions resolved against a selected descriptor bundle.

    Construct Channels(path) before accessing a channel definition.

    :param str | Path descriptor_path: The path of the descriptor .desc file
    """
    descriptor_path: Path
    """The resolved path of the descriptor .desc file"""

    def __init__(self, descriptor_path: str | Path) -> None:
        """
        Load the descriptor bundle and keep its resolved path.

        :param str | Path descriptor_path: The path of the descriptor .desc file
        :raises FileNotFoundError: If the path does not refer to a file
        """
        self.descriptor_path = DescriptorBundle(descriptor_path).descriptor_path

    def _channel(self, name: str, msg_type: str) -> Channel:
        """
        Factory function to build the Channel.

        :param str name: The name of the channel, e.g., "/apollo/canbus/chassis"
        :param str msg_type: The full name of the message, e.g., "apollo.canbus.Chassis"
        :returns: The concrete Channel
        :rtype: Channel
        """
        return Channel(name, DescriptorBundle(self.descriptor_path).message(msg_type))

    @property
    def Chassis(self) -> Channel:
        """
        Chassis channel.

        :returns: ``/apollo/canbus/chassis`` carrying ``apollo.canbus.Chassis``
        :rtype: Channel
        """
        return self._channel("/apollo/canbus/chassis", "apollo.canbus.Chassis")

    @property
    def Localization(self) -> Channel:
        """
        Localization channel.

        :returns: ``/apollo/localization/pose`` carrying ``apollo.localization.LocalizationEstimate``
        :rtype: Channel
        """
        return self._channel(
            "/apollo/localization/pose",
            "apollo.localization.LocalizationEstimate",
        )

    @property
    def Obstacles(self) -> Channel:
        """
        Perception obstacles channel.

        :returns: ``/apollo/perception/obstacles`` carrying ``apollo.perception.PerceptionObstacles``
        :rtype: Channel
        """
        return self._channel(
            "/apollo/perception/obstacles",
            "apollo.perception.PerceptionObstacles",
        )

    @property
    def TrafficLight(self) -> Channel:
        """
        Traffic light channel.

        :returns: ``/apollo/perception/traffic_light`` carrying ``apollo.perception.TrafficLightDetection``
        :rtype: Channel
        """
        return self._channel(
            "/apollo/perception/traffic_light",
            "apollo.perception.TrafficLightDetection",
        )

    @property
    def Planning(self) -> Channel:
        """
        Planning channel.

        :returns: ``/apollo/planning`` carrying ``apollo.planning.ADCTrajectory``
        :rtype: Channel
        """
        return self._channel("/apollo/planning", "apollo.planning.ADCTrajectory")

    @property
    def Prediction(self) -> Channel:
        """
        Prediction channel.

        :returns: ``/apollo/prediction`` carrying ``apollo.prediction.PredictionObstacles``
        :rtype: Channel
        """
        return self._channel("/apollo/prediction", "apollo.prediction.PredictionObstacles")

    @property
    def LegacyRoutingRequest(self) -> Channel:
        """
        Pre-9.0 routing input; Apollo 9.0+ reads it only via old_routing_adapter.
        Use ``bridge.services.LaneFollow`` or ``bridge.services.ValetParking`` instead.

        :returns: ``/apollo/routing_request`` carrying ``apollo.routing.RoutingRequest``
        :rtype: Channel
        """
        return self._channel("/apollo/routing_request", "apollo.routing.RoutingRequest")
