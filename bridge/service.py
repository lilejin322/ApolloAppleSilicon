"""
Service that communicate with CyberRT framework in the Docker container.
"""

from pathlib import Path
from typing import Type
from dataclasses import dataclass
from google.protobuf.message import Message
from message.descriptor import DescriptorBundle


class ServiceCallError(RuntimeError):
    """The bridge returned an unsuccessful service response (op 6, ok=0)."""


@dataclass
class Service:
    """
    Class representing information regarding Cyber services

    :param str name: The name of the service
    :param Type[Message] request_cls: The protobuf request message class, not an instance
    :param Type[Message] response_cls: The protobuf response message class, not an instance
    """
    name: str
    request_cls: Type[Message]
    response_cls: Type[Message]


class Services:
    """
    Apollo 9.0+ external command endpoints with selectable protobuf definitions.

    Construct Services(path) before accessing a service definition.

    :param str | Path: The path of the descriptor .desc file
    """
    descriptor_path: str | Path
    """The path of the descriptor .desc file"""

    def __init__(self, descriptor_path: str | Path):
        """
        Constructor

        :param str | Path: The path of the descriptor .desc file
        """
        self.descriptor_path = DescriptorBundle(descriptor_path).descriptor_path

    def _service(self, name: str, request_type: str, response_type: str) -> Service:
        """
        Factory function to build the Service.

        :param str name: The name of the service, e.g., "/apollo/external_command/lane_follow"
        :param str request_type: The full name of the request message, e.g., "apollo.external_command.LaneFollowCommand"
        :param str response_type: The full name of the response message, e.g., "apollo.external_command.CommandStatus"
        :returns: The concrete Service
        :rtype: Service
        """
        bundle = DescriptorBundle(self.descriptor_path)
        return Service(name, bundle.message(request_type), bundle.message(response_type))

    @property
    def LaneFollow(self) -> Service:
        """Drive along lanes to an end pose; replaces RoutingRequest"""
        return self._service(
            "/apollo/external_command/lane_follow",
            "apollo.external_command.LaneFollowCommand",
            "apollo.external_command.CommandStatus",
        )

    @property
    def ValetParking(self) -> Service:
        """Drive to and park in a parking spot"""
        return self._service(
            "/apollo/external_command/valet_parking",
            "apollo.external_command.ValetParkingCommand",
            "apollo.external_command.CommandStatus",
        )

    @property
    def Action(self) -> Service:
        """Stop, pull over, switch driving mode, etc."""
        return self._service(
            "/apollo/external_command/action",
            "apollo.external_command.ActionCommand",
            "apollo.external_command.CommandStatus",
        )

    @property
    def QueryStatus(self) -> Service:
        """Query the status of a previously sent command"""
        return self._service(
            "/apollo/external_command/command_status",
            "apollo.external_command.CommandStatusRequest",
            "apollo.external_command.CommandStatus",
        )
