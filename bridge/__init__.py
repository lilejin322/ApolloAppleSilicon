"""
CyberRT bridge client, channel definitions, and service definitions.

``CyberBridge`` is the TCP client of Apollo's bridge: it publishes and
subscribes on CyberRT channels and calls Cyber services (``BridgeOp`` names the
wire operations, ``ServiceCallError`` an unsuccessful service reply).
``Channels`` and ``Services`` hold the definitions it uses, a ``Channel`` or
``Service`` each: a name with its protobuf message classes, resolved from one
descriptor bundle.
"""

from .channel import Channel, Channels
from .cyber_bridge import CyberBridge
from .operation import BridgeOp
from .service import Service, ServiceCallError, Services

__all__ = ["BridgeOp", "Channel", "Channels", "CyberBridge", "Service", "ServiceCallError", "Services"]
