"""
Cyber Bridge module's Operations definition.
"""

class BridgeOp:
    """
    Class representing cyber bridge operations.
    """

    RegisterDesc = (1).to_bytes(1, byteorder="big")
    """Register a descriptor of protobuf message, let the cyber_bridge in apollo know the format"""

    AddReader = (2).to_bytes(1, byteorder="big")
    """Register a subscriber"""

    AddWriter = (3).to_bytes(1, byteorder="big")
    """Register a publisher"""

    Publish = (4).to_bytes(1, byteorder="big")
    """Publish a protobuf message"""

    CallService = (5).to_bytes(1, byteorder="big")
    """Call a Cyber service with a uint32 request ID."""

    ServiceResponse = (6).to_bytes(1, byteorder="big")
    """Receive the request ID, success byte, and serialized response."""
