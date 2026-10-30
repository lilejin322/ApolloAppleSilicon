"""
``SignalProgram``: the frozen runtime view of a ``SignalSection``.

The chromosome (``SignalSection``, genes, ``validate``) stays in the genetic side;
at the run boundary the caller materializes ``SignalProgram.from_section(section)``
once (like SUMO loading a ``tlLogic`` or an OpenSCENARIO ``TrafficSignalController``),
and ``TrafficLightManager`` only reads this program. It holds rebuilt ``SignalPlan``
copies and no reference to the section, so mutating or replacing the chromosome
later cannot change what is published. It has no genetic API (no ``validate``,
no ``to_dict`` / ``from_dict``, no mutation helpers).
"""

from typing import Any, Dict, Iterable, Tuple

from .traffic_signal import PedState, SignalColor, SignalPlan, SignalSection

__all__ = ["SignalProgram"]


class SignalProgram:
    """
    Immutable signal state machine for one run.

    :param Iterable[SignalPlan] plans: Plans to run; each one is rebuilt (deep copy).
    :raises ValueError: If one signal ID belongs to two plans.
    """

    __slots__ = ("_plans", "_by_signal", "_by_crosswalk")

    def __init__(self, plans: Iterable[SignalPlan] = ()) -> None:
        copied = tuple(SignalPlan.from_dict(p.to_dict()) for p in plans)
        by_signal: Dict[str, SignalPlan] = {}
        for plan in copied:
            for s in plan.signals:
                if s in by_signal:
                    raise ValueError(f"signal {s} in two plans")
                by_signal[s] = plan
        object.__setattr__(self, "_plans", copied)
        object.__setattr__(self, "_by_signal", by_signal)
        object.__setattr__(self, "_by_crosswalk", {w[0]: p for p in copied for w in p.walks})

    def __setattr__(self, name, value):
        raise AttributeError("SignalProgram is immutable")

    @classmethod
    def from_section(cls, section: SignalSection) -> "SignalProgram":
        """
        Materialize a program from a (validated) chromosome section; keeps no reference to it.

        :param SignalSection section: The scenario's signal section.
        :rtype: SignalProgram
        """
        if not isinstance(section, SignalSection):
            raise TypeError("from_section expects a SignalSection")
        return cls(section.plans)

    @property
    def plans(self) -> Tuple[SignalPlan, ...]:
        """The copied plans (frozen ``SignalPlan`` values)."""
        return self._plans

    def __len__(self) -> int:
        return len(self._plans)

    @property
    def signals(self) -> Tuple[str, ...]:
        """Signal IDs, sorted."""
        return tuple(sorted(self._by_signal))

    @property
    def crosswalks(self) -> Tuple[str, ...]:
        """Crosswalk IDs with a pedestrian head, sorted."""
        return tuple(sorted(self._by_crosswalk))

    def color(self, signal_id: str, t: float, force_green: bool = False) -> SignalColor:
        """Same as ``SignalSection.color``."""
        plan = self._by_signal[signal_id]
        return SignalColor.GREEN if force_green else plan.color(signal_id, t)

    def colors(self, t: float, force_green: bool = False) -> Dict[str, SignalColor]:
        """Same as ``SignalSection.colors``."""
        return {s: self.color(s, t, force_green) for s in sorted(self._by_signal)}

    def ped_state(self, crosswalk_id: str, t: float) -> PedState:
        """Same as ``SignalSection.ped_state``."""
        return self._by_crosswalk[crosswalk_id].ped_state(crosswalk_id, t)

    def detection(self, t: float, message_cls: Any, force_green: bool = False) -> Any:
        """
        An ``apollo.perception.TrafficLightDetection`` for time ``t``; same semantics as
        ``SignalSection.detection`` (one light per signal in ID order, confidence 1,
        ``remaining_time`` omitted when ``force_green``).
        """
        message = message_cls()
        message.header.timestamp_sec = t
        message.contain_lights = bool(self._by_signal)
        for signal_id, color in self.colors(t, force_green).items():
            light = message.traffic_light.add()
            light.id = signal_id
            light.color = int(color)
            light.confidence = 1.0
            if not force_green:
                light.remaining_time = self._remaining(signal_id, t)
        return message

    def _remaining(self, signal_id: str, t: float) -> float:
        plan = self._by_signal[signal_id]
        now = plan.color(signal_id, t)
        step, ahead = 0.1, 0.0
        while ahead < plan.cycle and plan.color(signal_id, t + ahead + step) == now:
            ahead += step
        return round(ahead + step, 1)
