"""
GPS back-projection helpers — adapted from skydock2/tools/log_server/services/projection.py.
Skydock2-internal imports (ai_class, utils) removed.
Only drone_state_from_dict is kept (used by dashboard for drone_state JSON parsing).

Full lat/lon projection requires skydock2 geometry code and is NOT available here.
The dashboard uses this only for drone_state parsing (lat/lon/altitude fields).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def drone_state_from_dict(ds: dict | None) -> Any | None:
    """
    Build a minimal drone state object from a mission log drone_state dict.
    Exposes: latitude, longitude, altitude_rel_home, rangefinder_m, rotaion.
    """
    if not ds or not isinstance(ds, dict):
        return None

    rot_d = ds.get("rotaion")
    if isinstance(rot_d, dict):
        rx = float(rot_d.get("x") or 0.0)
        ry = float(rot_d.get("y") or 0.0)
        rz = float(rot_d.get("z") or 0.0)
    else:
        rx = float(ds.get("rotaion_x") or 0.0)
        ry = float(ds.get("rotaion_y") or 0.0)
        rz = float(ds.get("rotaion_z") or 0.0)

    class _LogDroneState:
        __slots__ = (
            "latitude", "longitude", "altitude_rel_home",
            "rangefinder_m", "rotaion", "_rx", "_ry", "_rz",
        )

        def __init__(self) -> None:
            self.latitude          = float(ds.get("latitude") or 0.0)
            self.longitude         = float(ds.get("longitude") or 0.0)
            self.altitude_rel_home = float(ds.get("altitude_rel_home") or 0.0)
            self.rangefinder_m     = float(ds.get("rangefinder_m") or 0.0)
            self._rx, self._ry, self._rz = rx, ry, rz
            self.rotaion = SimpleNamespace(x=self._rx, y=self._ry, z=self._rz)

        def get_rotation_at_time(self, _t: Any) -> Any:
            return SimpleNamespace(x=self._rx, y=self._ry, z=self._rz)

        def get_position_at_time(self, _t: Any) -> Any:
            return SimpleNamespace(lat=self.latitude, lon=self.longitude)

    return _LogDroneState()
