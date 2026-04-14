"""
Tests for flights/projection.py — drone_state_from_dict().

No YOLO / GPS hardware required.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flights.projection import drone_state_from_dict


# ---------------------------------------------------------------------------
# drone_state_from_dict
# ---------------------------------------------------------------------------

def test_none_input():
    assert drone_state_from_dict(None) is None


def test_empty_dict():
    assert drone_state_from_dict({}) is None


def test_non_dict():
    assert drone_state_from_dict("not a dict") is None


def test_basic_fields():
    ds = drone_state_from_dict({
        "latitude": 51.5,
        "longitude": -0.1,
        "altitude_rel_home": 15.0,
        "rangefinder_m": 14.8,
    })
    assert ds is not None
    assert ds.latitude == pytest.approx(51.5)
    assert ds.longitude == pytest.approx(-0.1)
    assert ds.altitude_rel_home == pytest.approx(15.0)
    assert ds.rangefinder_m == pytest.approx(14.8)


def test_rotation_as_nested_dict():
    ds = drone_state_from_dict({
        "latitude": 0, "longitude": 0, "altitude_rel_home": 0,
        "rotaion": {"x": 1.0, "y": 2.0, "z": 3.0},
    })
    assert ds.rotaion.x == pytest.approx(1.0)
    assert ds.rotaion.y == pytest.approx(2.0)
    assert ds.rotaion.z == pytest.approx(3.0)


def test_rotation_as_flat_fields():
    ds = drone_state_from_dict({
        "latitude": 0, "longitude": 0, "altitude_rel_home": 0,
        "rotaion_x": 4.0, "rotaion_y": 5.0, "rotaion_z": 6.0,
    })
    assert ds.rotaion.x == pytest.approx(4.0)
    assert ds.rotaion.y == pytest.approx(5.0)
    assert ds.rotaion.z == pytest.approx(6.0)


def test_get_rotation_at_time():
    ds = drone_state_from_dict({
        "latitude": 0, "longitude": 0, "altitude_rel_home": 0,
        "rotaion": {"x": 1.0, "y": 2.0, "z": 3.0},
    })
    rot = ds.get_rotation_at_time(12345)
    assert rot.x == pytest.approx(1.0)
    assert rot.z == pytest.approx(3.0)


def test_get_position_at_time():
    ds = drone_state_from_dict({
        "latitude": 51.5, "longitude": -0.1, "altitude_rel_home": 0,
    })
    pos = ds.get_position_at_time(0)
    assert pos.lat == pytest.approx(51.5)
    assert pos.lon == pytest.approx(-0.1)


def test_missing_fields_default_to_zero():
    ds = drone_state_from_dict({"latitude": 10.0})
    assert ds.longitude == pytest.approx(0.0)
    assert ds.altitude_rel_home == pytest.approx(0.0)
    assert ds.rangefinder_m == pytest.approx(0.0)


def test_string_numeric_values():
    # Values might arrive as strings from JSON parsing edge cases
    ds = drone_state_from_dict({
        "latitude": "51.5",
        "longitude": "-0.1",
        "altitude_rel_home": "10",
    })
    assert ds.latitude == pytest.approx(51.5)
