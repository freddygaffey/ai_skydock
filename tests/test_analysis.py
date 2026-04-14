"""
Tests for flights/analysis.py — haversine_m(), parse_ts(),
build_timeline_payload(), build_summary_payload(), fsm_tick_path_points().

No YOLO / GPS hardware required.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flights.analysis import (
    build_summary_payload,
    build_timeline_payload,
    fsm_tick_path_points,
    haversine_m,
    iter_events,
    parse_ts,
    telemetry_path_points,
)


# ---------------------------------------------------------------------------
# haversine_m
# ---------------------------------------------------------------------------

def test_haversine_same_point():
    assert haversine_m(51.5, -0.1, 51.5, -0.1) == pytest.approx(0.0)


def test_haversine_known_distance():
    # London → Paris ≈ 340 km
    d = haversine_m(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330_000 < d < 350_000


def test_haversine_short_distance():
    # ~1 degree latitude ≈ 111 km
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert 110_000 < d < 112_000


def test_haversine_symmetric():
    d1 = haversine_m(51.0, 0.0, 52.0, 1.0)
    d2 = haversine_m(52.0, 1.0, 51.0, 0.0)
    assert d1 == pytest.approx(d2)


# ---------------------------------------------------------------------------
# parse_ts
# ---------------------------------------------------------------------------

def test_parse_ts_iso_utc():
    ts = parse_ts("2026-04-15T12:00:00Z")
    assert ts > 0


def test_parse_ts_iso_offset():
    ts = parse_ts("2026-04-15T12:00:00+00:00")
    assert ts > 0


def test_parse_ts_empty():
    assert parse_ts("") == 0.0


def test_parse_ts_none():
    assert parse_ts(None) == 0.0


def test_parse_ts_garbage():
    assert parse_ts("not-a-date") == 0.0


def test_parse_ts_ordering():
    t1 = parse_ts("2026-04-15T10:00:00Z")
    t2 = parse_ts("2026-04-15T11:00:00Z")
    assert t2 > t1


# ---------------------------------------------------------------------------
# iter_events helpers
# ---------------------------------------------------------------------------

def make_jsonl(tmp_path: Path, events: list[dict]) -> Path:
    path = tmp_path / "mission.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_iter_events_basic(tmp_path):
    path = make_jsonl(tmp_path, [{"event": "a"}, {"event": "b"}])
    evs = list(iter_events(path))
    assert len(evs) == 2


def test_iter_events_skips_bad_lines(tmp_path):
    path = tmp_path / "mission.jsonl"
    path.write_text('{"event": "ok"}\nnot json\n{"event": "also_ok"}\n')
    evs = list(iter_events(path))
    assert len(evs) == 2


# ---------------------------------------------------------------------------
# build_timeline_payload
# ---------------------------------------------------------------------------

def _make_transition(ts: str, from_: str, to: str) -> dict:
    return {"event": "fsm_transition", "ts": ts, "state_from": from_, "state_to": to}


def test_build_timeline_empty(tmp_path):
    path = make_jsonl(tmp_path, [{"event": "other", "ts": "2026-04-15T10:00:00Z"}])
    result = build_timeline_payload(path)
    assert result["segments"] == []
    assert result["summary"] == []


def test_build_timeline_single_transition(tmp_path):
    path = make_jsonl(tmp_path, [
        _make_transition("2026-04-15T10:00:00Z", "IDLE", "FLYING"),
        {"event": "other", "ts": "2026-04-15T10:01:00Z"},
    ])
    result = build_timeline_payload(path)
    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["state"] == "FLYING"
    assert seg["duration_s"] == pytest.approx(60.0)


def test_build_timeline_multiple_transitions(tmp_path):
    path = make_jsonl(tmp_path, [
        _make_transition("2026-04-15T10:00:00Z", "IDLE", "ARMING"),
        _make_transition("2026-04-15T10:00:05Z", "ARMING", "FLYING"),
        _make_transition("2026-04-15T10:02:00Z", "FLYING", "LANDING"),
    ])
    result = build_timeline_payload(path)
    assert len(result["segments"]) == 3
    states = [s["state"] for s in result["segments"]]
    assert states == ["ARMING", "FLYING", "LANDING"]


def test_build_timeline_summary_totals(tmp_path):
    path = make_jsonl(tmp_path, [
        _make_transition("2026-04-15T10:00:00Z", "IDLE", "FLYING"),
        _make_transition("2026-04-15T10:01:00Z", "FLYING", "IDLE"),
        _make_transition("2026-04-15T10:01:30Z", "IDLE", "FLYING"),
        {"event": "other", "ts": "2026-04-15T10:03:30Z"},
    ])
    result = build_timeline_payload(path)
    summary = {r["state"]: r for r in result["summary"]}
    assert summary["FLYING"]["total_s"] == pytest.approx(60.0 + 120.0)
    assert summary["FLYING"]["visits"] == 2
    assert summary["IDLE"]["total_s"] == pytest.approx(30.0)


def test_build_timeline_normalises_dotted_state(tmp_path):
    path = make_jsonl(tmp_path, [
        {"event": "fsm_transition", "ts": "2026-04-15T10:00:00Z",
         "state_from": "states.IDLE", "state_to": "states.FLYING"},
    ])
    result = build_timeline_payload(path)
    assert result["segments"][0]["state"] == "FLYING"


# ---------------------------------------------------------------------------
# build_summary_payload
# ---------------------------------------------------------------------------

def test_build_summary_basic(tmp_path):
    path = make_jsonl(tmp_path, [
        {"event": "mission_start", "ts": "2026-04-15T10:00:00Z", "mission_id": "001"},
        {
            "event": "telemetry_sample",
            "ts": "2026-04-15T10:00:01Z",
            "drone_state": {"latitude": 51.5, "longitude": -0.1, "altitude_rel_home": 10.0},
        },
        {
            "event": "telemetry_sample",
            "ts": "2026-04-15T10:00:02Z",
            "drone_state": {"latitude": 51.501, "longitude": -0.1, "altitude_rel_home": 12.0},
        },
    ])
    result = build_summary_payload(path)
    assert result["duration_s"] == pytest.approx(2.0)
    assert result["altitude_min_m"] == pytest.approx(10.0)
    assert result["altitude_max_m"] == pytest.approx(12.0)
    assert result["path_length_m"] > 0


def test_build_summary_skips_zero_coords(tmp_path):
    path = make_jsonl(tmp_path, [
        {
            "event": "telemetry_sample",
            "ts": "2026-04-15T10:00:00Z",
            "drone_state": {"latitude": 0.0, "longitude": 0.0, "altitude_rel_home": 5.0},
        },
    ])
    result = build_summary_payload(path)
    assert result["path_length_m"] == 0.0
    # altitude still captured
    assert result["altitude_min_m"] == pytest.approx(5.0)


def test_build_summary_event_counts(tmp_path):
    path = make_jsonl(tmp_path, [
        {"event": "fsm_tick", "ts": "2026-04-15T10:00:00Z"},
        {"event": "fsm_tick", "ts": "2026-04-15T10:00:01Z"},
        {"event": "telemetry_sample", "ts": "2026-04-15T10:00:02Z", "drone_state": {}},
    ])
    result = build_summary_payload(path)
    assert result["event_counts"]["fsm_tick"] == 2
    assert result["event_counts"]["telemetry_sample"] == 1


# ---------------------------------------------------------------------------
# fsm_tick_path_points
# ---------------------------------------------------------------------------

def test_fsm_tick_path_points(tmp_path):
    path = make_jsonl(tmp_path, [
        {"event": "fsm_tick", "ts": "2026-04-15T10:00:00Z", "state": "FLYING",
         "drone_state": {"latitude": 51.5, "longitude": -0.1, "altitude_rel_home": 10.0}},
        {"event": "fsm_tick", "ts": "2026-04-15T10:00:01Z", "state": "FLYING",
         "drone_state": {"latitude": 51.501, "longitude": -0.101, "altitude_rel_home": 11.0}},
    ])
    pts = fsm_tick_path_points(path, stride=1)
    assert len(pts) == 2
    assert pts[0]["lat"] == pytest.approx(51.5)
    assert pts[0]["state"] == "FLYING"


def test_fsm_tick_path_points_skips_zero(tmp_path):
    path = make_jsonl(tmp_path, [
        {"event": "fsm_tick", "ts": "2026-04-15T10:00:00Z",
         "drone_state": {"latitude": 0.0, "longitude": 0.0}},
        {"event": "fsm_tick", "ts": "2026-04-15T10:00:01Z",
         "drone_state": {"latitude": 51.5, "longitude": -0.1}},
    ])
    pts = fsm_tick_path_points(path, stride=1)
    assert len(pts) == 1


def test_fsm_tick_path_points_stride(tmp_path):
    events = [
        {"event": "fsm_tick", "ts": f"2026-04-15T10:00:0{i}Z",
         "drone_state": {"latitude": 51.5 + i * 0.001, "longitude": -0.1}}
        for i in range(6)
    ]
    path = make_jsonl(tmp_path, events)
    pts = fsm_tick_path_points(path, stride=2)
    assert len(pts) == 3


def test_telemetry_path_points(tmp_path):
    path = make_jsonl(tmp_path, [
        {"event": "telemetry_sample", "ts": "2026-04-15T10:00:00Z",
         "drone_state": {"latitude": 51.5, "longitude": -0.1, "altitude_rel_home": 10}},
        {"event": "other"},
        {"event": "telemetry_sample", "ts": "2026-04-15T10:00:01Z",
         "drone_state": {"latitude": 51.501, "longitude": -0.1, "altitude_rel_home": 11}},
    ])
    pts = telemetry_path_points(path)
    assert len(pts) == 2
