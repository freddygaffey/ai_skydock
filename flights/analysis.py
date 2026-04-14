"""
Mission log analysis — adapted from skydock2/tools/log_server/services/analysis.py.
Skydock2-internal imports (ai_class, utils, services.*) removed.
GPS projection depends only on flights/projection.py in this repo.

All functions parse mission.jsonl (JSONL format, one event dict per line).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Low-level JSONL helpers
# ---------------------------------------------------------------------------

def iter_events(path: Path):
    """Yield each JSON event dict from a mission.jsonl file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def iter_events_of_kind(path: Path, event_kind: str):
    for ev in iter_events(path):
        if ev.get("event") == event_kind:
            yield ev


def parse_ts(ts_str: Any) -> float:
    """Parse ISO timestamp string to float seconds since epoch. Returns 0 on error."""
    if not ts_str:
        return 0.0
    try:
        from datetime import datetime, timezone
        s = str(ts_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        return 0.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Timeline (FSM state transitions)
# ---------------------------------------------------------------------------

def _normalize_state(raw: str | None) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s.upper() or None


def build_timeline_payload(path: Path) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    last_ts = None

    for ev in iter_events(path):
        ts = parse_ts(ev.get("ts", ""))
        if ts > 0:
            last_ts = ts
        if ev.get("event") == "fsm_transition":
            transitions.append({
                "ts": ts,
                "state_from": _normalize_state(ev.get("state_from")) or "",
                "state_to":   _normalize_state(ev.get("state_to")) or "",
            })

    segments: list[dict[str, Any]] = []
    visit_counts: dict[str, int] = {}
    for i, t in enumerate(transitions):
        state = t["state_to"]
        start_ts = t["ts"]
        end_ts = transitions[i + 1]["ts"] if i + 1 < len(transitions) else (last_ts or start_ts)
        visit_counts[state] = visit_counts.get(state, 0) + 1
        segments.append({
            "state":      state,
            "start_ts":   start_ts,
            "end_ts":     end_ts,
            "duration_s": max(0.0, end_ts - start_ts),
            "visit_num":  visit_counts[state],
        })

    summary: dict[str, dict[str, Any]] = {}
    for seg in segments:
        s = seg["state"]
        if s not in summary:
            summary[s] = {"state": s, "total_s": 0.0, "visits": 0}
        summary[s]["total_s"] += seg["duration_s"]
        summary[s]["visits"] += 1

    return {
        "segments": segments,
        "summary":  sorted(summary.values(), key=lambda x: -x["total_s"]),
    }


# ---------------------------------------------------------------------------
# GPS path
# ---------------------------------------------------------------------------

def telemetry_path_points(path: Path, stride: int = 1) -> list[dict[str, Any]]:
    pts: list[dict[str, Any]] = []
    count = 0
    for ev in iter_events(path):
        if ev.get("event") != "telemetry_sample":
            continue
        count += 1
        if count % stride != 0:
            continue
        ds = ev.get("drone_state", {})
        lat, lon = ds.get("latitude", 0), ds.get("longitude", 0)
        if lat == 0 and lon == 0:
            continue
        pts.append({
            "lat": lat, "lon": lon,
            "alt": ds.get("altitude_rel_home", 0),
            "ts":  ev.get("ts", ""),
        })
    return pts


def fsm_tick_path_points(path: Path, stride: int = 5) -> list[dict[str, Any]]:
    pts: list[dict[str, Any]] = []
    count = 0
    for ev in iter_events_of_kind(path, "fsm_tick"):
        count += 1
        if stride > 1 and count % stride != 0:
            continue
        ds = ev.get("drone_state", {})
        if not isinstance(ds, dict):
            continue
        lat = ds.get("latitude", 0)
        lon = ds.get("longitude", 0)
        try:
            la, lo = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if la == 0.0 and lo == 0.0:
            continue
        pts.append({
            "lat":   la,
            "lon":   lo,
            "alt":   ds.get("altitude_rel_home", 0),
            "ts":    ev.get("ts", ""),
            "state": ev.get("state", ""),
        })
    return pts


# ---------------------------------------------------------------------------
# Mission summary
# ---------------------------------------------------------------------------

def build_summary_payload(path: Path) -> dict[str, Any]:
    header: dict[str, Any] = {}
    event_counts: dict[str, int] = {}
    first_ts = last_ts = None
    prev_ll: tuple[float, float] | None = None
    path_length_m = 0.0
    alts: list[float] = []
    frames_with_detections = 0

    for ev in iter_events(path):
        ev_type = ev.get("event", "")
        event_counts[ev_type] = event_counts.get(ev_type, 0) + 1

        ts = parse_ts(ev.get("ts", ""))
        if ts > 0:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        if ev_type == "mission_start":
            header = ev

        fr = ev.get("frame")
        if fr and (fr.get("detections") or []):
            frames_with_detections += 1

        if ev_type == "telemetry_sample":
            ds = ev.get("drone_state") or {}
            lat, lon = ds.get("latitude"), ds.get("longitude")
            if lat is not None and lon is not None:
                la, lo = float(lat), float(lon)
                if not (la == 0.0 and lo == 0.0):
                    if prev_ll is not None:
                        path_length_m += haversine_m(prev_ll[0], prev_ll[1], la, lo)
                    prev_ll = (la, lo)
                alt = ds.get("altitude_rel_home")
                if alt is not None:
                    try:
                        alts.append(float(alt))
                    except (TypeError, ValueError):
                        pass

    duration_s = (last_ts - first_ts) if first_ts and last_ts else 0.0
    alt_min  = min(alts) if alts else None
    alt_max  = max(alts) if alts else None
    alt_mean = sum(alts) / len(alts) if alts else None

    return {
        "header":               header,
        "duration_s":           duration_s,
        "event_counts":         event_counts,
        "frames_with_detections": frames_with_detections,
        "path_length_m":        round(path_length_m, 2),
        "altitude_min_m":       round(alt_min, 3) if alt_min is not None else None,
        "altitude_max_m":       round(alt_max, 3) if alt_max is not None else None,
        "altitude_mean_m":      round(alt_mean, 3) if alt_mean is not None else None,
    }


# ---------------------------------------------------------------------------
# Frame events (for flight browser)
# ---------------------------------------------------------------------------

def build_frame_events(path: Path, mission_dir: Path | None = None) -> list[dict[str, Any]]:
    """
    Build per-frame rows for the flight browser.
    For real missions: scans mission_dir/raw_frames/ and matches each JPEG
    to the nearest fsm_tick by time_ns.
    """
    from bisect import bisect_right

    frames_dir = (mission_dir / "raw_frames") if mission_dir else None
    frame_files: list[tuple[int, str]] = []

    if frames_dir and frames_dir.is_dir():
        for f in frames_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg") and f.stem.isdigit():
                frame_files.append((int(f.stem), str(f)))
        frame_files.sort()

    if not frame_files:
        return []

    # Build sorted fsm_tick index by time_ns
    ticks: list[tuple[int, dict[str, Any]]] = []
    if path and path.exists():
        for ev in iter_events(path):
            if ev.get("event") != "fsm_tick":
                continue
            t = ev.get("time_ns")
            if t is not None:
                ticks.append((int(t), ev))
        ticks.sort(key=lambda x: x[0])

    def nearest(ts_ns: int) -> dict[str, Any] | None:
        if not ticks:
            return None
        keys = [t[0] for t in ticks]
        idx = bisect_right(keys, ts_ns) - 1
        if idx < 0:
            idx = 0
        if idx + 1 < len(ticks) and abs(ticks[idx + 1][0] - ts_ns) < abs(ticks[idx][0] - ts_ns):
            idx += 1
        return ticks[idx][1]

    results = []
    for i, (ts_ns, img_path) in enumerate(frame_files):
        ev = nearest(ts_ns)
        dets = []
        ds_dict = None
        ts_str = None
        state = ""
        if ev:
            ds_dict = ev.get("drone_state")
            fr = ev.get("frame") or {}
            dets = fr.get("detections") or []
            ts_str = ev.get("ts")
            state = ev.get("state", "")

        results.append({
            "frame_index": i,
            "ts":          ts_str,
            "ts_ns":       ts_ns,
            "photo_path":  img_path,
            "detections":  dets,
            "drone_state": ds_dict,
            "state":       state,
        })
    return results
