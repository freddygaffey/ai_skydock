"""Training data generation: YOLO inference + GPS-projected bounding box validation."""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterator

# Safe Ultralytics hub / COCO checkpoint names (no path separators).
_HUB_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(\.[Pp][Tt])?$")

# Lazy YOLO model cache
_yolo_model = None
_yolo_model_path: str | None = None


def _repo_root() -> Path:
    """``skydock2`` root (parent of ``tools/log_server``)."""
    return Path(__file__).resolve().parents[3]


def default_model_path() -> str:
    """Path to the YOLO best.pt. Override with ``SKYDOCK_YOLO_MODEL`` env var."""
    env = os.environ.get("SKYDOCK_YOLO_MODEL", "").strip()
    if env:
        return env
    repo_pt = _repo_root() / "ai_train" / "best.pt"
    if repo_pt.is_file():
        return str(repo_pt)
    return str(Path.home() / "ai_train" / "best.pt")


def is_ultralytics_hub_model(spec: str) -> bool:
    """True for names like ``yolov8n.pt`` / ``yolo11s.pt`` (Ultralytics will fetch weights)."""
    s = (spec or "").strip()
    if not s or "/" in s or "\\" in s or s.startswith("."):
        return False
    if ".." in s:
        return False
    return bool(_HUB_MODEL_RE.fullmatch(s))


def coco_model_presets() -> list[str]:
    """COCO checkpoints listed in the training UI.

    Override with comma-separated ``SKYDOCK_YOLO_COCO_MODELS`` (same ``SKYDOCK_*`` style as ``SKYDOCK_YOLO_MODEL``).
    """
    raw = os.environ.get("SKYDOCK_YOLO_COCO_MODELS", "").strip()
    if raw:
        seen: set[str] = set()
        out: list[str] = []
        for part in raw.split(","):
            s = part.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out
    return [
        "yolov5n.pt",
        "yolov5s.pt",
        "yolov5m.pt",
        "yolov5l.pt",
        "yolov5x.pt",
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt",
    ]


def coco_model_presets_for_ui() -> tuple[list[str], str | None]:
    """Presets + optional ``SKYDOCK_YOLO_COCO_MODEL`` to pre-select in the UI."""
    preferred = os.environ.get("SKYDOCK_YOLO_COCO_MODEL", "").strip() or None
    presets = coco_model_presets()
    if preferred and preferred not in presets:
        presets = [preferred] + presets
    return presets, preferred


def default_yolo_model_presets_for_ui() -> tuple[list[str], str | None]:
    """Paths / hub names for the training UI (``SKYDOCK_YOLO_DEFAULT_MODELS``).

    Optional ``SKYDOCK_YOLO_DEFAULT_MODEL`` pre-selects one entry (added to the list if missing).
    """
    preferred = os.environ.get("SKYDOCK_YOLO_DEFAULT_MODEL", "").strip() or None
    raw = os.environ.get("SKYDOCK_YOLO_DEFAULT_MODELS", "").strip()
    if not raw:
        presets: list[str] = []
    else:
        seen: set[str] = set()
        presets = []
        for part in raw.split(","):
            s = part.strip()
            if s and s not in seen:
                seen.add(s)
                presets.append(s)
    if preferred and preferred not in presets:
        presets = [preferred] + presets
    return presets, preferred


def auto_pick_training_model() -> str | None:
    """Small / cheap default when nothing else is chosen (hub or local ``.pt``).

    - If ``SKYDOCK_YOLO_AUTO_MODEL`` is **unset** → ``yolov8n.pt`` (Ultralytics nano COCO).
    - If set **empty** or ``0``/``none``/``off`` → disabled (use true *Auto* in the UI).
    - Otherwise use that hub name or existing file path.
    """
    if "SKYDOCK_YOLO_AUTO_MODEL" not in os.environ:
        return "yolov8n.pt"
    raw = os.environ.get("SKYDOCK_YOLO_AUTO_MODEL", "").strip()
    if not raw or raw.lower() in ("0", "no", "none", "off", "false"):
        return None
    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())
    if is_ultralytics_hub_model(raw):
        return raw
    return "yolov8n.pt"


def training_yolo_models_for_ui() -> dict[str, Any]:
    """Template vars: default preset list, COCO list (deduped), and which option to pre-select."""
    default_presets, default_pick = default_yolo_model_presets_for_ui()
    coco_presets, coco_pick = coco_model_presets_for_ui()
    seen = set(default_presets)
    coco_only = [m for m in coco_presets if m not in seen]
    auto = auto_pick_training_model()
    preferred = default_pick or coco_pick or auto
    return {
        "yolo_default_presets": default_presets,
        "yolo_coco_presets": coco_only,
        "yolo_auto_pick": auto,
        "yolo_preferred": preferred,
    }


def predownload_ultralytics_weights(models: list[str]) -> dict[str, dict[str, Any]]:
    """Load each hub spec once so Ultralytics caches weights under ``~/.cache/ultralytics``."""
    out: dict[str, dict[str, Any]] = {}
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        msg = f"ultralytics not installed: {exc}"
        for raw in models:
            spec = (raw or "").strip()
            if spec:
                out[spec] = {"ok": False, "error": msg}
        return out
    for raw in models:
        spec = (raw or "").strip()
        if not spec:
            continue
        p = Path(spec).expanduser()
        if p.is_file():
            out[spec] = {"ok": True, "note": "already local"}
            continue
        if not is_ultralytics_hub_model(spec):
            out[spec] = {"ok": False, "error": "not a local file or allowed hub name"}
            continue
        try:
            YOLO(spec)
            out[spec] = {"ok": True}
        except Exception as exc:
            out[spec] = {"ok": False, "error": str(exc)}
    return out


def collect_hub_specs_for_prefetch() -> list[str]:
    """Unique hub names from auto-pick, default list, and COCO presets."""
    seen: set[str] = set()
    ordered: list[str] = []
    for m in (
        auto_pick_training_model(),
        *default_yolo_model_presets_for_ui()[0],
        *coco_model_presets(),
    ):
        if not m:
            continue
        s = str(m).strip()
        if not is_ultralytics_hub_model(s):
            continue
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def predownload_training_preset_weights() -> dict[str, dict[str, Any]]:
    """Prefetch all hub models the training UI might offer."""
    return predownload_ultralytics_weights(collect_hub_specs_for_prefetch())


def training_model_key_component(model_path: str) -> str:
    """Cache-key fragment: resolved path for local files, else hub id string."""
    s = (model_path or "").strip()
    p = Path(s).expanduser()
    try:
        if p.is_file():
            return str(p.resolve())
    except OSError:
        pass
    return s


def resolve_training_model_path(override: str | None) -> str:
    """Resolve weights: empty → :func:`default_model_path`; file → absolute path; else hub name."""
    raw = (override if override is not None else "").strip()
    if not raw:
        return default_model_path()
    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())
    if is_ultralytics_hub_model(raw):
        return raw
    raise FileNotFoundError(raw)


def default_stream_batch_size() -> int:
    """Default YOLO batch size (env ``SKYDOCK_YOLO_STREAM_BATCH`` or 24)."""
    return _training_batch_size()


def parse_request_batch_size(raw: Any) -> int | None:
    """``None`` / empty → use env default. Raises ``ValueError`` if not an int in 1..256."""
    if raw is None or raw == "":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValueError("batch_size must be an integer")
    if n < 1 or n > 256:
        raise ValueError("batch_size must be between 1 and 256")
    return n


def _yolo_half_enabled() -> bool:
    """FP16 inference when ``SKYDOCK_YOLO_HALF=1`` (CUDA only; ignored on CPU / MPS)."""
    return os.environ.get("SKYDOCK_YOLO_HALF", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _yolo_predict_use_half(force_device_cpu: bool, predict_kw: dict[str, Any]) -> bool:
    if force_device_cpu or not _yolo_half_enabled():
        return False
    if predict_kw.get("device") == "cpu":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _yolo_predict_imgsz() -> int | None:
    raw = os.environ.get("SKYDOCK_YOLO_IMGSZ", "").strip()
    if raw.isdigit():
        return max(32, int(raw))
    return None


def _yolo_predict_workers() -> int | None:
    raw = os.environ.get("SKYDOCK_YOLO_WORKERS", "").strip()
    if raw.isdigit():
        return max(0, int(raw))
    return None


def training_analysis_cache_key(
    mission_dir: Path,
    real_mission_name: str,
    *,
    model_path: str,
    conf_thresh: float,
    dist_thresh: float,
    frame_stride: int,
    yolo_batch: int | None = None,
) -> str:
    """Stable key for in-process analysis cache (invalidates on frames dir mtime, model, thresholds)."""
    frames_dir = mission_dir / "frames"
    try:
        mtime = int(frames_dir.stat().st_mtime_ns)
    except OSError:
        mtime = 0
    half = 1 if _yolo_half_enabled() else 0
    imgsz = os.environ.get("SKYDOCK_YOLO_IMGSZ", "").strip() or "def"
    mp = training_model_key_component(model_path)
    bkey = 0 if yolo_batch is None else int(yolo_batch)
    # bump when bbox-to-640px mapping changes (invalidates in-process cache)
    geom = "xyxy_to_640_v1|lbl1"
    return (
        f"{mission_dir.resolve()}|{real_mission_name}|{mtime}|{mp}|"
        f"{conf_thresh:.8g}|{dist_thresh:.8g}|{frame_stride}|{half}|{imgsz}|b{bkey}|{geom}"
    )


def yolo_predict_device_kw() -> Any | None:
    """``device`` argument for ``YOLO.predict``.

    - ``SKYDOCK_YOLO_FORCE_CPU=1`` → ``\"cpu\"`` (skip GPU; use when CUDA driver/toolkit mismatch or flaky GPU).
    - Unset ``SKYDOCK_YOLO_DEVICE`` → ``None`` (Ultralytics default: CUDA if PyTorch sees a GPU, else CPU).
    - ``cpu`` / ``cuda`` / ``cuda:0`` / ``0`` / ``mps`` etc. → passed through (see Ultralytics docs).
    """
    if os.environ.get("SKYDOCK_YOLO_FORCE_CPU", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return "cpu"
    raw = os.environ.get("SKYDOCK_YOLO_DEVICE", "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    return raw


def _exception_looks_like_cuda_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    hints = (
        "cuda",
        "cudnn",
        "cublas",
        "gpu",
        "nvidia",
        "device-side assert",
        "out of memory",
        "illegal memory access",
        "ecc",
        "no kernel image",
        "not compiled with cuda",
        "nvrtc",
        "driver",
    )
    return any(h in msg for h in hints)


def _yolo_predict_batch(
    model_path: str,
    paths: list[str],
    predict_kw: dict[str, Any],
    *,
    force_device_cpu: bool,
) -> tuple[Any, Any, bool]:
    """Run ``predict``; on likely CUDA failure reload model and retry on CPU.

    Returns ``(yolo_results, model, used_cpu_fallback)``.
    """
    global _yolo_model, _yolo_model_path
    model = _get_yolo_model(model_path)
    kw: dict[str, Any] = dict(predict_kw)
    if force_device_cpu:
        kw["device"] = "cpu"
        kw.pop("half", None)
    try:
        results = model.predict(paths, **kw)
        return results, model, force_device_cpu
    except Exception as first:
        if kw.get("device") == "cpu" or force_device_cpu:
            raise RuntimeError(
                "YOLO inference failed on CPU. "
                "Check that `pip install ultralytics torch` works, the model path is valid, "
                "and images are readable.\n"
                f"Original error: {first}"
            ) from first
        if not _exception_looks_like_cuda_failure(first):
            raise RuntimeError(
                "YOLO inference failed. If this is a CUDA / driver issue, set "
                "`SKYDOCK_YOLO_FORCE_CPU=1` or fix your NVIDIA driver + PyTorch CUDA build.\n"
                f"Original error: {first}"
            ) from first
        print(
            f"[training] YOLO CUDA error; reloading model and retrying batch on CPU. ({first})",
            flush=True,
        )
        _yolo_model = None
        _yolo_model_path = None
        model = _get_yolo_model(model_path)
        kw = dict(predict_kw)
        kw["device"] = "cpu"
        kw.pop("half", None)
        results = model.predict(paths, **kw)
        return results, model, True


def _torch_module_device_str(model) -> str:
    """Best-effort PyTorch device string for an Ultralytics ``YOLO`` wrapper."""
    try:
        inner = model.model
        dev = next(inner.parameters()).device
        return str(dev)
    except (StopIteration, AttributeError):
        return "unknown"


def yolo_loaded_device(model_path: str) -> str:
    """PyTorch device for the cached YOLO model (call after at least one ``predict``)."""
    return _torch_module_device_str(_get_yolo_model(model_path))


def _get_yolo_model(model_path: str):
    global _yolo_model, _yolo_model_path
    if _yolo_model is None or _yolo_model_path != model_path:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. "
                "Run: pip install ultralytics"
            )
        if not Path(model_path).exists() and not is_ultralytics_hub_model(model_path):
            raise RuntimeError(f"YOLO model not found: {model_path}")
        _yolo_model = YOLO(model_path)
        _yolo_model_path = model_path
    return _yolo_model


def _nearest_by_ts(sorted_entries: list[tuple[int, Any]], ts_ns: int) -> Any | None:
    if not sorted_entries:
        return None
    keys = [e[0] for e in sorted_entries]
    idx = bisect_right(keys, ts_ns) - 1
    if idx < 0:
        idx = 0
    if idx + 1 < len(sorted_entries):
        if abs(sorted_entries[idx + 1][0] - ts_ns) < abs(sorted_entries[idx][0] - ts_ns):
            idx += 1
    return sorted_entries[idx][1]


def _build_tick_index(log_path: Path) -> list[tuple[int, dict]]:
    from services.mission_store import iter_events
    ticks: list[tuple[int, dict]] = []
    for ev in iter_events(log_path):
        if ev.get("event") != "fsm_tick":
            continue
        t = ev.get("time_ns")
        if t is not None:
            ticks.append((int(t), ev))
    ticks.sort(key=lambda x: x[0])
    return ticks


def count_training_frames(mission_dir: Path, stride: int = 1) -> int:
    """Number of numeric-stem JPEGs in ``mission_dir/frames/`` (same set as analysis)."""
    return len(collect_training_frame_files(mission_dir, stride=stride))


def collect_training_frame_files(
    mission_dir: Path,
    stride: int = 1,
) -> list[tuple[int, Path]]:
    """Sorted ``(timestamp_ns, path)`` for analyzable JPEGs.

    ``stride``: keep every Nth frame after time sort (``1`` = all). Speeds up analysis
    when you do not need every frame labeled.
    """
    frames_dir = mission_dir / "frames"
    if not frames_dir.is_dir():
        return []
    st = max(1, int(stride))
    frame_files: list[tuple[int, Path]] = []
    for f in frames_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg") and f.stem.isdigit():
            frame_files.append((int(f.stem), f))
    frame_files.sort(key=lambda x: x[0])
    if st > 1:
        frame_files = frame_files[::st]
    return frame_files


def reorder_training_frame_files_focus(
    frame_files: list[tuple[int, Path]],
    focus_timestamp_ns: int | None,
    *,
    radius: int = 40,
) -> list[tuple[int, Path]]:
    """Place a time window around the frame closest to ``focus_timestamp_ns`` first, then the rest.

    Used so streaming YOLO results populate the filmstrip near where the reviewer already is
    before filling in earlier/later mission segments. Order is unchanged when ``focus`` is None.
    """
    if focus_timestamp_ns is None or not frame_files:
        return list(frame_files)
    r = max(0, min(2000, int(radius)))
    ft = int(focus_timestamp_ns)
    stems = [t for t, _ in frame_files]
    anchor = min(range(len(stems)), key=lambda i: abs(stems[i] - ft))
    n = len(frame_files)
    lo = max(0, anchor - r)
    hi = min(n, anchor + r + 1)
    front = frame_files[lo:hi]
    rest = frame_files[:lo] + frame_files[hi:]
    return front + rest


def _training_batch_size() -> int:
    raw = os.environ.get("SKYDOCK_YOLO_STREAM_BATCH", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return 24


def training_runtime_diagnostics(
    model_path: str,
    batch_size: int | None = None,
    *,
    frame_stride: int = 1,
) -> dict[str, Any]:
    """Lightweight hints for why CPU/RAM look 'idle' (often GPU is doing YOLO)."""
    if batch_size is None:
        batch_size = _training_batch_size()
    out: dict[str, Any] = {
        "batch_size": batch_size,
        "frame_stride": max(1, int(frame_stride)),
        "model": Path(model_path).name,
        "pid": os.getpid(),
        "cuda_available": False,
        "mps_available": False,
        "half_env": _yolo_half_enabled(),
        "imgsz_env": os.environ.get("SKYDOCK_YOLO_IMGSZ", "").strip() or None,
        "force_cpu_env": (
            os.environ.get("SKYDOCK_YOLO_FORCE_CPU", "").strip().lower()
            in ("1", "true", "yes", "on")
        ),
    }
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["cuda_device"] = torch.cuda.get_device_name(0)
        if hasattr(torch.backends, "mps") and callable(getattr(torch.backends.mps, "is_available", None)):
            out["mps_available"] = bool(torch.backends.mps.is_available())
    except ImportError:
        out["torch_installed"] = False
    else:
        out["torch_installed"] = True
    return out


# ``latlon_to_pixel`` / training canvas use 640×640; Ultralytics ``xyxy`` is in the
# loaded JPEG's pixel space (often full camera resolution from the Pi pipeline).
CAMERA_MODEL_PX = 640


def _yolo_orig_hw(yolo_result: Any) -> tuple[int, int]:
    """``(height, width)`` of the image YOLO read (same space as ``boxes.xyxy``)."""
    sh = getattr(yolo_result, "orig_shape", None)
    if sh is not None and len(sh) >= 2:
        try:
            return int(sh[0]), int(sh[1])
        except (TypeError, ValueError):
            pass
    oi = getattr(yolo_result, "orig_img", None)
    if oi is not None and hasattr(oi, "shape") and len(getattr(oi, "shape", ())) >= 2:
        h, w = int(oi.shape[0]), int(oi.shape[1])
        if h > 0 and w > 0:
            return h, w
    return CAMERA_MODEL_PX, CAMERA_MODEL_PX


def _scale_xyxy_to_camera_model(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    orig_h: int,
    orig_w: int,
) -> tuple[float, float, float, float]:
    """Map detector coords to 640×640 to match ``utils.latlon_to_pixel`` and the UI canvas."""
    if orig_w <= 0 or orig_h <= 0:
        return x1, y1, x2, y2
    sx = CAMERA_MODEL_PX / float(orig_w)
    sy = CAMERA_MODEL_PX / float(orig_h)
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def _yolo_detection_class_and_label(yolo_result: Any, boxes: Any, i: int) -> tuple[int | None, str]:
    """Ultralytics class index and human-readable name (COCO etc.) for box ``i``."""
    cls_tensor = getattr(boxes, "cls", None)
    cls_i: int | None = None
    if cls_tensor is not None and len(cls_tensor) > i:
        cls_i = int(round(float(cls_tensor[i].item())))
    names = getattr(yolo_result, "names", None)
    label = ""
    if cls_i is not None and isinstance(names, dict):
        raw = names.get(cls_i)
        label = str(raw) if raw is not None else f"class_{cls_i}"
    elif cls_i is not None:
        label = f"class_{cls_i}"
    return cls_i, label


def yolo_dets_from_result(yolo_result: Any) -> list[dict]:
    """Convert one Ultralytics ``predict`` result to 640×640-space detection dicts."""
    yolo_dets: list[dict] = []
    boxes = yolo_result.boxes
    oh, ow = _yolo_orig_hw(yolo_result)
    if boxes is None or not len(boxes):
        return yolo_dets
    for i in range(len(boxes)):
        rx1, ry1, rx2, ry2 = boxes.xyxy[i].tolist()
        x1, y1, x2, y2 = _scale_xyxy_to_camera_model(
            float(rx1), float(ry1), float(rx2), float(ry2), oh, ow
        )
        x1, y1, x2, y2 = round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)
        conf = round(float(boxes.conf[i]), 4)
        cls_i, det_label = _yolo_detection_class_and_label(yolo_result, boxes, i)
        yolo_dets.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf,
            "cls": cls_i,
            "label": det_label,
            "cx": round((x1 + x2) / 2, 1),
            "cy": round((y1 + y2) / 2, 1),
        })
    return yolo_dets


_TRAINING_IGNORE_DET_LABELS = frozenset({"dog"})


def _strip_ignored_training_dets(dets: list[dict]) -> list[dict]:
    """Drop COCO labels that are never valid weeds here (common aerial false positives)."""
    ign = _TRAINING_IGNORE_DET_LABELS
    return [
        d
        for d in dets
        if str(d.get("label") or "").strip().lower() not in ign
    ]


def training_compare_default_model_specs() -> list[str]:
    """Deduped default + COCO preset hub/file names (same sets as the training UI)."""
    ui = training_yolo_models_for_ui()
    seen: set[str] = set()
    out: list[str] = []
    for m in (*ui["yolo_default_presets"], *ui["yolo_coco_presets"]):
        s = str(m).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def training_compare_max_models() -> int:
    raw = os.environ.get("SKYDOCK_TRAINING_COMPARE_MAX_MODELS", "").strip()
    if raw.isdigit():
        return max(1, min(int(raw), 48))
    return 24


def safe_training_frame_image_path(mission_dir: Path, frame_path_raw: str) -> Path | None:
    """Resolve ``frames/<digits>.jpg`` under ``mission_dir``; reject traversal."""
    raw = (frame_path_raw or "").strip().replace("\\", "/")
    if not raw or ".." in raw or raw.startswith("/"):
        return None
    p = Path(raw)
    if len(p.parts) != 2 or p.parts[0] != "frames":
        return None
    name = p.parts[1]
    if not name or Path(name).name != name:
        return None
    stem = Path(name).stem
    suf = Path(name).suffix.lower()
    if not stem.isdigit() or suf not in (".jpg", ".jpeg"):
        return None
    full = (mission_dir / "frames" / name).resolve()
    frames_resolved = (mission_dir / "frames").resolve()
    try:
        full.relative_to(frames_resolved)
    except ValueError:
        return None
    if not full.is_file():
        return None
    return full


def compare_yolo_models_on_image(
    image_path: Path,
    model_specs: list[str],
) -> list[dict[str, Any]]:
    """Run YOLO once per model on ``image_path``; same low conf as full training analysis.

    Each successful row includes ``predict_s``: wall seconds for that model's ``predict()``
    on this image (first model in a session may include weight load / CUDA warmup).
    """
    max_n = training_compare_max_models()
    specs = [str(s).strip() for s in model_specs if str(s).strip()][:max_n]
    predict_kw: dict[str, Any] = {
        "conf": 0.05,
        "verbose": False,
        "stream": False,
    }
    dev_kw = yolo_predict_device_kw()
    if dev_kw is not None:
        predict_kw["device"] = dev_kw
    imgsz = _yolo_predict_imgsz()
    if imgsz is not None:
        predict_kw["imgsz"] = imgsz
    workers = _yolo_predict_workers()
    if workers is not None:
        predict_kw["workers"] = workers

    rows: list[dict[str, Any]] = []
    cpu_fallback = False
    for spec in specs:
        try:
            mp = resolve_training_model_path(spec)
        except FileNotFoundError:
            rows.append({
                "model_spec": spec,
                "model_path": "",
                "ok": False,
                "error": "weights not found",
                "dets": [],
                "n_dets": 0,
                "inference_device": None,
                "predict_s": None,
            })
            continue
        try:
            kw_batch = dict(predict_kw)
            if _yolo_predict_use_half(cpu_fallback, predict_kw):
                kw_batch["half"] = True
            t0 = time.perf_counter()
            yolo_results, _, cpu_fallback = _yolo_predict_batch(
                mp,
                [str(image_path)],
                kw_batch,
                force_device_cpu=cpu_fallback,
            )
            predict_s = round(time.perf_counter() - t0, 4)
            r0 = yolo_results[0]
            dets = _strip_ignored_training_dets(yolo_dets_from_result(r0))
            rows.append({
                "model_spec": spec,
                "model_path": mp,
                "ok": True,
                "error": None,
                "dets": dets,
                "n_dets": len(dets),
                "inference_device": yolo_loaded_device(mp),
                "predict_s": predict_s,
            })
        except Exception as exc:
            rows.append({
                "model_spec": spec,
                "model_path": spec,
                "ok": False,
                "error": str(exc),
                "dets": [],
                "n_dets": 0,
                "inference_device": None,
                "predict_s": None,
            })
    return rows


def _process_one_training_frame(
    ts_ns: int,
    frame_path: Path,
    yolo_result: Any,
    ticks: list[tuple[int, dict]],
    weed_locations: list[dict],
    conf_thresh: float,
    dist_thresh: float,
) -> dict:
    from services.projection import drone_state_from_dict
    from utils import latlon_to_pixel

    ev = _nearest_by_ts(ticks, ts_ns)
    ds_dict = ev.get("drone_state") if ev else None
    ds = drone_state_from_dict(ds_dict)

    gps_pts: list[dict] = []
    if ds and ds.altitude_rel_home > 0:
        for weed in weed_locations:
            pt = latlon_to_pixel(ds, weed["lat"], weed["lon"], ts_ns)
            if pt is not None:
                gps_pts.append({
                    "weed_id": weed.get("id", 0),
                    "px": round(pt[0], 1),
                    "py": round(pt[1], 1),
                })

    yolo_dets = _strip_ignored_training_dets(yolo_dets_from_result(yolo_result))

    matches: list[dict] = []
    for gps in gps_pts:
        gx, gy = gps["px"], gps["py"]
        best_det: dict | None = None
        best_dist = float("inf")
        for det in yolo_dets:
            d = ((det["cx"] - gx) ** 2 + (det["cy"] - gy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_det = det

        if best_det is None:
            status = "no_det"
        elif best_det["conf"] >= conf_thresh and best_dist <= dist_thresh:
            status = "auto"
        else:
            status = "review"

        matches.append({
            "weed_id": gps["weed_id"],
            "gps_px": gx,
            "gps_py": gy,
            "yolo_bbox": best_det,
            "dist_px": round(best_dist, 1) if best_det else None,
            "status": status,
        })

    # "Auto" only if every in-view weed aligns (nearest YOLO within thresholds).
    # Partial overlap (some weeds auto, some not) -> review.
    if not matches:
        frame_status = "no_weed"
    elif all(m["status"] == "auto" for m in matches):
        frame_status = "auto"
    elif any(m["status"] == "review" for m in matches):
        frame_status = "review"
    elif any(m["status"] == "auto" for m in matches):
        frame_status = "review"
    else:
        frame_status = "no_det"

    return {
        "timestamp_ns": ts_ns,
        "frame_path": f"frames/{frame_path.name}",
        "drone_state": ds_dict,
        "matches": matches,
        "all_yolo_dets": yolo_dets,
        "status": frame_status,
    }


def iter_analyze_mission_batches(
    mission_dir: Path,
    log_path: Path,
    weed_locations: list[dict],
    model_path: str,
    conf_thresh: float,
    dist_thresh: float,
    batch_size: int | None = None,
    frame_stride: int = 1,
    focus_timestamp_ns: int | None = None,
    focus_radius: int = 40,
) -> Iterator[list[dict]]:
    """Run YOLO in chunks; yield frame result dicts per chunk (for streaming UI)."""
    if batch_size is None:
        batch_size = _training_batch_size()

    frame_files = collect_training_frame_files(mission_dir, stride=frame_stride)
    if not frame_files:
        return
    frame_files = reorder_training_frame_files_focus(
        frame_files,
        focus_timestamp_ns,
        radius=focus_radius,
    )

    ticks = _build_tick_index(log_path)
    model = _get_yolo_model(model_path)
    predict_kw: dict[str, Any] = {
        "conf": 0.05,
        "verbose": False,
        "stream": False,
    }
    dev_kw = yolo_predict_device_kw()
    if dev_kw is not None:
        predict_kw["device"] = dev_kw
    imgsz = _yolo_predict_imgsz()
    if imgsz is not None:
        predict_kw["imgsz"] = imgsz
    workers = _yolo_predict_workers()
    if workers is not None:
        predict_kw["workers"] = workers

    cpu_fallback = False
    for i in range(0, len(frame_files), batch_size):
        chunk = frame_files[i : i + batch_size]
        paths = [str(fp) for _, fp in chunk]
        kw_batch = dict(predict_kw)
        if _yolo_predict_use_half(cpu_fallback, predict_kw):
            kw_batch["half"] = True
        if os.environ.get("SKYDOCK_TRAINING_VERBOSE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            print(
                f"[training] YOLO predict batch {i // batch_size + 1} "
                f"({len(paths)} imgs) pid={os.getpid()} "
                f"device={'cpu' if cpu_fallback else predict_kw.get('device', 'auto')}"
                f"{' half' if kw_batch.get('half') else ''}",
                flush=True,
            )
        yolo_results, model, cpu_fallback = _yolo_predict_batch(
            model_path,
            paths,
            kw_batch,
            force_device_cpu=cpu_fallback,
        )
        batch_out: list[dict] = []
        for (ts_ns, frame_path), yolo_result in zip(chunk, yolo_results):
            batch_out.append(
                _process_one_training_frame(
                    ts_ns,
                    frame_path,
                    yolo_result,
                    ticks,
                    weed_locations,
                    conf_thresh,
                    dist_thresh,
                )
            )
        yield batch_out


_TRAINING_WEED_PROXY_LABELS = frozenset({"sports ball", "frisbee"})


def _weed_proxy_label(label: str) -> bool:
    """Training sort: only COCO weed-proxy class names (no substring match — avoids e.g. giraffe, baseball glove)."""
    s = (label or "").strip().lower()
    return s in _TRAINING_WEED_PROXY_LABELS


def frame_has_weed_proxy_detection(fr: dict) -> bool:
    for d in fr.get("all_yolo_dets") or []:
        if not isinstance(d, dict):
            continue
        if _weed_proxy_label(str(d.get("label") or "")):
            return True
    return False


def sort_training_frames_proxy_first(results: list[dict]) -> None:
    """Stable ordering: frames with a weed-proxy YOLO class first, then by ``timestamp_ns``."""
    results.sort(
        key=lambda fr: (
            0 if frame_has_weed_proxy_detection(fr) else 1,
            int(fr.get("timestamp_ns") or 0),
        )
    )


def analyze_mission(
    mission_dir: Path,
    log_path: Path,
    weed_locations: list[dict],
    model_path: str,
    conf_thresh: float = 0.6,
    dist_thresh: float = 80.0,
    frame_stride: int = 1,
    batch_size: int | None = None,
    focus_timestamp_ns: int | None = None,
    focus_radius: int = 40,
) -> tuple[list[dict], str]:
    """Run YOLO inference + GPS matching on all frames in mission_dir/frames/.

    Returns ``(frame_results, inference_device)`` where ``inference_device`` is the
    PyTorch device used after ``predict`` (e.g. ``cuda:0`` or ``cpu``).

    Each frame dict contains:
      timestamp_ns, frame_path, drone_state, matches, all_yolo_dets, status
    where status is one of: "auto", "review", "no_det", "no_weed".
    Frame ``auto`` means every projected weed in that frame matched YOLO within
    thresholds; if any weed is off, the frame is ``review`` (or ``no_det``).
    """
    output: list[dict] = []
    for batch in iter_analyze_mission_batches(
        mission_dir,
        log_path,
        weed_locations,
        model_path,
        conf_thresh,
        dist_thresh,
        frame_stride=frame_stride,
        batch_size=batch_size,
        focus_timestamp_ns=focus_timestamp_ns,
        focus_radius=focus_radius,
    ):
        output.extend(batch)
    if not output:
        return [], "n/a"
    sort_training_frames_proxy_first(output)
    inference_device = yolo_loaded_device(model_path)
    return output, inference_device


def save_labels(mission_dir: Path, approved: list[dict]) -> int:
    """Write YOLO .txt label files next to approved frames in mission_dir/frames/.

    approved: list of {timestamp_ns, yolo_bbox: {x1,y1,x2,y2}} (multiple rows per
    timestamp are merged into one multi-line label file).
    Returns count of label files written.
    """
    from collections import defaultdict

    IMG_SIZE = 640
    frames_dir = mission_dir / "frames"
    grouped: dict[int, list[dict]] = defaultdict(list)
    empty_ts: set[int] = set()
    for item in approved:
        ts = int(item["timestamp_ns"])
        bbox = item.get("yolo_bbox")
        if bbox is None:
            empty_ts.add(ts)
            continue
        grouped[ts].append(bbox)

    count = 0
    all_ts = sorted(set(grouped.keys()) | empty_ts)
    for ts in all_ts:
        bboxes = grouped.get(ts, [])
        if bboxes:
            seen: set[tuple[float, float, float, float]] = set()
            lines: list[str] = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                key = (x1, y1, x2, y2)
                if key in seen:
                    continue
                seen.add(key)
                cx_n = (x1 + x2) / 2 / IMG_SIZE
                cy_n = (y1 + y2) / 2 / IMG_SIZE
                w_n = (x2 - x1) / IMG_SIZE
                h_n = (y2 - y1) / IMG_SIZE
                lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
            label_path = frames_dir / f"{ts}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            count += 1
        elif ts in empty_ts:
            label_path = frames_dir / f"{ts}.txt"
            label_path.write_text("")
            count += 1
    return count


def write_training_metadata(
    mission_dir: Path,
    approved_ts: list[int],
    skipped_ts: list[int],
    thresholds: dict,
) -> None:
    """Write training_labels.json next to mission.jsonl."""
    meta = {
        "approved": approved_ts,
        "skipped": skipped_ts,
        "thresholds": thresholds,
        "date": datetime.datetime.utcnow().isoformat() + "Z",
        "approved_count": len(approved_ts),
        "skipped_count": len(skipped_ts),
    }
    (mission_dir / "training_labels.json").write_text(
        json.dumps(meta, indent=2)
    )


TRAINING_REVIEW_PROGRESS_NAME = "training_review_progress.json"


def save_review_progress(mission_dir: Path, data: dict[str, Any]) -> Path:
    """Persist training UI snapshot (raw YOLO dets, matches, approvals, manual boxes).

    Written next to ``mission.jsonl`` as ``training_review_progress.json``.
    """
    path = mission_dir / TRAINING_REVIEW_PROGRESS_NAME
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_review_progress(mission_dir: Path) -> dict[str, Any] | None:
    """Return parsed progress JSON, or ``None`` if missing."""
    path = mission_dir / TRAINING_REVIEW_PROGRESS_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_real_dataset(
    mission_ids: list[str],
    missions_root: Path,
    dest_root: Path | None = None,
) -> tuple[str, int]:
    """Copy frame/label pairs (``*.jpg`` + sibling ``*.txt``) into ``ai_train/real_data`` train/valid (90/10 by ``ts %% 10``).

    Filenames are ``m{mission_id}_{timestamp}.jpg`` to avoid collisions across missions.
    Returns ``(dest_root_str, files_copied)``.
    """
    import shutil

    if dest_root is None:
        dest_root = _repo_root() / "ai_train" / "real_data"
    train_d = dest_root / "train"
    valid_d = dest_root / "valid"
    train_d.mkdir(parents=True, exist_ok=True)
    valid_d.mkdir(parents=True, exist_ok=True)

    n = 0
    for mid in mission_ids:
        mid = str(mid).strip()
        if not mid.isdigit():
            continue
        frames = missions_root / mid / "frames"
        if not frames.is_dir():
            continue
        for img_path in sorted(frames.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg"):
                continue
            if not img_path.stem.isdigit():
                continue
            label_path = img_path.with_suffix(".txt")
            if not label_path.is_file():
                continue
            ts = int(img_path.stem)
            use_train = (ts % 10) < 9
            dest_sub = train_d if use_train else valid_d
            base = f"m{mid}_{ts}"
            ext = img_path.suffix.lower()
            shutil.copy2(img_path, dest_sub / f"{base}{ext}")
            shutil.copy2(label_path, dest_sub / f"{base}.txt")
            n += 1

    dest_root.mkdir(parents=True, exist_ok=True)
    yaml_text = (
        f"path: {dest_root.resolve()}\n"
        "train: train\n"
        "val: valid\n"
        "nc: 1\n"
        "names: ['ball']\n"
    )
    (dest_root / "data.yaml").write_text(yaml_text)
    return str(dest_root.resolve()), n
