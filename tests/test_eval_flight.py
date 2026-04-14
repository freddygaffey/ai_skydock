"""
Tests for eval_flight.py — iou_box(), match_boxes(),
parse_yolo_label(), resolve_label_dir().

No YOLO / GPU required.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval_flight import iou_box, match_boxes, parse_yolo_label, resolve_label_dir


# ---------------------------------------------------------------------------
# iou_box
# ---------------------------------------------------------------------------

def test_iou_identical_boxes():
    box = [0.5, 0.5, 0.2, 0.2]
    assert iou_box(box, box) == pytest.approx(1.0)


def test_iou_non_overlapping():
    b1 = [0.1, 0.1, 0.1, 0.1]
    b2 = [0.9, 0.9, 0.1, 0.1]
    assert iou_box(b1, b2) == pytest.approx(0.0)


def test_iou_partial_overlap():
    # Two boxes that share exactly half their area
    # b1: x [0.0, 0.4], y [0.0, 0.4]   (cx=0.2, cy=0.2, w=0.4, h=0.4)
    # b2: x [0.2, 0.6], y [0.0, 0.4]   (cx=0.4, cy=0.2, w=0.4, h=0.4)
    b1 = [0.2, 0.2, 0.4, 0.4]
    b2 = [0.4, 0.2, 0.4, 0.4]
    iou = iou_box(b1, b2)
    # intersection = 0.2*0.4 = 0.08, union = 2*0.16 - 0.08 = 0.24
    assert iou == pytest.approx(0.08 / 0.24, rel=1e-4)


def test_iou_zero_area_box():
    b1 = [0.5, 0.5, 0.0, 0.0]
    b2 = [0.5, 0.5, 0.2, 0.2]
    assert iou_box(b1, b2) == pytest.approx(0.0)


def test_iou_symmetry():
    b1 = [0.3, 0.3, 0.2, 0.3]
    b2 = [0.4, 0.4, 0.3, 0.2]
    assert iou_box(b1, b2) == pytest.approx(iou_box(b2, b1))


# ---------------------------------------------------------------------------
# match_boxes
# ---------------------------------------------------------------------------

def test_match_perfect_single():
    box = [0.5, 0.5, 0.2, 0.2]
    tp, fp, fn, avg_iou = match_boxes([box], [box], iou_thresh=0.5)
    assert tp == 1
    assert fp == 0
    assert fn == 0
    assert avg_iou == pytest.approx(1.0)


def test_match_no_gt():
    preds = [[0.5, 0.5, 0.2, 0.2]]
    tp, fp, fn, avg_iou = match_boxes(preds, [], iou_thresh=0.5)
    assert tp == 0
    assert fp == 1
    assert fn == 0


def test_match_no_pred():
    gt = [[0.5, 0.5, 0.2, 0.2]]
    tp, fp, fn, avg_iou = match_boxes([], gt, iou_thresh=0.5)
    assert tp == 0
    assert fp == 0
    assert fn == 1


def test_match_both_empty():
    tp, fp, fn, avg_iou = match_boxes([], [], iou_thresh=0.5)
    assert tp == fp == fn == 0


def test_match_iou_below_threshold():
    # Non-overlapping boxes → FP + FN
    b1 = [0.1, 0.1, 0.1, 0.1]
    b2 = [0.9, 0.9, 0.1, 0.1]
    tp, fp, fn, _ = match_boxes([b1], [b2], iou_thresh=0.5)
    assert tp == 0
    assert fp == 1
    assert fn == 1


def test_match_greedy_one_pred_two_gt():
    # One pred close to gt1, gt2 is far → TP=1, FP=0, FN=1
    pred = [0.1, 0.1, 0.1, 0.1]
    gt1  = [0.1, 0.1, 0.1, 0.1]
    gt2  = [0.9, 0.9, 0.1, 0.1]
    tp, fp, fn, _ = match_boxes([pred], [gt1, gt2], iou_thresh=0.5)
    assert tp == 1
    assert fp == 0
    assert fn == 1


def test_match_two_preds_two_gt_perfect():
    b1 = [0.2, 0.2, 0.1, 0.1]
    b2 = [0.8, 0.8, 0.1, 0.1]
    tp, fp, fn, avg_iou = match_boxes([b1, b2], [b1, b2], iou_thresh=0.5)
    assert tp == 2
    assert fp == 0
    assert fn == 0
    assert avg_iou == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# parse_yolo_label
# ---------------------------------------------------------------------------

def test_parse_yolo_label_normal(tmp_path):
    lbl = tmp_path / "frame.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.1\n0 0.2 0.3 0.05 0.05\n")
    boxes = parse_yolo_label(lbl)
    assert len(boxes) == 2
    assert boxes[0] == pytest.approx([0.5, 0.5, 0.1, 0.1])


def test_parse_yolo_label_filters_class(tmp_path):
    lbl = tmp_path / "frame.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    boxes = parse_yolo_label(lbl)
    assert len(boxes) == 1  # class 1 filtered out


def test_parse_yolo_label_empty_file(tmp_path):
    lbl = tmp_path / "frame.txt"
    lbl.write_text("")
    assert parse_yolo_label(lbl) == []


def test_parse_yolo_label_missing_file(tmp_path):
    assert parse_yolo_label(tmp_path / "nonexistent.txt") == []


def test_parse_yolo_label_ignores_blank_lines(tmp_path):
    lbl = tmp_path / "frame.txt"
    lbl.write_text("\n0 0.5 0.5 0.1 0.1\n\n")
    assert len(parse_yolo_label(lbl)) == 1


# ---------------------------------------------------------------------------
# resolve_label_dir
# ---------------------------------------------------------------------------

def test_resolve_prefers_truth(tmp_path):
    flight = tmp_path / "flight01"
    truth = flight / "labels_truth"
    auto  = flight / "labels_auto_v001"
    truth.mkdir(parents=True)
    auto.mkdir()
    assert resolve_label_dir(flight, None) == truth


def test_resolve_falls_back_to_highest_auto(tmp_path):
    flight = tmp_path / "flight01"
    (flight / "labels_auto_v001").mkdir(parents=True)
    (flight / "labels_auto_v003").mkdir()
    (flight / "labels_auto_v002").mkdir()
    result = resolve_label_dir(flight, None)
    assert result.name == "labels_auto_v003"


def test_resolve_explicit_hint(tmp_path):
    flight = tmp_path / "flight01"
    truth = flight / "labels_truth"
    custom = flight / "labels_auto_v002"
    truth.mkdir(parents=True)
    custom.mkdir()
    result = resolve_label_dir(flight, "labels_auto_v002")
    assert result == custom


def test_resolve_explicit_hint_missing(tmp_path):
    flight = tmp_path / "flight01"
    flight.mkdir(parents=True)
    assert resolve_label_dir(flight, "labels_truth") is None


def test_resolve_no_labels(tmp_path):
    flight = tmp_path / "flight01"
    flight.mkdir(parents=True)
    assert resolve_label_dir(flight, None) is None
