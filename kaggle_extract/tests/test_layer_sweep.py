import math

from evaluation.layer_sweep import summarize_sweep, _frac

FLOOR = {"det_action": 0.0, "H_action": 2.26, "mlp_z_op": 0.37}
CEILING = {"det_action": 0.59, "H_action": 0.39, "mlp_z_op": 0.77}


def test_frac_basic_and_degenerate():
    assert _frac(0.0, 0.6, 0.3) == 0.5
    # Entropy axis: floor high, ceiling low -> fraction of the way *down*.
    assert _frac(2.26, 0.39, 0.39) == 1.0
    assert math.isnan(_frac(0.5, 0.5, 0.5))  # coincident anchors -> NaN, no crash


def test_best_layer_is_most_deterministic():
    rows = [
        {"layer": 4, "det_action": 0.05, "H_action": 2.0, "mlp_z_op": 0.40},
        {"layer": 12, "det_action": 0.31, "H_action": 1.1, "mlp_z_op": 0.60},
        {"layer": 22, "det_action": 0.11, "H_action": 1.6, "mlp_z_op": 0.55},
    ]
    s = summarize_sweep(rows, FLOOR, CEILING)
    assert s["best_layer"] == 12
    # Layer 12 det 0.31 over a 0->0.59 axis ~ 53% of the way up.
    assert abs(s["best_pos_det"] - 0.31 / 0.59) < 1e-9


def test_reaches_ceiling_when_a_layer_is_deterministic():
    rows = [
        {"layer": 8, "det_action": 0.10, "H_action": 1.9, "mlp_z_op": 0.45},
        {"layer": 16, "det_action": 0.45, "H_action": 0.6, "mlp_z_op": 0.71},
    ]
    s = summarize_sweep(rows, FLOOR, CEILING)
    assert s["best_layer"] == 16
    assert s["reaches_ceiling"] is True          # det 0.45 >= 0.20 and >= 50% of ceiling


def test_no_layer_reaches_ceiling_near_floor():
    rows = [
        {"layer": 4, "det_action": 0.02, "H_action": 2.2, "mlp_z_op": 0.39},
        {"layer": 12, "det_action": 0.11, "H_action": 1.9, "mlp_z_op": 0.52},
        {"layer": 22, "det_action": 0.09, "H_action": 2.0, "mlp_z_op": 0.50},
    ]
    s = summarize_sweep(rows, FLOOR, CEILING)
    assert s["best_layer"] == 12
    assert s["reaches_ceiling"] is False         # 0.11 < 0.20 bar
    assert len(s["per_layer"]) == 3


def test_tie_broken_by_mlp():
    rows = [
        {"layer": 6, "det_action": 0.20, "H_action": 1.4, "mlp_z_op": 0.58},
        {"layer": 18, "det_action": 0.20, "H_action": 1.2, "mlp_z_op": 0.66},
    ]
    s = summarize_sweep(rows, FLOOR, CEILING)
    assert s["best_layer"] == 18                 # equal det, higher MLP wins
