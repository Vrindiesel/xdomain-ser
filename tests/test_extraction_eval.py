# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0.
"""End-to-end regression test for extraction.eval's SER aggregation.

Guards the Counter micro-average against shape changes in
``compute_ser``'s return value: the dict-valued ``errors`` slot detail
(added 2026-06) must stay out of the ``Counter.update`` projection, or
the eval crashes with ``dict + dict`` -- which is exactly how
reproduce_table4.sh failed on 2026-06-12.
"""
import json
import subprocess
import sys

FIXTURE = [
    {
        "mr": [["a.slot", "x"], ["b.slot", "y"]],
        "pred_mr": "<LIST>(a.slot: x)",
        "surface_form": "text one",
        "hint_map_id": "topic_a",
    },
    {
        "mr": [["a.slot", "x"]],
        "pred_mr": ["<LIST>(a.slot: x)", "<LIST>(a.slot: z)"],
        "pred_scores": [1.0, 2.0],
        "surface_form": "text two",
        "hint_map_id": "topic_a",
    },
]


def _run(tmp_path, rank_method=None):
    inp = tmp_path / "preds.json"
    out = tmp_path / "scores.json"
    inp.write_text(json.dumps(FIXTURE))
    cmd = [sys.executable, "-m", "xdomain_ser.extraction.eval", str(inp),
           "--save_path", str(out)]
    if rank_method:
        cmd += ["--rank_method", rank_method]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(out.read_text())[str(inp)]


def test_eval_micro_average_counts(tmp_path):
    res = _run(tmp_path)
    # ex1 drops b.slot (D=1, N_ref=2); ex2's top-of-list pick is exact
    # (N_ref=1) -> micro SER = 1/3.
    assert res["ser_score"] == {"SER": 1 / 3, "D": 1, "I": 0, "S": 0, "N": 3}


def test_eval_rank_method_scores(tmp_path):
    res = _run(tmp_path, rank_method="scores")
    # score-ranked pick for ex2 is the higher-scored wrong candidate
    # (a.slot: z), so it contributes one substitution.
    assert res["ser_score"]["S"] == 1
    assert res["ser_score"]["D"] == 1
