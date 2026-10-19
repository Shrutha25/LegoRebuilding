"""The comparison report and the look-check calibration tool."""

import json

from backend import design_report as dr
from backend import look_calibration as lc


def run(kept, produced, stats, first=2000, last=4000):
    return {"designs_kept": kept, "designs_produced": produced, "first_design_ms": first, "last_design_ms": last, "done_ms": 9000, "gemini_calls": 2,
            "stats": {"concepts": 7, "sketch_ok": 5, "solver_ok": 3, "checks_ok": 3, "diverse_ok": 3, "look_checked": produced, "look_ok": kept, "look_ok_child": kept, "look_ok_choice": 1,
                      "concept_in_chars": 7000, "concept_out_chars": 3500, "look_in_chars": 1000, "look_out_chars": 200, **stats}}


def bench(runs):
    return {"photos": [{"photo": "bench_photos/photo1.jpg", "designs_runs": runs}, {"photo": "bench_photos/photo2.jpg", "designs_runs": [run(0, 0, {}, None, None)]}]}


def test_the_report_compares_models_on_shown_designs_pass_rates_latency_and_cost():
    a = dr.summarise("flash-lite", bench([run(2, 3, {"rescued": 1})]))
    b = dr.summarise("unpriced", bench([run(3, 3, {})]))
    assert a["shown_per_photo"] == {"photo1": 2, "photo2": 0} and a["rescued"] == 1 and a["look_ok_child"] == 2 and a["look_ok_choice"] == 2
    assert a["cost_usd_per_request"] > 0 and a["tokens_in_per_request"] > 0 and b["tokens_in_per_request"] > 0
    text = dr.render([a, b])
    for needle in ("flash-lite", "unpriced", "child question (new)", "pick-the-name question (old)", "est. cost per request", "designs shown, photo1", "Stage funnel", "rescued"):
        assert needle in text


def test_the_calibration_agreement_counts_kept_and_removed_against_the_labels(tmp_path):
    b = {"photos": [{"photo": "p/photo1.jpg", "designs_all": [
        {"label": "photo1#1", "title": "Duck", "removed": False, "verdict": {"ok": True, "ok_child": True, "ok_choice": True}},
        {"label": "photo1#2", "title": "Tree", "removed": True, "verdict": {"ok": False, "ok_child": False, "ok_choice": False}},
        {"label": "photo1#3", "title": "Boat", "removed": False, "verdict": {"ok": True, "ok_child": True, "ok_choice": False}}]}]}
    f = tmp_path / "labels.csv"
    lc.template(b, f)
    assert "photo1#2,Tree," in f.read_text(encoding="utf-8")
    f.write_text("label,title,recognisable\nphoto1#1,Duck,y\nphoto1#2,Tree,y\nphoto1#3,Boat,n\nphoto1#9,x,\n", encoding="utf-8")
    labels, ds = lc.load_labels(f), lc.designs(b)
    assert labels == {"photo1#1": True, "photo1#2": True, "photo1#3": False}
    child = lc.confusion(labels, {k: d["child"] for k, d in ds.items()})
    old = lc.confusion(labels, {k: d["choice"] for k, d in ds.items()})
    assert child["agreement"] == round(1 / 3, 3) and child["removed_recognisable"] == 1 and child["kept_unrecognisable"] == 1
    assert old["agreement"] == round(2 / 3, 3) and old["kept_unrecognisable"] == 0
    older = {"photos": [{"photo": "p/photo1.jpg", "designs_all": [{"title": "Duck", "removed": True}]}]}
    assert lc.designs(older)["photo1#1"] == {"title": "Duck", "kept": False, "child": None, "choice": False}
