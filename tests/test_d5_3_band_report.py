import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "d5"))
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import band_report  # noqa: E402
import train as d5_train  # noqa: E402


def write_validation(path, scenes, bands=3, best_epoch=7):
    """scenes: {scene_id: (dataset_id, [band_relative_l2], [band_rmse])}"""
    rows = [dict(scene_id=scene, dataset_id=dataset,
                 metrics=dict(band_relative_l2=list(rel), band_rmse=list(rmse)))
            for scene, (dataset, rel, rmse) in scenes.items()]
    path.write_text(json.dumps(dict(best_epoch=best_epoch, results=rows)), encoding="utf-8")
    return path


def test_band_labels_are_the_21_named_bands():
    labels = band_report.band_labels()
    assert labels[0] == 450 and labels[-1] == 650 and len(labels) == 21
    assert labels[10] == 550
    with pytest.raises(ValueError):
        band_report.band_labels(450, 650, 20)


def test_scene_mean_is_a_mean_of_ratios_not_a_ratio_of_norms(tmp_path):
    # Truth/error pairs would pool to 13/11 = 1.1818, but the recorded per-scene
    # ratios are 1.0 and 3.0, so the scene mean must be 2.0.
    path = write_validation(tmp_path / "validation.json", {
        "a": ("CAVE", [1.0, 2.0], [0.5, 0.5]),
        "b": ("KAIST", [3.0, 4.0], [0.5, 0.5]),
    })
    loaded = band_report.load_scenes(path, 2)
    summary = band_report.summarise(loaded["scenes"], "arm")
    assert summary["ALL"]["band_relative_l2"]["scene_mean"] == [2.0, 3.0]
    assert summary["ALL"]["band_relative_l2"]["scene_mean"] != pytest.approx([13 / 11, 0])
    assert summary["ALL"]["band_relative_l2"]["scene_sd"] == pytest.approx([np.std([1.0, 3.0], ddof=1),
                                                                           np.std([2.0, 4.0], ddof=1)])
    assert summary["ALL"]["band_relative_l2"]["scenes"] == 2


def test_dataset_blocks_are_separate(tmp_path):
    path = write_validation(tmp_path / "validation.json", {
        "a": ("CAVE", [1.0], [0.1]),
        "b": ("CAVE", [3.0], [0.1]),
        "c": ("KAIST", [9.0], [0.1]),
    })
    loaded = band_report.load_scenes(path, 1)
    summary = band_report.summarise(loaded["scenes"], "arm")
    assert summary["by_dataset"]["CAVE"]["band_relative_l2"]["scene_mean"] == [2.0]
    assert summary["by_dataset"]["KAIST"]["band_relative_l2"]["scene_mean"] == [9.0]
    assert summary["ALL"]["band_relative_l2"]["scene_mean"] == [13 / 3]


def test_load_scenes_rejects_bad_rows(tmp_path):
    bad_length = write_validation(tmp_path / "short.json", {"a": ("CAVE", [1.0, 2.0], [0.1, 0.1])})
    with pytest.raises(ValueError, match="expected 3"):
        band_report.load_scenes(bad_length, 3)

    nonfinite = write_validation(tmp_path / "nan.json", {"a": ("CAVE", [1.0, np.nan, 1.0], [0.1] * 3)})
    with pytest.raises(ValueError, match="non-finite"):
        band_report.load_scenes(nonfinite, 3)

    negative = write_validation(tmp_path / "neg.json", {"a": ("CAVE", [1.0, -1.0, 1.0], [0.1] * 3)})
    with pytest.raises(ValueError, match="negative"):
        band_report.load_scenes(negative, 3)

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(dict(results=[dict(scene_id="a", dataset_id="CAVE",
                                                     metrics=dict(band_rmse=[0.1]))])),
                       encoding="utf-8")
    with pytest.raises(ValueError, match="band_relative_l2"):
        band_report.load_scenes(missing, 1)

    duplicate = tmp_path / "dup.json"
    row = dict(scene_id="a", dataset_id="CAVE",
               metrics=dict(band_relative_l2=[1.0], band_rmse=[0.1]))
    duplicate.write_text(json.dumps(dict(results=[row, row])), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate scene_id"):
        band_report.load_scenes(duplicate, 1)


def test_paired_delta_is_aligned_by_scene_id_not_file_order(tmp_path):
    reference = write_validation(tmp_path / "ref.json", {
        "a": ("CAVE", [1.0, 1.0], [0.1, 0.1]),
        "b": ("KAIST", [1.0, 1.0], [0.1, 0.1]),
    })
    # Same scenes written in the opposite order.
    condition = write_validation(tmp_path / "con.json", {
        "b": ("KAIST", [3.0, 5.0], [0.1, 0.1]),
        "a": ("CAVE", [2.0, 4.0], [0.1, 0.1]),
    })
    ref = dict(name="ref", scenes=["a", "b"],
               _scenes=band_report.load_scenes(reference, 2)["scenes"])
    con = dict(name="con", scenes=["a", "b"],
               _scenes=band_report.load_scenes(condition, 2)["scenes"])
    result = band_report.paired(con, ref)
    assert result["delta"]["band_relative_l2"]["delta_mean"] == [1.5, 3.5]
    assert result["scenes"] == 2


def test_paired_refuses_mismatched_scene_sets(tmp_path):
    reference = write_validation(tmp_path / "ref.json", {"a": ("CAVE", [1.0], [0.1]),
                                                        "b": ("CAVE", [1.0], [0.1])})
    condition = write_validation(tmp_path / "con.json", {"a": ("CAVE", [2.0], [0.2])})
    ref = dict(name="ref", scenes=["a", "b"],
               _scenes=band_report.load_scenes(reference, 1)["scenes"])
    con = dict(name="con", scenes=["a"],
               _scenes=band_report.load_scenes(condition, 1)["scenes"])
    with pytest.raises(ValueError, match="scene sets differ"):
        band_report.paired(con, ref)


def test_delta_sign_conventions_and_agreement(tmp_path):
    # condition is lower on band 0 in every scene, higher on band 1 in every scene,
    # mixed on band 2.
    reference = write_validation(tmp_path / "ref.json", {
        s: ("CAVE", [1.0, 1.0, 1.0], [0.1, 0.1, 0.1]) for s in ("a", "b", "c")})
    condition = write_validation(tmp_path / "con.json", {
        "a": ("CAVE", [0.5, 1.5, 0.5], [0.1] * 3),
        "b": ("CAVE", [0.4, 1.4, 1.5], [0.1] * 3),
        "c": ("CAVE", [0.6, 1.6, 0.5], [0.1] * 3),
    })
    ref = dict(name="ref", scenes=["a", "b", "c"],
               _scenes=band_report.load_scenes(reference, 3)["scenes"])
    con = dict(name="con", scenes=["a", "b", "c"],
               _scenes=band_report.load_scenes(condition, 3)["scenes"])
    block = band_report.paired(con, ref)["delta"]["band_relative_l2"]
    assert block["scenes_delta_negative"] == [3, 0, 2]
    assert block["scenes_delta_positive"] == [0, 3, 1]
    assert block["full_agreement"] == [True, True, False]
    assert block["delta_mean"][0] < 0 and block["delta_mean"][1] > 0
    assert block["scenes_with_negative_scene_mean"] == 2


def test_cli_writes_definitions_and_deltas(tmp_path):
    rng = np.random.default_rng(0)
    scenes_ref = {f"s{i}": ("CAVE", rng.random(21) * 0.2 + 0.2, rng.random(21) * 0.1)
                  for i in range(3)}
    scenes_con = {k: (v[0], (v[1] - 0.01).tolist(), v[2]) for k, v in scenes_ref.items()}
    ref = write_validation(tmp_path / "ref.json", scenes_ref)
    con = write_validation(tmp_path / "con.json", scenes_con)

    sys.argv = ["band_report.py", "--condition", f"chromatic={con}",
                "--condition", f"achromatic={ref}",
                "--output", str(tmp_path / "out.json")]
    band_report.main()
    document = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert document["bands_nm"][0] == 450 and len(document["bands_nm"]) == 21
    assert "mean of per-scene ratios" in document["definitions"]["scene_mean"]
    assert "negative delta means the condition is better" in document["definitions"]["sign"]
    assert [c["name"] for c in document["conditions"]] == ["chromatic", "achromatic"]
    assert document["deltas"][0]["condition"] == "achromatic"
    assert document["deltas"][0]["reference"] == "chromatic"
    assert "_scenes" not in json.dumps(document)
    assert document["conditions"][0]["provenance"]["scenes"] == 3


def test_cli_refuses_existing_output_and_bad_condition(tmp_path):
    path = write_validation(tmp_path / "ref.json", {"a": ("CAVE", [1.0] * 21, [0.1] * 21)})
    existing = tmp_path / "out.json"
    existing.write_text("{}", encoding="utf-8")
    sys.argv = ["band_report.py", "--condition", f"x={path}", "--output", str(existing)]
    with pytest.raises(FileExistsError):
        band_report.main()
    sys.argv = ["band_report.py", "--condition", str(path), "--output", str(tmp_path / "y.json")]
    with pytest.raises(ValueError, match="name=path"):
        band_report.main()


def test_compare_recorded_metrics_handles_band_lists():
    recorded = dict(psnr_unit_range_db=30.0,
                    band_relative_l2_mean=[0.1, 0.2, 0.3])
    recomputed = dict(psnr_unit_range_db=30.5,
                      band_relative_l2_mean=np.array([0.1, 0.25, 0.3]))
    differences = d5_train.compare_recorded_metrics(recorded, recomputed)
    assert differences["psnr_unit_range_db"] == pytest.approx(0.5)
    assert differences["band_relative_l2_mean"] == pytest.approx(0.05)


def test_compare_recorded_metrics_is_infinite_when_shapes_differ():
    differences = d5_train.compare_recorded_metrics(
        dict(band_rmse_mean=[0.1, 0.2]), dict(band_rmse_mean=np.array([0.1])))
    assert differences["band_rmse_mean"] == float("inf")


def test_compare_recorded_metrics_ignores_absent_keys():
    differences = d5_train.compare_recorded_metrics(dict(a=1.0), dict(a=1.0, b=[1.0]))
    assert set(differences) == {"a"}
