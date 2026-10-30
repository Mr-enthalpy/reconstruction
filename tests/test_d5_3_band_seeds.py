import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "d5"))
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import band_seeds  # noqa: E402
import band_report  # noqa: E402
import summarize_seeds  # noqa: E402

SCENES = [f"s{i}" for i in range(1, 10)]


def write_run(root, arm, seed, ratios, dataset="CAVE", scenes=None):
    """ratios: {scene_id: band_relative_l2 list}; band_rmse is derived from it."""
    scenes = scenes or SCENES
    directory = Path(root) / f"{arm}_s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [dict(scene_id=scene, dataset_id=dataset,
                 metrics=dict(band_relative_l2=list(ratios[scene]),
                              band_rmse=[value / 10.0 for value in ratios[scene]]))
            for scene in scenes]
    (directory / "validation.json").write_text(
        json.dumps(dict(best_epoch=7, results=rows)), encoding="utf-8")
    return directory


def paired_pair(root, seed, deltas, scenes=SCENES, base=0.30):
    """chromatic = base, achromatic = base + delta, so the paired delta is -delta."""
    if isinstance(deltas, float):
        per_scene = {scene: deltas for scene in scenes}
    else:
        per_scene = deltas
    write_run(root, "s1c", seed, {scene: [base] for scene in scenes})
    write_run(root, "s3a", seed, {scene: [base + per_scene[scene]] for scene in scenes})


def run_document(root, bands=1):
    runs = band_seeds.discover(Path(root), bands)
    labels = band_seeds.labels_from_argument(str(bands))
    return band_seeds.summarise(runs, "s1c", "s3a", labels)


def test_labels_from_argument_accepts_count_or_span():
    assert band_seeds.labels_from_argument("21") == list(range(450, 651, 10))
    assert band_seeds.labels_from_argument("450:650:10") == list(range(450, 651, 10))
    assert band_seeds.labels_from_argument("1") == [450]
    with pytest.raises(ValueError):
        band_seeds.labels_from_argument("0")


def test_band_is_stable_only_when_all_three_clauses_hold(tmp_path):
    for seed, delta in zip((1, 2, 3), (0.02, 0.03, 0.025)):
        paired_pair(tmp_path, seed, delta)
    document = run_document(tmp_path)
    record = document["bands"][0]["metrics"]["band_relative_l2"]
    assert record["delta_mean"] == pytest.approx(-0.025)
    assert record["delta_sd_across_seeds"] == pytest.approx(np.std([0.02, 0.03, 0.025], ddof=1))
    assert record["scenes_favouring_chromatic"] == [9, 9, 9]
    assert record["seeds_with_negative_delta"] == 3
    assert record["delta_mean"] < 0
    assert record["stable"] is True
    assert record["unstable_reasons"] == []
    assert record["plateau_available"] is False
    assert document["summary"]["band_relative_l2"]["bands_stable"] == 1


def test_magnitude_clause_rejects_delta_below_cross_seed_sd(tmp_path):
    for seed, delta in zip((1, 2, 3), (0.002, -0.001, 0.003)):
        paired_pair(tmp_path, seed, delta)
    record = run_document(tmp_path)["bands"][0]["metrics"]["band_relative_l2"]
    assert record["exceeds_cross_seed_sd"] is False
    assert record["stable"] is False
    assert "|delta_mean| does not exceed the cross-seed sd" in record["unstable_reasons"]


def test_scene_clause_rejects_a_seed_with_fewer_than_eight_scenes(tmp_path):
    paired_pair(tmp_path, 1, 0.02)
    paired_pair(tmp_path, 2, 0.02)
    # Third seed: only six of nine scenes favour the chromatic arm.
    per_scene = {scene: (0.02 if index < 6 else -0.02) for index, scene in enumerate(SCENES)}
    paired_pair(tmp_path, 3, per_scene)
    record = run_document(tmp_path)["bands"][0]["metrics"]["band_relative_l2"]
    assert record["scenes_favouring_chromatic"] == [9, 9, 6]
    assert record["exceeds_cross_seed_sd"] is True
    assert record["seeds_with_negative_delta"] == 3
    assert record["stable"] is False
    assert record["unstable_reasons"] == [
        "a seed has fewer than 8/9 scenes in that direction"]


def test_sign_clause_rejects_a_reversed_seed(tmp_path):
    for seed, delta in zip((1, 2, 3), (0.02, -0.02, 0.02)):
        paired_pair(tmp_path, seed, delta)
    record = run_document(tmp_path)["bands"][0]["metrics"]["band_relative_l2"]
    assert record["seeds_with_negative_delta"] == 2
    assert "seed signs disagree" in record["unstable_reasons"]
    assert record["stable"] is False


def test_delta_is_paired_by_scene_id_not_file_order(tmp_path):
    # Same scenes, written in reverse order with the same values.
    forward = {scene: 0.01 * (index + 1) for index, scene in enumerate(SCENES)}
    backward = {scene: forward[scene] for scene in reversed(SCENES)}
    paired_pair(tmp_path, 1, forward)
    paired_pair(tmp_path, 2, backward)
    paired_pair(tmp_path, 3, forward)
    document = run_document(tmp_path)
    per_seed = [pair["metrics"]["band_relative_l2"]["mean_over_bands"]
                for pair in document["pairs_per_seed"]]
    assert per_seed == pytest.approx([-np.mean(list(forward.values()))] * 3)


def test_bands_are_independent_of_each_other(tmp_path):
    # One stable band, one band where only six scenes move, in each seed.
    for seed in (1, 2, 3):
        write_run(tmp_path, "s1c", seed,
                  {scene: [0.30, 0.30] for scene in SCENES})
        write_run(tmp_path, "s3a", seed,
                  {scene: [0.33, 0.33 if index < 6 else 0.27]
                   for index, scene in enumerate(SCENES)})
    document = run_document(tmp_path, bands=2)
    stable = [band["metrics"]["band_relative_l2"]["stable"] for band in document["bands"]]
    assert stable == [True, False]
    assert document["summary"]["band_relative_l2"]["bands_stable"] == 1
    assert document["summary"]["band_relative_l2"]["bands_unstable"] == [460]


def test_mismatched_scene_sets_are_refused(tmp_path):
    write_run(tmp_path, "s1c", 1, {scene: [0.3] for scene in SCENES})
    write_run(tmp_path, "s3a", 1, {scene: [0.3] for scene in SCENES[:-1]},
              scenes=SCENES[:-1])
    with pytest.raises(ValueError, match="scene sets differ"):
        run_document(tmp_path)


def test_unpaired_seed_is_reported_not_pooled(tmp_path):
    paired_pair(tmp_path, 1, 0.02)
    write_run(tmp_path, "s1c", 2, {scene: [0.3] for scene in SCENES})
    document = run_document(tmp_path)
    assert document["seeds"] == [1]
    assert "only_chromatic=[2]" in document["notes"][1]


def test_missing_arm_is_an_error(tmp_path):
    write_run(tmp_path, "s1c", 1, {scene: [0.3] for scene in SCENES})
    with pytest.raises(ValueError, match="no runs for arm"):
        run_document(tmp_path)


def test_budget_censored_is_copied_from_multiseed(tmp_path):
    paired_pair(tmp_path, 1, 0.02)
    multiseed = tmp_path / "multiseed.json"
    multiseed.write_text(json.dumps(dict(runs={
        "s1c_s1": dict(budget_censored=False),
        "s3a_s1": dict(budget_censored=True)})), encoding="utf-8")
    runs = band_seeds.discover(tmp_path, 1)
    document = band_seeds.summarise(runs, "s1c", "s3a", [450], multiseed)
    assert document["runs"]["s1c_s1"]["budget_censored"] is False
    assert document["runs"]["s3a_s1"]["budget_censored"] is True
    assert document["runs"]["s1c_s1"]["sha256"] == band_report.sha256_file(
        tmp_path / "s1c_s1" / "validation.json")


def test_cli_writes_constants_and_refuses_existing_output(tmp_path):
    paired_pair(tmp_path, 1, 0.02)
    output = tmp_path / "out.json"
    sys.argv = ["band_seeds.py", "--runs", str(tmp_path), "--output", str(output),
                "--bands", "1"]
    band_seeds.main()
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["bands_nm"] == [450]
    assert document["arms"] == dict(chromatic="s1c", achromatic="s3a")
    assert "mean of per-scene ratios" in document["definitions"]["scene_mean"]
    assert "n = 1" in document["definitions"]["unit"]
    sys.argv = ["band_seeds.py", "--runs", str(tmp_path), "--output", str(output),
                "--bands", "1"]
    with pytest.raises(FileExistsError):
        band_seeds.main()


def test_summarize_seeds_per_band_curve_keeps_the_dimension():
    rows = [dict(metrics=dict(band_relative_l2=[0.2, 0.4], band_rmse=[0.02, 0.04])),
            dict(metrics=dict(band_relative_l2=[0.4, 0.6], band_rmse=[0.04, 0.06]))]
    curve = summarize_seeds.per_band_curve(rows)
    assert curve["band_relative_l2"]["scene_mean"] == pytest.approx([0.3, 0.5])
    assert curve["band_relative_l2"]["scene_sd"] == pytest.approx(
        [np.std([0.2, 0.4], ddof=1), np.std([0.4, 0.6], ddof=1)])
    assert curve["band_rmse"]["scene_mean"] == pytest.approx([0.03, 0.05])


def test_summarize_seeds_analyse_emits_per_band(tmp_path):
    keys = summarize_seeds.KEYS

    def run_dir(arm, seed, scale):
        directory = tmp_path / f"{arm}_s{seed}"
        directory.mkdir()
        rows = [dict(scene_id=f"s{i}", dataset_id="CAVE",
                     metrics=dict({key: 1.0 for key in keys},
                                  band_relative_l2=[scale, scale * 2],
                                  band_rmse=[scale / 10, scale / 5]))
                for i in range(3)]
        aggregate = {key: 1.0 for key in keys}
        (directory / "validation.json").write_text(json.dumps(dict(
            best_epoch=5, best_validation_loss=0.5, results=rows,
            selection=dict(rule="min_validation_loss", best_epoch=5,
                           metrics_in_checkpoint={source: dict(aggregate)
                                                  for source in ("CAVE", "KAIST", "ALL")})),
            ), encoding="utf-8")
        history = [dict(epoch=i + 1, global_step=(i + 1) * 10, validation_loss=1.0 / (i + 1),
                        stale=0, metrics={source: dict(aggregate)
                                          for source in ("CAVE", "KAIST", "ALL")})
                   for i in range(5)]
        (directory / "history.json").write_text(json.dumps(history), encoding="utf-8")
        (directory / "execution.json").write_text(json.dumps(dict(
            config=dict(seed=seed, patience=60), kernel_sha256_array="abc",
            measured_forward_rms=1.0, train_noise_std_absolute=0.01)), encoding="utf-8")

    run_dir("s1c", 1, 0.2)
    run_dir("s3a", 1, 0.3)
    document = summarize_seeds.analyse(tmp_path)
    assert document["runs"]["s1c_s1"]["per_band"]["band_relative_l2"]["scene_mean"] == [
        pytest.approx(0.2), pytest.approx(0.4)]
    pair = document["pairs"][0]["per_band"]["band_relative_l2"]
    assert pair["delta"] == pytest.approx([-0.1, -0.2])
    assert document["runs"]["s1c_s1"]["metrics"]["psnr_unit_range_db"] == pytest.approx(1.0)
