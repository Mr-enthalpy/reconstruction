"""The SNR summary must be stratified, and must not pretend to know per-case extremes."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "d5_3"))

import snr_strata  # noqa: E402


def record(pair, mask, strength, pixel, matched, fraction=0.0, cases=9):
    return dict(bands_nm=list(pair), mask=mask, strength=strength, cases=cases,
                mean_relative_chromatic=1e-4, max_achromatic_abs=1e-6,
                mean_analytic_relative_to_measurement=1e-7,
                mean_energy_snr=pixel, mean_matched_filter_snr=matched,
                mean_detectable_fraction=fraction)


def document():
    rows = []
    for strength in (0.25, 1.0):
        for pair, pixel, matched in ((( 450, 460), 0.004, 2.0),
                                     ((450, 650), 0.30, 700.0)):
            for mask in ("block", "stripes"):
                rows.append(record(pair, mask, strength, pixel * strength * 4,
                                   matched * strength * 4))
    return dict(summary=dict(per_configuration=rows), noise=dict(sigma_chromatic=0.017),
                case_count=len(rows) * 9)


def test_ranges_are_per_strength_not_one_collapsed_interval():
    result = snr_strata.analyse(document(), "x.json", "deadbeef")
    by_strength = result["strata"]["by_strength"]
    assert set(by_strength) == {"0.25", "1"}
    low = by_strength["0.25"]["matched_filter_snr"]
    high = by_strength["1"]["matched_filter_snr"]
    assert low["min"] == pytest.approx(2.0) and low["max"] == pytest.approx(700.0)
    assert high["max"] == pytest.approx(2800.0)          # collapses would hide this
    assert result["strata"]["overall"]["matched_filter_snr"]["max"] == pytest.approx(2800.0)
    assert by_strength["0.25"]["configurations"] == 4
    assert result["counts"] == dict(configurations=8, cases=72, band_pairs=2,
                                    masks=2, strengths=2)


def test_band_separation_buckets():
    assert snr_strata.separation(record((450, 460), "block", 0.25, .1, 2)) == 10
    assert snr_strata.bucket_of(record((450, 460), "block", 0.25, .1, 2)) == "adjacent (10 nm)"
    assert snr_strata.bucket_of(record((490, 510), "block", 0.25, .1, 2)) == "near (20-40 nm)"
    assert snr_strata.bucket_of(record((530, 570), "block", 0.25, .1, 2)) == "near (20-40 nm)"
    assert snr_strata.bucket_of(record((450, 650), "block", 0.25, .1, 2)) == "distant (>= 50 nm)"
    result = snr_strata.analyse(document(), "x.json", "deadbeef")
    bucket = result["strata"]["by_strength_and_separation"]["0.25"]["distant (>= 50 nm)"]
    assert bucket["matched_filter_snr"]["max"] == pytest.approx(700.0)
    merged = result["strata"]["by_separation"]["adjacent (10 nm)"]
    assert merged["configurations"] == 4
    assert merged["matched_filter_snr"]["max"] == pytest.approx(8.0)


def test_a_missing_separation_bucket_is_null():
    rows = [record((450, 460), "block", 0.25, .1, 2)]
    rows.append(record((450, 650), "block", 1.0, .1, 2))
    result = snr_strata.analyse(dict(summary=dict(per_configuration=rows), noise={}))
    buckets = result["strata"]["by_strength_and_separation"]["0.25"]
    assert buckets["adjacent (10 nm)"]["configurations"] == 1
    assert buckets["distant (>= 50 nm)"] is None


def test_output_says_the_unit_is_the_configuration_mean():
    result = snr_strata.analyse(document(), "x.json", "deadbeef")
    assert "configuration means" in result["unit"]
    assert any("per-case" in note for note in result["notes"])
    assert any("known" in note for note in result["notes"])
    assert result["source"] == dict(path="x.json", sha256="deadbeef")


def test_cli_writes_the_document(tmp_path):
    source = tmp_path / "encoding_observability.json"
    source.write_text(json.dumps(document()), encoding="utf-8")
    output = tmp_path / "snr_strata.json"
    sys.argv = ["snr_strata.py", "--input", str(source), "--output", str(output)]
    snr_strata.main()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["counts"]["cases"] == 72
    assert written["source"]["sha256"] == snr_strata.sha256_file(source)
    with pytest.raises(FileExistsError):
        snr_strata.main()
