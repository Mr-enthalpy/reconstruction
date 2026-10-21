import numpy as np
import pytest

from lcd_reconstruction.datasets.general_scenes import (
    CAVE_DEVELOPMENT_TEST, CAVE_RESERVED_FINAL, CAVE_TEST, CAVE_VALIDATION,
    KAIST_TEST_IDS, KAIST_TRAIN_IDS, KAIST_VAL_IDS, WAVELENGTHS,
    cave_group, cave_split, interp_28_to_21,
)


def test_cave_split_keeps_prior_groups_and_reserves_final():
    assert CAVE_DEVELOPMENT_TEST <= {s for s in CAVE_TEST | CAVE_VALIDATION |
                                     CAVE_DEVELOPMENT_TEST if cave_split(s) == "train"}
    assert CAVE_VALIDATION == {s for s in CAVE_TEST | CAVE_VALIDATION |
                               CAVE_DEVELOPMENT_TEST if cave_split(s) == "val"}
    assert CAVE_RESERVED_FINAL <= CAVE_TEST
    assert "watercolors" in CAVE_TEST
    assert cave_split("face") == cave_split("photo_and_face") == "train"
    assert cave_group("face") == cave_group("photo_and_face") == "face_family"
    assert cave_group("chart_and_stuffed_toy") == cave_group("stuffed_toys") == "toy_chart_family"


def test_kaist_ids_are_disjoint_and_complete():
    train, val, test = set(KAIST_TRAIN_IDS), set(KAIST_VAL_IDS), set(KAIST_TEST_IDS)
    assert len(train) == 16 and len(val) == 4 and len(test) == 10
    assert not train & val and not train & test and not val & test
    assert train | val | test == set(range(1, 31))
    assert 23 in test and 24 in test


def test_28_to_21_interpolation_preserves_declared_endpoints_and_ordering():
    source = np.linspace(0.0, 1.0, 28, dtype=np.float32)[None, None, :]
    out = interp_28_to_21(np.repeat(source, 3, axis=0))[0, 0]
    assert out.shape == (21,)
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(out) > 0)
    assert out[10] == pytest.approx(0.5, abs=1e-5)


def test_official_band_assumption_is_declared():
    assert len(WAVELENGTHS) == 21
    assert WAVELENGTHS[0] == 450 and WAVELENGTHS[-1] == 650

