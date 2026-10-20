import numpy as np
import pytest

from terrasignal.baseline import (
    FALLBACK_THRESHOLD_DB, FLOOD, MASKED, NODATA, PERMANENT, remove_small, run_baseline, split_otsu,
)
from terrasignal.preprocess import lee_filter, slope_degrees

RNG = np.random.default_rng(0)
LAND_DB, WATER_DB = -8.0, -21.0


def speckled(db_mean: np.ndarray) -> np.ndarray:
    """Linear gamma0 with gamma-distributed speckle (ENL ~4.4)."""
    lin = 10 ** (db_mean / 10)
    return (lin * RNG.gamma(4.4, 1 / 4.4, size=lin.shape)).astype(np.float32)


def scene(water: np.ndarray) -> np.ndarray:
    return speckled(np.where(water, WATER_DB, LAND_DB))


def test_lee_reduces_speckle_and_keeps_nans():
    img = speckled(np.full((128, 128), LAND_DB))
    img[10, 10] = np.nan
    out = lee_filter(img)
    assert np.isnan(out[10, 10]) and np.isfinite(out[11, 11])
    assert np.nanstd(out) < 0.6 * np.nanstd(img)


def test_split_otsu_finds_water_threshold():
    water = np.zeros((512, 512), bool)
    water[:, :200] = True
    db = 10 * np.log10(lee_filter(scene(water)))
    t = split_otsu(db)
    assert t.method == "tiles" and t.n_tiles >= 1
    assert WATER_DB < t.value_db < LAND_DB


def test_split_otsu_falls_back_without_water():
    db = 10 * np.log10(lee_filter(scene(np.zeros((512, 512), bool))))
    t = split_otsu(db)
    assert t.method == "fallback" and t.value_db == FALLBACK_THRESHOLD_DB


def test_slope():
    dem = np.tile(np.arange(100, dtype=np.float32) * 10 * np.tan(np.radians(10)), (50, 1))
    assert np.nanmedian(slope_degrees(dem, 10.0)) == pytest.approx(10, abs=0.1)


def test_buildings_in_surface_model_are_not_steep():
    # 15 m tall 30x30 m blocks on flat ground: a DSM "city"
    dem = np.zeros((200, 200), np.float32)
    for y in range(10, 190, 8):
        for x in range(10, 190, 8):
            dem[y:y + 3, x:x + 3] = 15.0
    assert (slope_degrees(dem, 10.0, smooth_m=0) > 5).mean() > 0.2
    assert (slope_degrees(dem, 10.0) > 5).mean() < 0.05


def test_remove_small():
    m = np.zeros((50, 50), bool)
    m[1:3, 1:3] = True        # 4 px
    m[20:30, 20:30] = True    # 100 px
    out = remove_small(m, 50)
    assert not out[1:3, 1:3].any() and out[20:30, 20:30].all()


def test_flood_logic_end_to_end():
    H = W = 512
    river = np.zeros((H, W), bool)
    river[:, 240:270] = True                     # permanent river, water both dates
    flood = np.zeros((H, W), bool)
    flood[100:300, 50:230] = True                # new water after the event
    hill = np.zeros((H, W), bool)
    hill[400:480, 350:450] = True                # dark radar shadow on a steep slope

    before = scene(river | hill)
    after = scene(river | flood | hill)
    dem = np.zeros((H, W), np.float32)
    dem[400:480, 350:450] = np.arange(100) * 5.0  # ~26° slope
    jrc = np.zeros((H, W), np.uint8)
    jrc[river] = 90
    after[0, 0] = np.nan

    classes, stats = run_baseline(before, after, dem, jrc, 10.0)

    assert (classes[150:250, 80:200] == FLOOD).mean() > 0.95
    assert (classes[river] == FLOOD).mean() < 0.01
    assert (classes[river] == PERMANENT).mean() > 0.9
    assert (classes[410:470, 360:440] == FLOOD).sum() == 0
    assert (classes[410:470, 360:440] == MASKED).mean() > 0.9
    assert classes[0, 0] == NODATA
    assert stats["flood_km2"] == pytest.approx(200 * 180 * 100 / 1e6, rel=0.1)


def test_no_event_means_no_flood():
    # Control: unchanged land should produce (almost) nothing.
    land = np.zeros((512, 512), bool)
    classes, stats = run_baseline(scene(land), scene(land), np.zeros((512, 512), np.float32),
                                  np.zeros((512, 512), np.uint8), 10.0)
    assert stats["flood_km2"] < 0.01
