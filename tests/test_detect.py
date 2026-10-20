import numpy as np

from terrasignal.baseline import FLOOD, NO_FLOOD, PROBABLE
from terrasignal.detect import DetectParams, run_model
from test_infer import DarkIsWater

# Stand-in boundary at -15 dB VV (logit = -4x - 2 with x = (dB - mean) / std),
# so -12 dB flooded vegetation is "not open water", as with the real U-Net.
META = {"mean": [-14.0, -21.0], "std": [2.0, 2.0]}

RNG = np.random.default_rng(1)


def lin(db, shape):
    return (10 ** (np.asarray(db) / 10) * RNG.gamma(4.4, 1 / 4.4, size=shape)).astype(np.float32)


def test_hybrid_separates_open_water_and_probable_flood():
    H = W = 512
    vv_b_db = np.full((H, W), -8.0); vh_b_db = np.full((H, W), -15.0)
    vv_a_db, vh_a_db = vv_b_db.copy(), vh_b_db.copy()
    vv_a_db[50:200, 50:200], vh_a_db[50:200, 50:200] = -22.0, -27.0      # open water
    vv_a_db[300:450, 300:450], vh_a_db[300:450, 300:450] = -12.0, -21.0  # flooded vegetation

    model = DarkIsWater().eval()
    zeros = np.zeros((H, W), np.float32)
    classes, stats, _ = run_model(
        model, META, lin(vv_b_db, (H, W)), lin(vh_b_db, (H, W)), lin(vv_a_db, (H, W)),
        lin(vh_a_db, (H, W)), zeros, zeros.astype(np.uint8), 10.0, DetectParams(),
    )
    # the stand-in model sees raw speckle pixel by pixel, so a few holes are expected
    assert (classes[70:180, 70:180] == FLOOD).mean() > 0.9
    assert (classes[320:430, 320:430] == PROBABLE).mean() > 0.9
    assert (classes[0:40, 250:500] == NO_FLOOD).mean() > 0.99
    assert stats["probable_flood_km2"] > 1.5 and stats["flood_km2"] > 1.5

    classes_off, stats_off, _ = run_model(
        model, META, lin(vv_b_db, (H, W)), lin(vh_b_db, (H, W)), lin(vv_a_db, (H, W)),
        lin(vh_a_db, (H, W)), zeros, zeros.astype(np.uint8), 10.0, DetectParams(change=False),
    )
    assert stats_off["probable_flood_km2"] == 0 and not (classes_off == PROBABLE).any()
