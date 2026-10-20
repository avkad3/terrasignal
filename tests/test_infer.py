import numpy as np
import torch
from torch import nn

from terrasignal.infer import _tiles, water_probability

META = {"mean": [-10.0, -17.0], "std": [4.0, 4.0]}


class DarkIsWater(nn.Module):
    """Stand-in model: logit rises as normalized VV falls, plus a dropout
    layer so MC sampling has something to sample."""

    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        return self.drop(-4 * x[:, :1] - 2)


def test_tiles_cover_every_pixel():
    for H, W in [(512, 512), (700, 1300), (1001, 1096)]:
        cov = np.zeros((H, W), int)
        for y, x in _tiles(H, W, 512, 384):
            cov[y:y + 512, x:x + 512] += 1
        assert cov.min() >= 1


def test_water_probability_shapes_nans_and_values():
    H, W = 700, 900  # not a multiple of the tile, forces edge tiles
    vv = np.full((H, W), 10 ** (-8 / 10), np.float32)
    vv[:, :300] = 10 ** (-22 / 10)            # water on the left
    vh = vv * 0.3
    vv[5, 5] = np.nan
    prob, std = water_probability(DarkIsWater(), META, vv, vh, device="cpu")
    assert prob.shape == (H, W) and std is None
    assert np.isnan(prob[5, 5])
    assert np.nanmean(prob[:, :250]) > 0.9 and np.nanmean(prob[:, 400:]) < 0.1


def test_small_scene_is_padded():
    vv = np.full((100, 130), 10 ** (-20 / 10), np.float32)
    prob, _ = water_probability(DarkIsWater(), META, vv, vv * 0.3, device="cpu")
    assert prob.shape == (100, 130) and np.nanmean(prob) > 0.9


def test_mc_dropout_gives_spread_and_restores_eval():
    torch.manual_seed(0)
    m = DarkIsWater()
    vv = np.full((256, 256), 10 ** (-16 / 10), np.float32)  # logit != 0, so dropout matters
    _, std = water_probability(m, META, vv, vv * 0.3, mc_samples=8, device="cpu")
    assert std is not None and np.nanmean(std) > 0.01
    assert not m.training and not m.drop.training
