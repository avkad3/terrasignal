import torch

from terrasignal.model import build_model, logit_samples


def test_split_forward_matches_full_forward_and_mc_varies():
    torch.manual_seed(0)
    m = build_model("resnet18", weights=None, dropout=0.3).eval()
    x = torch.randn(2, 2, 64, 64)
    with torch.no_grad():
        full = m(x)[:, 0]
        det = logit_samples(m, x)
        mc = logit_samples(m, x, mc_samples=6)
        hot = logit_samples(m, x, temperature=2.0)
    assert det.shape == (1, 2, 64, 64) and torch.allclose(det[0], full, atol=1e-5)
    assert mc.shape == (6, 2, 64, 64) and mc.std(0).mean() > 1e-3
    assert torch.allclose(hot[0], full / 2, atol=1e-5)
    assert not any(mod.training for mod in m.modules())


def test_shift_nan_matches_roll_without_wraparound():
    import numpy as np
    import sys
    sys.path.insert(0, "train")
    from s1f11 import shift_nan
    x = np.arange(2 * 6 * 7, dtype=np.float32).reshape(2, 6, 7)
    for dy, dx in [(1, 1), (0, 3), (-2, 1), (2, -3)]:
        s = shift_nan(x, dy, dx)
        r = np.roll(np.roll(x, dy, 1), dx, 2)
        ok = np.isfinite(s)
        assert ok.sum() == 2 * (6 - abs(dy)) * (7 - abs(dx))
        assert np.array_equal(s[ok], r[ok])
