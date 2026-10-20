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


def test_export_round_trip_is_exact(tmp_path):
    from terrasignal.model import export_checkpoint, load_checkpoint, save_checkpoint
    torch.manual_seed(0)
    m = build_model("resnet18", weights=None, dropout=0.2).eval()
    meta = {"encoder": "resnet18", "dropout": 0.2, "mean": [-10.0, -17.0], "std": [4.0, 4.5], "temperature": 1.5}
    save_checkpoint(tmp_path / "m.pt", m, meta)
    export_checkpoint(tmp_path / "m.pt", tmp_path / "export")
    assert sorted(p.name for p in (tmp_path / "export").iterdir()) == ["config.json", "model.safetensors"]
    m2, meta2 = load_checkpoint(tmp_path / "export")
    x = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))
    assert meta2 == meta


def test_hf_reference_parsing(monkeypatch):
    import huggingface_hub
    from terrasignal import model as tm
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, fn, subfolder=None: calls.append((repo, fn, subfolder)) or f"/cache/{fn}")
    tm._resolve("hf://someone/terrasignal-water/gee_baseline")
    tm._resolve("hf://someone/terrasignal-water")
    assert calls[0] == ("someone/terrasignal-water", "config.json", "gee_baseline")
    assert calls[2] == ("someone/terrasignal-water", "config.json", None)
