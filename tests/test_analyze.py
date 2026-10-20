import numpy as np
from affine import Affine
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from terrasignal.analyze import backdrop, polygons
from terrasignal.baseline import FLOOD, NO_FLOOD, PROBABLE

# 10 m UTM 44N grid near Vijayawada
T = Affine(10, 0, 457000, 0, -10, 1831000)
EPSG = 32644


def test_one_feature_per_region_and_areas_add_up():
    c = np.full((200, 200), NO_FLOOD, np.uint8)
    c[10:60, 10:60] = FLOOD                      # 2500 px block
    for k in range(30):                           # diagonal-only chain: one 8-connected region
        c[100 + k, 100 + k] = PROBABLE
    c[150:160, 20:40] = PROBABLE                  # 200 px block
    conf = np.where(c == FLOOD, 0.8, 0.0).astype(np.float32)

    fc, n = polygons(c, conf, T, EPSG, 10.0)
    props = [f["properties"] for f in fc["features"]]
    assert n == 3 and len(props) == 3
    area = {p["class"]: 0.0 for p in props}
    for p in props:
        area[p["class"]] += p["area_km2"]
    assert abs(area["open_water_flood"] - 2500 * 100 / 1e6) < 1e-9
    assert abs(area["probable_flood"] - (30 + 200) * 100 / 1e6) < 1e-9
    assert [p.get("confidence") for p in props if p["class"] == "open_water_flood"] == [0.8]
    assert props[0]["area_km2"] >= props[-1]["area_km2"]  # sorted largest first


def test_polygons_are_wgs84_and_in_place():
    c = np.zeros((100, 100), np.uint8)
    c[20:40, 20:40] = FLOOD
    fc, _ = polygons(c, None, T, EPSG, 10.0)
    g = shape(fc["features"][0]["geometry"])
    lon, lat = g.centroid.x, g.centroid.y
    assert 80.4 < lon < 80.8 and 16.4 < lat < 16.7  # Vijayawada
    back = shp_transform(Transformer.from_crs(4326, EPSG, always_xy=True).transform, g)
    assert abs(back.area - 400 * 100) / 40000 < 0.05  # simplification keeps area within 5%


def test_backdrop_scaling():
    lin = 10 ** (np.array([[-25.0, 0.0, -12.5, np.nan]]) / 10)
    assert backdrop(lin).tolist() == [[0, 254, 127, 255]]


def test_analyze_registers_rio_accessor_on_its_own():
    # Regression: the API path (fetch -> analyze) never imported rioxarray,
    # so .rio was missing. Check in a fresh interpreter with only analyze imported.
    import subprocess
    import sys
    code = "import terrasignal.analyze, xarray as xr; xr.DataArray([[1.0]], dims=('y', 'x')).rio"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
