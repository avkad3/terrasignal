from datetime import date

import pytest

from terrasignal.aoi import AOI, AOIError
from terrasignal.fetch import NoPairError, Pass, pick_pair


def P(state, orbit, d, cov=1.0):
    return Pass(state, orbit, date.fromisoformat(d), coverage=cov)


def test_area_and_utm():
    aoi = AOI(92.10, 26.25, 92.50, 26.55)
    assert 1200 < aoi.area_km2 < 1450
    assert aoi.utm_epsg == 32646
    assert AOI(-58.5, -34.7, -58.3, -34.5).utm_epsg == 32721  # Buenos Aires, south


def test_area_cap_and_bad_boxes():
    with pytest.raises(AOIError, match="limit"):
        AOI(60, 20, 70, 30).check_size()
    with pytest.raises(AOIError):
        AOI(10, 5, 9, 6)
    with pytest.raises(AOIError):
        AOI(10, 80, 11, 81)


def test_pair_requires_same_geometry_and_order():
    passes = [
        P("descending", 63, "2024-06-28"),
        P("ascending", 12, "2024-07-01"),   # nearest to before, wrong geometry
        P("descending", 63, "2024-07-10"),  # after
        P("ascending", 12, "2024-07-13"),
    ]
    b, a = pick_pair(passes, date(2024, 7, 1), date(2024, 7, 10))
    assert (a.day, b.day) == (date(2024, 7, 10), date(2024, 6, 28))
    assert a.key == b.key


def test_pair_skips_low_coverage_and_falls_back():
    passes = [
        P("descending", 63, "2024-07-10", cov=0.4),  # nearest after but clipped
        P("descending", 63, "2024-06-28"),
        P("ascending", 12, "2024-07-13"),
        P("ascending", 12, "2024-07-01"),
    ]
    b, a = pick_pair(passes, date(2024, 7, 1), date(2024, 7, 10))
    assert a.key == ("ascending", 12) and b.day == date(2024, 7, 1)


def test_no_pair():
    with pytest.raises(NoPairError):
        pick_pair([P("descending", 63, "2024-07-10")], date(2024, 7, 1), date(2024, 7, 10))
