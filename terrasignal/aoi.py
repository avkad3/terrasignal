"""Area of interest: a lon/lat bounding box anywhere on Earth, validated and
assigned a local metric CRS (UTM) for processing."""
from __future__ import annotations

from dataclasses import dataclass

from pyproj import Geod

DEFAULT_MAX_AREA_KM2 = 2500.0
_GEOD = Geod(ellps="WGS84")


class AOIError(ValueError):
    pass


@dataclass(frozen=True)
class AOI:
    west: float
    south: float
    east: float
    north: float

    def __post_init__(self):
        if not (-180 <= self.west < self.east <= 180):
            raise AOIError(f"invalid longitudes: west={self.west}, east={self.east}")
        if not (-90 <= self.south < self.north <= 90):
            raise AOIError(f"invalid latitudes: south={self.south}, north={self.north}")
        if abs(self.south) > 78 or abs(self.north) > 78:
            # Sentinel-1 IW coverage over land is sparse near the poles and
            # UTM breaks down; keep the scope honest.
            raise AOIError("latitudes beyond ±78° are not supported")

    @classmethod
    def from_list(cls, bbox) -> "AOI":
        if len(bbox) != 4:
            raise AOIError("bbox must be [west, south, east, north]")
        return cls(*map(float, bbox))

    @property
    def bbox(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    @property
    def area_km2(self) -> float:
        lons = [self.west, self.east, self.east, self.west]
        lats = [self.south, self.south, self.north, self.north]
        area, _ = _GEOD.polygon_area_perimeter(lons, lats)
        return abs(area) / 1e6

    @property
    def utm_epsg(self) -> int:
        lon = (self.west + self.east) / 2
        lat = (self.south + self.north) / 2
        zone = int((lon + 180) // 6) + 1
        return (32600 if lat >= 0 else 32700) + min(zone, 60)

    def check_size(self, max_area_km2: float = DEFAULT_MAX_AREA_KM2) -> None:
        a = self.area_km2
        if a > max_area_km2:
            raise AOIError(
                f"AOI is {a:.0f} km², above the {max_area_km2:.0f} km² limit; draw a smaller box"
            )
