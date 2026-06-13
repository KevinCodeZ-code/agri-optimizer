"""
Terrain analysis module.
Loads SRTM DEM, computes slope, and calculates surface area.
"""
import os
import numpy as np
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEM_PATH = os.path.join(DATA_DIR, "Uganda_SRTM30meters.tif")
DEM_URL = "https://raw.githubusercontent.com/KevinCodeZ-code/agri-optimizer/main/data/Uganda_SRTM30meters.tif"


@st.cache_data(max_entries=5, ttl=3600)
def load_dem(clip_bounds_utm=None):
    """Load DEM raster, optionally clipped to UTM bounds (minx, miny, maxx, maxy).
    Returns (elevation, bounds, crs, transform) or None.
    """
    if not os.path.exists(DEM_PATH):
        try:
            import urllib.request
            os.makedirs(DATA_DIR, exist_ok=True)
            with st.spinner("Downloading DEM data (~13MB)..."):
                urllib.request.urlretrieve(DEM_URL, DEM_PATH)
        except Exception:
            return _generate_synthetic_dem("DEM file not found.")
    try:
        import rasterio
        from rasterio.windows import from_bounds
        BUFFER = 150
        with rasterio.open(DEM_PATH) as src:
            if clip_bounds_utm:
                minx, miny, maxx, maxy = clip_bounds_utm
                minx -= BUFFER
                miny -= BUFFER
                maxx += BUFFER
                maxy += BUFFER
                dem_minx, dem_miny, dem_maxx, dem_maxy = src.bounds
                minx = max(minx, dem_minx)
                miny = max(miny, dem_miny)
                maxx = min(maxx, dem_maxx)
                maxy = min(maxy, dem_maxy)
                if minx >= maxx or miny >= maxy:
                    return _generate_synthetic_dem("Field bounds do not overlap with DEM coverage.")
                window = from_bounds(minx, miny, maxx, maxy, src.transform)
                elevation = src.read(1, window=window).astype(np.float64)
                elevation[elevation == src.nodata] = np.nan
                clipped_transform = src.window_transform(window)
                clipped_bounds = src.window_bounds(window)
                return elevation, clipped_bounds, src.crs, clipped_transform
            elevation = src.read(1).astype(np.float64)
            elevation[elevation == src.nodata] = np.nan
            return elevation, src.bounds, src.crs, src.transform
    except Exception as e:
        return _generate_synthetic_dem(f"DEM error: {e}")


def _generate_synthetic_dem(reason=None):
    """Generate a synthetic 100x100 DEM as fallback."""
    if reason:
        st.info(f"{reason} Using generated terrain data for preview.")
    np.random.seed(42)
    x = np.linspace(0, 3000, 100)
    y = np.linspace(0, 3000, 100)
    xx, yy = np.meshgrid(x, y)
    elevation = (1100
                 + 30 * np.sin(xx / 500) * np.cos(yy / 400)
                 + 15 * np.sin(xx / 200 + yy / 300)
                 + 2 * np.random.randn(100, 100))
    from rasterio.coords import BoundingBox
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    bounds = BoundingBox(0, 0, 3000, 3000)
    crs = CRS.from_epsg(32636)
    transform = from_origin(0, 3000, 30, 30)
    return elevation, bounds, crs, transform


def compute_slope(elevation, transform):
    """Compute slope in radians from DEM using gradient (matching MATLAB gradient)."""
    if elevation is None:
        return None
    cell_size = abs(transform[0]) if transform else 30.0
    dy, dx = np.gradient(elevation, cell_size, cell_size)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return slope


def compute_slope_stats(slope):
    """Return mean, max, min slope in degrees."""
    if slope is None or np.all(np.isnan(slope)):
        return 0.0, 0.0, 0.0
    valid = slope[~np.isnan(slope)]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0
    mean_deg = float(np.degrees(np.mean(valid)))
    max_deg = float(np.degrees(np.max(valid)))
    min_deg = float(np.degrees(np.min(valid)))
    return mean_deg, max_deg, min_deg


def compute_surface_area(area_2d, mean_slope_rad):
    """Compute 3D surface area from 2D area and mean slope."""
    if mean_slope_rad is None or mean_slope_rad == 0:
        return area_2d
    return area_2d / np.cos(mean_slope_rad)
