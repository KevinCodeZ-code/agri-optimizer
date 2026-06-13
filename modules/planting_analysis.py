"""
Planting analysis module.
Generates planting grid and computes plant density.
"""
import numpy as np
import streamlit as st


def generate_planting_grid(polygon_coords, plant_spacing, row_spacing, area_ha=None):
    """
    Generate planting grid within a polygon.
    
    Equivalent MATLAB logic:
        x_vals = min_x : plant_spacing : max_x
        y_vals = min_y : row_spacing : max_y
        meshgrid -> keep only points inside polygon
    
    Returns: (points_array, n_plants, density_per_ha)
    """
    if not polygon_coords or len(polygon_coords) < 3:
        return np.array([]), 0, 0.0

    from shapely.geometry import Polygon, Point
    polygon = Polygon(polygon_coords)
    if not polygon.is_valid or polygon.is_empty:
        return np.array([]), 0, 0.0

    min_x, min_y, max_x, max_y = polygon.bounds

    # Generate grid
    plant_spacing = max(plant_spacing, 0.01)
    row_spacing = max(row_spacing, 0.01)
    x_vals = np.arange(min_x, max_x + plant_spacing, plant_spacing)
    y_vals = np.arange(min_y, max_y + row_spacing, row_spacing)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Filter points inside polygon
    points = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            pt = Point(xx[i, j], yy[i, j])
            if polygon.contains(pt) or polygon.touches(pt):
                points.append((xx[i, j], yy[i, j]))

    points_array = np.array(points)
    n_plants = len(points_array)

    # Compute density
    if area_ha and area_ha > 0:
        density = n_plants / area_ha
    else:
        area_ha_calc = polygon.area / 10000.0 if polygon.area > 0 else 1.0
        density = n_plants / area_ha_calc if area_ha_calc > 0 else 0.0

    return points_array, n_plants, density
