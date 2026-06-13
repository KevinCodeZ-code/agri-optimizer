"""
Agro-economic analysis module.
Computes revenue, cost, and profit.
"""
import numpy as np


def compute_economics(area_ha, planting_density=None):
    """
    Compute economic analysis for the selected crop and area.
    
    Parameters:
        area_ha: Farm area in hectares
        planting_density: Plants per hectare (optional, for display)
    
    Returns: dict with revenue, cost, profit, and summary info
    """
    YIELD_T_HA = 7.8        # tonnes per hectare
    CROP_PRICE = 450.0      # USD per tonne
    PRODUCTION_COST = 1850.0  # USD per hectare

    total_yield = YIELD_T_HA * area_ha
    revenue = total_yield * CROP_PRICE
    total_cost = PRODUCTION_COST * area_ha
    profit = revenue - total_cost
    profit_margin = (profit / revenue * 100) if revenue > 0 else 0.0

    return {
        "area_ha": area_ha,
        "yield_t_ha": YIELD_T_HA,
        "total_yield_t": total_yield,
        "crop_price_usd": CROP_PRICE,
        "production_cost_usd_ha": PRODUCTION_COST,
        "total_revenue_usd": revenue,
        "total_cost_usd": total_cost,
        "profit_usd": profit,
        "profit_margin_pct": profit_margin,
        "planting_density": planting_density,
    }
