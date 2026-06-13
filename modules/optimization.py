"""
Optimization module.
Transforms environmental parameters and crop database into actionable recommendations.
"""
import numpy as np
import pandas as pd


def temperature_score(tmean, tmin, topt, tmax):
    """Score temperature suitability 0-1 using triangular membership."""
    if tmean <= tmin or tmean >= tmax:
        return 0.0
    if tmean <= topt:
        return (tmean - tmin) / (topt - tmin)
    return (tmax - tmean) / (tmax - topt)


def rainfall_score(total_rain, rain_min, rain_opt):
    """Score rainfall suitability 0-1."""
    if total_rain < rain_min * 0.5:
        return 0.0
    if total_rain < rain_min:
        return (total_rain - rain_min * 0.5) / (rain_min * 0.5)
    if total_rain <= rain_opt * 1.3:
        return 1.0
    if total_rain <= rain_opt * 2.0:
        return 1.0 - (total_rain - rain_opt * 1.3) / (rain_opt * 0.7)
    return 0.3


def slope_score(mean_slope_deg):
    """Score slope suitability 0-1 (flatter is better for most crops)."""
    if mean_slope_deg <= 5:
        return 1.0
    if mean_slope_deg <= 15:
        return 1.0 - (mean_slope_deg - 5) / 10 * 0.4
    if mean_slope_deg <= 30:
        return 0.6 - (mean_slope_deg - 15) / 15 * 0.4
    return max(0.1, 0.2 - (mean_slope_deg - 30) / 30 * 0.1)


WEIGHTS = {"temperature": 0.35, "rainfall": 0.35, "slope": 0.15, "kc_match": 0.15}


def compute_suitability(crop_row, avg_tmean, total_rain, mean_slope_deg, avg_kc_actual):
    """Compute overall suitability score (0-100) for a single crop."""
    t_score = temperature_score(
        avg_tmean, crop_row["Tmin_C"], crop_row["Topt_C"], crop_row["Tmax_C"]
    )
    r_score = rainfall_score(total_rain, crop_row["Rain_Min_mm"], crop_row["Rain_Opt_mm"])
    s_score = slope_score(mean_slope_deg)

    # Kc match: how well the crop's typical Kc matches the environmental demand
    kc_crop = crop_row.get("Avg_Kc", 1.0)
    if avg_kc_actual > 0:
        kc_ratio = min(kc_crop, avg_kc_actual) / max(kc_crop, avg_kc_actual, 0.01)
    else:
        kc_ratio = 0.5
    kc_score = kc_ratio

    overall = (
        WEIGHTS["temperature"] * t_score
        + WEIGHTS["rainfall"] * r_score
        + WEIGHTS["slope"] * s_score
        + WEIGHTS["kc_match"] * kc_score
    )
    return min(overall * 100, 100), {
        "temperature": round(t_score * 100, 1),
        "rainfall": round(r_score * 100, 1),
        "slope": round(s_score * 100, 1),
        "kc_match": round(kc_score * 100, 1),
    }


def rank_crops(
    crop_db,
    avg_tmean,
    total_rain,
    mean_slope_deg,
    avg_kc_actual,
    min_score=10,
):
    """
    Rank all crops by suitability.
    Returns list of dicts sorted descending.
    """
    id_col = "Crop_ID" if "Crop_ID" in crop_db.columns else "Crop_Number"
    results = []
    for _, row in crop_db.iterrows():
        score, breakdown = compute_suitability(
            row, avg_tmean, total_rain, mean_slope_deg, avg_kc_actual
        )
        if score >= min_score:
            results.append({
                "crop_id": int(row[id_col]),
                "name": row["Common_Name"],
                "score": round(score, 1),
                **breakdown,
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def estimate_yield(crop_row, suitability_score, total_cwr, total_etc):
    """
    Estimate expected yield in tonnes/ha.
    Uses yield potential adjusted by suitability and water availability.
    """
    base_yield = crop_row["Yield_Potential_t_ha"]
    # Water limitation factor
    if total_cwr > 0 and total_etc > 0:
        water_factor = min(1.0, (total_etc - total_cwr) / total_etc + 0.3)
    else:
        water_factor = 1.0
    water_factor = max(0.2, min(1.0, water_factor))
    suitability_factor = suitability_score / 100.0
    expected = base_yield * suitability_factor * water_factor
    return round(expected, 2)


def compute_water_allocation(total_cwr, field_area_ha, irrigation_days):
    """Compute optimal water allocation."""
    if irrigation_days == 0:
        irrigation_days = 1
    total_volume_m3 = (total_cwr / 1000.0) * field_area_ha * 10000
    per_event_m3 = total_volume_m3 / irrigation_days if irrigation_days > 0 else 0
    return {
        "total_cwr_mm": round(total_cwr, 1),
        "total_volume_m3": round(total_volume_m3, 0),
        "per_event_m3": round(per_event_m3, 0),
        "irrigation_days": irrigation_days,
    }


def compute_wue(expected_yield_t_ha, total_etc_mm):
    """
    Water Use Efficiency: crop production per unit water consumed.
    WUE = Yield (kg/ha) / ETc (mm)  -> kg/m³
    """
    yield_kg_ha = expected_yield_t_ha * 1000
    etc_m3_ha = total_etc_mm * 10  # 1 mm = 10 m³/ha
    wue = yield_kg_ha / etc_m3_ha if etc_m3_ha > 0 else 0
    return round(wue, 2)


def compute_iwue(expected_yield_t_ha, total_irrigation_mm):
    """
    Irrigation Water Use Efficiency: crop production per unit irrigation water.
    IWUE = Yield (kg/ha) / Irrigation (mm) -> kg/m³
    """
    yield_kg_ha = expected_yield_t_ha * 1000
    irr_m3_ha = total_irrigation_mm * 10
    iwue = yield_kg_ha / irr_m3_ha if irr_m3_ha > 0 else 0
    return round(iwue, 2)


def compute_profit(expected_yield_t_ha, field_area_ha, crop_row):
    """
    Compute expected profit using crop database prices.
    Returns amounts in UGX.
    """
    price_per_kg = crop_row["Price_UGX_per_kg"]
    cost_col = [c for c in crop_row.index if "Production_Cost" in c or "production_cost" in c]
    cost_per_ha = crop_row[cost_col[0]] if cost_col else 0

    total_yield_kg = expected_yield_t_ha * 1000 * field_area_ha
    revenue = total_yield_kg * price_per_kg
    total_cost = cost_per_ha * field_area_ha
    profit = revenue - total_cost
    margin = (profit / revenue * 100) if revenue > 0 else 0.0

    return {
        "expected_yield_t_ha": expected_yield_t_ha,
        "total_yield_kg": round(total_yield_kg, 0),
        "price_ugx_per_kg": int(price_per_kg),
        "revenue_ugx": round(revenue, 0),
        "cost_per_ha_ugx": int(cost_per_ha),
        "total_cost_ugx": round(total_cost, 0),
        "profit_ugx": round(profit, 0),
        "margin_pct": round(margin, 1),
    }


def generate_irrigation_schedule(
    cwr_df, irrigation_days, growth_stages_str, avg_kc
):
    """
    Generate irrigation schedule based on CWR, growth stages, and Kc.
    Returns list of schedule events.
    """
    if cwr_df is None or cwr_df.empty:
        return []

    stages = [s.strip() for s in str(growth_stages_str).split("|")]
    n_stages = len(stages)
    n_days = len(cwr_df)
    if n_stages == 0:
        n_stages = 1

    days_per_stage = max(1, n_days // n_stages)
    schedule = []

    for i, stage in enumerate(stages):
        start = i * days_per_stage
        end = min((i + 1) * days_per_stage, n_days)
        if start >= n_days:
            break

        stage_cwr = cwr_df.iloc[start:end]
        total_stage_cwr = stage_cwr["CWR"].sum()
        n_irr = max(1, int(total_stage_cwr / 5))  # irrigate every ~5mm

        if n_irr > 0:
            interval = max(1, (end - start) // n_irr)
            water_per_event = total_stage_cwr / n_irr if n_irr > 0 else 0

            for j in range(n_irr):
                day_idx = start + j * interval
                if day_idx >= n_days:
                    break
                date = cwr_df.iloc[day_idx]["Date"]
                schedule.append({
                    "stage": stage,
                    "date": date,
                    "day": int(day_idx + 1),
                    "water_mm": round(water_per_event, 1),
                })

    if not schedule:
        return []

    return schedule
