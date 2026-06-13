"""
Soil and Fertility Analysis Module.
Provides soil suitability assessment, pH analysis, and fertilizer recommendations.
"""
import numpy as np
import pandas as pd


FERTILIZERS = {
    "Urea": {"N": 46, "P": 0, "K": 0, "type": "N", "desc": "High-nitrogen, quick release"},
    "CAN": {"N": 27, "P": 0, "K": 0, "type": "N", "desc": "Calcium Ammonium Nitrate"},
    "ASN": {"N": 26, "P": 0, "K": 0, "type": "N", "desc": "Ammonium Sulphate Nitrate"},
    "DAP": {"N": 18, "P": 46, "K": 0, "type": "NP", "desc": "Diammonium Phosphate (N+P)"},
    "TSP": {"N": 0, "P": 46, "K": 0, "type": "P", "desc": "Triple Super Phosphate"},
    "SSP": {"N": 0, "P": 20, "K": 0, "type": "P", "desc": "Single Super Phosphate"},
    "MOP": {"N": 0, "P": 0, "K": 60, "type": "K", "desc": "Muriate of Potash (0-0-60)"},
    "SOP": {"N": 0, "P": 0, "K": 50, "type": "K", "desc": "Sulphate of Potash (0-0-50)"},
    "NPK 17-17-17": {"N": 17, "P": 17, "K": 17, "type": "NPK", "desc": "Balanced compound"},
    "NPK 23-10-5": {"N": 23, "P": 10, "K": 5, "type": "NPK", "desc": "High-N compound"},
    "NPK 25-5-5": {"N": 25, "P": 5, "K": 5, "type": "NPK", "desc": "High-N compound"},
}


def get_crop_fertilizer_recommendations(crop_name, n_kg_ha, p_kg_ha, k_kg_ha, recommended_str=None):
    """Calculate fertilizer recommendations based on crop NPK requirements.

    Returns a DataFrame with columns: Fertilizer, Type, Rate_kg_ha, N_kg, P_kg, K_kg.
    """
    if recommended_str:
        fert_names = [f.strip() for f in recommended_str.replace("+", ",").split(",")]
    else:
        fert_names = []

    available = {k: v for k, v in FERTILIZERS.items()}
    matched = []
    for name in fert_names:
        if name in available:
            matched.append((name, available[name]))
        else:
            for alias in ["NPK " + name, name.replace(" ", "")]:
                if alias in available:
                    matched.append((alias, available[alias]))
                    break

    if not matched:
        if n_kg_ha > 0 or p_kg_ha > 0 or k_kg_ha > 0:
            if p_kg_ha > 0 and n_kg_ha > 0:
                matched.append(("DAP", available["DAP"]))
            if k_kg_ha > 0:
                matched.append(("MOP", available["MOP"]))
            if n_kg_ha > 0:
                matched.append(("Urea", available["Urea"]))
        else:
            return pd.DataFrame()

    remaining_n, remaining_p, remaining_k = n_kg_ha, p_kg_ha, k_kg_ha
    rows = []
    for name, info in matched:
        if remaining_n <= 0 and remaining_p <= 0 and remaining_k <= 0:
            break
        if info["type"] in ("NPK", "NP") and (remaining_p > 0 or remaining_n > 0):
            if info["P"] > 0 and remaining_p > 0:
                p_rate = remaining_p / (info["P"] / 100)
                n_from_fert = p_rate * (info["N"] / 100)
                k_from_fert = p_rate * (info["K"] / 100)
                rate = p_rate
                remaining_p = 0
            elif info["N"] > 0 and remaining_n > 0:
                n_rate = remaining_n / (info["N"] / 100)
                p_from_fert = n_rate * (info["P"] / 100)
                k_from_fert = n_rate * (info["K"] / 100)
                rate = n_rate
            else:
                continue
            if rate <= 0:
                continue
            n_applied = rate * (info["N"] / 100)
            p_applied = rate * (info["P"] / 100)
            k_applied = rate * (info["K"] / 100)
            remaining_n = max(0, remaining_n - n_applied)
            remaining_k = max(0, remaining_k - k_applied)
            rows.append({"Fertilizer": name, "Type": info["type"], "Rate_kg_ha": round(rate, 1),
                         "N_kg_ha": round(n_applied, 1), "P_kg_ha": round(p_applied, 1),
                         "K_kg_ha": round(k_applied, 1)})
        elif info["type"] == "P" and remaining_p > 0:
            rate = remaining_p / (info["P"] / 100)
            p_applied = rate * (info["P"] / 100)
            rows.append({"Fertilizer": name, "Type": info["type"], "Rate_kg_ha": round(rate, 1),
                         "N_kg_ha": 0, "P_kg_ha": round(p_applied, 1), "K_kg_ha": 0})
            remaining_p = 0
        elif info["type"] == "K" and remaining_k > 0:
            rate = remaining_k / (info["K"] / 100)
            k_applied = rate * (info["K"] / 100)
            rows.append({"Fertilizer": name, "Type": info["type"], "Rate_kg_ha": round(rate, 1),
                         "N_kg_ha": 0, "P_kg_ha": 0, "K_kg_ha": round(k_applied, 1)})
            remaining_k = 0
        elif info["type"] == "N" and remaining_n > 0:
            rate = remaining_n / (info["N"] / 100)
            n_applied = rate * (info["N"] / 100)
            rows.append({"Fertilizer": name, "Type": info["type"], "Rate_kg_ha": round(rate, 1),
                         "N_kg_ha": round(n_applied, 1), "P_kg_ha": 0, "K_kg_ha": 0})
            remaining_n = 0

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df


SOIL_TYPES = {
    "Sandy Loam": {"ph_range": (5.5, 7.5), "organic_matter": "Low", "drainage": "Good",
                   "suitable_crops": ["Maize", "Cassava", "Sweet Potato", "Groundnuts"]},
    "Clay Loam": {"ph_range": (5.0, 7.0), "organic_matter": "Medium", "drainage": "Moderate",
                  "suitable_crops": ["Rice", "Sugarcane", "Banana", "Coffee"]},
    "Silt Loam": {"ph_range": (5.5, 7.5), "organic_matter": "High", "drainage": "Moderate",
                  "suitable_crops": ["Maize", "Wheat", "Soybean", "Vegetables"]},
    "Loam": {"ph_range": (6.0, 7.5), "organic_matter": "High", "drainage": "Good",
             "suitable_crops": ["All crops generally suitable"]},
    "Sandy Clay": {"ph_range": (5.0, 6.5), "organic_matter": "Low", "drainage": "Good",
                   "suitable_crops": ["Cassava", "Pineapple", "Cowpeas"]},
}

PH_SUITABILITY = {
    (0.0, 4.5): "Very Strongly Acid - Liming recommended",
    (4.5, 5.5): "Strongly Acid - Liming recommended for most crops",
    (5.5, 6.5): "Slightly Acid - Suitable for most crops",
    (6.5, 7.5): "Neutral - Optimal for most crops",
    (7.5, 8.5): "Slightly Alkaline - Manage with organic matter",
    (8.5, 14.0): "Strongly Alkaline - Soil amendment required",
}

NUTRIENT_DEFICIENCY = {
    "Nitrogen (N)": {"symptoms": "Yellowing of older leaves, stunted growth",
                     "remedy": "Apply Urea (46% N) at 50-100 kg/ha"},
    "Phosphorus (P)": {"symptoms": "Purplish discoloration, poor root development",
                       "remedy": "Apply DAP (18-46-0) or TSP at 40-60 kg/ha"},
    "Potassium (K)": {"symptoms": "Yellowing leaf margins, weak stems",
                      "remedy": "Apply Muriate of Potash (0-0-60) at 30-50 kg/ha"},
    "Zinc (Zn)": {"symptoms": "Interveinal chlorosis, stunted internodes",
                  "remedy": "Apply Zinc Sulphate at 10-20 kg/ha"},
    "Magnesium (Mg)": {"symptoms": "Interveinal chlorosis of older leaves",
                       "remedy": "Apply Magnesium Sulphate or Dolomitic Lime"},
}


def get_ph_suitability(ph_value):
    for (lo, hi), desc in PH_SUITABILITY.items():
        if lo <= ph_value < hi:
            return desc
    return "Unknown pH range"


def get_soil_type_analysis(soil_type):
    return SOIL_TYPES.get(soil_type, {})


def get_nutrient_recommendations(crop_name=None, soil_type=None):
    recs = []
    for nutrient, info in NUTRIENT_DEFICIENCY.items():
        recs.append({"Nutrient": nutrient, **info})
    return pd.DataFrame(recs)


def compute_soil_suitability(ph_value, soil_type=None):
    if 5.5 <= ph_value <= 7.5:
        ph_score = 100
    elif 5.0 <= ph_value < 5.5 or 7.5 < ph_value <= 8.0:
        ph_score = 60
    elif 4.5 <= ph_value < 5.0 or 8.0 < ph_value <= 8.5:
        ph_score = 30
    else:
        ph_score = 10
    return ph_score


def run_soil_analysis(ph_value=6.5, soil_type="Loam", crop_name=None, n_kg_ha=None, p_kg_ha=None, k_kg_ha=None, recommended_fertilizers=None):
    """Run full soil and fertility analysis, including crop-specific fertilizer recommendations."""
    ph_suitability = get_ph_suitability(ph_value)
    soil_info = get_soil_type_analysis(soil_type)
    soil_score = compute_soil_suitability(ph_value, soil_type)
    nutrient_df = get_nutrient_recommendations()

    fert_df = pd.DataFrame()
    if crop_name and (n_kg_ha is not None or recommended_fertilizers):
        fert_df = get_crop_fertilizer_recommendations(
            crop_name, n_kg_ha or 0, p_kg_ha or 0, k_kg_ha or 0, recommended_fertilizers
        )

    return {
        "ph_value": ph_value,
        "ph_suitability": ph_suitability,
        "soil_type": soil_type if soil_type else "Not specified",
        "soil_info": soil_info,
        "soil_score": soil_score,
        "nutrient_recommendations": nutrient_df,
        "crop_fertilizer_recommendations": fert_df,
        "crop_name": crop_name,
        "soil_types_available": list(SOIL_TYPES.keys()),
    }
