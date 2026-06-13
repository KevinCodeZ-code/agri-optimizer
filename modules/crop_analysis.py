"""
Crop database analysis module.
Loads Master_Crop_Database_40_Crops.xlsx and provides crop selection.
"""
import os
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CROP_DB_PATH = os.path.join(DATA_DIR, "Master_Crop_Database_40_Crops.xlsx")


@st.cache_data
def load_crop_database():
    """Load crop database from Excel file. Returns DataFrame or None."""
    if not os.path.exists(CROP_DB_PATH):
        # Generate synthetic crop database when file is missing
        return _generate_synthetic_crops()
    try:
        df = pd.read_excel(CROP_DB_PATH)
        return df
    except Exception as e:
        st.warning(f"Could not load crop database: {e}. Using synthetic data.")
        return _generate_synthetic_crops()


def _generate_synthetic_crops():
    """Generate 40 synthetic crops as fallback."""
    crops = []
    data = [
        (1, "Maize", 0.75, 0.25), (2, "Rice", 0.20, 0.15),
        (3, "Wheat", 0.15, 0.12), (4, "Sorghum", 0.60, 0.20),
        (5, "Millet", 0.50, 0.15), (6, "Cassava", 1.00, 0.80),
        (7, "Sweet Potato", 0.90, 0.30), (8, "Irish Potato", 0.60, 0.25),
        (9, "Yam", 1.00, 0.50), (10, "Banana", 2.00, 2.00),
        (11, "Plantain", 2.00, 2.00), (12, "Beans", 0.50, 0.15),
        (13, "Cowpeas", 0.60, 0.20), (14, "Groundnuts", 0.45, 0.15),
        (15, "Soybean", 0.50, 0.10), (16, "Sunflower", 0.60, 0.30),
        (17, "Sesame", 0.45, 0.10), (18, "Cotton", 0.75, 0.30),
        (19, "Coffee Arabica", 2.50, 2.00), (20, "Coffee Robusta", 3.00, 2.50),
        (21, "Cocoa", 3.00, 2.50), (22, "Tea", 1.20, 0.60),
        (23, "Tobacco", 1.00, 0.50), (24, "Sugarcane", 1.50, 0.40),
        (25, "Onion", 0.30, 0.10), (26, "Tomato", 0.60, 0.40),
        (27, "Cabbage", 0.50, 0.40), (28, "Carrot", 0.30, 0.08),
        (29, "Spinach", 0.30, 0.20), (30, "Okra", 0.60, 0.40),
        (31, "Eggplant", 0.60, 0.45), (32, "Watermelon", 2.00, 1.00),
        (33, "Pineapple", 0.60, 0.30), (34, "Mango", 8.00, 8.00),
        (35, "Orange", 5.00, 5.00), (36, "Avocado", 6.00, 6.00),
        (37, "Papaya", 2.50, 2.50), (38, "Passion Fruit", 2.00, 2.00),
        (39, "Coconut", 7.50, 7.50), (40, "Cashew", 8.00, 8.00),
    ]
    for num, name, row_sp, plant_sp in data:
        crops.append({
            "Crop_Number": num,
            "Common_Name": name,
            "Row_Spacing_m": row_sp,
            "Plant_Spacing_m": plant_sp,
        })
    return pd.DataFrame(crops)


def _get_crop_id_col(df):
    """Return the crop ID column name (Crop_ID or Crop_Number)."""
    if "Crop_ID" in df.columns:
        return "Crop_ID"
    return "Crop_Number"


def get_crop_options(df):
    """Return list of crop display strings for dropdown."""
    if df is None or df.empty:
        return []
    id_col = _get_crop_id_col(df)
    return [f"{row[id_col]}. {row['Common_Name']}" for _, row in df.iterrows()]


def get_selected_crop(df, display_str):
    """Parse dropdown selection and return crop row as dict."""
    if df is None or df.empty or not display_str:
        return None
    try:
        num = int(display_str.split(".")[0])
        id_col = _get_crop_id_col(df)
        row = df[df[id_col] == num].iloc[0]
        return {
            "number": int(row[id_col]),
            "name": str(row["Common_Name"]),
            "row_spacing": float(row["Row_Spacing_m"]),
            "plant_spacing": float(row["Plant_Spacing_m"]),
        }
    except (IndexError, ValueError, KeyError):
        return None
