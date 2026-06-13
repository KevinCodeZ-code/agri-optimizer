"""
Meteorological analysis module.
Retrieves NASA POWER weather data or generates synthetic data.
Computes ET0, crop water requirements, and irrigation demand.
"""
import numpy as np
import pandas as pd
import requests
import streamlit as st

LAT = 0.35
LON = 33.75
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

NASA_API = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"parameters=T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN,WS2M,PRECTOTCORR"
    f"&community=RE&longitude={LON}&latitude={LAT}"
    f"&start={START_DATE}&end={END_DATE}&format=JSON"
)


@st.cache_data
def fetch_weather_data():
    """Fetch weather data from NASA POWER API or generate synthetic."""
    try:
        resp = requests.get(NASA_API, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return _parse_nasa_data(data)
        else:
            st.warning(f"NASA API returned {resp.status_code}. Using synthetic data.")
            return _generate_synthetic_weather()
    except Exception as e:
        st.warning(f"NASA API request failed: {e}. Using synthetic weather data.")
        return _generate_synthetic_weather()


def _parse_nasa_data(data):
    """Parse NASA POWER JSON response into a DataFrame."""
    properties = data.get("properties", {}).get("parameter", {})
    dates = list(properties.get("T2M_MAX", {}).keys())
    if not dates:
        return _generate_synthetic_weather()

    records = []
    for date in dates:
        tmax = properties.get("T2M_MAX", {}).get(date)
        tmin = properties.get("T2M_MIN", {}).get(date)
        rh = properties.get("RH2M", {}).get(date)
        rs = properties.get("ALLSKY_SFC_SW_DWN", {}).get(date)
        ws = properties.get("WS2M", {}).get(date)
        rain = properties.get("PRECTOTCORR", {}).get(date)

        if tmax is None or tmin is None:
            continue

        records.append({
            "Date": pd.to_datetime(date),
            "Tmax": tmax,
            "Tmin": tmin,
            "RH": rh if rh is not None else 60.0,
            "Rs": rs if rs is not None else 15.0,
            "WS": ws if ws is not None else 2.0,
            "Rainfall": rain if rain is not None else 0.0,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return _generate_synthetic_weather()
    return df.sort_values("Date").reset_index(drop=True)


def _generate_synthetic_weather():
    """Generate synthetic daily weather data for one year."""
    np.random.seed(42)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    n = len(dates)
    day_of_year = np.arange(n)

    # Seasonal patterns for tropical climate (Uganda)
    tmax = 28 + 3 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.randn(n) * 1.5
    tmin = 18 + 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.randn(n) * 1.0
    rh = 65 + 10 * np.sin(2 * np.pi * (day_of_year - 120) / 365) + np.random.randn(n) * 5
    rs = 18 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.randn(n) * 2
    ws = 2.5 + 0.5 * np.random.randn(n)
    ws = np.clip(ws, 0.5, 6.0)

    # Bimodal rainfall pattern (Uganda)
    rainfall = 2 + 3 * np.sin(2 * np.pi * day_of_year / 365) + \
               3 * np.sin(2 * np.pi * (day_of_year - 100) / 180) + \
               np.random.exponential(1.5, n)
    rainfall = np.clip(rainfall, 0, None)

    return pd.DataFrame({
        "Date": dates,
        "Tmax": tmax,
        "Tmin": tmin,
        "RH": rh,
        "Rs": rs,
        "WS": ws,
        "Rainfall": rainfall,
    })


def compute_eto(df):
    """
    Compute reference evapotranspiration (ET0) using Hargreaves-Samani equation.
    ET0 = 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin) * Rs * 0.408
    """
    result = df.copy()
    result["Tmean"] = (result["Tmax"] + result["Tmin"]) / 2.0
    tr = result["Tmax"] - result["Tmin"]
    tr = tr.clip(lower=0.5)  # avoid sqrt of zero/negative
    result["ET0"] = 0.0023 * (result["Tmean"] + 17.8) * np.sqrt(tr) * result["Rs"] * 0.408
    result["ET0"] = result["ET0"].clip(lower=0)
    return result


def compute_crop_water_requirement(df_eto, kc_values):
    """
    Compute ETc = ET0 * Kc, effective rainfall, and CWR.
    kc_values: array of Kc coefficients for each day.
    """
    result = df_eto.copy()
    result["Kc"] = kc_values[:len(result)] if len(kc_values) >= len(result) else np.ones(len(result))
    result["ETc"] = result["ET0"] * result["Kc"]
    result["Peff"] = result["Rainfall"] * 0.75
    result["CWR"] = (result["ETc"] - result["Peff"]).clip(lower=0)
    return result


def generate_kc_values(n_days):
    """
    Generate crop coefficient curve: starts at 0.3, gradually increases to 1.0.
    """
    kc = np.ones(n_days)
    growth_phase = min(n_days // 3, 60)
    kc[:growth_phase] = np.linspace(0.3, 1.0, growth_phase)
    return kc


def compute_irrigation_days(df_cwr):
    """Count days where CWR > 0 (irrigation required)."""
    return int((df_cwr["CWR"] > 0).sum())
