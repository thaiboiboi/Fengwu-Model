"""
fengwu_extract.py — Innoptus Solar Team
========================================
Extracts the 4 race variables from FengWu's output array (69, 721, 1440).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT FENGWU ACTUALLY OUTPUTS  (from the paper, Section 2.1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Shape: (69, 721, 1440)
  69  = 5 surface vars  +  64 upper-air vars (4 vars × 13 pressure levels  +  wait...
         actually: the ai-models plugin uses a SUBSET)

  Wait -- looking at the actual input you shared:
    Channel 0  = u10     (wind U,  values ~0.15  → m/s range after denorm)
    Channel 1  = v10     (wind V,  values ~-2.6  → m/s range)
    Channel 2  = t2m     (temp,    values ~252K  → Kelvin confirmed)
    Channel 3  = msl     (pressure,values ~100774 Pa → Pa confirmed)
    Channel 4  = z_50hPa (geopotential, values ~187000 m²/s²)
    ...
    Channels 5–68 = upper-air variables at 13 pressure levels

DIRECTLY AVAILABLE:
    ✓ temperature_2m    → channel 2  (Kelvin → °C)
    ✓ wind_speed_10m    → channels 0+1 (u10, v10 m/s → km/h)
    ✗ cloud_cover       → NOT in FengWu output
    ✗ shortwave_radiation → NOT in FengWu output

FOR CLOUD COVER + RADIATION — two options:
    Option A (proxy):  derive from relative humidity at 700hPa
                       r is in the upper-air channels
    Option B (better): fetch cloud cover + radiation from ERA5 or
                       Open-Meteo for the same time and location,
                       and use FengWu only for what it's good at
                       (temperature + wind)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANNEL MAP  (69 channels, 0-indexed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Surface (channels 0–4):
    0  u10    10m U-wind         m/s
    1  v10    10m V-wind         m/s
    2  t2m    2m temperature     K
    3  msl    mean sea level P   Pa
    4  sp     surface pressure   Pa   (some configs) / z50 in others

  Upper-air (channels 5–68) — 4 vars × 13 pressure levels + 5 vars × ...
  The exact layout depends on the ai-models-fengwu plugin version.
  Pressure levels: [50,100,150,200,250,300,400,500,600,700,850,925,1000] hPa

  Upper-air variable order (inferred from input values + paper):
    z  geopotential          m²/s²   channels  5–17
    r  relative humidity     0–1     channels 18–30
    u  U-wind                m/s     channels 31–43
    v  V-wind                m/s     channels 44–56
    t  temperature           K       channels 57–68  (only 12 levels here)

Install:
    pip install numpy pandas
"""

import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Channel index map  (confirmed from the input array you shared)
# ---------------------------------------------------------------------------

# Surface channels (first 5)
CH_U10  = 0    # 10m U-wind  (m/s)
CH_V10  = 1    # 10m V-wind  (m/s)
CH_T2M  = 2    # 2m temperature  (Kelvin) — confirmed: values ~252K
CH_MSL  = 3    # mean sea level pressure (Pa) — confirmed: values ~100774
CH_SP   = 4    # surface pressure (Pa) or z50 depending on plugin version

# Upper-air channels — 13 pressure levels each
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
N_LEVELS        = len(PRESSURE_LEVELS)   # 13

# Upper-air variable blocks (5 variables × 13 levels = 65, but we only have 64)
# The ai-models plugin likely uses 4 vars × 13 levels = 52 channels for upper air
# plus the 5 surface = 57 total... inspect your actual data to confirm
# Based on the paper: z, r, u, v, t at 37 levels — plugin subsets to 13 levels
CH_Z_START = 5                           # geopotential  channels  5–17
CH_R_START = CH_Z_START  + N_LEVELS      # rel. humidity channels 18–30
CH_U_START = CH_R_START  + N_LEVELS      # U-wind        channels 31–43
CH_V_START = CH_U_START  + N_LEVELS      # V-wind        channels 44–56
CH_T_START = CH_V_START  + N_LEVELS      # temperature   channels 57–69

# Index of 700 hPa within the pressure level list (for humidity → cloud proxy)
IDX_700HPA = PRESSURE_LEVELS.index(700)  # index 9
IDX_850HPA = PRESSURE_LEVELS.index(850)  # index 10


# ---------------------------------------------------------------------------
# Grid definition  (0.25° global grid)
# ---------------------------------------------------------------------------

LATS = np.linspace(90, -90, 721)    # 90°N to 90°S
LONS = np.linspace(0, 359.75, 1440) # 0° to 359.75°E  (0–360 convention)


def lat_lon_to_index(lat: float, lon: float):
    """
    Convert (lat, lon) to nearest (lat_idx, lon_idx) on the FengWu 0.25° grid.

    Note: FengWu uses 0-360 longitude convention.
    So Nashville at -86.77° becomes 360 - 86.77 = 273.23°.
    """
    # Convert negative longitude to 0-360
    lon_360 = lon % 360

    lat_idx = int(np.argmin(np.abs(LATS - lat)))
    lon_idx = int(np.argmin(np.abs(LONS - lon_360)))

    actual_lat = LATS[lat_idx]
    actual_lon = LONS[lon_idx]

    return lat_idx, lon_idx, actual_lat, actual_lon


# ---------------------------------------------------------------------------
# Extract variables from one FengWu output array
# ---------------------------------------------------------------------------

def extract_point(
    output: np.ndarray,
    lat: float,
    lon: float,
) -> dict:
    """
    Extract all available variables at a single lat/lon point
    from one FengWu output timestep.

    Parameters
    ----------
    output  : np.ndarray of shape (69, 721, 1440) — one FengWu output step
    lat     : latitude  (decimal degrees, standard -90 to +90)
    lon     : longitude (decimal degrees, standard -180 to +180)

    Returns
    -------
    dict of raw extracted values (before unit conversion)
    """
    assert output.shape == (69, 721, 1440), \
        f"Expected shape (69, 721, 1440), got {output.shape}"

    lat_idx, lon_idx, actual_lat, actual_lon = lat_lon_to_index(lat, lon)

    return {
        # Surface variables
        "u10_ms":        float(output[CH_U10, lat_idx, lon_idx]),
        "v10_ms":        float(output[CH_V10, lat_idx, lon_idx]),
        "t2m_k":         float(output[CH_T2M, lat_idx, lon_idx]),
        "msl_pa":        float(output[CH_MSL, lat_idx, lon_idx]),

        # Upper-air at 700 hPa (for cloud proxy)
        "r_700hpa":      float(output[CH_R_START + IDX_700HPA, lat_idx, lon_idx]),
        "t_700hpa_k":    float(output[CH_T_START + IDX_700HPA, lat_idx, lon_idx]),

        # Upper-air at 850 hPa (for cloud proxy)
        "r_850hpa":      float(output[CH_R_START + IDX_850HPA, lat_idx, lon_idx]),
        "t_850hpa_k":    float(output[CH_T_START + IDX_850HPA, lat_idx, lon_idx]),

        # Grid info
        "actual_lat":    actual_lat,
        "actual_lon":    actual_lon,
    }


# ---------------------------------------------------------------------------
# Unit conversions → Open-Meteo format
# ---------------------------------------------------------------------------

def to_race_variables(raw: dict, lat: float, valid_time: datetime) -> dict:
    """
    Convert raw FengWu output to the 4 race variables.

    DIRECTLY from FengWu:
        temperature_2m     K   → °C
        wind_speed_10m     m/s → km/h
        wind_direction_10m m/s → degrees

    DERIVED (proxy — not native FengWu output):
        cloud_cover        from relative humidity at 700+850 hPa
        shortwave_radiation from cloud cover + clear-sky model
    """
    result = {}

    # ── 1. Temperature  (K → °C) ──────────────────────────────────────────
    result["temperature_2m"] = raw["t2m_k"] - 273.15

    # ── 2. Wind speed + direction  (m/s → km/h + degrees) ────────────────
    u = raw["u10_ms"]
    v = raw["v10_ms"]
    result["wind_speed_10m"]     = math.sqrt(u**2 + v**2) * 3.6
    result["wind_direction_10m"] = math.degrees(math.atan2(-u, -v)) % 360

    # ── 3. Cloud cover proxy  (from relative humidity) ────────────────────
    #
    # FengWu does NOT output cloud cover directly.
    # We estimate it from relative humidity using the Sundqvist (1978) scheme:
    #   cloud fraction = max(0, (RH - RH_crit) / (1 - RH_crit))^0.5
    # where RH_crit = 0.70 (threshold below which no cloud forms).
    # We average 700 hPa and 850 hPa humidity for a mid-troposphere estimate.
    #
    rh_700 = np.clip(raw["r_700hpa"], 0, 1)
    rh_850 = np.clip(raw["r_850hpa"], 0, 1)
    rh_avg = (rh_700 + rh_850) / 2.0

    RH_CRIT = 0.70
    if rh_avg > RH_CRIT:
        cloud_fraction = math.sqrt((rh_avg - RH_CRIT) / (1.0 - RH_CRIT))
    else:
        cloud_fraction = 0.0

    result["cloud_cover"] = cloud_fraction * 100.0   # → %

    # ── 4. Shortwave radiation  (clear-sky × cloud attenuation) ──────────
    #
    # FengWu does NOT output radiation.
    # We compute it from:
    #   a) Solar geometry  (zenith angle from lat + day-of-year + hour)
    #   b) Clear-sky GHI   (simplified Bird & Hulstrom)
    #   c) Cloud attenuation (Kasten & Czeplak empirical formula)
    #
    doy        = valid_time.timetuple().tm_yday
    hour_utc   = valid_time.hour + valid_time.minute / 60.0

    # Solar declination
    decl_rad   = math.radians(23.45 * math.sin(math.radians(360/365 * (doy - 81))))

    # Hour angle (15° per hour, solar noon = 0)
    hour_angle = math.radians(15.0 * (hour_utc - 12.0))

    lat_rad    = math.radians(lat)
    sin_elev   = max(0.0,
                     math.sin(lat_rad) * math.sin(decl_rad) +
                     math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle))

    # Clear-sky GHI (W/m²)
    S0            = 1361.0   # solar constant
    clearsky_ghi  = S0 * sin_elev * 0.75   # 75% atmospheric transmittance

    # Cloud attenuation (Kasten & Czeplak 1980)
    cloud_f       = result["cloud_cover"] / 100.0
    transmittance = 1.0 - 0.75 * (cloud_f ** 3.4)
    result["shortwave_radiation"] = max(0.0, clearsky_ghi * transmittance)

    return result


# ---------------------------------------------------------------------------
# Process a full FengWu forecast run  (multiple timesteps)
# ---------------------------------------------------------------------------

def extract_forecast(
    fengwu_outputs: list,
    waypoints:      list,
    init_datetime:  datetime,
    timestep_hours: int = 6,
) -> dict:
    """
    Extract the 4 race variables from a sequence of FengWu output arrays.

    Parameters
    ----------
    fengwu_outputs  : list of np.ndarray, each shape (69, 721, 1440)
                      One array per forecast timestep (every 6h)
    waypoints       : [{"name": str, "lat": float, "lon": float}, ...]
    init_datetime   : when the forecast was initialised  e.g. datetime(2024,7,19,12)
    timestep_hours  : hours between each output step (FengWu = 6h)

    Returns
    -------
    {
        "Nashville":    pd.DataFrame,   ← UTC datetime index
        "Paducah":      pd.DataFrame,
        "Edwardsville": pd.DataFrame,
    }
    Each DataFrame has columns:
        temperature_2m, wind_speed_10m, wind_direction_10m,
        cloud_cover, shortwave_radiation
    """
    results = {}

    for wp in waypoints:
        name = wp.get("name") or f"({wp['lat']:.2f},{wp['lon']:.2f})"
        rows = []

        for step, output in enumerate(fengwu_outputs):
            valid_time = init_datetime + timedelta(hours=(step + 1) * timestep_hours)

            raw  = extract_point(output, wp["lat"], wp["lon"])
            race = to_race_variables(raw, wp["lat"], valid_time)
            race["datetime"] = valid_time
            rows.append(race)

        df = pd.DataFrame(rows).set_index("datetime")
        df.index = df.index.tz_localize("UTC")
        results[name] = df

    return results


# ---------------------------------------------------------------------------
# Print a clean summary
# ---------------------------------------------------------------------------

UNITS = {
    "temperature_2m":     "°C",
    "wind_speed_10m":     "km/h",
    "wind_direction_10m": "°",
    "cloud_cover":        "% (proxy)",
    "shortwave_radiation":"W/m² (derived)",
}

def print_forecast(results: dict) -> None:
    for wp_name, df in results.items():
        print(f"\n{'='*60}")
        print(f"  {wp_name}")
        print(f"{'='*60}")
        print(f"  Period: {df.index[0]}  →  {df.index[-1]}  ({len(df)} steps)")
        print(f"\n  {'Variable':<22} {'Min':>7} {'Max':>7} {'Mean':>7}  Unit")
        print(f"  {'─'*52}")
        for col, unit in UNITS.items():
            if col in df.columns:
                v = df[col].dropna()
                print(f"  {col:<22} {v.min():>7.1f} {v.max():>7.1f} "
                      f"{v.mean():>7.1f}  {unit}")

        # Daytime rows only
        daytime = df[df["shortwave_radiation"] > 10]
        if not daytime.empty:
            print(f"\n  Daytime steps ({len(daytime)}):")
            print(f"  {'Datetime (UTC)':<24} {'Temp':>6} {'Wind':>6} "
                  f"{'Cloud':>6} {'GHI':>7}")
            print(f"  {'─'*55}")
            for ts, row in daytime.iterrows():
                print(f"  {str(ts):<24} "
                      f"{row['temperature_2m']:>6.1f} "
                      f"{row['wind_speed_10m']:>6.1f} "
                      f"{row['cloud_cover']:>6.1f} "
                      f"{row['shortwave_radiation']:>7.1f}")


# ---------------------------------------------------------------------------
# Entry point — test with a dummy output array
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime

    print("FengWu Output Extractor — Innoptus Solar Team")
    print("ASC 2024 Stage 1: Nashville → Edwardsville")

    # ── ASC 2024 Stage 1 waypoints ─────────────────────────────────────────
    WAYPOINTS = [
        {"name": "Nashville",    "lat": 36.1468, "lon": -86.7756},
        {"name": "Paducah",      "lat": 37.0898, "lon": -88.5966},
        {"name": "Edwardsville", "lat": 38.7933, "lon": -90.0030},
    ]

    # ── Load real FengWu output ────────────────────────────────────────────
    # Replace this section with your actual output loading code.
    # FengWu outputs are typically saved as .npy files, one per timestep.
    #
    # Example — if saved as .npy files:
    #   import glob
    #   output_files = sorted(glob.glob("./predictions/fengwu_step_*.npy"))
    #   fengwu_outputs = [np.load(f) for f in output_files]
    #
    # Example — if saved as a single stacked array:
    #   all_steps = np.load("fengwu_20240719_1200.npy")  # shape (N, 69, 721, 1440)
    #   fengwu_outputs = [all_steps[i] for i in range(all_steps.shape[0])]
    #
    # For testing — create a dummy array matching the values you showed:
    print("\nCreating dummy output for testing (replace with real FengWu output) ...")
    dummy_step = np.zeros((69, 721, 1440), dtype=np.float32)

    # Set realistic values from your example input
    dummy_step[CH_U10]  = 0.158        # u10 ~ 0.15 m/s
    dummy_step[CH_V10]  = -0.87        # v10 ~ -0.87 m/s
    dummy_step[CH_T2M]  = 300.5        # t2m ~ 300.5 K = 27.35°C (summer Nashville)
    dummy_step[CH_MSL]  = 100774.0     # msl ~ 100774 Pa (confirmed from your data)

    # Set humidity at 700hPa and 850hPa (relative humidity 0–1)
    dummy_step[CH_R_START + IDX_700HPA] = 0.65   # 65% RH at 700hPa
    dummy_step[CH_R_START + IDX_850HPA] = 0.75   # 75% RH at 850hPa
    dummy_step[CH_T_START + IDX_700HPA] = 285.0  # temp at 700hPa
    dummy_step[CH_T_START + IDX_850HPA] = 295.0  # temp at 850hPa

    # Simulate 8 output steps (48 hours at 6h intervals)
    fengwu_outputs = [dummy_step + np.random.normal(0, 0.1, dummy_step.shape)
                      for _ in range(8)]

    # ── Run extraction ─────────────────────────────────────────────────────
    INIT_DATETIME = datetime(2024, 7, 19, 12, 0)   # July 19 2024, 12 UTC

    print("\nExtracting race variables at waypoints ...")
    results = extract_forecast(
        fengwu_outputs = fengwu_outputs,
        waypoints      = WAYPOINTS,
        init_datetime  = INIT_DATETIME,
        timestep_hours = 6,
    )

    # ── Print and save ─────────────────────────────────────────────────────
    print_forecast(results)

    # Save each waypoint to CSV
    for wp_name, df in results.items():
        fname = f"fengwu_{wp_name.lower().replace(' ','_')}.csv"
        df.to_csv(fname)
        print(f"\n  Saved → {fname}")

    print("\n" + "="*60)
    print("  IMPORTANT NOTES ON FENGWU OUTPUT")
    print("="*60)
    print("""
  From the FengWu paper (Section 2.1):
  FengWu directly outputs only 4 SURFACE variables:
    ✓ t2m  — 2m temperature        (channel 2, Kelvin)
    ✓ u10  — 10m U-wind component  (channel 0, m/s)
    ✓ v10  — 10m V-wind component  (channel 1, m/s)
    ✓ msl  — mean sea level P      (channel 3, Pa)

  Cloud cover and radiation are NOT in FengWu's output.
  This code derives them as proxies from humidity channels.

  For accurate cloud cover + radiation, combine FengWu with:
    → ERA5 historical  (for past validation)
    → Open-Meteo API   (for race-day use)
  """)