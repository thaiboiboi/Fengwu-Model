import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry

# 1. Setup API Client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 2. Define Model Constants
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SURFACE_VARS = ["msl", "u10", "v10", "t2m"]
UPPER_VARS = ["geopotential_height", "specific_humidity", "temperature", "windspeed_u", "windspeed_v"]

# 3. Create the massive variable list for the API
all_hourly_vars = []
all_hourly_vars.extend(SURFACE_VARS)
for var in UPPER_VARS:
    for lvl in LEVELS:
        all_hourly_vars.append(f"{var}_{lvl}hPa")

# 4. API Request (Bounding Box for SASOL Race Area)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": [-27.0, -26.0],   # Min/Max Lat
    "longitude": [27.5, 29.5],   # Min/Max Lon
    "start_date": "2024-09-22",
    "end_date": "2024-09-22",
    "hourly": all_hourly_vars,
    "models": "era5"
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]
hourly = response.Hourly()

# 5. Extract Dimensions
# We assume the API returns a grid. We need to know the shape (n_lat, n_lon)
# Open-Meteo returns flattened arrays; we reshape them based on the coordinate grid
lats = np.arange(-26.0, -27.25, -0.25) # ERA5 Order: North to South
lons = np.arange(27.5, 29.75, 0.25)
n_lat, n_lon = len(lats), len(lons)
target_hour_idx = 6 # Example for 06:00 UTC

# 6. Build input_surface (4, n_lat, n_lon)
# Order: MSLP, U10, V10, T2M
input_surface = np.zeros((4, n_lat, n_lon))
for i in range(4):
    data = hourly.Variables(i).ValuesAsNumpy()
    # Note: reshape to (hours, lats, lons) then pick target hour
    input_surface[i] = data.reshape(-1, n_lat, n_lon)[target_hour_idx]

# 7. Build input_upper (5, 13, n_lat, n_lon)
# Order: Z, Q, T, U, V
input_upper = np.zeros((5, 13, n_lat, n_lon))
var_counter = 4 # Start after surface variables

for v_idx in range(5):        # For Z, Q, T, U, V
    for l_idx in range(13):   # For each level
        data = hourly.Variables(var_counter).ValuesAsNumpy()
        val = data.reshape(-1, n_lat, n_lon)[target_hour_idx]
        
        # APPLY UNIT CONVERSION FOR Z (Geopotential)
        if v_idx == 0: # If it is Geopotential Height
            val = val * 9.80665
            
        input_upper[v_idx, l_idx] = val
        var_counter += 1

# 8. Save
np.save('input_surface.npy', input_surface.astype(np.float32))
np.save('input_upper.npy', input_upper.astype(np.float32))

print(f"Successfully adapted! Surface: {input_surface.shape}, Upper: {input_upper.shape}")