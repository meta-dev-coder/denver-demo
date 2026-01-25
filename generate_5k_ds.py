import pandas as pd
import numpy as np

# Grounded Denver Neighborhood Data
DENVER_NBHDS = {
    "Five Points": {"SVI": 0.82, "Elevation": 5200, "Canopy": 12},
    "Washington Park": {"SVI": 0.15, "Elevation": 5310, "Canopy": 35},
    "Elyria-Swansea": {"SVI": 0.94, "Elevation": 5150, "Canopy": 8},
    "Cherry Creek": {"SVI": 0.10, "Elevation": 5280, "Canopy": 28},
    "Montbello": {"SVI": 0.75, "Elevation": 5400, "Canopy": 15}
}

import pandas as pd
import numpy as np

def generate_flood_history(canopy_df):
    """Generates 5,000+ rows of synthetic Denver flood history (2006-2026)"""
    data = []
    # Using real SVI archetypes for Denver neighborhoods
    nbhd_profiles = {
        "Elyria-Swansea": {"svi": 0.94, "wealth": 0.4},  # High Vulnerability
        "Five Points": {"svi": 0.78, "wealth": 0.8},
        "Washington Park": {"svi": 0.12, "wealth": 2.5}, # Low Vulnerability
        "Cherry Creek": {"svi": 0.10, "wealth": 3.0},
        "Montbello": {"svi": 0.72, "wealth": 0.6}
    }

    for year in range(2006, 2027):
        for month in range(1, 13):
            # ~21 scenarios per month across Denver neighborhoods
            for _ in range(21):
                name = np.random.choice(list(nbhd_profiles.keys()))
                profile = nbhd_profiles[name]
                canopy = canopy_df[canopy_df['NBHD_NAME'] == name]['PCT_CANOPY'].values[0] if name in canopy_df['NBHD_NAME'].values else 15

                # Denver Rainfall Logic: 100-year storm is ~1.5-2.0 inches in 1 hour
                rain = np.random.gamma(shape=1.5, scale=0.8)
                season_mult = 1.3 if month in [5, 6] else 1.0 # Spring Saturation

                # Target: Depth (Physical)
                depth = (rain * 0.5 * season_mult) - (canopy * 0.02)
                depth = max(0, round(depth, 2))

                # Target: Property Loss (Economic)
                loss = depth * 150000 * profile['wealth'] if depth > 0.1 else 0

                data.append({
                    "Year": year, "Month": month, "Neighborhood": name,
                    "Rainfall_In": round(rain, 2), "Canopy_Pct": canopy,
                    "SVI": profile['svi'], "Depth_m": depth,
                    "Property_Loss_USD": round(loss, 2)
                })
    return pd.DataFrame(data)

def generate_robust_denver_data():
    data = []
    # 20 Years (2006 - 2026)
    for year in range(2006, 2027):
        for month in range(1, 13):
            # Generate ~21 scenarios per month to exceed 5000 rows
            for _ in range(21):
                name = np.random.choice(list(DENVER_NBHDS.keys()))
                profile = DENVER_NBHDS[name]

                # Denver Precipitation Logic: May/June are wettest
                base_rain = 1.5 if month in [5, 6] else 0.5
                rain = np.random.gamma(shape=2, scale=base_rain)

                # Depth logic: More rain + lower elevation + lower canopy = More flood
                depth = (rain * 0.4) + ( (5400 - profile['Elevation']) * 0.001) - (profile['Canopy'] * 0.01)
                depth = max(0, depth)

                # Economic Loss: Wealthy areas have higher $ loss for same depth
                wealth_factor = 2.0 if profile['SVI'] < 0.3 else 0.5
                loss = depth * 100000 * wealth_factor

                data.append({
                    "Year": year, "Month": month, "Neighborhood": name,
                    "Rainfall_In": round(rain, 2), "SVI": profile['SVI'],
                    "Canopy_Pct": profile['Canopy'], "Depth_m": round(depth, 2),
                    "Property_Loss_USD": round(loss, 2)
                })
    return pd.DataFrame(data)

# Export for app.py
df_final = generate_robust_denver_data()
df_final.to_csv("denver_flood_5k.csv", index=False)