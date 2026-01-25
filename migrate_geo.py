import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# 1. Load Data
# Assuming 'storm_inlets.csv' (with x,y) and 'boundaries.geojson'
inlets_df = pd.read_csv("ODC_UTIL_STMINLET_P_1223013160178616384.csv")
boundaries_gdf = gpd.read_file("ODC_UTIL_STMCOLSYSBASIN_A_631212441909972652.geojson")

# 2. Cleanup (Ambiguity A: Delete missing data)
inlets_df = inlets_df.dropna(subset=['x', 'y', 'FACILITYID'])

# 3. Convert CSV to GeoDataFrame
# Note: Denver x/y is usually EPSG:2232 (NAD83 Colorado Central)
inlets_gdf = gpd.GeoDataFrame(
    inlets_df,
    geometry=gpd.points_from_xy(inlets_df.x, inlets_df.y),
    crs="EPSG:2232"
)

# 4. Align Coordinate Systems
# GeoJSON is usually 4326; we must match the inlets to the boundaries
if boundaries_gdf.crs != inlets_gdf.crs:
    boundaries_gdf = boundaries_gdf.to_crs(inlets_gdf.crs)

# 5. Spatial Join (Point-in-Polygon)
# This finds which BASIN_NAME contains each FACILITYID
mapping_results = gpd.sjoin(inlets_gdf, boundaries_gdf[['BASIN_NAME', 'geometry']],
                            how="left", predicate="within")

# 6. Final Mapping Table
# We only need the ID and its Basin name
final_mapping = mapping_results[['FACILITYID', 'BASIN_NAME']].dropna()
final_mapping.to_csv("inlet_to_basin_mapping.csv", index=False)

print(f"✅ Mapping Complete! {len(final_mapping)} inlets linked to basins.")