import pandas as pd
import streamlit as st

@st.cache_data
def load_standard_data():
    """Loads basic canopy and ground-truth flood data."""
    try:
        canopy_df = pd.read_csv("ODC_ENV_TREECANOPY2020_A_-5786833496851207437.csv")
        flood_df = pd.read_csv("denver_flood_5k.csv")
        return canopy_df, flood_df
    except Exception as e:
        st.error(f"Failed to load standard datasets: {e}")
        return None, None

@st.cache_data
def load_storm_network_data():
    """
    Loads and cleans the hydraulic network datasets.
    Implements Ambiguity A: Drop rows with missing values.
    """
    try:
        # 1. Storm Mains (Edges)
        mains = pd.read_csv("ODC_UTIL_STMMAIN_L_-1375342829723093326.csv").dropna(subset=[
            'UPSTREAMNODEID', 'DOWNSTREAMNODEID', 'HSIZE_INCHES', 'SLOPE'
        ])

        # 2. Inlets (Nodes)
        # inlets = pd.read_csv("ODC_UTIL_STMINLET_P_1223013160178616384.csv").dropna(subset=['FACILITYID', 'x', 'y'])

        # 3. Basin Mapping (From Point-in-Polygon Activity)
        # Using BASIN_NAME as the unique identifier per your instruction
        mapping = pd.read_csv("inlet_to_basin_mapping.csv").dropna()

        # 4. Basin Boundaries (Attributes like Area)
        basins = pd.read_csv("ODC_UTIL_STMCOLSYSBASIN_A_6966226426837139631.csv").dropna(subset=['Basin Name', 'Area_AC'])

        return mains, mapping, basins
    except Exception as e:
        st.error(f"Failed to load Storm Network datasets: {e}")
        return None, None, None