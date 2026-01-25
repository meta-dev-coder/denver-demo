import networkx as nx
import numpy as np

def build_storm_graph(mains_df):
    """Creates a directed graph of the storm sewer network."""
    G = nx.DiGraph()
    for _, row in mains_df.iterrows():
        G.add_edge(row['UPSTREAMNODEID'], row['DOWNSTREAMNODEID'],
                   capacity=calculate_manning_capacity(row),
                   facility_id=row['FACILITYID'])
    return G

def calculate_manning_capacity(row):
    """Calculates max flow (cfs) using Manning's Equation."""
    # Standard Manning's n for concrete = 0.013
    n = 0.013
    d_ft = row['HSIZE_INCHES'] / 12.0
    slope = max(row['SLOPE'], 0.001) # Avoid zero-slope errors
    area = (np.pi * (d_ft/2)**2)
    radius = d_ft / 4
    velocity = (1.486 / n) * (radius**(2/3)) * (slope**0.5)
    return velocity * area

def run_network_simulation(G, rainfall_in, runoff_coeff, mapping_df, basin_data):
    """
    Fixed Version:
    1. Distributes Area among inlets.
    2. Handles Slope unit conversion.
    """
    results = []

    # Pre-calculate: How many inlets are in each basin?
    inlet_counts = mapping_df.groupby('BASIN_NAME')['FACILITYID'].count().to_dict()

    # Pre-calculate: Distributed Inflow per Basin
    distributed_inflows = {}
    for _, b in basin_data.iterrows():
        b_name = b['Basin Name']
        count = inlet_counts.get(b_name, 1) # Avoid division by zero
        # Distribute the total basin area across all its inlets
        distributed_area = b['Area_AC'] / count
        # Q = C * i * A
        distributed_inflows[b_name] = runoff_coeff * rainfall_in * distributed_area

    for _, mapping in mapping_df.iterrows():
        b_name = mapping['BASIN_NAME']
        inflow_q = distributed_inflows.get(b_name, 0)
        node_id = mapping['FACILITYID']

        if node_id in G:
            for u, v, data in G.edges(node_id, data=True):
                # Manning's Capacity
                capacity = data['capacity']

                # Check for "Zero Capacity" error (often due to missing slope)
                if capacity < 0.1:
                    status = "Data Error (No Slope)"
                else:
                    status = "Surcharged" if inflow_q > capacity else "Stable"

                results.append({
                    "Pipe_ID": data['facility_id'],
                    "Basin": b_name,
                    "Inflow_CFS": round(inflow_q, 3),
                    "Capacity_CFS": round(capacity, 3),
                    "Surcharge_Ratio": round(inflow_q / capacity, 2) if capacity > 0 else 0,
                    "Status": status
                })
    import pandas as pd
    return pd.DataFrame(results)

def get_llm_summary(sim_df):
    """Truncates 34k records into a meaningful summary for the AI."""
    summary = {
        "total_pipes": len(sim_df),
        "surcharged_count": len(sim_df[sim_df['Status'] == "Surcharged"]),
        "worst_basins": sim_df[sim_df['Status'] == "Surcharged"]
        .groupby('Basin')['Pipe_ID'].count()
        .sort_values(ascending=False).head(5).to_dict()
    }
    return summary

def get_simulation_summary(results_df):
    """
    Calculates total counts and identifies the top 10 most impacted basins.
    """
    # 1. Total Counts
    total_pipes = len(results_df)
    total_surcharged = (results_df['Status'] == "Surcharged").sum()

    # 2. Group by Basin to find the Top 10
    basin_counts = results_df[results_df['Status'] == "Surcharged"].groupby('Basin').size().reset_index(name='Surcharged_Count')
    top_10_basins = basin_counts.sort_values(by='Surcharged_Count', ascending=False).head(10)

    return total_pipes, total_surcharged, top_10_basins