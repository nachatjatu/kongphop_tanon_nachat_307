import pickle
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
import osmnx as ox
import networkx as nx
from copy import deepcopy
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
from tqdm import tqdm

# --------------------------------
# Shortest-path computation
# --------------------------------
def compute_shortest_time(G, row, gas_stations):
    warnings.simplefilter("ignore", category=FutureWarning)
    G_traffic = deepcopy(G)

    # Apply congestion multipliers
    for j, edge in enumerate(G_traffic.edges):
        G_traffic[edge[0]][edge[1]][edge[2]]["speed_kph"] *= 1 - row[j]

    G_traffic = ox.routing.add_edge_travel_times(G_traffic)

    nodes, _ = ox.graph_to_gdfs(G_traffic)
    hospital_nodes = nodes[nodes["vtype"] == "hosp"].index

    depot_nodes = list(gas_stations) + list(hospital_nodes)

    accident_nodes = list(nodes[nodes["vtype"] == "accident"].index)

    A = np.zeros((len(depot_nodes), len(accident_nodes)))

    for i, depot_id in enumerate(depot_nodes):
        length = nx.single_source_dijkstra_path_length(
            G_traffic, depot_id, weight="travel_time"
        )
        for j, accident_id in enumerate(accident_nodes):
            A[i, j] = length[accident_id]

    return A


# --------------------------------
# Precompute one scenario
# --------------------------------
def process_scenario(G, day, time, gas_stations):
    tag = f"{day}_{time}"
    print(f"\n=== Processing {tag} ===")

    acc_path = f"processed_data/accident_tables/{tag}_accidents.csv"
    cong_path = f"processed_data/congestion_tables/{tag}_congestion.csv"

    accident_df = pd.read_csv(acc_path)
    congestion_df = pd.read_csv(cong_path)

    accident_df["date"] = pd.to_datetime(accident_df["date"])
    congestion_df["date"] = pd.to_datetime(congestion_df["date"])

    # Step 2: temporary dfs
    acc_tmp = accident_df.drop("date", axis=1)
    cong_tmp = congestion_df.drop("date", axis=1)

    # Step 3: validity mask
    mask_valid = ~(
        acc_tmp.isna().any(axis=1)
        | cong_tmp.isna().any(axis=1)
        | (acc_tmp == 0).all(axis=1)
    )

    # Now filter final dfs + A_traffics
    accident_df_filtered = accident_df.loc[mask_valid].reset_index().drop('index', axis=1)
    congestion_df_filtered = congestion_df.loc[mask_valid].reset_index().drop('index', axis=1)

    print(f"  Computing A_traffics for {len(congestion_df_filtered)} scenarios...")
    A_list = Parallel(n_jobs=12, verbose=10)(
        delayed(compute_shortest_time)(G, row, gas_stations)
        for _, row in congestion_df_filtered.drop('date', axis=1).iterrows()
    )

    A = np.stack(A_list, axis=0)

    # Save result
    A_out_path = f"processed_data/filtered/A_traffics_{tag}_filtered"
    accident_out_path = f"processed_data/filtered/{tag}_accidents_filtered"
    congestion_out_path = f"processed_data/filtered/{tag}_congestion_filtered"

    print(f"  Saving to processed_data/filtered/")
    np.save(A_out_path, A)
    accident_df_filtered.to_csv(accident_out_path)
    congestion_df_filtered.to_csv(congestion_out_path)

    print(f"  ✔ Done: {tag}")


# --------------------------------
# Main: Loop over all combos
# --------------------------------
if __name__ == "__main__":
    days = ["wd", "we"]
    times = ["early", "morning", "midday", "evening", "night"]

    with open('depots/gas_stations.pickle', 'rb') as f:
        gas_stations = pickle.load(f)

    # Load graph once
    print("Loading graph...")
    with open("processed_data/bkk_augmented_graph.pickle", "rb") as f:
        G = pickle.load(f)

    G = ox.routing.add_edge_speeds(G)

    print("Starting batch computation...")

    for day in days:
        for time in times:
            if day == "wd" and time == "early":
                continue
            process_scenario(G, day, time, gas_stations)

    print("\n=== All combinations processed ===")
