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

# --------------------------------
# Shortest-path computation
# --------------------------------
def compute_shortest_time(G, row):
    G_traffic = deepcopy(G)

    # Apply congestion multipliers
    for j, edge in enumerate(G_traffic.edges):
        G_traffic[edge[0]][edge[1]][edge[2]]["speed_kph"] *= 1 - row[j]

    G_traffic = ox.routing.add_edge_travel_times(G_traffic)

    nodes, _ = ox.graph_to_gdfs(G_traffic)
    hospital_nodes = nodes[nodes["vtype"] == "hosp"]
    accident_nodes = nodes[nodes["vtype"] == "accident"]

    A = np.zeros((len(hospital_nodes), len(accident_nodes)))

    for i, hosp_id in enumerate(hospital_nodes.index):
        length = nx.single_source_dijkstra_path_length(
            G_traffic, hosp_id, weight="travel_time"
        )
        for j, accident_id in enumerate(accident_nodes.index):
            A[i, j] = length[accident_id]

    return A


# --------------------------------
# Precompute one scenario
# --------------------------------
def process_scenario(G, day, time):
    tag = f"{day}_{time}"
    print(f"\n=== Processing {tag} ===")

    acc_path = f"processed_data/accident_tables/{tag}_accidents.csv"
    cong_path = f"processed_data/congestion_tables/{tag}_congestion.csv"
    out_path = f"processed_data/A_traffics_{tag}.npz"

    if not os.path.exists(acc_path):
        print(f"  ⚠️ Missing file: {acc_path}, skipping.")
        return

    if not os.path.exists(cong_path):
        print(f"  ⚠️ Missing file: {cong_path}, skipping.")
        return

    # Load CSVs (faithful to your logic)
    accident_df = pd.read_csv(acc_path).drop("date", axis=1)
    congestion_df = pd.read_csv(cong_path).drop("date", axis=1)

    # Filtering logic EXACTLY as your script
    to_drop = (
        accident_df.isna().any(axis=1)
        | congestion_df.isna().any(axis=1)
        | (accident_df == 0).all(axis=1)
    )

    accident_df = accident_df[~to_drop]
    congestion_df = congestion_df[~to_drop]

    # Compute shortest paths for all scenarios
    print(f"  Computing A_traffics for {len(congestion_df)} scenarios...")
    A_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_shortest_time)(G, row)
        for _, row in congestion_df.iterrows()
    )

    A = np.stack(A_list, axis=0)

    # Save result
    print(f"  Saving to {out_path}")
    np.savez_compressed(out_path, A=A)

    print(f"  ✔ Done: {tag}")


# --------------------------------
# Main: Loop over all combos
# --------------------------------
if __name__ == "__main__":
    days = ["wd", "we"]
    times = ["early", "morning", "midday", "evening", "night"]

    # Load graph once
    print("Loading graph...")
    with open("processed_data/bkk_augmented_graph.pickle", "rb") as f:
        G = pickle.load(f)

    G = ox.routing.add_edge_speeds(G)

    print("Starting batch computation...")

    for day in days:
        for time in times:
            process_scenario(G, day, time)

    print("\n=== All combinations processed ===")
