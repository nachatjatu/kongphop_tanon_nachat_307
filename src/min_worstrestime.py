import pickle
from copy import deepcopy

import geopandas as gpd
import gurobipy as gp
import networkx as nx
import numpy as np
import osmnx as ox
from gurobipy import GRB

if __name__ == "__main__":
    with open("processed_data/bkk_augmented_graph.pickle", "rb") as f:
        G = pickle.load(f)

    # compute travel time
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)

    # impute travel time for each scenario
    G_traffics = []
    n_realizations = 10
    for i in range(n_realizations):
        G_traffics.append(deepcopy(G))
        congestion_factor = np.random.choice(
            [1.0, 1.25, 1.5], size=len(G_traffics[-1].edges)
        )
        for idx, edge in enumerate(G_traffics[-1].edges):
            G_traffics[-1][edge[0]][edge[1]][edge[2]][
                "travel_time"
            ] *= congestion_factor[idx]

    # compute shortest time between depots and accidents
    A_traffics = []
    for G_traffic in G_traffics:
        nodes, edges = ox.graph_to_gdfs(G_traffic)
        hospital_nodes = nodes[nodes["vtype"] == "hosp"][:30]
        accident_nodes = nodes[nodes["vtype"] == "accident"][:30]
        A_traffics.append(np.zeros((len(hospital_nodes), len(accident_nodes))))
        for i, hosp_id in enumerate(hospital_nodes.index):
            length = nx.single_source_dijkstra_path_length(
                G, hosp_id, weight="travel_time"
            )
            for j, accident_id in enumerate(accident_nodes.index):
                A_traffics[-1][i, j] = length[accident_id]

    # probability of accident sites and realization
    P = np.random.rand(len(accident_nodes))
    P /= np.linalg.norm(P)
    B = np.zeros((n_realizations, P.size), dtype=np.bool)
    for i in range(n_realizations):
        B[i, :] = np.random.rand(P.size) >= P

    # formulate coverage problem
    # problem variable
    num_depots = 10

    # Create a new model
    model = gp.Model("Accident coverage problem")

    # Create variables
    X = model.addMVar(
        (len(hospital_nodes), len(accident_nodes)), vtype=GRB.BINARY, name="X"
    )
    y = model.addMVar(len(hospital_nodes), vtype=GRB.BINARY, name="y")
    t = model.addVar()

    # Set objective
    model.setObjective(t, GRB.MINIMIZE)

    # Add constraint
    for i in range(n_realizations):
        model.addConstr(
            (A_traffics[i] * X)[:, B[i, :]].sum(axis=0) <= t, f"realization {i + 1}"
        )
    model.addConstr(X.sum(axis=0) == 1, "only one site")
    model.addConstr(
        X.sum(axis=1) <= len(accident_nodes) * y, "only assign to open depot"
    )
    model.addConstr(y.sum() <= num_depots, "depot opening")

    # Optimize model
    model.optimize()

    print(f"Minimum worst case response time: {model.ObjVal:.2f}")
