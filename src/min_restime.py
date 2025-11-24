import pickle

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

    # compute shortest time between depots and accidents
    nodes, edges = ox.graph_to_gdfs(G)
    hospital_nodes = nodes[nodes["vtype"] == "hosp"][:30]
    accident_nodes = nodes[nodes["vtype"] == "accident"][:30]
    A = np.zeros((len(hospital_nodes), len(accident_nodes)))
    for i, hosp_id in enumerate(hospital_nodes.index):
        length = nx.single_source_dijkstra_path_length(G, hosp_id, weight="travel_time")
        for j, accident_id in enumerate(accident_nodes.index):
            A[i, j] = length[accident_id]

    # probability of accident sites
    P = np.random.rand(len(accident_nodes))
    P /= np.linalg.norm(P)

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

    # Set objective
    model.setObjective(((A * X) @ P).sum(), GRB.MINIMIZE)

    # Add constraint
    model.addConstr(X.sum(axis=0) == 1, "only one site")
    model.addConstr(
        X.sum(axis=1) <= len(accident_nodes) * y, "only assign to open depot"
    )
    model.addConstr(y.sum() <= num_depots, "depot opening")

    # Optimize model
    model.optimize()

    print(f"Minimum average response time: {model.ObjVal:.2f}")
