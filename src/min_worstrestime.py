import pickle
from copy import deepcopy


import pandas as pd
# import cvxpy as cvx
import geopandas as gpd
import gurobipy as gp
import networkx as nx
import numpy as np
import osmnx as ox
from gurobipy import GRB
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(0)

if __name__ == "__main__":
    with open("processed_data/bkk_augmented_graph.pickle", "rb") as f:
        G = pickle.load(f)

    # load CSV
    accident_df = pd.read_csv('processed_data/accident_tables/wd_morning_accidents.csv').drop('date', axis=1)
    congestion_df = pd.read_csv('processed_data/congestion_tables/wd_morning_congestion.csv').drop('date', axis=1)

    to_drop = accident_df.isna().any(axis=1) | congestion_df.isna().any(axis=1) | (accident_df == 0).all(axis=1)

    accident_df = accident_df[~to_drop].head(30)
    congestion_df = congestion_df[~to_drop].head(30)

    # process accidents
    accident_np = accident_df.to_numpy()
    accident_counts = accident_np.sum(axis=1)
    accident_freqs = accident_np / accident_counts[:, np.newaxis]

    # compute travel time
    G = ox.routing.add_edge_speeds(G)

    # impute travel time for each scenario
    G_traffics = []
    for i, row in tqdm(congestion_df.iterrows()):
        G_traffics.append(deepcopy(G))

        # impute travel time for each scenario
        for j, edge in enumerate(G_traffics[-1].edges):
            G_traffics[-1][edge[0]][edge[1]][edge[2]][
                "speed_kph"
            ] *= (1 - row[j])

        G_traffics[-1] = ox.routing.add_edge_travel_times(G_traffics[-1])


    # compute shortest time between depots and accidents
    A_traffics = []
    for G_traffic in tqdm(G_traffics):
        nodes, edges = ox.graph_to_gdfs(G_traffic)
        hospital_nodes = nodes[nodes["vtype"] == "hosp"]
        accident_nodes = nodes[nodes["vtype"] == "accident"]
        A_traffics.append(np.zeros((len(hospital_nodes), len(accident_nodes))))
        for i, hosp_id in enumerate(hospital_nodes.index):
            length = nx.single_source_dijkstra_path_length(
                G_traffic, hosp_id, weight="travel_time"
            )
            for j, accident_id in enumerate(accident_nodes.index):
                A_traffics[-1][i, j] = length[accident_id]

    # probability of accident sites and realization
    P = accident_freqs

    # formulate coverage problem
    # problem variable
    num_depots = 10

    # Create a model using gurobi
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
    for i in range(len(congestion_df)):
        model.addConstr(
            ((A_traffics[i] * X) @ P[i, :]).sum() <= t, f"realization {i + 1}"
        )
    model.addConstr(X.sum(axis=0) == 1, "only one site")
    model.addConstr(
        X.sum(axis=1) <= len(accident_nodes) * y, "only assign to open depot"
    )
    model.addConstr(y.sum() <= num_depots, "depot opening")

    # Optimize model
    model.optimize()

    print(f"Minimum worst case response time: {model.ObjVal:.2f}")
    print(y.X)

    # # Create variables
    # X = cvx.Variable((len(hospital_nodes), len(accident_nodes)), boolean=True)
    # y = cvx.Variable(len(hospital_nodes), boolean=True)
    # t = cvx.Variable()

    # # Set objective
    # obj = cvx.Minimize(t)

    # # Add constraint
    # constraints = [
    #     cvx.sum(X, axis=0) == 1,
    #     X <= y[:, np.newaxis],
    #     cvx.sum(y) <= num_depots,
    # ]
    # for i in range(n_realizations):
    #     constraints.append(cvx.sum(cvx.multiply(A_traffics[i], X) @ P[i, :]) <= t)

    # # Optimize model
    # prob = cvx.Problem(obj, constraints)
    # prob.solve(solver=cvx.SCIP, verbose=True)

    # print(f"Minimum average response time: {prob.value:.2f}")
    # print(y.value)
