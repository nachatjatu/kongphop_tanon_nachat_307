import pickle

# import cvxpy as cvx
import geopandas as gpd
import gurobipy as gp
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from gurobipy import GRB
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse


def init_model(A, P, init_depots):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    # formulate coverage problem
    # problem variable
    n_hospitals = A.shape[0]
    n_accidents = A.shape[1]

    # Create a model using gurobi
    model = gp.Model("Accident coverage problem", env=env)

    # Create variables
    X = model.addMVar((n_hospitals, n_accidents), vtype=GRB.BINARY, name="X")
    y = model.addMVar(n_hospitals, vtype=GRB.BINARY, name="y")

    # Set objective
    model.setObjective(((A * X) @ P).sum(), GRB.MINIMIZE)

    # Add constraint
    model.addConstr(X.sum(axis=0) == 1, "only one site")
    model.addConstr(X <= y[:, np.newaxis], "only assign to open depot")

    depot_constr = model.addConstr(y.sum() <= init_depots, "depot opening")

    return model, depot_constr, X, y


def batch_opt(A_traffics, P, min_depots, max_depots, day, time):
    init_depots = min_depots
    print(f"Initializing model for {day}, {time}")
    model, depot_constr, X, y = init_model(A_traffics, P, init_depots)
    print(f"Model initialized")
    results = {}

    for num_depots in tqdm(range(min_depots, max_depots+1)):
        depot_constr.RHS = num_depots

        # Optimize model
        model.optimize()

        results[num_depots] = {
            "day": day,
            "time": time,
            "status": int(model.status),
            "objective": model.ObjVal if model.Status == GRB.OPTIMAL else None,
            "y": y.X.tolist(),
            "X": X.X.tolist(),
        }
    print(f"Completed optimization experiments for {day}, {time}")
    return results



def run_single(day, time, min_depots, max_depots, G):

    print(f"Loading accident and congestion data for {day}, {time}")
    A_traffics = np.load(f"processed_data/accident_tables/A_traffics_{day}_{time}.npy")
    accident_df = pd.read_csv(f"processed_data/accident_tables/{day}_{time}_accidents.csv")
    congestion_df = pd.read_csv(f"processed_data/congestion_tables/{day}_{time}_congestion.csv")

    print(f"Processing data")

    accident_df["date"] = pd.to_datetime(accident_df["date"])
    congestion_df["date"] = pd.to_datetime(congestion_df["date"])

    accident_df_filtered = accident_df[accident_df["date"].dt.year < 2023].drop("date", axis=1)
    congestion_df_filtered = congestion_df[congestion_df["date"].dt.year < 2023].drop("date", axis=1)

    to_drop = (
        accident_df_filtered.isna().any(axis=1)
        | congestion_df_filtered.isna().any(axis=1)
        | (accident_df_filtered == 0).all(axis=1)
    )

    accident_df_filtered = accident_df_filtered[~to_drop]
    congestion_df_filtered = congestion_df_filtered[~to_drop]

    # process accidents
    accident_counts = (accident_df_filtered.to_numpy()).sum(axis=0)
    accident_freqs = accident_counts / accident_counts.sum()
    # process congestion
    congestion_factors = congestion_df_filtered.mean(axis=0)

    G = ox.add_edge_speeds(G)

    # impute travel time for each scenario
    for idx, edge in enumerate(G.edges):
        G[edge[0]][edge[1]][edge[2]]["speed_kph"] *= 1 - congestion_factors[idx]

    G = ox.routing.add_edge_travel_times(G)

    # compute shortest time between depots and accidents
    nodes, edges = ox.graph_to_gdfs(G)
    hospital_nodes = nodes[nodes["vtype"] == "hosp"]
    accident_nodes = nodes[nodes["vtype"] == "accident"]
    A = np.zeros((len(hospital_nodes), len(accident_nodes)))
    for i, hosp_id in enumerate(hospital_nodes.index):
        length = nx.single_source_dijkstra_path_length(G, hosp_id, weight="travel_time")
        for j, accident_id in enumerate(accident_nodes.index):
            A[i, j] = length[accident_id]

    # probability of accident sites and realization
    P = accident_freqs

    return batch_opt(A, P, min_depots, max_depots, day, time)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_depots", type=int)
    parser.add_argument("--max_depots", type=int)
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    min_depots, max_depots = args.min_depots, args.max_depots

    with open("processed_data/bkk_augmented_graph.pickle", "rb") as f:
        G = pickle.load(f)

    print("Creating parallel jobs")

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_single)(day, time, min_depots, max_depots, G) 
        for day in ["wd", "we"] 
        for time in ["early", "morning", "midday", "evening", "night"]
    )

    print("Jobs completed - now saving")
    with open("results/avgrestime_raw.pickle", "wb") as f:
        pickle.dump(results, f)
    print("Success!")









