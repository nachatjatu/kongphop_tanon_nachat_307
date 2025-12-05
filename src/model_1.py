import pickle

# import cvxpy as cvx
import geopandas as gpd
import gurobipy as gp
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from gurobipy import GRB
from tqdm import tqdm
import argparse


def init_model(A, P, init_depots):
    # formulate coverage problem
    # problem variable
    n_hospitals = A.shape[0]
    n_accidents = A.shape[1]

    # Create a model using gurobi
    model = gp.Model("Accident coverage problem")

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


def batch_opt(A_traffics, P, min_depots, max_depots, day, time, step):
    # initialize model
    print(f"Initializing model for {day}, {time}")
    model, depot_constr, X, y = init_model(A_traffics, P, min_depots)
    print(f"Model initialized")
    results = {}

    # re-run optimizations with different RHS values
    for num_depots in tqdm(range(min_depots, max_depots+1, step)):
        depot_constr.RHS = num_depots
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



def run_single(day, time, min_depots, max_depots, G, gas_stations, step=1):
    # load data
    print(f"Loading accident and congestion data for {day}, {time}")
    accident_df = pd.read_csv(f"processed_data/accident_tables/{day}_{time}_accidents.csv")
    congestion_df = pd.read_csv(f"processed_data/congestion_tables/{day}_{time}_congestion.csv")

    # process data
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
    hospital_nodes = nodes[nodes["vtype"] == "hosp"].index

    depot_nodes = list(gas_stations) + list(hospital_nodes)

    accident_nodes = list(nodes[nodes["vtype"] == "accident"].index)
    A = np.zeros((len(depot_nodes), len(accident_nodes)))
    for i, depot_id in enumerate(depot_nodes):
        length = nx.single_source_dijkstra_path_length(G, depot_id, weight="travel_time")
        for j, accident_id in enumerate(accident_nodes):
            A[i, j] = length[accident_id]

    # probability of accident sites and realization
    P = accident_freqs
    # solve batch
    return batch_opt(A, P, min_depots, max_depots, day, time, step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_depots", type=int)
    parser.add_argument("--max_depots", type=int)
    parser.add_argument("--step", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    min_depots, max_depots, step = args.min_depots, args.max_depots, args.step

    # load supplementary data
    with open("processed_data/bkk_augmented_graph.pickle", "rb") as f:
        G = pickle.load(f)
    with open('depots/gas_stations.pickle', 'rb') as f:
        gas_stations = pickle.load(f)

    # perform optimization experiments
    results = {}
    for day in ["wd", "we"] :
        results[day] = {}
        for time in ["early", "morning", "midday", "evening", "night"]:
            res = run_single(day, time, min_depots, max_depots, G, gas_stations, step=step)
            results[day][time] = res

    # save files
    print("Jobs completed - now saving")
    with open("results/model_1_raw.pickle", "wb") as f:
        pickle.dump(results, f)
    print("Success!")









