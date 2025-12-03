import pickle
import warnings
from copy import deepcopy
from joblib import Parallel, delayed

# import cvxpy as cvx
import geopandas as gpd
import gurobipy as gp
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from gurobipy import GRB
import argparse
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


def init_model(A_traffics, P, congestion_df, init_depots):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    # formulate coverage problem
    # problem variable
    n_hospitals = A_traffics[0].shape[0]
    n_accidents = A_traffics[0].shape[1]

    # Create a model using gurobi
    model = gp.Model("Accident coverage problem", env=env)

    # Create variables
    X = model.addMVar((n_hospitals, n_accidents), vtype=GRB.BINARY, name="X")
    y = model.addMVar(n_hospitals, vtype=GRB.BINARY, name="y")
    t = model.addVar()

    # Set objective
    model.setObjective(t, GRB.MINIMIZE)

    # Add constraint
    for i in range(len(congestion_df)):
        model.addConstr(
            ((A_traffics[i] * X) @ P[i, :]).sum() <= t, f"realization {i + 1}"
        )
    model.addConstr(X.sum(axis=0) == 1, "only one site")
    model.addConstr(X <= y[:, np.newaxis], "only assign to open depot")

    depot_constr = model.addConstr(y.sum() <= init_depots, "depot opening")

    return model, depot_constr, X, y, t


def batch_opt(A_traffics, P, congestion_df, min_depots, max_depots, day, time):
    init_depots = min_depots
    print(f"Initializing model for {day}, {time}")
    model, depot_constr, X, y, t = init_model(A_traffics, P, congestion_df, init_depots)
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
            "response_time": float(t.X),
        }
    print(f"Completed optimization experiments for {day}, {time}")
    return results


def run_single(day, time, min_depots, max_depots):

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
    accident_np = accident_df_filtered.to_numpy()
    accident_counts = accident_np.sum(axis=1)
    P = accident_np / accident_counts[:, np.newaxis]

    return batch_opt(A_traffics, P, congestion_df_filtered, min_depots, max_depots, day, time)


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
        delayed(run_single)(day, time, min_depots, max_depots) 
        for day in ["wd", "we"] 
        for time in ["early", "morning", "midday", "evening", "night"]
    )

    print("Jobs completed - now saving")
    with open("results/experimental_results.pickle", "wb") as f:
        pickle.dump(results, f)
    print("Success!")



    