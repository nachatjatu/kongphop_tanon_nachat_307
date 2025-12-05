import pickle
import warnings
from copy import deepcopy

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
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    # formulate coverage problem
    # problem variable
    n_hospitals = A_traffics[0].shape[0]
    n_accidents = A_traffics[0].shape[1]

    # Create a model using gurobi
    model = gp.Model("Accident coverage problem")
    
    model.setParam("Threads", 12)

    # Create variables
    X = model.addMVar((n_hospitals, n_accidents), vtype=GRB.BINARY, name="X")
    y = model.addMVar(n_hospitals, vtype=GRB.BINARY, name="y")


    # Add constraint
    scenario_times = []
    for i in range(len(congestion_df)):
        T_i = ((A_traffics[i] * X) @ P[i, :]).sum()
        scenario_times.append(T_i)

    model.setObjective(gp.quicksum(scenario_times) / len(scenario_times), GRB.MINIMIZE)

    model.addConstr(X.sum(axis=0) == 1, "only one site")
    model.addConstr(X <= y[:, np.newaxis], "only assign to open depot")

    depot_constr = model.addConstr(y.sum() <= init_depots, "depot opening")

    return model, depot_constr, X, y


def batch_opt(A_traffics, P, congestion_df, min_depots, max_depots, day, time, step):
    # initialize model
    print(f"Initializing model for {day}, {time}")
    model, depot_constr, X, y = init_model(A_traffics, P, congestion_df, min_depots)
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
    # for memory
    model.dispose()
    del model, X, y, depot_constr
    return results


def run_single(day, time, min_depots, max_depots, step=1):
    tag = f'{day}_{time}'
    print(f"Loading accident and congestion data for {tag}")
    accident_df = pd.read_csv(f"processed_data/train/{tag}_accidents_train.csv").drop('date', axis=1)
    congestion_df = pd.read_csv(f"processed_data/train/{tag}_congestion_train.csv").drop('date', axis=1)
    A_traffics = np.load(f'processed_data/train/A_traffics_{tag}_train.npy')

    # process accidents
    accident_np = accident_df.to_numpy()
    accident_counts = accident_np.sum(axis=1)
    P = accident_np / accident_counts[:, np.newaxis]

    # solve batch
    return batch_opt(A_traffics, P, congestion_df, min_depots, max_depots, day, time, step)


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
    for day in ["wd", "we"]:
        results[day] = {}
        for time in ["early", "morning", "midday", "evening", "night"]:
            res = run_single(day, time, min_depots, max_depots, step)
            results[day][time] = res

            
    # save files
    print("Jobs completed - now saving")
    with open("results/model_2_raw.pickle", "wb") as f:
        pickle.dump(results, f)
    print("Success!")



    