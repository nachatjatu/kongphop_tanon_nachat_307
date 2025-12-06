import argparse
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
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


def init_model1(A_traffics, P, congestion_df, init_depots):
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    # formulate coverage problem
    # problem variable
    n_hospitals = A_traffics[0].shape[0]
    n_accidents = A_traffics[0].shape[1]

    # Create a model using gurobi
    model = gp.Model("Accident coverage problem")

    model.setParam('MIPGap', 0.03)
    model.setParam("Threads", 12)

    # Create variables
    X = model.addMVar((n_hospitals, n_accidents), vtype=GRB.BINARY, name="X")
    y = model.addMVar(n_hospitals, vtype=GRB.BINARY, name="y")
    t = model.addVar()

    # Set objective
    model.setObjective(t, GRB.MINIMIZE)

    # Add constraint
    for i in range(len(congestion_df)):
        P_broadcast = P[i, np.newaxis, :]
        model.addConstr(
            (A_traffics[i] * X * P_broadcast).sum() <= t, f"realization {i + 1}"
        )
    model.addConstr(X.sum(axis=0) == 1, "only one site")
    model.addConstr(X <= y[:, np.newaxis], "only assign to open depot")

    depot_constr = model.addConstr(y.sum() <= init_depots, "depot opening")

    return model, depot_constr, X, y, t


def init_model2(A_traffics, P, congestion_df, init_depots, init_worst_times):
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    # formulate coverage problem
    # problem variable
    n_hospitals = A_traffics[0].shape[0]
    n_accidents = A_traffics[0].shape[1]

    # Create a model using gurobi
    model = gp.Model("Accident coverage problem")

    model.setParam('MIPGap', 0.01)
    model.setParam("Threads", 12)

    # Create variables
    X = model.addMVar((n_hospitals, n_accidents), vtype=GRB.BINARY, name="X")
    y = model.addMVar(n_hospitals, vtype=GRB.BINARY, name="y")

    # Add constraint
    scenario_times = []
    for i in range(len(congestion_df)):
        P_broadcast = P[i, np.newaxis, :]
        T_i = (A_traffics[i] * X * P_broadcast).sum()
        scenario_times.append(T_i)

    model.setObjective(gp.quicksum(scenario_times) / len(scenario_times), GRB.MINIMIZE)

    # Add constraint
    worst_case_constrs = []
    for i in range(len(congestion_df)):
        P_broadcast = P[i, np.newaxis, :]
        worst_case_constrs.append(
            model.addConstr(
                (A_traffics[i] * X * P_broadcast).sum() <= init_worst_times,
                f"realization {i + 1}",
            )
        )
    model.addConstr(X.sum(axis=0) == 1, "only one site")
    model.addConstr(X <= y[:, np.newaxis], "only assign to open depot")

    depot_constr = model.addConstr(y.sum() <= init_depots, "depot opening")

    return model, depot_constr, worst_case_constrs, X, y


def batch_opt(A_traffics, P, congestion_df, min_depots, max_depots, day, time, step):
    # initialize model
    print(f"Initializing model for {day}, {time}")
    model1, depot_constr1, X1, y1, t1 = init_model1(
        A_traffics, P, congestion_df, min_depots
    )
    model2, depot_constr2, worst_case_constrs2, X2, y2 = init_model2(
        A_traffics, P, congestion_df, min_depots, 1e6
    )
    print(f"Model initialized")
    results = {}

    # re-run optimizations with different RHS values
    for num_depots in tqdm(range(min_depots, max_depots + 1, step)):
        print(f"Optimizing {day}, {time}, {num_depots} depots")

        depot_constr1.RHS = num_depots
        model1.optimize()

        if model1.Status == GRB.OPTIMAL:
        
            depot_constr2.RHS = num_depots
            for worse_case_constr2 in worst_case_constrs2:
                worse_case_constr2.RHS = t1.X

            model2.optimize()

            results[num_depots] = {
                "day": day,
                "time": time,
                "status": int(model2.status),
                "objective": model2.ObjVal if model2.Status == GRB.OPTIMAL else None,
                "y": y2.X.tolist(),
                "X": X2.X.tolist(),
                "response_time": float(t1.X),
            }
        else:
            results[num_depots] = {
                "day": day,
                "time": time,
                "status": int(model1.status),
                "objective": model1.ObjVal if model1.Status == GRB.OPTIMAL else None,
                "y": y1.X.tolist(),
                "X": X1.X.tolist(),
                "response_time": float(t1.X),
            }
    print(f"Completed optimization experiments for {day}, {time}")
    # for memory
    model1.dispose()
    del model1, X1, y1, t1, depot_constr1
    model2.dispose()
    del model2, X2, y2, depot_constr2, worst_case_constrs2
    return results


def run_single(day, time, min_depots, max_depots, step=1, mode="train"):
    # load data
    print(f"Loading accident and congestion data for {day}, {time}")
    tag = f'{day}_{time}'
    accident_df = pd.read_csv(f"processed_data/{mode}/{tag}_accidents_{mode}.csv").drop('date', axis=1)
    congestion_df = pd.read_csv(f"processed_data/{mode}/{tag}_congestion_{mode}.csv").drop('date', axis=1)
    A_traffics = np.load(f'processed_data/{mode}/A_traffics_{tag}_{mode}.npy')

    print(len(accident_df), len(congestion_df), A_traffics.shape)

    # process data
    print(f"Processing data")

    # process accidents
    accident_np = accident_df.to_numpy()
    accident_counts = accident_np.sum(axis=1)
    P = accident_np / accident_counts[:, np.newaxis]

    # solve batch
    return batch_opt(
        A_traffics, P, congestion_df, min_depots, max_depots, day, time, step
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_depots", type=int)
    parser.add_argument("--max_depots", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--mode", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    min_depots, max_depots, step, mode = args.min_depots, args.max_depots, args.step, args.mode

    # load supplementary data
    with open("processed_data/bkk_augmented_graph.pickle", "rb") as f:
        G = pickle.load(f)
    with open("depots/gas_stations.pickle", "rb") as f:
        gas_stations = pickle.load(f)

    # perform optimization experiments
    results = {}
    
    for day in ["wd", "we"]:
        results[day] = {}
        for time in ["early", "morning", "midday", "evening", "night"]:
            res = run_single(day, time, min_depots, max_depots, step, mode)
            results[day][time] = res

            # Save after each (day, time) combination
            with open(f"results/model3_checkpoint_{mode}.pickle", "wb") as f:
                pickle.dump(results, f)


    # save files
    print("Jobs completed - now saving")
    with open(f"results/model_3_raw_{mode}.pickle", "wb") as f:
        pickle.dump(results, f)
    print("Success!")
