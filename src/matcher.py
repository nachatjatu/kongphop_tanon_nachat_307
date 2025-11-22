import networkx as nx
import pandas as pd
import geopandas as gpd
import osmnx as ox
import warnings
import numpy as np
from tqdm import tqdm
from shapely.ops import substring
from collections import defaultdict
from pprint import pprint
import heapq
import itertools

warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
warnings.filterwarnings("ignore", category=FutureWarning, module="mappymatch")


def filter_distance(gdf, distance=15):
    """
    Filter TMC nodes by distance to the matched OSM segment and clean their
    positive/negative offset links.

    Steps:
      1. Keep only nodes within `distance` meters of their matched OSM edge.
      2. Cast POSITIVE OFFSET / NEGATIVE OFFSET to nullable Int64.
      3. Remove offsets that do not point to another node in the filtered set.
      4. Drop orphan nodes that have neither a valid positive nor negative offset.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input TMC dataset containing 'dist_to_osm_m', 'LOCATION CODE',
        'POSITIVE OFFSET', and 'NEGATIVE OFFSET'.
    distance : float, default=15
        Maximum allowed distance from the matched OSM edge.

    Returns
    -------
    GeoDataFrame
        Cleaned TMC nodes with valid offset topology.
    """

    gdf_near = gdf[gdf["dist_to_osm_m"] <= distance].copy()

    # Cast offsets correctly
    gdf_near["POSITIVE OFFSET"] = gdf_near["POSITIVE OFFSET"].astype("Int64")
    gdf_near["NEGATIVE OFFSET"] = gdf_near["NEGATIVE OFFSET"].astype("Int64")

    # Validate positive offsets
    valid_codes = set(gdf_near["LOCATION CODE"])
    gdf_near["positive_ok"] = gdf_near["POSITIVE OFFSET"].isin(valid_codes)

    # Remove invalid positive links
    gdf_near.loc[~gdf_near["positive_ok"], "POSITIVE OFFSET"] = pd.NA

    # Validate negative offsets
    gdf_near["negative_ok"] = gdf_near["NEGATIVE OFFSET"].isin(valid_codes)
    gdf_near.loc[~gdf_near["negative_ok"], "NEGATIVE OFFSET"] = pd.NA

    # Remove orphans: keep only nodes that have at least ONE valid neighbor
    gdf_clean = gdf_near[
        gdf_near["POSITIVE OFFSET"].notna() | gdf_near["NEGATIVE OFFSET"].notna()
    ].copy()

    return gdf_clean


def find_full_chains(df_indexed, min_length=3):
    """
    Build full forward-linked TMC chains by walking POSITIVE OFFSET links
    starting from every node with no NEGATIVE OFFSET.

    Only chains with length ≥ min_length are returned.

    Parameters
    ----------
    df_indexed : DataFrame
        TMC dataset indexed by LOCATION CODE, containing POSITIVE/NEGATIVE OFFSET.
    min_length : int, default=3
        Minimum chain length to include.

    Returns
    -------
    list[list]
        A list of chains, each a list of LOCATION CODEs.
    """

    chains = []

    starting_points = df_indexed[df_indexed['NEGATIVE OFFSET'].isna()]

    for start_code in tqdm(starting_points.index):
        curr = start_code
        full_chain = []
        while True:
            full_chain.append(curr)
            nxt = df_indexed.at[curr, "POSITIVE OFFSET"]
            if pd.isna(nxt):
                break
            curr = nxt
        if len(full_chain) >= min_length:
            chains.append(full_chain)

    return chains


def find_rolling_chains(chains, max_length=7, min_length=3):
    """
    Expand full chains into rolling subchains (sliding windows), keeping only
    windows whose lengths fall between min_length and max_length.

    Parameters
    ----------
    chains : list[list]
        Output from find_full_chains.
    max_length : int, default=7
        Maximum window size.
    min_length : int, default=3
        Minimum window size.

    Returns
    -------
    list[list]
        All rolling subchains across all input chains.
    """

    rolling_chains = []
    for chain in tqdm(chains):
        n = len(chain)
        # --- Emit rolling windows with length ∈ [min_length, max_length] ---
        for i in range(n):
            # window ends at j (exclusive), so j ranges from i+min_length to i+max_length
            for j in range(i + min_length, min(i + max_length, n) + 1):
                subchain = chain[i:j]
                rolling_chains.append(subchain)
    return rolling_chains



def candidate_radius_join(points_gdf, edges_gdf, radius=40):
    sindex = edges_gdf.sindex
    results = []

    for loc_code, pt in points_gdf.geometry.items():
        buff = pt.buffer(radius)
        idx = list(sindex.intersection(buff.bounds))
        if len(idx) == 0:
            continue

        cand = edges_gdf.iloc[idx]
        cand = cand[cand.intersects(buff)]

        for _, row in cand.iterrows():
            results.append({
                "loc_code": loc_code,
                "pt_geom": pt,
                "edge_id": (row["u"], row["v"], row["key"]),
                "edge_geom": row.geometry
            })

    return pd.DataFrame(results)

def compute_candidate_projections(df):
    proj_pts = []
    dists = []

    for edge_geom, pt in zip(df["edge_geom"], df["pt_geom"]):
        d = edge_geom.project(pt)
        proj = edge_geom.interpolate(d)
        dist = pt.distance(proj)

        proj_pts.append(proj)
        dists.append(dist)

    df["projected_point"] = proj_pts
    df["dist_to_candidate"] = dists
    return df

def build_matchings(df):
    matchings = {}
    for loc_code, sub in df.groupby("loc_code"):
        sub = sub.reset_index(drop=True)

        matchings[loc_code] = {
            "point_geometry": sub["pt_geom"].tolist(),
            "projected_point": sub["projected_point"].tolist(),
            "edge_geometry": sub["edge_geom"].tolist(),
            "edge_id": sub["edge_id"].tolist(),
            "dist_to_candidate": sub["dist_to_candidate"].tolist(),
            "point_id": list(sub.index)           # 0..K-1
        }

    return matchings



def add_points_to_edge(G, edges_gdf, edge_id, split_list, vtype):
    """
    Split a graph edge at multiple projected match points and insert virtual nodes.

    Parameters
    ----------
    G : MultiDiGraph (projected, EPSG:3857)
    edge_id : tuple (u, v, k)
        The ID of the edge to split.
    split_list : list of (loc_code, projected_point, point_id)
        projected_point MUST be a POINT IN EPSG:3857 (on the road)
        loc_code used for naming, point_id ensures unique candidate.
    vtype : str
        Tag to assign to virtual nodes (e.g., 'virt')

    Notes
    -----
    - NO CRS transformation is performed here.
    - Edge geometry MUST be in EPSG:3857.
    - projected_point MUST be in EPSG:3857.
    """

    u, v, k = edge_id

    # 1. Get the original edge geometry (already in 3857)
    attrs = G.edges[u, v, k].copy()
    geom_3857 = edges_gdf.loc[(u, v, k)]["geometry"]      # must be a LineString in 3857

    # 2. Compute distances along the line for each projected point
    splits = []
    for loc_code, proj_pt, pt_id in split_list:
        try:
            s = geom_3857.project(proj_pt)   # safe: both in 3857
        except Exception:
            print("ERROR computing projection")
            print("edge_id:", edge_id)
            print("edge_geom:", geom_3857)
            print("proj_pt:", proj_pt)
            raise
        splits.append((s, loc_code, proj_pt, pt_id))

    # Sort by distance along the edge
    splits.sort(key=lambda x: x[0])

    # 3. Remove original forward edge
    G.remove_edge(u, v, k)

    # 4. Build list of nodes in sequence: u -> virtual nodes -> v
    nodes = [u]
    dists = [0] + [s for (s, _, _, _) in splits] + [geom_3857.length]
    new_nodes = []

    # Create virtual nodes for each split point
    for (_, loc_code, proj_pt, pt_id) in splits:
        new_node = f"{vtype}_{loc_code}_{pt_id}"

        # Add virtual node
        G.add_node(
            new_node,
            x=proj_pt.x,        # coordinates in 3857
            y=proj_pt.y,
            geometry=proj_pt,
            vtype=vtype
        )

        nodes.append(new_node)
        new_nodes.append(new_node)

    # End at v
    nodes.append(v)

    # 5. Add new forward subedges
    for i in range(len(nodes) - 1):

        start, end = dists[i], dists[i+1]

        seg_3857 = substring(geom_3857, start, end)    # stays in 3857
        seg_len  = seg_3857.length

        # copy base attributes
        attrs2 = attrs.copy()
        attrs2["geometry"] = seg_3857
        attrs2["length"] = seg_len
        attrs2["is_subedge"] = True

        G.add_edge(nodes[i], nodes[i+1], **attrs2)


def build_augmented_graph_batch(G, edges_gdf, matchings):
    """
    Use candidate matchings to add virtual nodes onto edges in G.
    """

    G_aug = G.copy()

    # 1. Group by edge_id
    edge_groups = defaultdict(list)

    for loc_code, m in matchings.items():
        for edge_id, proj_pt, pid in zip(m["edge_id"], 
                                         m["projected_point"], 
                                         m["point_id"]):
            edge_groups[edge_id].append((loc_code, proj_pt, pid))

    # 2. Split edges
    edges_gdf_indexed = (
        edges_gdf
        .assign(key=lambda df: df["key"].astype(int))   # ensure integer keys
        .set_index(["u","v","key"])
        .sort_index()
    )

    for edge_id, split_list in edge_groups.items():
        assert edge_id in edges_gdf_indexed.index, f"Missing edge geometry for {edge_id}"
        add_points_to_edge(G_aug, edges_gdf_indexed, edge_id, split_list, vtype="virt")

    return G_aug


def dijkstra_multi_target(G, source, targets, weight="length"):
    """
    Multi-target Dijkstra with early stopping and no type errors
    when node IDs include mixed types (e.g., int + str).
    """
    targets = set(targets)
    dist = {source: 0}
    found = {}

    counter = itertools.count() 
    pq = [(0, next(counter), source)]

    while pq and targets:
        d, _, u = heapq.heappop(pq)

        # stale entry
        if d > dist[u]:
            continue

        # reached a target
        if u in targets:
            found[u] = d
            targets.remove(u)
            if not targets:
                break

        # relax edges
        for v, attrdict in G[u].items():
            # MultiDiGraph: attrdict is {key: edge_attrs}
            # pick first key
            key = next(iter(attrdict))
            attrs = attrdict[key]
            w = attrs.get(weight, 1)

            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, next(counter), v))

    return found

def shortest_paths_multi_target(G, source, next_loc_code, next_pt_ids):
    """
    Return distances from source to each next candidate.
    Unreachable targets get np.inf.
    """

    targets = [f"virt_{next_loc_code}_{pid}" for pid in next_pt_ids]
    found = dijkstra_multi_target(G, source, targets, weight="length")

    # Assign infinity for unreachable targets
    distances = [found.get(t, np.inf) for t in targets]

    return np.array(distances, dtype=float)


def backward(G, chain, matchings, alpha=1.0, beta=1.0):
    """
    Dynamic programming (backward pass) to select best candidate for each TMC point.

    Uses:
      emission cost  = dist_to_candidate (meters)
      transition cost = shortest path in augmented graph between virtual nodes

    virtual node naming convention:
        virt_{loc_code}_{point_id}
    """

    N = len(chain)
    V = {}
    backpointer = {}

    # --------------------------------------------------------------
    #                DP backward pass
    # --------------------------------------------------------------
    for i in range(N - 1, -1, -1):

        loc_code = chain[i]
        m_i = matchings[loc_code]

        pt_ids_i   = m_i["point_id"]                # e.g. [0,1,2]
        emission_i = np.array(m_i["dist_to_candidate"], dtype=float)
        K = len(pt_ids_i)

        # ---------------- Base case ----------------
        if i == N - 1:
            V[i] = -alpha * emission_i
            backpointer[i] = np.full(K, -1)
            continue

        # ---------------- Recursive case ----------------
        V[i] = np.empty(K)
        backpointer[i] = np.empty(K, dtype=int)

        next_loc = chain[i+1]
        m_next = matchings[next_loc]
        next_point_ids = m_next["point_id"]           # e.g. [0,1,2,3]

        for j in range(K):

            pt_id_j = pt_ids_i[j]
            source_node = f"virt_{loc_code}_{pt_id_j}"

            # ---- Transition distances (source → each next candidate) ----
            trans_dists = shortest_paths_multi_target(
                G,
                source_node,
                next_loc,
                next_point_ids
            )  # array of shape (K_next,)

            # Bellman backup:
            #  value_j = - (alpha*emission[j] + beta*transition[j→k]) + V_next[k]
            values = -(alpha * emission_i[j] + beta * trans_dists) + V[i+1]

            k_best = np.argmax(values)
            V[i][j] = values[k_best]
            backpointer[i][j] = k_best

    # --------------------------------------------------------------
    #                   Backtracking
    # --------------------------------------------------------------
    path_idx = np.empty(N, dtype=int)   # best candidate index per TMC index
    chosen_point_ids = []

    # best candidate at first location
    path_idx[0] = np.argmax(V[0])
    chosen_point_ids.append(matchings[chain[0]]["point_id"][path_idx[0]])

    # follow backpointers
    for i in range(N-1):
        path_idx[i+1] = backpointer[i][path_idx[i]]
        chosen_point_ids.append(matchings[chain[i+1]]["point_id"][path_idx[i+1]])

    return V, path_idx, chosen_point_ids

def reconstruct_chosen_mapping(chain, matchings, path_idx):
    mapping = {}
    for i, loc_code in enumerate(chain):
        j = int(path_idx[i])
        m = matchings[loc_code]

        mapping[loc_code] = {
            "loc_code": loc_code,
            "point_geom": m["point_geometry"][j],     # correct
            "projected_point": m["projected_point"][j], # correct (POINT)
            "point_id": m["point_id"][j],             # correct (INT)
            "edge_geom": m["edge_geometry"][j],
            "edge_id": m["edge_id"][j],
            "candidate_index": j
        }
    return mapping



def chain_is_valid(chain, matchings):
    """
    Check that every LOCATION CODE in a chain has at least one
    valid candidate match in the matchings table.

    Parameters
    ----------
    chain : list
        Sequence of LOCATION CODEs.
    matchings : dict
        Mapping loc_code -> candidate match info.

    Returns
    -------
    bool
        True if all nodes have ≥1 candidate, else False.
    """

    return all(cid in matchings and len(matchings[cid]["point_geometry"]) > 0
               for cid in chain)

