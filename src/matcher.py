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

from shapely.geometry import Point

warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
warnings.filterwarnings("ignore", category=FutureWarning, module="mappymatch")

M = 1e9

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


def find_full_chains(df_indexed, min_length=1):
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
                "point_geom": pt,
                "edge_id": (row["u"], row["v"], row["key"]),
                "edge_geom": row.geometry,
                "edge_name": str(row["name"])
            })

    return pd.DataFrame(results)

def compute_candidate_projections(df):
    proj_pts = []
    dists = []

    for edge_geom, pt in zip(df["edge_geom"], df["point_geom"]):
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
            "point_geometry": sub["point_geom"].tolist(),
            "projected_point": sub["projected_point"].tolist(),
            "edge_geometry": sub["edge_geom"].tolist(),
            "edge_id": sub["edge_id"].tolist(),
            "edge_name": sub["edge_name"].tolist(),
            "dist_to_candidate": sub["dist_to_candidate"].tolist(),
            "point_id": list(sub.index)           # 0..K-1
        }

    return matchings

def add_points_to_edge(G, edges_gdf, edge_id, split_list, vtype):
    """
    Split a graph edge at multiple projected match points and insert virtual nodes.
    Enhanced to avoid zero-length or POINT segments using epsilon spacing.
    """

    from shapely.geometry import LineString, Point
    from shapely.ops import substring

    # Spacing / safety parameters
    eps = 0.05          # minimum spacing between splits (meters)
    min_seg_len = 0.5   # minimum edge segment length (meters)

    u, v, k = edge_id

    # 1. Original edge geometry
    attrs = edges_gdf.loc[u, v, k].copy()
    geom_3857 = edges_gdf.loc[(u, v, k)]["geometry"]
    total_len = geom_3857.length

    # 2. Compute projected distances
    raw_splits = []
    for loc_code, proj_pt, pt_id in split_list:
        try:
            s = geom_3857.project(proj_pt)
        except Exception:
            print("ERROR computing projection")
            print("edge_id:", edge_id)
            print("proj_pt:", proj_pt)
            raise
        raw_splits.append((s, loc_code, proj_pt, pt_id))

    # Sort by distance
    raw_splits.sort(key=lambda x: x[0])

    # 3. Remove original forward edge
    G.remove_edge(u, v, k)

    # ------------------------------
    # 4. EPSILON SPACING / DEDUP
    # ------------------------------
    clean_s = []
    for (s, loc_code, proj_pt, pt_id) in raw_splits:
        if not clean_s:
            clean_s.append(s)
        else:
            if s - clean_s[-1] < eps:
                clean_s.append(clean_s[-1] + eps)
            else:
                clean_s.append(s)

    # Clip to valid range
    clean_s = [max(0, min(total_len, s)) for s in clean_s]

    # 5. Build node list
    nodes = [u]
    dists = [0]
    new_nodes = []

    for (dist, (_, loc_code, proj_pt, pt_id)) in zip(clean_s, raw_splits):

        name_comps = []

        name_comps.append(str(loc_code))
        if pt_id != "":
            name_comps.append(str(pt_id))


        new_name = "_".join(name_comps)

        G.add_node(
            new_name,
            x=proj_pt.x,
            y=proj_pt.y,
            geometry=proj_pt,
            vtype=vtype,
            street_count=1,
        )

        nodes.append(new_name)
        dists.append(dist)
        new_nodes.append(new_name)

    nodes.append(v)
    dists.append(total_len)

    # ------------------------------
    # 6. SAFE SUBSTRING
    # ------------------------------
    def safe_substring(line, start, end):
        # Ensure ordering
        if end < start:
            start, end = end, start

        # Enforce minimum segment size
        if end - start < min_seg_len:
            end = start + min_seg_len

        # Clip
        start = max(0, min(start, line.length))
        end   = max(0, min(end,   line.length))

        if end <= start:         # fallback
            end = min(line.length, start + min_seg_len)

        seg = substring(line, start, end)

        if seg.is_empty or seg.geom_type != "LineString":
            p1 = line.interpolate(start)
            p2 = line.interpolate(end)
            seg = LineString([p1, p2])

        return seg

    # ------------------------------
    # 7. ADD SUB-EDGES
    # ------------------------------
    for i in range(len(nodes) - 1):

        start, end = dists[i], dists[i+1]
        seg_3857 = safe_substring(geom_3857, start, end)
        seg_len  = seg_3857.length

        attrs2 = attrs.copy()
        attrs2["geometry"] = seg_3857
        attrs2["length"] = seg_len
        attrs2["is_subedge"] = True

        G.add_edge(nodes[i], nodes[i+1], **attrs2)


def build_augmented_graph_batch(G, edges_gdf, matchings, vtype):
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
        add_points_to_edge(G_aug, edges_gdf_indexed, edge_id, split_list, vtype=vtype)

    return G_aug


def dijkstra_multi_target(
    G, source, targets, weight="length", cutoff=5000
):
    """
    Multi-target Dijkstra with:
    - Early stopping when all targets found
    - Early stopping when frontier distance > best_found target
    - Optional cutoff distance
    """
    targets = set(targets)
    dist = {source: 0}
    found = {}

    counter = itertools.count()
    pq = [(0, next(counter), source)]

    best_found = float("inf")

    while pq and targets:
        d, _, u = heapq.heappop(pq)

        # stale entry
        if d > dist[u]:
            continue

        # cutoff pruning
        if cutoff is not None and d > cutoff:
            break

        # reached a target
        if u in targets:
            found[u] = d
            targets.remove(u)
            best_found = min(best_found, d)

            # if no targets left, stop
            if not targets:
                break

        # relax edges
        for v, attrdict in G[u].items():
            key = next(iter(attrdict))       # pick first key for MultiDiGraph
            attrs = attrdict[key]
            w = attrs.get(weight, 1)

            nd = d + w

            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, next(counter), v))

    return found


def shortest_paths_multi_target(G, source, next_loc_code, next_pt_ids):
    """
    Return distances from source to each next candidate.
    Unreachable targets get np.inf.
    """

    targets = [f"{next_loc_code}_{pid}" for pid in next_pt_ids]
    found = dijkstra_multi_target(G, source, targets, weight="length")

    # Assign infinity for unreachable targets
    distances = [found.get(t, M) for t in targets]

    return np.array(distances, dtype=float)



def backward(G, chain, matchings, alpha=1.0, beta=1.0, verbose=False):
    """
    Dynamic programming (backward pass) to select best candidate for each TMC point.

    emission cost  = dist_to_candidate
    transition cost = shortest-path cost
    """

    N = len(chain)
    V = {}
    ptr = {}

    if verbose:
        print("\n=================== BACKWARD DP START ===================\n")

    for i in range(N - 1, -1, -1):

        loc_code = chain[i]
        m_i = matchings[loc_code]

        pt_ids_i   = m_i["point_id"]
        emission_i = np.array(m_i["dist_to_candidate"], dtype=float)

        K = len(pt_ids_i)

        if verbose:
            print("\n-----------------------------------------------------------")
            print(f" STEP i = {i}   (TMC point = {loc_code})")
            print(f" Candidates at this step: {pt_ids_i}")
            print(f" Emission costs: {emission_i}")

        # ---------------- Base Case ----------------
        if i == N - 1:
            V[i] = -alpha * emission_i
            if verbose:
                print("\n  Base case (last point).")
                print(f"  V[{i}] = {V[i]}")
            continue

        # ---------------- Recursive Case ----------------
        V[i] = np.empty(K)
        ptr[i] = np.empty(K, dtype=int)

        next_loc = chain[i+1]
        m_next = matchings[next_loc]
        next_point_ids = m_next["point_id"]
        K_next = len(next_point_ids)

        if verbose:
            print(f"\n  Next step: i+1 = {i+1}, TMC = {next_loc}")
            print(f"  Next candidates: {next_point_ids}")
            print(f"  Future values V[{i+1}]: {V[i+1]}")

        for j in range(K):
            pt_id_j = pt_ids_i[j]
            source_node = f"{loc_code}_{pt_id_j}"

            if verbose:
                print(f"\n    Candidate j = {j}, point_id = {pt_id_j}")
                print(f"    Emission cost (alpha * dist): {alpha * emission_i[j]:.3f}")
                print(f"    Source virtual node: {source_node}")

            # Transition distances
            trans_dists = shortest_paths_multi_target(
                G,
                source_node,
                next_loc,
                next_point_ids
            )

            if verbose:
                print(f"    Transition distances to next candidates:")
                for b in range(K_next):
                    print(f"      -> to {next_point_ids[b]}: dist={trans_dists[b]:.3f}")

            # Bellman backup
            values = -(alpha * emission_i[j] + beta * trans_dists) + V[i+1]

            if verbose:
                print("    Value contributions (for each next candidate b):")
                for b in range(K_next):
                    cost_component = -(alpha * emission_i[j] + beta * trans_dists[b])
                    print(f"      b={b} (pt={next_point_ids[b]}):")
                    print(f"        cost_component = {cost_component:.3f}")
                    print(f"        future value V[i+1][b] = {V[i+1][b]:.3f}")
                    print(f"        total value = {values[b]:.3f}")

            k_best = np.argmax(values)

            if verbose:
                print(f"    >>> Best next candidate = b={k_best} "
                      f"(point_id={next_point_ids[k_best]}) "
                      f"with value {values[k_best]:.3f}")

            V[i][j] = values[k_best]
            ptr[i][j] = k_best

        if verbose:
            print(f"\n  Final V[{i}] = {V[i]}")
            print(f"  Pointers ptr[{i}] = {ptr[i]}")

    if verbose:
        print("\n==================== BACKWARD DP END =====================\n")

    return V, ptr



def reconstruct_best_path(V, ptr, chain, matchings, verbose=False):
    """
    Reconstruct the best candidate index at each TMC step using V and ptr.

    Returns:
      path_indices : list[int]   # index of chosen candidate at each step
    """
    N = len(chain)

    # 1. Best candidate for step 0
    if 0 not in V:
        raise ValueError("V[0] missing — backward DP incomplete.")

    j0 = int(np.argmax(V[0]))
    path = [j0]

    if verbose:
        print("\n--- RECONSTRUCTING BEST PATH ---")
        print(f"Step 0: best j = {j0}, point_id = {matchings[chain[0]]['point_id'][j0]}")

    # 2. Follow pointers forward
    for i in range(N - 1):
        j = path[-1]  # index chosen at step i

        if i not in ptr:
            raise ValueError(f"ptr[{i}] missing — backward DP did not create pointer for step {i}.")

        if j < 0 or j >= len(ptr[i]):
            raise ValueError(f"Invalid pointer access: ptr[{i}][{j}]")

        next_j = int(ptr[i][j])
        path.append(next_j)

        if verbose:
            curr_loc = chain[i]
            next_loc = chain[i+1]
            print(f"Step {i+1}: from {curr_loc} j={j} → {next_loc} next_j={next_j} "
                  f"(point_id={matchings[next_loc]['point_id'][next_j]})")

    return path



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
            "edge_name": m["edge_name"][j],
            "dist_to_candidate": m["dist_to_candidate"][j],
            "candidate_index": j
        }
    return mapping


def build_final_graph(G, edges_gdf, all_mappings):
    """
    Use candidate matchings to add virtual nodes onto edges in G.
    """

    G_final = G.copy()

    # 1. Group by edge_id
    edge_groups = defaultdict(list)

    for loc_code, m in all_mappings.items():
        edge_id = m["edge_id"]
        proj_pt = m["projected_point"]
        pid = m["point_id"]
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
        add_points_to_edge(G_final, edges_gdf_indexed, edge_id, split_list, vtype="virt")

    return G_final