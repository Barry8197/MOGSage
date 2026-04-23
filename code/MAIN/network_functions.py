"""
Utilities for bipartite projection and kNN thresholding on NetworkX graphs.

Functions exported
- normalize_edges_similarity_from_node_attr
- summarize_graph
- threshold_graph¨
- integrate_networks_weighted
- gen_knn_network
- create_similarity_matrix
- abs_bicorr
- pearson_corr
- cosine_corr
- hamming_dist
- knn_graph_generation
- fix_node_attrs
- fix_pheno_onehot_in_graph
- build_phenotype_KG
- stratified_inductive_splits
- normalize_edge_weights_symmetric
- transfer_attributes
- attach_phenotype_flags_from_df
- nx_to_pyg
- drop_attrs_netx
- load_graph
- node_labels
- knn_threshold
"""

import os
import copy
import heapq
import pickle
import torch
from statistics import mean, median
import numpy as np
import networkx as nx
import astropy.stats
from scipy.sparse import coo_matrix
import pandas as pd
from typing import Optional, Tuple, Iterable, Union, Callable, Any, Dict, Hashable, Literal, Union, List, Sequence
import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
from collections.abc import Iterable
import warnings
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.cm import get_cmap
from scipy.spatial.distance import pdist, squareform
from palettable import wesanderson
import math
from collections import defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

__all__ = [
    "normalize_edges_similarity_from_node_attr",
    "summarize_graph",
    "threshold_graph",
    "integrate_networks_weighted",
    "create_similarity_matrix",
    "gen_knn_network"
    "abs_bicorr",
    "pearson_corr",
    "cosine_corr",
    "hamming_dist",
    "knn_graph_generation",
    "fix_node_attrs",
    "fix_pheno_onehot_in_graph",
    "build_phenotype_KG",
    "attach_phenotype_flags_from_df",
    "stratified_inductive_splits",
    "normalize_edge_weights_symmetric",
    "transfer_attributes",
    "nx_to_pyg",
    "drop_attrs_netx",
    "load_graph",
    "node_labels",
    "knn_threshold"
    
]

def knn_threshold(
    G: nx.Graph,
    k: int,
    weight: str = "weight",
    kind: str = "distance",      # "distance" (smaller is closer) or "similarity" (larger is closer)
    mode: str = "mutual",        # "mutual" (intersection), "union" (either side), or "out" (directed only)
    direction: str = "out",      # for DiGraph: "out"|"in"|"both"
    ensure_min_k: bool = False,  # NEW: ensure each node keeps its own k nearest edges (if available)
    keep_isolates: bool = True
):
    """
    Return a kNN-pruned copy of G by keeping only edges that satisfy the k-nearest rule.
    Preserves node/edge attributes.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph. Edges should carry a numeric attribute named `weight`.
    k : int
        Number of nearest neighbors to consider per node.
    weight : str
        Edge attribute to use as weight.
    kind : {"distance", "similarity"}
        Interpretation of weight: "distance" -> lower is closer; "similarity" -> higher is closer.
    mode : {"mutual", "union", "out"}
        - mutual: keep {u,v} if v in kNN(u) AND u in kNN(v)
        - union:  keep {u,v} if v in kNN(u) OR  u in kNN(v)  [guarantees >= k when undirected]
        - out (DiGraph only): keep u->v if v in kNN_out(u)
    direction : {"out","in","both"} for DiGraph
        Defines neighbor set used to compute kNN for directed graphs.
    ensure_min_k : bool
        If True, always keep edges from each node to its own top-k neighbors (if they exist).
        Reciprocity (per `mode`) can add more edges on top.
        For undirected graphs this guarantees degree(u) >= min(k, deg_G(u)).
    keep_isolates : bool
        If False, drop nodes with degree 0 after pruning.

    Returns
    -------
    H : graph of same class as G
        The kNN-pruned graph.
    knn : dict
        Mapping node -> set of its selected k nearest neighbors (according to `direction` for DiGraph).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("Multigraphs not supported by this helper (collapse multi-edges first).")

    is_directed = G.is_directed()
    select = heapq.nsmallest if kind == "distance" else heapq.nlargest

    # Neighbor accessor depending on graph type/direction
    if is_directed:
        if direction == "out":
            def neigh(u):
                return G[u]  # successors and their edge-attr dicts
        elif direction == "in":
            def neigh(u):
                return {v: G[v][u] for v in G.predecessors(u)}
        elif direction == "both":
            def neigh(u):
                # Combine in/out; if both directions exist, merge by best weight
                d = {}
                for v in G.successors(u):
                    d[v] = G[u][v]
                for v in G.predecessors(u):
                    if v in d:
                        w1 = d[v].get(weight, 1.0)
                        w2 = G[v][u].get(weight, 1.0)
                        best = min(w1, w2) if kind == "distance" else max(w1, w2)
                        merged = {**d[v], **G[v][u]}
                        merged[weight] = best
                        d[v] = merged
                    else:
                        d[v] = G[v][u]
                return d
        else:
            raise ValueError("direction must be 'out', 'in', or 'both'")
    else:
        def neigh(u):
            return G[u]  # adjacency for undirected graph

    # Build kNN set for each node
    knn = {}
    for u in G.nodes():
        nb = []
        for v, attr in neigh(u).items():
            if u == v:
                continue
            w = attr.get(weight, 1.0)
            nb.append((v, w))

        if len(nb) <= k:
            chosen = [v for v, _ in nb]
        else:
            chosen = [v for v, _ in select(k, nb, key=lambda x: x[1])]
        knn[u] = set(chosen)

    # Decide which edges to keep
    keep_edges = set()

    def add_edge(u, v):
        if is_directed:
            keep_edges.add((u, v))
        else:
            a, b = (u, v) if u <= v else (v, u)
            keep_edges.add((a, b))

    # 1) Ensure minimum k per node by adding each node's own kNN edges
    if ensure_min_k:
        for u, vs in knn.items():
            for v in vs:
                add_edge(u, v)

    # 2) Apply reciprocity rule (mutual/union/out)
    if mode == "mutual":
        if is_directed:
            # For mutual in DiGraph, treat reciprocity on the underlying undirected relation
            for u in G.nodes():
                for v in knn[u]:
                    if u in knn.get(v, ()):
                        add_edge(u, v)
                        add_edge(v, u)  # keep both directions if they exist
        else:
            for u in G.nodes():
                for v in knn[u]:
                    if u in knn.get(v, ()):
                        add_edge(u, v)

    elif mode == "union":
        if is_directed:
            # Keep u->v if v in kNN(u); also keep v->u if u in kNN(v)
            for u in G.nodes():
                for v in knn[u]:
                    add_edge(u, v)
            for v in G.nodes():
                for u in knn[v]:
                    add_edge(v, u)
        else:
            for u in G.nodes():
                for v in knn[u]:
                    add_edge(u, v)

    elif mode == "out":
        if not is_directed:
            raise ValueError("mode='out' only valid for directed graphs")
        for u in G.nodes():
            for v in knn[u]:
                add_edge(u, v)
    else:
        raise ValueError("mode must be 'mutual', 'union', or 'out' (directed only)")

    # Build pruned graph with same type, nodes, and kept edges
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))

    # Add edges with original data
    if is_directed:
        for u, v in keep_edges:
            if G.has_edge(u, v):
                H.add_edge(u, v, **G[u][v])
    else:
        for u, v in keep_edges:
            if G.has_edge(u, v):
                H.add_edge(u, v, **G[u][v])

    if not keep_isolates:
        H.remove_nodes_from([n for n in H.nodes() if H.degree(n) == 0])

    return H, knn

def node_labels(g):
    return pd.DataFrame(nx.get_node_attributes(g, "pheno_onehot")).T.astype(np.float32)

def load_graph(path, name):
    with open(path, "rb") as f:
        g = pickle.load(f)
    g.name = name
    return g

def drop_attrs_netx(G , attrs_to_keep) : 
    for _, data in G.nodes(data=True):
        for k in list(data.keys()):
            if k not in attrs_to_keep:
                del data[k]

def nx_to_pyg(g , attrs_to_keep, get_weight=False) : 
    drop_attrs_netx(g, attrs_to_keep)

    # Direct edge_index creation (no Data object)
    node_order = pd.Series(sorted(g.nodes))
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    edges = list(g.edges())
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if get_weight : 
        edge_weight = torch.tensor(
            [g[u][v].get("weight") for u, v in ei.t().tolist()],
            dtype=torch.float,
        )
        return ei, node_order, edge_weight
    else : 
        return ei, node_order

def transfer_attributes(
    G_src: nx.Graph,
    G_dst: nx.Graph,
    clear_source: bool = False,
    overwrite: bool = True,
    deep_copy: bool = True,
) -> nx.Graph:
    """
    Transfer attributes from G_src to G_dst.

    - Node attributes are always copied for all nodes.
    - If clear_source is True, all node, edge, and graph-level attributes are removed
      from G_src after copying.
    - If overwrite is False, existing attributes in G_dst are preserved (only missing keys are added).
    - If deep_copy is True, values are deep-copied (safer if values are mutable, e.g., numpy arrays).

    Assumes G_src and G_dst have the same set of nodes.

    Returns:
        The destination graph (G_dst), modified in place.
    """
    # Sanity check: same nodes
    if set(G_src.nodes) != set(G_dst.nodes):
        raise ValueError("G_src and G_dst must have the same set of nodes.")

    copier = copy.deepcopy if deep_copy else (lambda x: x)

    if clear_source:
        # Clear node attributes
        for n in G_dst.nodes:
            G_dst.nodes[n].clear()

    # Copy node attributes
    for n in G_dst.nodes:
        src_attrs = G_src.nodes[n]
        if not src_attrs:
            continue
        if overwrite:
            for k, v in src_attrs.items():
                G_dst.nodes[n][k] = copier(v)
        else:
            for k, v in src_attrs.items():
                G_dst.nodes[n].setdefault(k, copier(v))

    return G_dst


def normalize_edge_weights_symmetric(G, attr="weight", new_attr=None, add_self_loops=False, self_loop_weight=1.0):
    """
    Set d[new_attr] = w / sqrt(deg(u)*deg(v)), where deg is weighted degree.
    Optionally add self-loops before computing degrees.
    """
    if add_self_loops and not G.is_multigraph():
        for n in G.nodes():
            if not G.has_edge(n, n):
                G.add_edge(n, n, **{attr: self_loop_weight})

    target_attr = new_attr or attr

    # Precompute weighted degrees
    deg = dict(G.degree(weight=attr))

    if G.is_multigraph():
        for u, v, k, d in G.edges(keys=True, data=True):
            w = d.get(attr, 1.0)
            du = max(deg.get(u, 0.0), 1e-12)
            dv = max(deg.get(v, 0.0), 1e-12)
            d[target_attr] = w / math.sqrt(du * dv)
    else:
        for u, v, d in G.edges(data=True):
            w = d.get(attr, 1.0)
            du = max(deg.get(u, 0.0), 1e-12)
            dv = max(deg.get(v, 0.0), 1e-12)
            d[target_attr] = w / math.sqrt(du * dv)

    return G


def stratified_inductive_splits(
    G,
    y_attr='pheno_onehot',
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    """
    Returns:
      G_train, G_val, G_test: induced subgraphs with only intra-split edges
      splits: dict with node arrays {'train': nodes_train, 'val': nodes_val, 'test': nodes_test}
      stats: dict with quick diagnostics on label proportions and cross edges
    """
    # Sanity checks
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    nodes = np.array(list(G.nodes()))
    N = len(nodes)
    if N == 0:
        raise ValueError("Graph has no nodes.")

    # Build multi-label matrix y in node order
    try:
        y = np.stack([np.asarray(G.nodes[n][y_attr], dtype=int) for n in nodes], axis=0)
    except KeyError as e:
        raise KeyError(f"Missing node attribute '{y_attr}' on at least one node.") from e

    if y.ndim != 2 or set(np.unique(y)) - {0, 1}:
        raise ValueError(f"'{y_attr}' must be a 2D 0/1 array per node (multi-label one-hot).")

    # 1) Train vs temp (val+test)
    temp_size = val_size + test_size
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=temp_size, random_state=random_state
    )
    train_idx, temp_idx = next(msss1.split(np.zeros(N), y))

    # 2) Split temp into val and test
    rel_test = test_size / (val_size + test_size)  # fraction of temp to test
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=rel_test, random_state=random_state
    )
    val_rel, test_rel = next(msss2.split(np.zeros(len(temp_idx)), y[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    # Map indices back to node IDs
    nodes_train = nodes[train_idx]
    nodes_val   = nodes[val_idx]
    nodes_test  = nodes[test_idx]

    # Ensure disjointness
    assert len(set(nodes_train) & set(nodes_val)) == 0
    assert len(set(nodes_train) & set(nodes_test)) == 0
    assert len(set(nodes_val)   & set(nodes_test)) == 0

    # Build induced subgraphs (no edges to nodes outside each split)
    G_train = G.subgraph(nodes_train).copy()
    G_val   = G.subgraph(nodes_val).copy()
    G_test  = G.subgraph(nodes_test).copy()

    # Diagnostics: label proportion differences
    def proportions(y_, idx):
        return y_[idx].mean(axis=0)

    p_all   = proportions(y, np.arange(N))
    p_train = proportions(y, train_idx)
    p_val   = proportions(y, val_idx)
    p_test  = proportions(y, test_idx)

    diffs = {
        'max_abs_diff_train_vs_all': float(np.abs(p_train - p_all).max()),
        'max_abs_diff_val_vs_all':   float(np.abs(p_val   - p_all).max()),
        'max_abs_diff_test_vs_all':  float(np.abs(p_test  - p_all).max()),
    }

    # Count cross edges in the original graph (for info only)
    # These edges will NOT appear in the induced subgraphs
    set_train, set_val, set_test = set(nodes_train), set(nodes_val), set(nodes_test)
    cross_tv = cross_tt = cross_vt = 0
    for u, v in G.edges():
        in_train = (u in set_train) and (v in set_train)
        in_val   = (u in set_val)   and (v in set_val)
        in_test  = (u in set_test)  and (v in set_test)
        if not (in_train or in_val or in_test):
            # Edge spans different splits
            if (u in set_train and v in set_val) or (u in set_val and v in set_train):
                cross_tv += 1
            elif (u in set_train and v in set_test) or (u in set_test and v in set_train):
                cross_tt += 1
            elif (u in set_val and v in set_test) or (u in set_test and v in set_val):
                cross_vt += 1

    stats = {
        'n_nodes': {'all': int(N), 'train': int(len(nodes_train)), 'val': int(len(nodes_val)), 'test': int(len(nodes_test))},
        'n_edges': {'all': int(G.number_of_edges()),
                    'train': int(G_train.number_of_edges()),
                    'val': int(G_val.number_of_edges()),
                    'test': int(G_test.number_of_edges())},
        'cross_edges_in_original': {'train_val': cross_tv, 'train_test': cross_tt, 'val_test': cross_vt},
        'label_proportion_diffs': diffs
    }

    splits = {'train': nodes_train, 'val': nodes_val, 'test': nodes_test}
    return G_train, G_val, G_test, splits, stats

def _phe_attr_key(phecode, prefix='phe_'):
    # Convert phecode to a stable string key
    return f"{prefix}{phecode}"

def attach_phenotype_flags_from_df(
    G,
    df,
    idx_col='ID',
    phenotype_col='Phenotype',
    attr_prefix='pheno_',
    value=True,
    clear_existing=False,
    drop_list_attr=True,
    list_attr_name='phenotypes'
):
    """
    For each node (eid) in G, attach per-phenotype node attributes indicating presence.
    Example: node['phe_208.0'] = True

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes are eids (must match df[idx_col]).
    df : pd.DataFrame
        Must contain columns [idx_col, phenotype_col]. Duplicates are dropped.
    idx_col, phenotype_col : str
        Column names for eid and phenotype.
    attr_prefix : str
        Prefix used for per-phenotype attributes.
    value : any
        Value to store for presence (e.g., True or 1).
    clear_existing : bool
        If True, remove any existing attributes that start with attr_prefix before adding flags.
    drop_list_attr : bool
        If True, remove the legacy list attribute (e.g., 'phenotype') if present.
    list_attr_name : str
        Name of the legacy list attribute to drop if drop_list_attr is True.
    """
    x = (df[[idx_col, phenotype_col]]
         .dropna()
         .drop_duplicates())

    # Optional: clear existing per-phenotype flags
    if clear_existing:
        for n, data in G.nodes(data=True):
            to_del = [k for k in data.keys() if isinstance(k, str) and k.startswith(attr_prefix)]
            for k in to_del:
                del data[k]

    # Group phenotypes by eid and set flags
    for eid, sub in x.groupby(idx_col, observed=True):
        if eid not in G:
            # Skip eids that aren't nodes in the graph
            continue
        node = G.nodes[eid]
        # Optionally remove legacy list attribute
        if drop_list_attr and list_attr_name in node:
            del node[list_attr_name]
        # Set flags
        for p in sub[phenotype_col].unique():
            node[_phe_attr_key(p, prefix=attr_prefix)] = value

    # Keep a hint in graph about how to find these attributes later
    G.graph['phenotype_attr_prefix'] = attr_prefix
    return G

def normalize_edges_similarity_from_node_attr(
    G,
    code_attr='phenotypes',
    metric='jaccard',            # 'jaccard' or 'cosine'
    out_attr=None,               # default: 'jaccard' or 'cosine' based on metric
    replace_weight=False,
    use_edge_weight_if_available=True  # use existing edge 'weight' as intersection size
):
    """
    Compute edge similarity from node attribute `code_attr` (list/set of phenotypes per node),
    using either Jaccard or Cosine (binary) similarity, and store it as an edge attribute.

    - Jaccard(u,v) = |Su ∩ Sv| / |Su ∪ Sv|
    - Cosine(u,v)  = |Su ∩ Sv| / sqrt(|Su| * |Sv|)  (binary cosine on sets)

    Parameters
    ----------
    G : nx.Graph
    code_attr : str
        Node attribute containing the phenotypes (list or set).
    metric : {'jaccard', 'cosine'}
        Which similarity to compute.
    out_attr : str or None
        Edge attribute name to store the similarity. If None, uses the metric name.
    replace_weight : bool
        If True, overwrite edge 'weight' with the similarity value.
    use_edge_weight_if_available : bool
        If True and the edge already has 'weight' equal to the count of shared phenotypes,
        use it as |Su ∩ Sv| to avoid recomputing intersections.

    Returns
    -------
    G : nx.Graph
        Modified in place (also returned for convenience).
    """
    metric = metric.lower()
    if metric not in {'jaccard', 'cosine'}:
        raise ValueError("metric must be 'jaccard' or 'cosine'")
    if out_attr is None:
        out_attr = metric

    # Precompute sets and sizes per node
    code_sets = {}
    sizes = {}
    for n, data in G.nodes(data=True):
        vals = data.get(code_attr, [])
        s = vals if isinstance(vals, set) else set(vals)
        code_sets[n] = s
        sizes[n] = len(s)

    for u, v, d in G.edges(data=True):
        Su = code_sets.get(u, set())
        Sv = code_sets.get(v, set())
        ku = sizes.get(u, 0)
        kv = sizes.get(v, 0)

        # intersection size
        if use_edge_weight_if_available and ('weight' in d) and isinstance(d['weight'], (int, float)):
            w = int(d['weight'])
        else:
            w = len(Su & Sv)

        if metric == 'jaccard':
            denom = ku + kv - w
            sim = 0.0 if denom <= 0 else (w / denom)
        else:  # cosine
            denom = math.sqrt(ku * kv)
            sim = 0.0 if denom == 0 else (w / denom)

        d[out_attr] = float(sim)
        if replace_weight:
            d['weight'] = float(sim)

    return G

def build_phenotype_KG(df, id_col = 'ID', phenotype_col = 'Phenotype', min_weight=1, onehot_dtype=np.uint8):
    """
    Build an undirected NetworkX graph where nodes are eids, edges connect eids
    that share >=1 phecode, and edge weight is the number of shared phenotypes.
    Additionally, attach node features:
      - 'pheno_onehot': 0/1 one-hot vector over all phenotypes
      - 'phenotypes': list of phenotypes the eid has

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns [id_col, phenotype_col] (other columns are ignored).
    min_weight : int
        Keep edges with weight >= min_weight.
    onehot_dtype : np.dtype
        Dtype for the one-hot arrays (np.uint8 or bool recommended).

    Returns
    -------
    G : networkx.Graph
        Graph with node attributes:
          - G.nodes[id_col][phenotype_col_onehot] : np.ndarray shape (n_phenotypes,)
          - G.nodes[id_col][phenotype_col]       : list of phecode values
        And graph-level attribute:
          - G.graph['phenotype_col_categories']  : list giving the index order used in one-hots
    """
    # Use only eid/phecode, drop NaNs and duplicates
    df = df[[id_col, phenotype_col]].dropna().drop_duplicates()

    # Map ids to category codes (gives us a consistent index order)
    ids = df[id_col].astype('category')
    phe  = df[phenotype_col].astype('category')
    id_codes = ids.cat.codes.to_numpy()
    phe_codes = phe.cat.codes.to_numpy()

    n_e = ids.cat.categories.size
    n_p = phe.cat.categories.size

    # Build bipartite incidence matrix B (ids x phenotypes), entries 1 if present
    data = np.ones(len(df), dtype=np.int32)
    B = coo_matrix((data, (id_codes, phe_codes)), shape=(n_e, n_p)).tocsr()

    # Initialize graph and record the phecode order used in one-hots
    G = nx.Graph()
    id_values = ids.cat.categories.tolist()
    phe_values = phe.cat.categories.tolist()
    print(phe_values)
    G.add_nodes_from(id_values)
    G.graph['phecode_categories'] = phe_values  # interpret one-hot indices using this list

    # Attach node features
    # For each id (row i of B), get indices of nonzero phenotypes and build:
    #  - list of phenotypes (original values)
    #  - one-hot vector (dense). For large n_p, this can be memory-heavy.
    #    If memory is tight, consider:
    #      - storing just the indices as 'phecode_indices' instead of a dense vector, or
    #      - using np.packbits(onehot.astype(np.uint8)) to store a compact bytes array.
    indptr = B.indptr
    indices = B.indices
    for i, idx in enumerate(id_values):
        col_idx = indices[indptr[i]:indptr[i+1]]  # phecode indices for this id
        # Dense one-hot vector
        onehot = np.zeros(n_p, dtype=onehot_dtype)
        if len(col_idx) > 0:
            onehot[col_idx] = 1
        # List of phenotypes (original values)
        phe_list = [phe_values[j] for j in col_idx]
        # Set node attributes
        G.nodes[idx]['pheno_onehot'] = onehot
        G.nodes[idx]['phenotypes'] = phe_list

    # Project: S = B * B^T gives number of shared phenotypes between ids
    S = B @ B.T
    S.setdiag(0)
    S.eliminate_zeros()

    # Add weighted edges (iterate upper triangle to avoid duplicates)
    S_coo = S.tocoo()
    for i, j, w in zip(S_coo.row, S_coo.col, S_coo.data):
        if i < j and w >= min_weight:
            ei = id_values[i]
            ej = id_values[j]
            G.add_edge(ei, ej, weight=int(w))

    return G

def sample_ids_with_phecode_min_counts(
    df,
    n=1000,
    min_per_code=10,
    idx_col='eid',
    phenotype_col='phecode',
    random_state=42
):
    """
    Select up to `n` eids such that as many phenotypes as possible appear at least
    `min_per_code` times among the selected eids. Infeasible phenotypes (fewer than
    `min_per_code` total eids in the dataset) are filtered out automatically.

    Greedy strategy:
      - Keep a deficit need[c] for each phecode c (initially = min_per_code).
      - At each step, pick the eid that covers the most unmet phenotypes (lazy heap).
      - Stop when either all feasible phenotypes are satisfied or we pick `n` eids.
      - If we satisfy all feasible phenotypes early and still have room, fill with
        random eids to reach `n`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [idx_col, phenotype_col]. Duplicates ignored.
    n : int
        Number of eids to select (upper bound).
    min_per_code : int
        Minimum number of selected eids required per phecode.
    idx_col : str
        Column name for eid.
    phenotype_col : str
        Column name for phecode.
    random_state : int or np.random.Generator or None
        Random seed or RNG for reproducibility.

    Returns
    -------
    result : dict with keys
        - 'selected_eids': list of selected eids (length <= n)
        - 'kept_phenotypes': set of phenotypes that were feasible (freq >= min_per_code)
        - 'satisfied_phenotypes': set of phenotypes that reached the target among selected
        - 'unsatisfied_phenotypes': dict {phecode: remaining_deficit}
        - 'coverage_counts': dict {phecode: count_in_selection} for kept phenotypes
        - 'status': 'all_satisfied' or 'budget_exhausted'
    """
    # RNG
    rng = np.random.default_rng(random_state) if not isinstance(random_state, np.random.Generator) else random_state

    # Deduplicate pairs and drop NaNs
    x = df[[idx_col, phenotype_col]].dropna().drop_duplicates()

    # Universe of eids
    all_eids = x[idx_col].unique().tolist()

    # Build mapping: phecode -> set(eids)
    code_to_eids = defaultdict(set)
    for eid, code in x.itertuples(index=False, name=None):
        code_to_eids[code].add(eid)

    # Keep only feasible phenotypes (those with at least min_per_code eids)
    kept_phenotypes = {c for c, eids in code_to_eids.items() if len(eids) >= min_per_code}
    if not kept_phenotypes:
        # No feasible phenotypes at all; just sample n eids at random
        k = min(n, len(all_eids))
        selected_eids = rng.choice(all_eids, size=k, replace=False).tolist()
        return {
            'selected_eids': selected_eids,
            'kept_phenotypes': set(),
            'satisfied_phenotypes': set(),
            'unsatisfied_phenotypes': {},
            'coverage_counts': {},
            'status': 'no_feasible_phenotypes'
        }

    # Restrict code_to_eids to kept phenotypes
    code_to_eids = {c: eids for c, eids in code_to_eids.items() if c in kept_phenotypes}

    # Build eid -> set of kept phenotypes (only those that count towards the targets)
    eid_to_codes = defaultdict(set)
    for c, eids in code_to_eids.items():
        for eid in eids:
            eid_to_codes[eid].add(c)

    candidate_eids = set(eid_to_codes.keys())  # eids that can help satisfy constraints

    # Deficits per phecode
    need = {c: min_per_code for c in kept_phenotypes}
    remaining_deficits = sum(need.values())

    # Lazy greedy heap: items are (-gain, tie_breaker, eid), where gain = count of codes with need>0 that this eid covers
    # Initialise with current gains (all codes have need>0 initially)
    heap = []
    # Shuffle candidates for random tie-breaking
    cand_list = list(candidate_eids)
    rng.shuffle(cand_list)
    for idx, eid in enumerate(cand_list):
        initial_gain = len(eid_to_codes[eid])
        if initial_gain > 0:
            heapq.heappush(heap, (-initial_gain, idx, eid))

    selected = []
    selected_set = set()

    # Greedy selection
    while heap and len(selected) < n and remaining_deficits > 0:
        neg_gain, _, eid = heapq.heappop(heap)

        # Compute the true current gain (the heap may be stale)
        true_gain = sum(1 for c in eid_to_codes[eid] if need[c] > 0)

        if true_gain < -neg_gain:
            # Update and reinsert with new (smaller) gain
            heapq.heappush(heap, (-true_gain, rng.integers(0, 1_000_000), eid))
            continue

        if true_gain == 0:
            # This eid doesn't help anymore; skip
            continue

        # Accept eid
        selected.append(eid)
        selected_set.add(eid)

        # Update deficits
        reduced = 0
        for c in eid_to_codes[eid]:
            if need[c] > 0:
                need[c] -= 1
                reduced += 1
        remaining_deficits -= reduced

    # Check satisfaction status
    all_satisfied = all(v == 0 for v in need.values())

    # If satisfied early, fill remaining slots with random eids
    if all_satisfied and len(selected) < n:
        remaining_pool = [e for e in all_eids if e not in selected_set]
        fill_k = min(n - len(selected), len(remaining_pool))
        if fill_k > 0:
            selected.extend(rng.choice(remaining_pool, size=fill_k, replace=False).tolist())

    # Compute coverage counts for kept phenotypes within the selected set
    selected_set = set(selected)
    coverage_counts = {c: sum(1 for eid in code_to_eids[c] if eid in selected_set) for c in kept_phenotypes}
    satisfied_phenotypes = {c for c, cnt in coverage_counts.items() if cnt >= min_per_code}
    unsatisfied_phenotypes = {c: max(0, min_per_code - coverage_counts[c]) for c in kept_phenotypes if coverage_counts[c] < min_per_code}

    status = 'all_satisfied' if all_satisfied else 'budget_exhausted'

    return {
        'selected_eids': selected,
        'kept_phenotypes': kept_phenotypes,
        'satisfied_phenotypes': satisfied_phenotypes,
        'unsatisfied_phenotypes': unsatisfied_phenotypes,
        'coverage_counts': coverage_counts,
        'status': status
    }

def parse_pheno_onehot_string(s: str, dtype=np.int8) -> np.ndarray:
    """
    Convert a string like "[0 0 1 0\n 0 0 1]" into a NumPy array.
    Handles spaces and newlines; strips surrounding brackets.
    """
    if not isinstance(s, str):
        return np.asarray(s, dtype=dtype)
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    arr = np.fromstring(s, sep=' ', dtype=dtype)
    if arr.size == 0 and s:
        raise ValueError("Failed to parse pheno_onehot string.")
    return arr

def fix_pheno_onehot_in_graph(G, key='phenotype_onehot', dtype=np.int8, check_length=True):
    """
    Convert node attribute `key` from string to NumPy array for all nodes in a NetworkX graph.
    Updates the graph in place. Returns the number of nodes updated.
    """
    n_fixed = 0
    expected_len = None

    for _, attrs in G.nodes(data=True):
        if key in attrs and isinstance(attrs[key], str):
            arr = parse_pheno_onehot_string(attrs[key], dtype=dtype)
            attrs[key] = arr
            n_fixed += 1
            if check_length:
                if expected_len is None:
                    expected_len = arr.size
                elif arr.size != expected_len:
                    raise ValueError(
                        f"Inconsistent one-hot length: expected {expected_len}, got {arr.size}"
                    )
    return n_fixed


def fix_node_attrs(G, attr, default=0, allow_float_str=False):
    """
    Convert node attribute `attr` to int in place.

    - Missing/None/NaN values are set to `default`.
    - If `allow_float_str` is True, values like "3.0" (or floats) are handled via int(float(v)).
      Otherwise, values are cast with int(v) directly.
    """
    for _, data in G.nodes(data=True):
        v = data.get(attr, None)
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                data[attr] = default
            else:
                data[attr] = int(float(v)) if allow_float_str else int(v)
        except (ValueError, TypeError):
            data[attr] = default


def create_similarity_matrix(mat , method = 'euclidean') :
    """
    Creates a similarity matrix from the given data matrix using specified methods.

    Parameters:
        mat (pd.DataFrame): The matrix from which to calculate similarities (e.g., gene expression levels).
        method (str): The method to use for calculating similarities. Supported methods are 'bicorr', 'pearson', and 'euclidean'.

    Returns:
        pd.DataFrame: A DataFrame representing the similarity matrix.
    """
    if method == 'bicorr' : 
        adj = abs_bicorr(mat.T)
    elif method == 'pearson' : 
        adj = pearson_corr(mat.T)
    elif method == 'cosine' : 
        adj = cosine_corr(mat.T)
    elif method == 'hamming' : 
        adj = hamming_dist(mat.T)
    else : 
        distances = pdist(mat.values, metric='euclidean')
        dist_matrix = squareform(distances)

        adj = pd.DataFrame(data=dist_matrix , index=mat.index , columns=mat.index)
        
    return adj

def abs_bicorr(data , mat_means=True) : 
    """
    Calculates the absolute bicorrelation matrix for the given data.

    Parameters:
        data (pd.DataFrame): Data for which to compute the bicorrelation.
        mat_means (bool): If True, subtract the mean from each column before computing the correlation.

    Returns:
        pd.DataFrame: Bicorrelation matrix.
    """
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T
    if mat_means==True : 
        mat = mat - mat.mean(axis = 0)

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    bicorr = astropy.stats.biweight_midcovariance(mat)

    for i in range(K) : 
        correl[i , : ] = bicorr[i , :] / np.sqrt(bicorr[i,i] * np.diag(bicorr))
        
    return pd.DataFrame(data = correl , index=idx , columns=cols , dtype=np.float32)

def pearson_corr(data, mat_means=True) : 
    """
    Computes the Pearson correlation matrix for the given data.

    Parameters:
        data (pd.DataFrame): Data for which to compute the Pearson correlation.
        mat_means (bool): Normalizes data by its mean if set to True.

    Returns:
        pd.DataFrame: Pearson correlation matrix.
    """
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T
    if mat_means==True : 
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / norms

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    cov = np.cov(mat)

    for i in range(K) : 
        correl[i , : ] = cov[i , :] / np.sqrt(cov[i,i] * np.diag(cov))
        
    return pd.DataFrame(data = correl , index=idx , columns=cols , dtype=np.float32)

def cosine_corr(data, mat_means=True) : 
    """
    Computes cosine correlations for the given data, treated as vectors.

    Parameters:
        data (pd.DataFrame): Data for which to compute cosine correlations.
        mat_means (bool): If True, normalizes the data before computing correlation.

    Returns:
        pd.DataFrame: Cosine correlation matrix.
    """
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T
    if mat_means==True : 
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / norms

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    cov = np.dot(mat, mat.T)

    for i in range(K) : 
        correl[i , : ] = cov[i , :] / np.sqrt(cov[i,i] * np.diag(cov))
        
    return pd.DataFrame(data = correl , index=idx , columns=cols , dtype=np.float32)

def hamming_dist(data, mat_means=True) : 
    """
    Calculate the pairwise Hamming distance between rows (patients) in a DataFrame.
    
    Args:
    df (pandas.DataFrame): Input DataFrame where rows are patients and columns are features.
    
    Returns:
    pandas.DataFrame: A DataFrame representing the pairwise Hamming distances.
    """
    # Initialize a matrix to store Hamming distances
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T

    # Expand dimensions and compute difference array, using broadcasting
    differences = mat[:, np.newaxis, :] != mat[np.newaxis, :, :]
    
    # Sum differences along the features axis to compute the Hamming distance
    hamming_distances = np.sum(differences, axis=2)
    
    # Convert the resulting matrix back into a DataFrame
    return pd.DataFrame(hamming_distances, index=idx, columns=cols)
    

def knn_graph_generation(datExpr , datMeta , knn = 20 , method = 'euclidean' ,extracted_feats = None, **args) : 
    """
    Generates a k-nearest neighbor graph based on the specified data and method of similarity.

    Parameters:
        datExpr (pd.DataFrame): DataFrame containing expression data or other numerical data.
        datMeta (pd.DataFrame or pd.Series): Metadata for the nodes in the graph.
        knn (int): Number of nearest neighbors to connect to each node.
        method (str): Method used for calculating similarity or distance ('euclidean', 'bicorr', 'pearson', 'cosine').
        extracted_feats ([type]): Specific features extracted from the data to use for graph construction.
        **args: Additional arguments for customizing the node visualization (e.g., `node_colour`, `node_size`).

    Returns:
        nx.Graph: A NetworkX graph object representing the k-nearest neighbors graph.
    """    
    if extracted_feats is not None : 
        mat = datExpr.loc[: , extracted_feats]
    else : 
        mat = datExpr
        
    adj = create_similarity_matrix(mat , method)
    
    if 'node_colour' not in args.keys() : 
        node_colour = datMeta.astype('category').cat.set_categories(wesanderson.FantasticFox2_5.hex_colors , rename=True)
    else : 
        node_colour = args['node_colour']
    if 'node_size' not in args.keys() : 
        node_size = 300
    else : 
        node_size = args['node_size']

    if 'weight' not in args.keys() : 
        G = gen_knn_network(adj , knn , datMeta , node_colours=node_colour , node_size=node_size )
    else : 
        G = gen_knn_network(adj , knn , datMeta , node_colours=node_colour , node_size=node_size , weight=args['weight'])

    return G

def get_k_neighbors(matrix, k , corr=True):
    """
    Finds k-nearest neighbors for each row in the given matrix.

    Parameters:
        matrix (pd.DataFrame): The matrix from which neighbors are to be found.
        k (int): The number of neighbors to find for each row.
        corr (bool): Indicates whether to use correlation rather than distance for finding neighbors.

    Returns:
        dict: A dictionary where keys are indices (or node names) and values are lists of k-nearest neighbors' indices.
    """    
    #dist_mtx = scipy.spatial.distance_matrix(matrix.values ,  matrix.values)
    #dist_mtx = pd.DataFrame(dist_mtx , index = matrix.index , columns = matrix.index)
    #if corr == True : 
    #    matrix.loc[: , :] = 1 - matrix.values
    
    k_neighbors = {}
    for node in matrix:
        k_neighbors[node] = {}
        neighbors = matrix.loc[node].nlargest(k + 1) 
        k_neighbors[node]["neighbours"] = neighbors.index.tolist()[1:] # Exclude the node itself
        k_neighbors[node]["weights"] = neighbors.values.tolist()[1:] # Exclude the node itself
        
    return k_neighbors

def gen_knn_network(data , K , labels ,  node_colours = 'skyblue' , node_size = 300 , plot=True , weight=False) : 
    """
    Plots a k-nearest neighbors network using NetworkX.

    Parameters:
        data (pd.DataFrame): The similarity or distance matrix used to determine neighbors.
        K (int): The number of nearest neighbors for network connections.
        labels (pd.Series): Labels or categories for the nodes used in plotting.
        node_colours (str or list): Color or list of colors for the nodes.
        node_size (int): Size of the nodes in the plot.

    Returns:
        nx.Graph: A NetworkX graph object that has been plotted.
    """
    # Get k-nearest neighbors for each node (k=20 in this example)
    k_neighbors = get_k_neighbors(data, k=K)

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(data.index)
    
    nx.set_node_attributes(G , labels.astype('category').cat.codes , 'label')
    nx.set_node_attributes(G , pd.Series(np.arange(len(data.index)) , index=data.index) , 'idx')

    # Add edges based on the k-nearest neighbors
    for node, neighbors in k_neighbors.items():
        if weight : 
            for neighbor , weight in zip(neighbors['neighbours'] , neighbors['weights']):

                G.add_edge(node, neighbor , weight=weight)
        else : 
            for neighbor in neighbors["neighbours"]:
                G.add_edge(node, neighbor)

    if plot == True : 
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_colours, font_size=8)
        patches = []
        if type(node_colours) == pd.core.series.Series : 
            for col , lab in zip(node_colours.unique() , labels.unique()) : 
                patches.append(mpatches.Patch(color=col, label=lab))
            plt.legend(handles=patches)
        plt.show()
    
    return G



MissingPolicy = Literal["copy", "fill", "nan", "drop"]
Mode = Literal["intersection", "union"]

def integrate_networks_weighted(
    graphs: Sequence[Union[nx.Graph, nx.DiGraph]],
    *,
    weight_attrs: Union[str, Sequence[str]] = "weight",
    net_weights: Optional[Sequence[float]] = None,
    out_weight: str = "weight",
    mode: Mode = "intersection",
    missing_policy: MissingPolicy = "copy",
    fill_value: float = 0.0,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Integrate multiple simple graphs by computing a (possibly weighted) average
    of their edge weights.

    Parameters
    ----------
    graphs : sequence of nx.Graph or nx.DiGraph
        Input graphs. Must be simple graphs (not MultiGraphs). All must have the same directedness.
    weight_attrs : str or sequence of str
        Edge attribute name(s) holding the weight in each graph.
        - If str: the same attribute name is used for all graphs.
        - If sequence: must have the same length as `graphs`, specifying the attribute name per graph.
    net_weights : sequence of float, optional
        Per-graph integration weights. If None, defaults to equal weights for all graphs.
        These are normalized internally for averaging (except when missing_policy="copy"
        with missing edges, where renormalization is done over the present graphs only).
    out_weight : str
        Edge attribute name in the output graph that will store the averaged weight.
    mode : {"intersection", "union"}
        - "intersection": include only edges present in all graphs AND with non-missing
          weight attributes in all graphs.
        - "union": include edges present in at least one graph. Missing weights are
          handled via `missing_policy`.
    missing_policy : {"copy", "fill", "nan", "drop"} (used only when mode="union")
        How to handle edges/weights missing in some graphs:
        - "copy": average over the subset of graphs where the edge weight is present,
          renormalizing their weights (this generalizes the 2-graph "copy" behavior).
        - "fill": substitute `fill_value` for missing weights and average using all graphs.
        - "nan": set output weight to NaN if any graph is missing that edge/weight;
          otherwise average using all graphs.
        - "drop": skip edges that are missing in any graph; otherwise average using all graphs.
    fill_value : float
        Used with missing_policy="fill" as the substitute for missing weights.

    Returns
    -------
    H : nx.Graph or nx.DiGraph
        New graph with averaged edge weights stored in `out_weight`. Node attributes are
        merged in sequence order (later graphs override earlier on conflicts). Only the
        `out_weight` edge attribute is set for edges.

    Notes
    -----
    - MultiGraph/MultiDiGraph are not supported.
    - An edge's weight is considered "missing" if the edge is absent in a graph or if the
      specified weight attribute is not present for that edge.
    """
    if not graphs:
        raise ValueError("Provide at least one graph.")
    # Validate graph types and directedness
    directed = graphs[0].is_directed()
    for i, G in enumerate(graphs):
        if G.is_multigraph():
            raise NotImplementedError("MultiGraphs are not supported.")
        if G.is_directed() != directed:
            raise ValueError("All graphs must have the same directedness (all directed or all undirected).")

    # Normalize weight_attrs to per-graph list
    if isinstance(weight_attrs, str):
        weight_attrs = [weight_attrs] * len(graphs)
    else:
        if len(weight_attrs) != len(graphs):
            raise ValueError("Length of weight_attrs must match number of graphs.")

    # Normalize net_weights
    if net_weights is None:
        net_weights = [1.0] * len(graphs)
    else:
        if len(net_weights) != len(graphs):
            raise ValueError("Length of net_weights must match number of graphs.")
    # Ensure numeric and compute normalized weights for full-graph averaging
    try:
        net_weights = [float(w) for w in net_weights]
    except Exception as e:
        raise ValueError("net_weights must be numeric.") from e

    total_w = sum(net_weights)
    if total_w <= 0:
        # Fallback to equal weights if all provided weights sum to 0 or negative
        net_weights = [1.0] * len(graphs)
        total_w = float(len(graphs))
    norm_weights = [w / total_w for w in net_weights]

    # Helper: edge key for deduplication in undirected graphs
    def edge_key(u: Hashable, v: Hashable):
        return (u, v) if directed else frozenset((u, v))

    # Build edge key sets per graph
    edge_sets: List[set] = []
    for G in graphs:
        if directed:
            edge_sets.append(set(G.edges()))
        else:
            edge_sets.append(set(edge_key(u, v) for u, v in G.edges()))

    # Determine which edge keys to include
    if mode == "intersection":
        edge_keys = set.intersection(*edge_sets) if edge_sets else set()
    elif mode == "union":
        edge_keys = set.union(*edge_sets) if edge_sets else set()
    else:
        raise ValueError("mode must be 'intersection' or 'union'.")

    # Create output graph of same class as the first input
    H = graphs[0].__class__()

    # Merge nodes: later graphs override earlier attributes on conflicts
    for G in graphs:
        H.add_nodes_from(G.nodes(data=True))

    # Helper to get an edge weight (None if edge missing or attribute missing)
    def get_weight(G: Union[nx.Graph, nx.DiGraph], u: Hashable, v: Hashable, attr: str) -> Optional[float]:
        if G.has_edge(u, v):
            return G[u][v].get(attr, None)
        return None

    for k in edge_keys:
        # Recover endpoints for adding to H
        if directed:
            u, v = k  # tuple
        else:
            nodes_tuple = tuple(k)  # frozenset -> tuple of 1 or 2 elements
            if len(nodes_tuple) == 1:
                u = v = nodes_tuple[0]  # self-loop case
            else:
                u, v = nodes_tuple[0], nodes_tuple[1]

        # Gather weights across graphs
        vals: List[Optional[float]] = []
        present_mask: List[bool] = []
        for G, attr in zip(graphs, weight_attrs):
            # For undirected graphs, G.has_edge(u, v) works regardless of order
            w = get_weight(G, u, v, attr)
            vals.append(w)
            present_mask.append(w is not None)

        all_present = all(present_mask)
        any_present = any(present_mask)

        if mode == "intersection":
            # Require presence and non-missing weight in every graph
            if not all_present:
                continue
            # Weighted average with normalized weights across all graphs
            w_out = sum(wi * vi for wi, vi in zip(norm_weights, vals))  # type: ignore
        else:
            # mode == "union"
            if not any_present:
                # Edge exists structurally in at least one graph but weight attributes are all missing
                if missing_policy == "fill":
                    w_out = sum(wi * fill_value for wi in norm_weights)
                elif missing_policy == "nan":
                    w_out = float("nan")
                else:
                    # "copy" and "drop" -> nothing usable
                    continue
            else:
                if missing_policy == "copy":
                    # Average over present-only, renormalizing their weights
                    present_weights = [w for w, p in zip(net_weights, present_mask) if p]
                    denom = sum(present_weights)
                    if denom <= 0:
                        # Fallback: simple mean over present values
                        present_vals = [v for v, p in zip(vals, present_mask) if p]  # type: ignore
                        w_out = sum(present_vals) / len(present_vals)
                    else:
                        w_out = sum(w * v for w, v, p in zip(net_weights, vals, present_mask) if p) / denom  # type: ignore
                elif missing_policy == "fill":
                    # Substitute fill_value for missing and use normalized weights
                    use_vals = [v if p else fill_value for v, p in zip(vals, present_mask)]  # type: ignore
                    w_out = sum(wi * vi for wi, vi in zip(norm_weights, use_vals))
                elif missing_policy == "nan":
                    w_out = sum(wi * vi for wi, vi in zip(norm_weights, vals)) if all_present else float("nan")  # type: ignore
                elif missing_policy == "drop":
                    if not all_present:
                        continue
                    w_out = sum(wi * vi for wi, vi in zip(norm_weights, vals))  # type: ignore
                else:
                    raise ValueError("missing_policy must be one of {'copy', 'fill', 'nan', 'drop'}.")

        H.add_edge(u, v, **{out_weight: w_out})

    return H

def threshold_graph(
    G,
    weight="weight",
    threshold=None,         # absolute cutoff (float)
    percentile=None,        # percentile in [0,1], e.g. 0.9 keeps top 10% (similarity) or bottom 10% (distance)
    kind="similarity",      # "similarity" => keep w >= t; "distance" => keep w <= t
    keep_isolates=True,
    default_weight=None,    # if an edge lacks 'weight'; None = skip that edge
):
    """
    Return a thresholded copy of G by edge weight.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph. Edges should carry a numeric `weight` attribute.
    weight : str
        Edge attribute name for weight.
    threshold : float or None
        Absolute cutoff. For kind="similarity", keep edges with w >= threshold.
        For kind="distance", keep edges with w <= threshold.
    percentile : float in [0,1] or None
        If provided, compute threshold from global edge-weight distribution.
        For kind="similarity", threshold = np.quantile(weights, percentile) (keep top tail).
        For kind="distance", threshold = np.quantile(weights, percentile) (keep bottom tail).
    kind : {"similarity","distance"}
        Interpretation of weight.
    keep_isolates : bool
        If False, drop nodes with degree 0 after filtering.
    default_weight : float or None
        Fallback if an edge lacks the `weight` attribute. If None, skip such edges.

    Returns
    -------
    H : nx.Graph
        Thresholded graph (same class as G).
    cutoff : float
        The numeric cutoff actually used.
    """

    if threshold is None and percentile is None:
        raise ValueError("Provide either threshold or percentile.")

    # Collect edge weights
    ws = []
    for u, v, d in G.edges(data=True):
        w = d.get(weight, default_weight)
        if w is None or np.isnan(w):
            continue
        ws.append(float(w))

    if not ws:
        # no usable weights; return empty edge set
        H = G.__class__()
        H.add_nodes_from(G.nodes(data=True))
        return H, np.nan

    ws = np.array(ws, dtype=float)

    # Determine cutoff
    if percentile is not None:
        cutoff = float(np.quantile(ws, percentile))
        # For similarity: larger is better -> keep >= cutoff
        # For distance: smaller is better -> keep <= cutoff
        # (same cutoff, different comparator)
    else:
        cutoff = float(threshold)

    # Build filtered graph
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))

    if kind == "similarity":
        def keep(w): return w >= cutoff
    elif kind == "distance":
        def keep(w): return w <= cutoff
    else:
        raise ValueError("kind must be 'similarity' or 'distance'")

    for u, v, d in G.edges(data=True):
        w = d.get(weight, default_weight)
        if w is None or np.isnan(w):
            continue
        if keep(float(w)):
            H.add_edge(u, v, **d)

    if not keep_isolates:
        H.remove_nodes_from([n for n in H.nodes() if H.degree(n) == 0])

    return H, cutoff



def summarize_graph(G, heavy=False, k_top=5, weight=None, path_limit=20000):
    """
    Print a summary of classic network measures for a NetworkX graph.

    Parameters
    - G: nx.Graph / nx.DiGraph / nx.MultiGraph / nx.MultiDiGraph
    - heavy: bool, compute expensive metrics (diameter, avg shortest path, triangles, connectivity)
    - k_top: int, how many top nodes to show by degree/centrality
    - weight: str or None, edge attribute name to use as weight for degree (e.g., 'weight')
    - path_limit: int, max nodes allowed for all-pairs shortest-path heavy metrics

    Notes
    - Heavy metrics can be slow on large graphs; they are gated by heavy=True and size thresholds.
    - For directed graphs, some undirected measures are computed on an undirected view.
    """
    is_directed = G.is_directed()
    is_multi = G.is_multigraph()

    n = G.number_of_nodes()
    m = G.number_of_edges()

    print("=== Network summary ===")
    print(f"Directed: {is_directed} | MultiGraph: {is_multi}")
    print(f"Nodes: {n:,} | Edges: {m:,} | Density: {nx.density(G):.6f}")
    try:
        sl = nx.number_of_selfloops(G)
        print(f"Self-loops: {sl}")
    except Exception:
        print("Self-loops: n/a")

    # Degree stats
    def stats(vals):
        arr = list(vals)
        if not arr:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(mean(arr)),
            "median": float(median(arr)),
        }

    if is_directed:
        indeg = dict(G.in_degree(weight=weight))
        outdeg = dict(G.out_degree(weight=weight))
        indeg_stats = stats(indeg.values())
        outdeg_stats = stats(outdeg.values())
        print(f"In-degree  (min/median/mean/max): {indeg_stats['min']:.3g} / {indeg_stats['median']:.3g} / {indeg_stats['mean']:.3g} / {indeg_stats['max']:.3g}")
        print(f"Out-degree (min/median/mean/max): {outdeg_stats['min']:.3g} / {outdeg_stats['median']:.3g} / {outdeg_stats['mean']:.3g} / {outdeg_stats['max']:.3g}")

        top_in = sorted(indeg.items(), key=lambda x: x[1], reverse=True)[:k_top]
        top_out = sorted(outdeg.items(), key=lambda x: x[1], reverse=True)[:k_top]
        if top_in:
            print(f"Top-{k_top} by in-degree:  {top_in}")
        if top_out:
            print(f"Top-{k_top} by out-degree: {top_out}")
    else:
        deg = dict(G.degree(weight=weight))
        deg_stats = stats(deg.values())
        print(f"Degree (min/median/mean/max): {deg_stats['min']:.3g} / {deg_stats['median']:.3g} / {deg_stats['mean']:.3g} / {deg_stats['max']:.3g}")
        top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:k_top]
        if top_deg:
            print(f"Top-{k_top} by degree: {top_deg}")

    # Connectivity components
    if is_directed:
        try:
            nwcc = nx.number_weakly_connected_components(G)
            nscc = nx.number_strongly_connected_components(G)
            lwcc = max((len(c) for c in nx.weakly_connected_components(G)), default=0)
            lscc = max((len(c) for c in nx.strongly_connected_components(G)), default=0)
            print(f"Weakly CCs: {nwcc} (largest={lwcc}) | Strongly CCs: {nscc} (largest={lscc})")
        except Exception:
            print("Connectivity (directed): n/a")
        # Reciprocity
        try:
            rec = nx.reciprocity(G)
            print(f"Reciprocity: {rec:.6f}")
        except Exception:
            pass
    else:
        try:
            ncc = nx.number_connected_components(G)
            lcc = max((len(c) for c in nx.connected_components(G)), default=0)
            print(f"Connected components: {ncc} (largest={lcc})")
        except Exception:
            print("Connectivity (undirected): n/a")

    # Undirected view for clustering/transitivity/assortativity
    Gu = nx.Graph(G)  # simple undirected view (collapses parallel edges)
    try:
        trans = nx.transitivity(Gu)
        acc = nx.average_clustering(Gu)
        print(f"Transitivity: {trans:.6f} | Avg clustering: {acc:.6f}")
    except Exception:
        print("Clustering: n/a")

    # Degree assortativity (on undirected simple graph)
    try:
        r = nx.degree_assortativity_coefficient(Gu)
        print(f"Degree assortativity: {r:.6f}")
    except Exception:
        pass

    # Triangles
    if heavy and n <= path_limit:
        try:
            tri_total = sum(nx.triangles(Gu).values()) // 3
            print(f"Triangles: {tri_total}")
        except Exception:
            print("Triangles: n/a")
    else:
        print("Triangles: skipped (set heavy=True or reduce graph size)")

    # Shortest-path metrics (on largest component)
    if heavy and n <= path_limit:
        try:
            if is_directed:
                # use largest weakly connected component converted to undirected
                largest = max(nx.weakly_connected_components(G), key=len, default=None)
                if largest:
                    H = Gu.subgraph(largest).copy()
                else:
                    H = None
            else:
                largest = max(nx.connected_components(G), key=len, default=None)
                H = Gu.subgraph(largest).copy() if largest else None

            if H and nx.is_connected(H):
                diam = nx.diameter(H)
                aspl = nx.average_shortest_path_length(H)
                print(f"Diameter (largest comp): {diam} | Avg shortest path length: {aspl:.6f}")
            else:
                print("Shortest-path metrics: graph not connected or empty")
        except Exception:
            print("Shortest-path metrics: n/a")
    else:
        print("Shortest-path metrics: skipped (set heavy=True and ensure n ≤ path_limit)")

    # k-core (max core number)
    try:
        core_num = nx.core_number(Gu)
        kmax = max(core_num.values()) if core_num else 0
        print(f"Max k-core number: {kmax}")
    except Exception:
        pass

    # Edge/vertex connectivity (expensive)
    if heavy and n <= min(2000, path_limit):  # keep limits small; these are expensive
        try:
            ec = nx.edge_connectivity(Gu)
            vc = nx.node_connectivity(Gu)
            print(f"Edge connectivity: {ec} | Vertex connectivity: {vc}")
        except Exception:
            print("Global connectivity: n/a")
    else:
        print("Global connectivity: skipped (heavy=True for small graphs)")

    print("=== End summary ===")

