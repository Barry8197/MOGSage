"""
Utilities for bipartite projection and kNN thresholding on NetworkX graphs.

Functions exported
- collab_project_ids_fast_weighted
- knn_threshold
- summarize_graph
- threshold_graph
- build_bipartite_with_entropy
- export_graph_csv
- load_graph_csv
- add_node_features
- pat_freq_label
- plot_subset_network
- normalize_edge_weights
- integrate_networks
- integrate_networks_weighted
- network_enrichment_test
- network_enrichment_test_continuous
- label_nodes_by_best_centrality
- gen_knn_network
- create_similarity_matrix
- abs_bicorr
- pearson_corr
- cosine_corr
- hamming_dist
- knn_graph_generation
- fix_node_attrs
- fix_phecode_onehot_in_graph
- build_shared_phecode_graph_sparse
- sample_eids_with_phecode_min_counts
- stratified_inductive_splits
- normalize_edge_weights_symmetric
- transfer_attributes
"""

import os
import copy
import heapq
from statistics import mean, median
import numpy as np
import networkx as nx
import astropy.stats
from scipy.sparse import coo_matrix, csr_matrix, diags
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
from scipy.stats import chi2_contingency, fisher_exact , mannwhitneyu, ttest_ind, kruskal, f_oneway
from collections import defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

__all__ = [
    "collab_project_ids_fast_weighted",
    "knn_threshold",
    "summarize_graph",
    "threshold_graph",
    "build_bipartite_with_entropy",
    "export_graph_csv",
    "load_graph_csv",
    "add_node_features",
    "pat_freq_label",
    "plot_subset_network",
    "normalize_edge_weights",
    "integrate_networks",
    "integrate_networks_weighted",
    "network_enrichment_test",
    "network_enrichment_test_continuous",
    "label_nodes_by_best_centrality",
    "gen_knn_network",
    "create_similarity_matrix",
    "abs_bicorr",
    "pearson_corr",
    "cosine_corr",
    "hamming_dist",
    "knn_graph_generation",
    "fix_node_attrs",
    "fix_phecode_onehot_in_graph",
    "build_shared_phecode_graph_sparse",
    "sample_eids_with_phecode_min_counts",
    "attach_phecode_flags_from_df",
    "normalize_edges_similarity_from_node_attr",
    "stratified_inductive_splits",
    "normalize_edge_weights_symmetric",
    "transfer_attributes"
    
]

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
    y_attr='phecode_onehot',
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

def attach_phecode_flags_from_df(
    G,
    df,
    eid_col='eid',
    phecode_col='phecode',
    attr_prefix='phe_',
    value=True,
    clear_existing=False,
    drop_list_attr=True,
    list_attr_name='phecodes'
):
    """
    For each node (eid) in G, attach per-phecode node attributes indicating presence.
    Example: node['phe_208.0'] = True

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes are eids (must match df[eid_col]).
    df : pd.DataFrame
        Must contain columns [eid_col, phecode_col]. Duplicates are dropped.
    eid_col, phecode_col : str
        Column names for eid and phecode.
    attr_prefix : str
        Prefix used for per-phecode attributes.
    value : any
        Value to store for presence (e.g., True or 1).
    clear_existing : bool
        If True, remove any existing attributes that start with attr_prefix before adding flags.
    drop_list_attr : bool
        If True, remove the legacy list attribute (e.g., 'phecodes') if present.
    list_attr_name : str
        Name of the legacy list attribute to drop if drop_list_attr is True.
    """
    x = (df[[eid_col, phecode_col]]
         .dropna()
         .drop_duplicates())

    # Optional: clear existing per-phecode flags
    if clear_existing:
        for n, data in G.nodes(data=True):
            to_del = [k for k in data.keys() if isinstance(k, str) and k.startswith(attr_prefix)]
            for k in to_del:
                del data[k]

    # Group phecodes by eid and set flags
    for eid, sub in x.groupby(eid_col, observed=True):
        if eid not in G:
            # Skip eids that aren't nodes in the graph
            continue
        node = G.nodes[eid]
        # Optionally remove legacy list attribute
        if drop_list_attr and list_attr_name in node:
            del node[list_attr_name]
        # Set flags
        for p in sub[phecode_col].unique():
            node[_phe_attr_key(p, prefix=attr_prefix)] = value

    # Keep a hint in graph about how to find these attributes later
    G.graph['phecode_attr_prefix'] = attr_prefix
    return G

def normalize_edges_similarity_from_node_attr(
    G,
    code_attr='phecodes',
    metric='jaccard',            # 'jaccard' or 'cosine'
    out_attr=None,               # default: 'jaccard' or 'cosine' based on metric
    replace_weight=False,
    use_edge_weight_if_available=True  # use existing edge 'weight' as intersection size
):
    """
    Compute edge similarity from node attribute `code_attr` (list/set of phecodes per node),
    using either Jaccard or Cosine (binary) similarity, and store it as an edge attribute.

    - Jaccard(u,v) = |Su ∩ Sv| / |Su ∪ Sv|
    - Cosine(u,v)  = |Su ∩ Sv| / sqrt(|Su| * |Sv|)  (binary cosine on sets)

    Parameters
    ----------
    G : nx.Graph
    code_attr : str
        Node attribute containing the phecodes (list or set).
    metric : {'jaccard', 'cosine'}
        Which similarity to compute.
    out_attr : str or None
        Edge attribute name to store the similarity. If None, uses the metric name.
    replace_weight : bool
        If True, overwrite edge 'weight' with the similarity value.
    use_edge_weight_if_available : bool
        If True and the edge already has 'weight' equal to the count of shared phecodes,
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

def build_shared_phecode_graph_sparse(df, min_weight=1, onehot_dtype=np.uint8):
    """
    Build an undirected NetworkX graph where nodes are eids, edges connect eids
    that share >=1 phecode, and edge weight is the number of shared phecodes.
    Additionally, attach node features:
      - 'phecode_onehot': 0/1 one-hot vector over all phecodes
      - 'phecodes': list of phecodes the eid has

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ['eid', 'phecode'] (other columns are ignored).
    min_weight : int
        Keep edges with weight >= min_weight.
    onehot_dtype : np.dtype
        Dtype for the one-hot arrays (np.uint8 or bool recommended).

    Returns
    -------
    G : networkx.Graph
        Graph with node attributes:
          - G.nodes[eid]['phecode_onehot'] : np.ndarray shape (n_phecodes,)
          - G.nodes[eid]['phecodes']       : list of phecode values
        And graph-level attribute:
          - G.graph['phecode_categories']  : list giving the index order used in one-hots
    """
    # Use only eid/phecode, drop NaNs and duplicates
    df = df[['eid', 'phecode']].dropna().drop_duplicates()

    # Map ids to category codes (gives us a consistent index order)
    eids = df['eid'].astype('category')
    phe  = df['phecode'].astype('category')
    eid_codes = eids.cat.codes.to_numpy()
    phe_codes = phe.cat.codes.to_numpy()

    n_e = eids.cat.categories.size
    n_p = phe.cat.categories.size

    # Build bipartite incidence matrix B (eids x phecodes), entries 1 if present
    data = np.ones(len(df), dtype=np.int32)
    B = coo_matrix((data, (eid_codes, phe_codes)), shape=(n_e, n_p)).tocsr()

    # Initialize graph and record the phecode order used in one-hots
    G = nx.Graph()
    eid_values = eids.cat.categories.tolist()
    phe_values = phe.cat.categories.tolist()
    print(phe_values)
    G.add_nodes_from(eid_values)
    G.graph['phecode_categories'] = phe_values  # interpret one-hot indices using this list

    # Attach node features
    # For each eid (row i of B), get indices of nonzero phecodes and build:
    #  - list of phecodes (original values)
    #  - one-hot vector (dense). For large n_p, this can be memory-heavy.
    #    If memory is tight, consider:
    #      - storing just the indices as 'phecode_indices' instead of a dense vector, or
    #      - using np.packbits(onehot.astype(np.uint8)) to store a compact bytes array.
    indptr = B.indptr
    indices = B.indices
    for i, eid in enumerate(eid_values):
        col_idx = indices[indptr[i]:indptr[i+1]]  # phecode indices for this eid
        # Dense one-hot vector
        onehot = np.zeros(n_p, dtype=onehot_dtype)
        if len(col_idx) > 0:
            onehot[col_idx] = 1
        # List of phecodes (original values)
        phe_list = [phe_values[j] for j in col_idx]
        # Set node attributes
        G.nodes[eid]['phecode_onehot'] = onehot
        G.nodes[eid]['phecodes'] = phe_list

    # Project: S = B * B^T gives number of shared phecodes between eids
    S = B @ B.T
    S.setdiag(0)
    S.eliminate_zeros()

    # Add weighted edges (iterate upper triangle to avoid duplicates)
    S_coo = S.tocoo()
    for i, j, w in zip(S_coo.row, S_coo.col, S_coo.data):
        if i < j and w >= min_weight:
            ei = eid_values[i]
            ej = eid_values[j]
            G.add_edge(ei, ej, weight=int(w))

    return G


def sample_eids_with_phecode_min_counts(
    df,
    n=1000,
    min_per_code=10,
    eid_col='eid',
    phecode_col='phecode',
    random_state=42
):
    """
    Select up to `n` eids such that as many phecodes as possible appear at least
    `min_per_code` times among the selected eids. Infeasible phecodes (fewer than
    `min_per_code` total eids in the dataset) are filtered out automatically.

    Greedy strategy:
      - Keep a deficit need[c] for each phecode c (initially = min_per_code).
      - At each step, pick the eid that covers the most unmet phecodes (lazy heap).
      - Stop when either all feasible phecodes are satisfied or we pick `n` eids.
      - If we satisfy all feasible phecodes early and still have room, fill with
        random eids to reach `n`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [eid_col, phecode_col]. Duplicates ignored.
    n : int
        Number of eids to select (upper bound).
    min_per_code : int
        Minimum number of selected eids required per phecode.
    eid_col : str
        Column name for eid.
    phecode_col : str
        Column name for phecode.
    random_state : int or np.random.Generator or None
        Random seed or RNG for reproducibility.

    Returns
    -------
    result : dict with keys
        - 'selected_eids': list of selected eids (length <= n)
        - 'kept_phecodes': set of phecodes that were feasible (freq >= min_per_code)
        - 'satisfied_phecodes': set of phecodes that reached the target among selected
        - 'unsatisfied_phecodes': dict {phecode: remaining_deficit}
        - 'coverage_counts': dict {phecode: count_in_selection} for kept phecodes
        - 'status': 'all_satisfied' or 'budget_exhausted'
    """
    # RNG
    rng = np.random.default_rng(random_state) if not isinstance(random_state, np.random.Generator) else random_state

    # Deduplicate pairs and drop NaNs
    x = df[[eid_col, phecode_col]].dropna().drop_duplicates()

    # Universe of eids
    all_eids = x[eid_col].unique().tolist()

    # Build mapping: phecode -> set(eids)
    code_to_eids = defaultdict(set)
    for eid, code in x.itertuples(index=False, name=None):
        code_to_eids[code].add(eid)

    # Keep only feasible phecodes (those with at least min_per_code eids)
    kept_phecodes = {c for c, eids in code_to_eids.items() if len(eids) >= min_per_code}
    if not kept_phecodes:
        # No feasible phecodes at all; just sample n eids at random
        k = min(n, len(all_eids))
        selected_eids = rng.choice(all_eids, size=k, replace=False).tolist()
        return {
            'selected_eids': selected_eids,
            'kept_phecodes': set(),
            'satisfied_phecodes': set(),
            'unsatisfied_phecodes': {},
            'coverage_counts': {},
            'status': 'no_feasible_phecodes'
        }

    # Restrict code_to_eids to kept phecodes
    code_to_eids = {c: eids for c, eids in code_to_eids.items() if c in kept_phecodes}

    # Build eid -> set of kept phecodes (only those that count towards the targets)
    eid_to_codes = defaultdict(set)
    for c, eids in code_to_eids.items():
        for eid in eids:
            eid_to_codes[eid].add(c)

    candidate_eids = set(eid_to_codes.keys())  # eids that can help satisfy constraints

    # Deficits per phecode
    need = {c: min_per_code for c in kept_phecodes}
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

    # Compute coverage counts for kept phecodes within the selected set
    selected_set = set(selected)
    coverage_counts = {c: sum(1 for eid in code_to_eids[c] if eid in selected_set) for c in kept_phecodes}
    satisfied_phecodes = {c for c, cnt in coverage_counts.items() if cnt >= min_per_code}
    unsatisfied_phecodes = {c: max(0, min_per_code - coverage_counts[c]) for c in kept_phecodes if coverage_counts[c] < min_per_code}

    status = 'all_satisfied' if all_satisfied else 'budget_exhausted'

    return {
        'selected_eids': selected,
        'kept_phecodes': kept_phecodes,
        'satisfied_phecodes': satisfied_phecodes,
        'unsatisfied_phecodes': unsatisfied_phecodes,
        'coverage_counts': coverage_counts,
        'status': status
    }

def parse_phecode_onehot_string(s: str, dtype=np.int8) -> np.ndarray:
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
        raise ValueError("Failed to parse phecode_onehot string.")
    return arr

def fix_phecode_onehot_in_graph(G, key='phecode_onehot', dtype=np.int8, check_length=True):
    """
    Convert node attribute `key` from string to NumPy array for all nodes in a NetworkX graph.
    Updates the graph in place. Returns the number of nodes updated.
    """
    n_fixed = 0
    expected_len = None

    for _, attrs in G.nodes(data=True):
        if key in attrs and isinstance(attrs[key], str):
            arr = parse_phecode_onehot_string(attrs[key], dtype=dtype)
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


def label_nodes_by_best_centrality(
    graphs: Sequence[Union[nx.Graph, nx.DiGraph]],
    labels: Sequence[str],
    *,
    weight_attrs: Union[str, Sequence[str]] = "weight",
    centrality_func = nx.eigenvector_centrality_numpy,
    centrality_kwargs: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
    fill_missing: float = float("-inf"),
    normalize: Optional[str] = None,  # None | "zscore" | "minmax"
    tie_break: str = "first",         # "first" | "random"
    unknown_label: Optional[str] = None,  # label for nodes absent in all graphs
    rng: Optional[random.Random] = None,  # used if tie_break="random"
) -> Tuple[Dict[Hashable, str], List[Dict[Hashable, float]]]:
    """
    Compute a centrality per graph and assign each node the label of the graph
    where it attains its highest score.

    Parameters
    ----------
    graphs : sequence of NetworkX graphs
        n graphs (can be mixed directed/undirected, but centrality_func must support them).
    labels : sequence of str
        n labels, one per graph (same order as `graphs`).
    weight_attrs : str or sequence of str, default "weight"
        Edge attribute name(s) used as weights for each graph. If a single str,
        it is used for all graphs. If a sequence, length must equal len(graphs).
    centrality_func : callable, default nx.eigenvector_centrality_numpy
        Function taking (G, ...) and returning a dict {node: score}.
        If it supports a `weight=` kwarg, it will be passed.
    centrality_kwargs : dict or sequence of dict, optional
        Extra kwargs for the centrality function. If a single dict, applied to
        all graphs. If a sequence, must align with graphs.
    fill_missing : float, default -inf
        Score to assign when a node is absent in a graph (or not returned by
        the centrality function). Use -inf to ensure missing cannot win.
    normalize : None | "zscore" | "minmax"
        Optional per-graph normalization applied after computing centralities:
        - "zscore": (x - mean)/std across nodes present in that graph.
        - "minmax": (x - min)/(max - min); if max==min, all become 0.0.
        Use this when comparing scores across very different graphs.
    tie_break : "first" or "random", default "first"
        If multiple graphs share the same maximum score for a node, choose the
        first in input order, or pick randomly (requires `rng` or uses random).
    unknown_label : str or None, default None
        Label assigned to nodes that are absent in all graphs (i.e., all scores
        are fill_missing and equal).
    rng : random.Random, optional
        Random generator used if tie_break="random".

    Returns
    -------
    node_to_label : dict
        Mapping node -> label (one label per node).
    centralities_per_graph : list of dict
        List (aligned with `graphs`) of dicts {node: score} after normalization (if any).
    """
    if not graphs:
        raise ValueError("Provide at least one graph.")
    if len(labels) != len(graphs):
        raise ValueError("`labels` must have the same length as `graphs`.")

    n = len(graphs)

    # Normalize weight_attrs
    if isinstance(weight_attrs, str):
        weight_attrs_list = [weight_attrs] * n
    else:
        if len(weight_attrs) != n:
            raise ValueError("Length of weight_attrs must match number of graphs.")
        weight_attrs_list = list(weight_attrs)

    # Normalize centrality_kwargs
    if centrality_kwargs is None:
        centrality_kwargs_list = [dict() for _ in range(n)]
    elif isinstance(centrality_kwargs, dict):
        centrality_kwargs_list = [centrality_kwargs] * n
    else:
        if len(centrality_kwargs) != n:
            raise ValueError("Length of centrality_kwargs must match number of graphs.")
        centrality_kwargs_list = list(centrality_kwargs)

    # Compute per-graph centralities
    centr_list: List[Dict[Hashable, float]] = []
    all_nodes = set()
    for G in graphs:
        all_nodes.update(G.nodes())

    for G, wattr, kwargs in zip(graphs, weight_attrs_list, centrality_kwargs_list):
        # Try calling with weight kwarg; if unsupported, retry without it
        try:
            scores = centrality_func(G, weight=wattr, **kwargs)  # type: ignore
        except TypeError:
            scores = centrality_func(G, **kwargs)  # type: ignore
        # Ensure scores are floats
        scores = {u: float(s) for u, s in scores.items()}
        centr_list.append(scores)

    # Optional per-graph normalization (based on values present in that graph)
    if normalize is not None:
        norm = normalize.lower()
        if norm not in ("zscore", "minmax"):
            raise ValueError("normalize must be one of {None, 'zscore', 'minmax'}.")
        normed: List[Dict[Hashable, float]] = []
        for scores in centr_list:
            vals = list(scores.values())
            if not vals:
                normed.append(scores)
                continue
            if norm == "zscore":
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                std = math.sqrt(var)
                if std == 0.0:
                    normed.append({u: 0.0 for u in scores})
                else:
                    normed.append({u: (v - mean) / std for u, v in scores.items()})
            else:  # minmax
                vmin, vmax = min(vals), max(vals)
                if vmax == vmin:
                    normed.append({u: 0.0 for u in scores})
                else:
                    scale = vmax - vmin
                    normed.append({u: (v - vmin) / scale for u, v in scores.items()})
        centr_list = normed

    # Assign label per node by best score
    node_to_label: Dict[Hashable, str] = {}

    for node in all_nodes:
        scores = [centr.get(node, fill_missing) for centr in centr_list]
        max_val = max(scores)
        # If all scores equal and equal to fill_missing, treat as unknown
        if all(s == max_val for s in scores) and max_val == fill_missing:
            if unknown_label is not None:
                node_to_label[node] = unknown_label
            else:
                # Skip assignment; continue to next node
                continue
        else:
            # Indices of max
            best_idxs = [i for i, s in enumerate(scores) if s == max_val]
            if len(best_idxs) == 1 or tie_break == "first":
                chosen_idx = best_idxs[0]
            elif tie_break == "random":
                r = rng if rng is not None else random
                chosen_idx = r.choice(best_idxs)
            else:
                # Fallback to first if unrecognized tie_break
                chosen_idx = best_idxs[0]
            node_to_label[node] = labels[chosen_idx]

    return node_to_label, centr_list


try:
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR adjustment. Returns adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    cummin = 1.0
    # iterate from largest to smallest p-value
    for i, idx in enumerate(order[::-1]):
        rank = n - i
        val = pvals[idx] * n / rank
        cummin = min(cummin, val)
        ranked[idx] = cummin
    return np.minimum(ranked, 1.0)

try:
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

def network_enrichment_test_continuous(
    G: nx.Graph,
    value_col: str = "Score",             # continuous node attribute to test
    cluster_col: str = "CommunityCluster",# categorical cluster label on nodes
    alpha: float = 0.05,
    alternative: str = "greater",         # "greater" (higher in-cluster), "less", "two-sided"
    test: str = "mw",                     # "mw" (Mann–Whitney U) or "t" (Welch's t-test)
    fdr_method: str = "bh",
    min_n_in: int = 3,                    # skip if in-cluster n < min_n_in
    min_n_out: int = 3                    # skip if out-of-cluster n < min_n_out
) -> Dict[str, object]:
    """
    Test whether a continuous node attribute is enriched (elevated) within any cluster.

    Returns
    -------
    dict with:
      - "overall": dict with global tests across clusters:
          {"kruskal": {"H","p","k"}, "anova": {"F","p","k"}, "group_stats": DataFrame}
      - "by_cluster": DataFrame, one row per cluster with:
          cluster, n_in, n_out, mean_in, mean_out, median_in, median_out,
          std_in, std_out, diff_mean, diff_median, cohen_d, auc, cliffs_delta,
          stat, p_raw, p_adj, enriched (bool)
    Notes
    -----
    - AUC is derived from Mann–Whitney U: auc = U / (n_in * n_out); Cliff's delta = 2*auc - 1 (ties ignored).
    - Cohen's d uses pooled SD; Welch's t-test is used for "t".
    - P-values are adjusted across all tested clusters (BH-FDR).
    """
    # Pull node attributes
    vals = pd.Series(nx.get_node_attributes(G, value_col), name=value_col)
    clabs = pd.Series(nx.get_node_attributes(G, cluster_col), name=cluster_col)

    if vals.empty or clabs.empty:
        raise ValueError(f"Graph must contain node attributes '{value_col}' and '{cluster_col}'.")

    df = pd.merge(vals, clabs, left_index=True, right_index=True)
    # coerce to numeric, drop NaNs
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col, cluster_col])

    # Need at least two clusters
    k = df[cluster_col].nunique()
    if k < 2:
        raise ValueError("Need at least two clusters for this test.")

    # Global tests across clusters
    groups = [g[value_col].to_numpy() for _, g in df.groupby(cluster_col, sort=False)]
    # Kruskal–Wallis (nonparametric)
    try:
        H, p_kw = kruskal(*groups)
    except Exception:
        H, p_kw = (np.nan, np.nan)
    # One-way ANOVA (parametric; for reference)
    try:
        F, p_anova = f_oneway(*groups)
    except Exception:
        F, p_anova = (np.nan, np.nan)

    # Per-cluster tests (cluster vs rest)
    results = []
    overall_mean = float(df[value_col].mean())
    overall_std = float(df[value_col].std(ddof=1))

    for cl, sub in df.groupby(cluster_col, sort=False):
        x = sub[value_col].to_numpy()
        y = df.loc[df[cluster_col] != cl, value_col].to_numpy()
        n_in = int(x.size)
        n_out = int(y.size)

        if (n_in < min_n_in) or (n_out < min_n_out):
            results.append({
                "cluster": cl, "n_in": n_in, "n_out": n_out,
                "mean_in": np.mean(x) if n_in else np.nan,
                "mean_out": np.mean(y) if n_out else np.nan,
                "median_in": np.median(x) if n_in else np.nan,
                "median_out": np.median(y) if n_out else np.nan,
                "std_in": np.std(x, ddof=1) if n_in > 1 else np.nan,
                "std_out": np.std(y, ddof=1) if n_out > 1 else np.nan,
                "diff_mean": (np.mean(x) - np.mean(y)) if (n_in and n_out) else np.nan,
                "diff_median": (np.median(x) - np.median(y)) if (n_in and n_out) else np.nan,
                "cohen_d": np.nan, "auc": np.nan, "cliffs_delta": np.nan,
                "stat": np.nan, "p_raw": np.nan
            })
            continue

        mean_in = float(np.mean(x)); mean_out = float(np.mean(y))
        med_in = float(np.median(x)); med_out = float(np.median(y))
        sd_in = float(np.std(x, ddof=1)) if n_in > 1 else np.nan
        sd_out = float(np.std(y, ddof=1)) if n_out > 1 else np.nan

        # Effect sizes
        # Cohen's d (pooled SD)
        if (n_in > 1) and (n_out > 1) and np.isfinite(sd_in) and np.isfinite(sd_out):
            sp2 = ((n_in - 1) * sd_in**2 + (n_out - 1) * sd_out**2) / (n_in + n_out - 2)
            cohen_d = (mean_in - mean_out) / np.sqrt(sp2) if sp2 > 0 else np.nan
        else:
            cohen_d = np.nan

        # Statistical test
        p_raw = np.nan
        stat = np.nan
        auc = np.nan
        cliffs = np.nan

        if test.lower() in {"mw", "mannwhitneyu"}:
            # Mann–Whitney U; handles one-sided alternatives
            try:
                res = mannwhitneyu(x, y, alternative=alternative)
                stat = float(res.statistic)
                p_raw = float(res.pvalue)
                # AUC and Cliff's delta (ties ignored)
                auc = stat / (n_in * n_out)
                cliffs = 2.0 * auc - 1.0
            except Exception:
                pass
        elif test.lower() in {"t", "ttest", "ttest_ind"}:
            # Welch's t-test (unequal variances)
            try:
                # SciPy >=1.6 has 'alternative'; if not, do two-sided and halve
                try:
                    res = ttest_ind(x, y, equal_var=False, alternative=alternative)
                    stat, p_raw = float(res.statistic), float(res.pvalue)
                except TypeError:
                    res = ttest_ind(x, y, equal_var=False)
                    stat2, p2 = float(res.statistic), float(res.pvalue)
                    if alternative == "two-sided":
                        stat, p_raw = stat2, p2
                    else:
                        # one-sided from two-sided
                        if alternative == "greater":
                            tail = 0.5 * p2 if stat2 > 0 else 1 - 0.5 * p2
                        else:  # "less"
                            tail = 0.5 * p2 if stat2 < 0 else 1 - 0.5 * p2
                        stat, p_raw = stat2, tail
                # For t-test, we still report Cliff's delta estimate from normal approx if SDs > 0
            except Exception:
                pass
        else:
            raise ValueError("test must be 'mw' or 't'.")

        results.append({
            "cluster": cl,
            "n_in": n_in, "n_out": n_out,
            "mean_in": mean_in, "mean_out": mean_out,
            "median_in": med_in, "median_out": med_out,
            "std_in": sd_in, "std_out": sd_out,
            "diff_mean": mean_in - mean_out,
            "diff_median": med_in - med_out,
            "cohen_d": cohen_d,
            "auc": auc,
            "cliffs_delta": cliffs,
            "stat": stat,
            "p_raw": p_raw
        })

    res_df = pd.DataFrame(results)

    # Adjust p-values across clusters
    if res_df["p_raw"].notna().sum() > 0:
        mask = res_df["p_raw"].notna()
        pvals = res_df.loc[mask, "p_raw"].to_numpy()
        if _HAS_STATSMODELS and fdr_method.lower() in {"bh", "fdr_bh"}:
            _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        else:
            p_adj = _bh_adjust(pvals)
        res_df["p_adj"] = np.nan
        res_df.loc[mask, "p_adj"] = p_adj
    else:
        res_df["p_adj"] = np.nan

    # Enrichment call
    def _is_enriched(row) -> Optional[bool]:
        if pd.isna(row["p_adj"]):
            return None
        if alternative == "greater":
            return bool((row["p_adj"] < alpha) and (row["diff_mean"] > 0))
        elif alternative == "less":
            return bool((row["p_adj"] < alpha) and (row["diff_mean"] < 0))
        else:
            return bool(row["p_adj"] < alpha)

    res_df["enriched"] = res_df.apply(_is_enriched, axis=1)
    res_df = res_df.sort_values(["p_adj", "p_raw"], na_position="last").reset_index(drop=True)

    # Group-level descriptive stats
    grp_rows = []
    for cl, g in df.groupby(cluster_col, sort=False):
        arr = g[value_col].to_numpy()
        grp_rows.append({
            "cluster": cl,
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        })
    group_stats = pd.DataFrame(grp_rows).sort_values("cluster").reset_index(drop=True)

    overall = {
        "kruskal": {"H": H, "p": p_kw, "k": k},
        "anova": {"F": F, "p": p_anova, "k": k},
        "group_stats": group_stats,
        "overall_mean": overall_mean,
        "overall_std": overall_std
    }

    return {"overall": overall, "by_cluster": res_df}



def network_enrichment_test(
    G: nx.Graph,
    center_col: str = "Center",
    cluster_col: str = "CommunityCluster",
    alpha: float = 0.05,
    alternative: str = "greater",  # "greater" tests overrepresentation
    use_yates: bool = False,       # only relevant if the table is 2x2
    fdr_method: str = "bh",        # "bh" (Benjamini–Hochberg)
    min_count: int = 0             # optionally skip cells with very low counts
) -> Dict[str, object]:
    """
    Test whether any Center is overrepresented within any CommunityCluster.

    Returns a dictionary with:
      - "overall": dict with chi2, dof, pvalue, table (counts), expected (expected counts), residuals (standardized)
      - "by_pair": DataFrame with one row per (cluster, center) including:
          cluster, center, a_in_cluster, b_other_centers_in_cluster, c_center_not_in_cluster, d_rest,
          total_in_cluster, prop_in_cluster, baseline_prop, odds_ratio, log2_or, fisher_p, p_adj,
          std_residual, enriched (boolean)

    Parameters
    ----------
    df : DataFrame with at least columns [center_col, cluster_col]
    center_col : column with Center labels
    cluster_col : column with cluster labels
    alpha : FDR threshold for calling enrichment
    alternative : "greater" (enrichment), "less" (depletion), or "two-sided"
    use_yates : apply Yates correction in the global chi-square if 2x2
    fdr_method : "bh" (Benjamini–Hochberg). Additional methods can be added if needed.
    min_count : skip pairwise tests for cells with observed count < min_count

    Notes
    -----
    - Fisher’s exact test is used for pairwise tests (2x2 tables).
    - Multiple testing is controlled across all (cluster, center) pairs via FDR.
    - Standardized residuals come from the global chi-square expected counts and
      are helpful to gauge direction and magnitude at each cell.
    """
    df1 = pd.Series(nx.get_node_attributes(G , center_col , default=np.nan) , name=center_col)
    df2 = pd.Series(nx.get_node_attributes(G , cluster_col , default=np.nan ), name=cluster_col)
    if df1.isna().sum() == len(df1) or df2.isna().sum() == len(df2) : 
        raise ValueError(f"Network must contain attributes '{center_col}' and '{cluster_col}'.")
    
    df = pd.merge(df1 , df2 , right_index=True , left_index=True)
    
    # Validate input
    if center_col not in df.columns or cluster_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{center_col}' and '{cluster_col}'.")

    # Drop missing labels
    work = df[[center_col, cluster_col]].dropna()

    # Need at least 2 centers and 2 clusters to test association
    if work[center_col].nunique() < 2 or work[cluster_col].nunique() < 2:
        raise ValueError("Need at least two unique Centers and two unique Clusters for this test.")

    # Build contingency table: rows=clusters, cols=centers
    ct = pd.crosstab(work[cluster_col], work[center_col])
    total_all = ct.values.sum()

    # Global chi-squared test of independence
    chi2, p_global, dof, expected = chi2_contingency(ct.values, correction=use_yates)
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)

    # Standardized Pearson residuals: (obs - exp) / sqrt(exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = (ct - expected_df) / np.sqrt(expected_df)
    residuals = residuals.replace([np.inf, -np.inf], np.nan)

    # Prepare pairwise Fisher tests for enrichment per (cluster, center)
    results = []
    centers = ct.columns.tolist()
    clusters = ct.index.tolist()

    totals_per_cluster = ct.sum(axis=1)  # total rows
    totals_per_center = ct.sum(axis=0)   # total cols

    for cl in clusters:
        for ce in centers:
            a = int(ct.at[cl, ce])                            # in cluster & in center
            if a < min_count:
                # still include but mark as skipped by setting p to NaN
                skip = True
            else:
                skip = False
            b = int(totals_per_cluster[cl] - a)               # in cluster & not center
            c = int(totals_per_center[ce] - a)                # not in cluster & in center
            d = int(total_all - a - b - c)                    # not in cluster & not center

            table_2x2 = [[a, b], [c, d]]

            # Fisher exact test (one-sided "greater" to test enrichment of center in cluster)
            try:
                or_est, p_fisher = fisher_exact(table_2x2, alternative=alternative)
            except Exception:
                or_est, p_fisher = (np.nan, np.nan)

            # Effect sizes
            total_in_cluster = a + b
            baseline_prop = (a + c) / total_all if total_all > 0 else np.nan
            prop_in_cluster = a / total_in_cluster if total_in_cluster > 0 else np.nan
            log2_or = np.log2(or_est) if (or_est is not None and np.isfinite(or_est) and or_est > 0) else np.nan

            std_res = residuals.at[cl, ce] if pd.notna(residuals.at[cl, ce]) else np.nan

            results.append({
                cluster_col : cl,
                center_col : ce,
                "a_in_cluster": a,
                "b_other_centers_in_cluster": b,
                "c_center_not_in_cluster": c,
                "d_rest": d,
                "total_in_cluster": total_in_cluster,
                "prop_in_cluster": prop_in_cluster,
                "baseline_prop": baseline_prop,
                "odds_ratio": or_est,
                "log2_or": log2_or,
                "fisher_p": (np.nan if skip else p_fisher),
                "std_residual": std_res,
            })

    res_df = pd.DataFrame(results)

    # Multiple testing correction across all pairwise tests
    # Only adjust non-NaN p-values
    if res_df["fisher_p"].notna().sum() > 0:
        mask = res_df["fisher_p"].notna()
        pvals = res_df.loc[mask, "fisher_p"].to_numpy()

        if _HAS_STATSMODELS and fdr_method.lower() in {"bh", "fdr_bh"}:
            _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        else:
            p_adj = _bh_adjust(pvals)

        res_df["p_adj"] = np.nan
        res_df.loc[mask, "p_adj"] = p_adj
    else:
        res_df["p_adj"] = np.nan

    # Call enrichment
    # For "greater", enrichment implies odds_ratio > 1 and significant after FDR
    # For "less", depletion implies odds_ratio < 1 and significant after FDR
    # For "two-sided", use std_residual sign for direction.
    def _is_enriched(row) -> Optional[bool]:
        if pd.isna(row["p_adj"]):
            return None
        if alternative == "greater":
            return bool((row["p_adj"] < alpha) and (row["odds_ratio"] is not None) and (row["odds_ratio"] > 1))
        elif alternative == "less":
            return bool((row["p_adj"] < alpha) and (row["odds_ratio"] is not None) and (row["odds_ratio"] < 1))
        else:
            # two-sided: consider positive residual as enrichment
            return bool((row["p_adj"] < alpha) and (pd.notna(row["std_residual"])) and (row["std_residual"] > 0))

    res_df["enriched"] = res_df.apply(_is_enriched, axis=1)

    # Sort by adjusted p-value
    res_df = res_df.sort_values(["p_adj", "fisher_p"], na_position="last").reset_index(drop=True)

    overall = {
        "chi2": chi2,
        "dof": dof,
        "pvalue": p_global,
        "table": ct,
        "expected": expected_df,
        "residuals": residuals
    }

    return {"overall": overall, "by_pair": res_df}

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


def integrate_networks(
    G1: Union[nx.Graph, nx.DiGraph],
    G2: Union[nx.Graph, nx.DiGraph],
    *,
    weight1: str = "weight",
    weight2: str = "weight",
    out_weight: str = "weight",
    mode: Mode = "intersection",
    missing_policy: MissingPolicy = "copy",
    fill_value: float = 0.0,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Integrate two simple graphs by averaging their edge weights.

    Parameters
    ----------
    G1, G2 : nx.Graph or nx.DiGraph
        Input graphs. Must both be simple graphs (not MultiGraphs) and have the same directedness.
    weight1 : str
        Edge attribute name in G1 holding the weight.
    weight2 : str
        Edge attribute name in G2 holding the weight.
    out_weight : str
        Edge attribute name in the output graph to store the averaged weight.
    mode : {"intersection", "union"}
        - "intersection": only edges present in both graphs are included and averaged.
        - "union": edges present in either graph are included. When an edge is
          missing in one graph (or its weight attribute is missing), apply `missing_policy`.
    missing_policy : {"copy", "fill", "nan", "drop"}
        Policy for edges missing in one graph (only relevant when mode="union"):
        - "copy": use the available weight (no averaging).
        - "fill": average the available weight with `fill_value`.
        - "nan": set the output weight to NaN.
        - "drop": skip that edge.
    fill_value : float
        Value to use with missing_policy="fill".

    Returns
    -------
    H : nx.Graph or nx.DiGraph
        A new graph with averaged edge weights stored in `out_weight`.
        Node attributes are merged (G2 overrides G1 on conflicts). Only the
        `out_weight` edge attribute is set for edges.

    Notes
    -----
    - This function does not support MultiGraph/MultiDiGraph.
    - If an existing edge is missing the specified weight attribute, it is treated
      as "missing" for the purposes of averaging/missing_policy.
    """
    # Basic validations
    if G1.is_multigraph() or G2.is_multigraph():
        raise NotImplementedError("MultiGraphs are not supported.")
    if G1.is_directed() != G2.is_directed():
        raise ValueError("Both graphs must have the same directedness (both directed or both undirected).")

    # Create output graph of same kind as G1
    H = G1.__class__()

    # Merge nodes (G2 overrides G1 on attribute conflicts)
    H.add_nodes_from(G1.nodes(data=True))
    H.add_nodes_from(G2.nodes(data=True))

    # Build edge sets
    E1 = set(G1.edges())
    E2 = set(G2.edges())

    if mode == "intersection":
        edge_pairs = E1 & E2
    elif mode == "union":
        edge_pairs = E1 | E2
    else:
        raise ValueError("mode must be 'intersection' or 'union'.")

    # Helper to get weight if edge exists and attribute is present
    def get_w(G: Union[nx.Graph, nx.DiGraph], u: Hashable, v: Hashable, attr: str) -> Optional[float]:
        if G.has_edge(u, v):
            return G[u][v].get(attr, None)
        return None

    for u, v in edge_pairs:
        w1 = get_w(G1, u, v, weight1)
        w2 = get_w(G2, u, v, weight2)

        # Decide how to compute/assign the output weight
        if w1 is not None and w2 is not None:
            w = (w1 + w2) / 2.0
        else:
            if mode == "intersection":
                # In intersection mode, we only expected edges present in both graphs,
                # but a missing attribute still means we cannot average.
                continue
            # union mode handling of missing
            if missing_policy == "copy":
                w = w1 if w1 is not None else w2
            elif missing_policy == "fill":
                a = w1 if w1 is not None else fill_value
                b = w2 if w2 is not None else fill_value
                w = (a + b) / 2.0
            elif missing_policy == "nan":
                w = float("nan")
            elif missing_policy == "drop":
                continue
            else:
                raise ValueError("missing_policy must be one of {'copy', 'fill', 'nan', 'drop'}.")

        # Add edge with averaged weight
        H.add_edge(u, v, **{out_weight: w})

    return H


def normalize_edge_weights(G, attr='weight', new_attr='weight01'):
    # Collect existing weights (skip edges missing the attribute)
    weights = nx.get_edge_attributes(G, attr)
    if not weights:
        raise ValueError(f"No edges have attribute '{attr}'")

    vals = np.fromiter(weights.values(), dtype=float)
    wmin, wmax = float(vals.min()), float(vals.max())

    if wmax == wmin:
        # All weights equal -> map to 0.0 (or choose 1.0 / 0.5 as you prefer)
        norm = {e: 0.0 for e in weights.keys()}
    else:
        norm = {e: (w - wmin) / (wmax - wmin) for e, w in weights.items()}

    # Attach normalized weights as a new attribute
    nx.set_edge_attributes(G, norm, new_attr)
    return G

def plot_subset_network(
    G: nx.Graph,
    subset_attr: str,
    subset: Union[Any, Iterable, Callable[[Any], bool]],
    *,
    color_attr: Optional[str] = None,
    cmap_continuous: str = "viridis",
    cmap_categorical: str = "tab10",
    default_color: str = "#9e9e9e",
    add_colorbar: bool = True,
    drop_missing_color: bool = True, 
    cat_is_numeric: bool = False,
    node_size: int = 300,
    edge_color: str = "#cccccc",
    layout: Union[str, Dict] = "spring",
    seed: Optional[int] = 42,
    pos: Optional[Dict] = None,           # optional precomputed positions (overrides layout if provided)
    with_labels: bool = False,
    figsize: tuple = (8, 6),
    title: Optional[str] = None,
    return_subgraph: bool = False
):
    """
    Subset a NetworkX graph by a node attribute and plot the subgraph, optionally
    coloring nodes by another attribute.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    subset_attr : str
        Node attribute to filter on.
    subset : scalar | iterable | callable
        - scalar: keep nodes where node[subset_attr] == subset
        - iterable: keep nodes where node[subset_attr] is in subset
        - callable: keep nodes where subset(node[subset_attr]) is True
    color_attr : str, optional
        Node attribute to color nodes by. If None, all nodes use default_color.
        Numeric -> continuous colormap; otherwise -> categorical colors.
    cmap_continuous : str
        Matplotlib colormap name for numeric attributes.
    cmap_categorical : str
        Matplotlib colormap name for categorical attributes.
    default_color : str
        Color for nodes with missing color_attr or when color_attr is None.
    drop_missing_color : bool
        Drop nodes if missing from colour attr
    add_colorbar : bool
        Add colorbar for continuous coloring.
    cat_is_numeric : bool
        Specify if attribute colour is numeric or categorical (default)
    node_size : int
        Node size in the plot.
    edge_color : str
        Edge color.
    layout : "spring" | "kamada_kawai" | "fruchterman_reingold" | dict
        Layout algorithm name, or a positions dict {node: (x,y)}.
    seed : int, optional
        Random seed for layouts that support it.
    pos : dict, optional
        Explicit positions; if provided, overrides layout.
    with_labels : bool
        Whether to draw node labels.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    return_subgraph : bool
        If True, return the induced subgraph (and color mapping when categorical).

    Returns
    -------
    If return_subgraph:
        H : nx.Graph
        info : dict with optional keys {"cat2color": dict} for categorical coloring
    Else:
        None
    """
    def is_nan(x):
        try:
            return np.isnan(x)
        except Exception:
            return False

    def has_value(d, attr):
        if attr is None:
            return True
        v = d.get(attr, None)
        if v is None:
            return False
        # Treat NaN as missing for numeric types
        return not is_nan(v)

    # Build predicate from subset
    if callable(subset):
        pred = subset
    elif isinstance(subset, (set, list, tuple)) and not isinstance(subset, (str, bytes)):
        sset = set(subset)
        pred = lambda v: v in sset
    else:
        pred = lambda v: v == subset

    # Keep nodes that match subset AND (optionally) have a non-missing color_attr
    nodes_keep = []
    for n, d in G.nodes(data=True):
        # must have subset_attr with a non-missing value
        if not has_value(d, subset_attr):
            continue
        if not pred(d[subset_attr]):
            continue
        # optionally require color_attr to be present and non-missing
        if drop_missing_color and not has_value(d, color_attr):
            continue
        nodes_keep.append(n)

    H = G.subgraph(nodes_keep).copy()

    if H.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No nodes match the filter", ha="center", va="center")
        ax.axis("off")
        if return_subgraph:
            return H, {}
        return

    # Positions
    if pos is not None:
        P = {n: pos[n] for n in H.nodes if n in pos}
    else:
        if isinstance(layout, dict):
            P = {n: layout[n] for n in H.nodes if n in layout}
        elif layout == "spring":
            P = nx.spring_layout(H, seed=seed)
        elif layout == "kamada_kawai":
            P = nx.kamada_kawai_layout(H)
        elif layout in ("fruchterman_reingold", "fr"):
            P = nx.fruchterman_reingold_layout(H, seed=seed)
        else:
            # default fallback
            P = nx.spring_layout(H, seed=seed)

    # Compute node colors
    colorbar_obj = None
    cat2color = None
    if color_attr is None:
        node_colors = [default_color] * H.number_of_nodes()
    else:
        vals = [H.nodes[n].get(color_attr, None) for n in H.nodes()]
        # Decide numeric vs categorical
        valid_num = [
            (v is not None) and (isinstance(v, (int, float, np.number))) and not is_nan(v)
            for v in vals
        ]

        if cat_is_numeric:
            arr = np.array([np.nan if (v is None or is_nan(v)) else float(v) for v in vals], dtype=float)
            finite_vals = arr[np.isfinite(arr)]
            if finite_vals.size == 0:
                # All missing -> default color
                node_colors = [default_color] * H.number_of_nodes()
            else:
                vmin, vmax = np.min(finite_vals), np.max(finite_vals)
                if vmin == vmax:
                    vmin -= 1.0
                    vmax += 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap(cmap_continuous)
                node_colors = [cmap(norm(x)) if np.isfinite(x) else default_color for x in arr]
                # Prepare colorbar
                if add_colorbar:
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    colorbar_obj = sm
        else:
            # Categorical
            cats = []
            for v in vals:
                if v is None or is_nan(v):
                    cats.append("__MISSING__")
                else:
                    cats.append(str(v))
            unique_cats = [c for c in dict.fromkeys(cats)]  # preserve order
            n = len(unique_cats)
            
            # Resolve the colormap/palette source
            if isinstance(cmap_categorical, Colormap):
                cmap = cmap_categorical
            elif isinstance(cmap_categorical, (list, tuple, np.ndarray)):
                # Treat as a fixed palette
                palette = list(cmap_categorical)
                # Cycle if fewer colors than categories
                colors = [palette[i % len(palette)] for i in range(n)]
            else:
                # A string colormap name
                cmap = get_cmap(cmap_categorical)
            
            # If we have a Colormap, sample it at evenly spaced points in [0, 1]
            if 'cmap' in locals():
                if n == 1:
                    samples = [0.5]
                else:
                    samples = np.linspace(0, 1, num=n)
                colors = [cmap(s) for s in samples]
            
            # Map categories to colors
            cat2color = dict(zip(unique_cats, colors))
            node_colors = [cat2color[c] for c in cats]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(H, P, edge_color=edge_color, alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(H, P, node_color=node_colors, node_size=node_size, ax=ax)
    if with_labels:
        nx.draw_networkx_labels(H, P, font_size=9, ax=ax)

    if title:
        ax.set_title(title)

    ax.set_axis_off()

    # Add legend or colorbar
    if color_attr is not None:
        if cat2color is not None:
            # Categorical legend
            handles = []
            for cat, col in cat2color.items():
                label = "Missing" if cat == "__MISSING__" else cat
                handles.append(mpatches.Patch(color=col, label=label))
            ax.legend(handles=handles, title=color_attr, loc="best", frameon=False)
        elif colorbar_obj is not None:
            cbar = fig.colorbar(colorbar_obj, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(color_attr)

    plt.tight_layout()
    if return_subgraph:
        info = {}
        if cat2color is not None:
            # Expose category→color mapping (with "Missing" label normalized)
            info["cat2color"] = {("Missing" if k == "__MISSING__" else k): v for k, v in cat2color.items()}
        return H, info
    return


def add_node_features(
    G: nx.Graph,
    df: pd.DataFrame,
    key_col: str,
    feature_cols: Optional[Iterable[str]] = None,
    *,
    label_attr: Optional[str] = None,        # if nodes are identified by a node attribute instead of the node key
    agg: Union[str, Callable] = "first",      # how to resolve duplicates in df (by key_col): "first", "last", "mean", callable
    fillvalue: Optional[float] = None,        # value for nodes missing in df; default None means leave untouched
    inplace: bool = True
):
    """
    Add features from df to nodes of G.

    If label_attr is None, matches df[key_col] to node keys (G.nodes).
    If label_attr is provided, matches df[key_col] to G.nodes[node][label_attr].

    NaNs in df are kept; set fillvalue to assign a value to nodes not present in df.
    """
    if not inplace:
        G = G.copy()

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != key_col]

    # Deduplicate df by key_col according to agg
    if isinstance(agg, str):
        if agg == "first":
            dfu = df.drop_duplicates(key_col, keep="first").set_index(key_col)
        elif agg == "last":
            dfu = df.drop_duplicates(key_col, keep="last").set_index(key_col)
        elif agg in {"mean", "max", "min"}:
            dfu = df.groupby(key_col)[list(feature_cols)].agg(agg)
        else:
            raise ValueError("agg must be 'first', 'last', 'mean', 'max', 'min', or a callable.")
    else:
        dfu = df.groupby(key_col)[list(feature_cols)].agg(agg)

    # Determine mapping target: node keys vs a node attribute
    if label_attr is None:
        # Match directly on node keys
        present = dfu.index.intersection(G.nodes)
        if len(present) > 0:
            nx.set_node_attributes(G, dfu.loc[present, feature_cols].to_dict(orient="index"))
        missing_nodes = [n for n in G.nodes if n not in dfu.index]
        if fillvalue is not None and missing_nodes:
            for col in feature_cols:
                nx.set_node_attributes(G, {n: fillvalue for n in missing_nodes}, name=col)
    else:
        # Build map from label_attr -> node key
        label_to_node = {}
        for n, data in G.nodes(data=True):
            label = data.get(label_attr, None)
            if label is not None and label not in label_to_node:
                label_to_node[label] = n

        # For labels that exist in df and in the graph, set attributes on corresponding nodes
        common_labels = dfu.index.intersection(pd.Index(label_to_node.keys()))
        if len(common_labels) > 0:
            # node -> feature dict
            payload = {label_to_node[l]: attrs for l, attrs in dfu.loc[common_labels, feature_cols].iterrows()}
            nx.set_node_attributes(G, payload)

        # Optionally fill nodes with no matching label
        if fillvalue is not None:
            matched_nodes = set(label_to_node[l] for l in common_labels)
            missing_nodes = [n for n in G.nodes if n not in matched_nodes]
            if missing_nodes:
                for col in feature_cols:
                    nx.set_node_attributes(G, {n: fillvalue for n in missing_nodes}, name=col)

    return None if inplace else G


def pat_freq_label(df, col="CLIN_GROUP", weight="weight"):
    """
    For each ID, return the CLIN_GROUP with the largest total weight.
    Ties are broken by the earliest occurrence in the original row order.
    """
    tmp = (
        df.reset_index()  # original row order kept in 'index'
          .dropna(subset=[col])  # keep same behavior re: CLIN_GROUP NAs
          .groupby(['ID', col], as_index=False)
          .agg(total_weight=(weight, 'sum'), first_pos=('index', 'min'))
    )

    best = (
        tmp.sort_values(['ID', 'total_weight', 'first_pos'],
                        ascending=[True, False, True])
           .drop_duplicates('ID', keep='first')
    )

    return best.set_index('ID')[col]

def load_graph_csv(
    edges_path="edges.csv",
    nodes_path="nodes.csv",
    weight="weight",
    directed=None,
    multigraph=False,
    infer_node_types=True,
    keep_label=True,
    verbose=True,
):
    """
    Load a NetworkX graph from CSVs produced by export_graph_csv.

    Assumptions:
    - edges.csv has columns: Source, Target, [Weight], [other edge attributes...]
    - nodes.csv has columns: Id, Label, [other node attributes...]

    Parameters:
      edges_path: path to edges CSV
      nodes_path: path to nodes CSV (optional but recommended)
      weight: edge weight attribute name to use in the loaded graph
              (maps the exported 'Weight' column to this name)
      directed: True/False to force DiGraph/Graph. If None, a simple heuristic
                is used: graph is considered undirected if the edge set is
                perfectly symmetric; otherwise directed.
      multigraph: if True, construct a Multi(Graph|DiGraph)
      infer_node_types: try to recover Python types from string IDs using
                        ast.literal_eval (e.g., "1"->1, "(1,2)"->(1,2)).
                        Disable if your node IDs are strings like "00123".
      keep_label: if False, drop the 'Label' node attribute when present
      verbose: print a brief summary

    Returns:
      G: a NetworkX Graph/DiGraph/MultiGraph/MultiDiGraph
    """

    def _standardize_colnames(df, names=("Source", "Target", "Weight", "Id", "Label")):
        rename_map = {}
        lower_to_actual = {c.lower(): c for c in df.columns}
        for name in names:
            if name.lower() in lower_to_actual:
                rename_map[lower_to_actual[name.lower()]] = name
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

    def _literal_parse_series(s):
        # Parse series values with ast.literal_eval where possible
        def _parse(v):
            if pd.isna(v):
                return None
            if isinstance(v, (int, float, bool)):
                return v
            if not isinstance(v, str):
                return v
            # Handle common 'nan'/'None' strings
            v_strip = v.strip()
            if v_strip.lower() in ("nan", "none", ""):
                return None
            try:
                return ast.literal_eval(v_strip)
            except Exception:
                return v  # leave as string if not a Python literal
        return s.apply(_parse)

    # Read CSVs as strings first to preserve exact representations
    edf = pd.read_csv(edges_path, dtype=str)
    _standardize_colnames(edf)
    if not {"Source", "Target"}.issubset(edf.columns):
        raise ValueError("Edges CSV must contain 'Source' and 'Target' columns.")

    # Normalize columns and parse types if requested
    if infer_node_types:
        edf["Source"] = _literal_parse_series(edf["Source"])
        edf["Target"] = _literal_parse_series(edf["Target"])

    # Map exported 'Weight' to requested weight attribute name
    if "Weight" in edf.columns:
        if weight != "Weight":
            edf.rename(columns={"Weight": weight}, inplace=True)
        try:
            # best-effort numeric conversion
            edf[weight] = pd.to_numeric(edf[weight])
        except :
            edf[weight] = _literal_parse_series(edf[weight])

    # Decide graph type
    if directed is None:
        # Heuristic: if all edges have their reversed counterpart, assume undirected
        pairs = set(zip(edf["Source"], edf["Target"]))
        rev_pairs = set((v, u) for (u, v) in pairs)
        directed = not (pairs == rev_pairs)

    if multigraph:
        create_using = nx.MultiDiGraph() if directed else nx.MultiGraph()
    else:
        create_using = nx.DiGraph() if directed else nx.Graph()

    # Parse other edge attribute columns
    edge_attr_cols = [c for c in edf.columns if c not in ("Source", "Target")]
    for c in edge_attr_cols:
        if infer_node_types:
            edf[c] = pd.to_numeric(edf[c])

    # Build graph from edges
    G = nx.from_pandas_edgelist(
        edf,
        source="Source",
        target="Target",
        edge_attr=edge_attr_cols if edge_attr_cols else None,
        create_using=create_using,
    )

    # Nodes: load attributes if nodes CSV exists; otherwise create from edges
    if os.path.exists(nodes_path):
        ndf = pd.read_csv(nodes_path, dtype=str)
        _standardize_colnames(ndf)

        if "Id" not in ndf.columns:
            raise ValueError("Nodes CSV must contain 'Id' column.")

        if infer_node_types:
            ndf["Id"] = _literal_parse_series(ndf["Id"])

        node_attr_cols = [c for c in ndf.columns if c != "Id"]
        if not keep_label and "Label" in node_attr_cols:
            node_attr_cols.remove("Label")

        # Parse attribute columns
        for c in node_attr_cols:
            if infer_node_types:
                ndf[c] = _literal_parse_series(ndf[c])

        # Ensure nodes (including isolates) are present and set attributes
        for _, row in ndf.iterrows():
            nid = row["Id"]
            attrs = {k: row[k] for k in node_attr_cols}
            # Optionally provide default label if missing
            if keep_label and "Label" not in attrs:
                attrs["Label"] = str(nid)
            # Add/ensure node and set attributes
            if nid not in G:
                G.add_node(nid)
            if attrs:
                for k, v in attrs.items():
                    if v is not None:
                        G.nodes[nid][k] = v
    else:
        # Create nodes from edges and, if desired, give them a Label
        for n in set(edf["Source"]).union(set(edf["Target"])):
            if n not in G:
                G.add_node(n)
        if keep_label:
            nx.set_node_attributes(G, {n: str(n) for n in G.nodes()}, name="Label")

    if verbose:
        m = G.number_of_edges()
        n = G.number_of_nodes()
        gtype = G.__class__.__name__
        print(f"Loaded {gtype} with {n} nodes and {m} edges from {nodes_path if os.path.exists(nodes_path) else '(no nodes.csv)'} and {edges_path}.")

    return G


def export_graph_csv(
    G,
    edges_path="edges.csv",
    nodes_path="nodes.csv",
    weight="weight",
    directed=None
):
    """
    Fast export of a NetworkX graph to CSVs for Gephi import.
    - edges.csv: Source, Target, [Weight], [edge attributes...]
    - nodes.csv: Id, Label, [node attributes...]

    Gephi: Data Laboratory -> Import Spreadsheet (first import nodes, then edges)
    """
    if directed is None:
        directed = G.is_directed()

    # Edges
    edf = nx.to_pandas_edgelist(G)
    edf.rename(columns={"source": "Source", "target": "Target"}, inplace=True)
    if weight in edf.columns:
        edf.rename(columns={weight: "Weight"}, inplace=True)
    else:
        edf["Weight"] = 1.0

    # Nodes
    ndf = pd.DataFrame({"Id": list(G.nodes())})
    ndf["Label"] = ndf["Id"].astype(str)
    # Optionally merge node attributes
    if G.number_of_nodes() > 0:
        nattrs = nx.get_node_attributes(G, None)  # returns {}
        # Flatten node attributes if you have them:
        if any(G.nodes[n] for n in G.nodes()):
            na = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
            na.index.name = "Id"
            na.reset_index(inplace=True)
            ndf = ndf.merge(na, on="Id", how="left")

    ndf.to_csv(nodes_path, index=False)
    edf.to_csv(edges_path, index=False)
    print(f"Saved {len(ndf)} nodes to {nodes_path} and {len(edf)} edges to {edges_path}.")


def build_bipartite_with_entropy(
    clin: pd.DataFrame,
    id_col: str = "ID",
    clin_col: str = "CLIN_ID",
    weight_input_col: Optional[str] = None,   # if None, counts occurrences; else uses and sums this column
    dropna: bool = True,
    normalization: str = "w_norm_tfidf",      # one of: 'weight','w_norm_sum','w_norm_deg','w_norm_bi','w_norm_col','w_norm_col_sqrt','w_norm_tfidf'
    apply_entropy: bool = True,
    # Entropy sigmoid parameters
    h0: float = 0.4,                          # centre of sigmoid in [0,1]
    t: float = 8.0,                           # steepness
    s_min: float = 0.1,                       # minimum scaling factor
    s_max: float = 1.0,                       # maximum scaling factor
    lambda_entropy: float = 0.7,              # blend between original (0) and fully scaled (1)
    graph_class=nx.DiGraph                    # choose nx.DiGraph or nx.Graph
) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Build a bipartite graph (IDs -> CLIN_IDs) with multiple weight normalizations
    and optional per-ID entropy scaling.

    Parameters
    ----------
    clin : DataFrame with at least [id_col, clin_col] and optionally [weight_input_col]
    id_col, clin_col : column names for IDs and clinics
    weight_input_col : if None, edge weights are counts of (ID, CLIN_ID);
                       else use clin[weight_input_col] summed over duplicates
    dropna : drop rows with NA in id_col/clin_col before aggregation
    normalization : which weight to put on edges before entropy scaling
                    options: 'weight','w_norm_sum','w_norm_deg','w_norm_bi','w_norm_col','w_norm_col_sqrt','w_norm_tfidf'
    apply_entropy : whether to apply per-ID entropy scaling via a sigmoid on H_norm
    h0, t, s_min, s_max, lambda_entropy : controls for entropy scaling
    graph_class : nx.DiGraph (default) or nx.Graph

    Returns
    -------
    G_clin : NetworkX graph with edges (ID -> CLIN_ID) carrying:
             - weight: normalized+entropy-scaled weight
             - raw_weight: original (aggregated) weight
    edges  : DataFrame with all intermediate columns and the final used weight column
    """
    # 1) Aggregate to edges
    cols = [id_col, clin_col] + ([weight_input_col] if weight_input_col else [])
    df = clin[cols]
    if dropna:
        df = df.dropna(subset=[id_col, clin_col])

    if weight_input_col is None:
        # counts of (ID, CLIN_ID)
        edges = df.value_counts([id_col, clin_col]).reset_index(name="weight")
    else:
        # sum provided weights over duplicates
        edges = (
            df.groupby([id_col, clin_col], as_index=False)[weight_input_col]
              .sum()
              .rename(columns={weight_input_col: "weight"})
        )

    if edges.empty:
        # return empty graph with nodes inferred from unique IDs/CLINs in clin
        G_empty = graph_class()
        return G_empty, edges

    # 2) Patient- and clinic-level stats
    edges["sum_w"]   = edges.groupby(id_col)["weight"].transform("sum")
    edges["out_deg"] = edges.groupby(id_col)[clin_col].transform("nunique")

    edges["in_sum_w"] = edges.groupby(clin_col)["weight"].transform("sum")
    edges["in_deg"]   = edges.groupby(clin_col)[id_col].transform("nunique")

    N_patients = edges[id_col].nunique()

    # 3) Normalizations
    # Row-stochastic
    edges["w_norm_sum"] = edges["weight"] / edges["sum_w"].replace(0, np.nan)
    # Degree-normalized (patient side)
    edges["w_norm_deg"] = edges["weight"] / edges["out_deg"].replace(0, np.nan)
    # Bi-normalization: w / sqrt(sum_w_i * in_sum_w_c)
    edges["w_norm_bi"] = edges["weight"] / np.sqrt(
        edges["sum_w"].clip(lower=1) * edges["in_sum_w"].clip(lower=1)
    )
    # Clinic-side penalties
    edges["w_norm_col"]      = edges["in_sum_w"].replace(0, np.nan) / edges["weight"]
    edges["w_norm_col_sqrt"] = np.sqrt(edges["in_sum_w"].clip(lower=1)) / edges["weight"]
    # TF-IDF: row-stochastic * IDF(unique patients per clinic)
    idf = np.log1p(N_patients / edges["in_deg"].clip(lower=1))
    edges["w_norm_tfidf"] = (edges["weight"] / edges["sum_w"].replace(0, np.nan)) * idf

    for col in ["w_norm_sum","w_norm_deg","w_norm_bi","w_norm_col","w_norm_col_sqrt","w_norm_tfidf"]:
        edges[col] = edges[col].fillna(0.0)

    # 3b) Per-ID entropy from w_norm_sum (probabilities)
    eps = 1e-12
    p = edges["w_norm_sum"].clip(lower=eps)
    h_term = -p * np.log(p)
    H = h_term.groupby(edges[id_col]).transform("sum")
    m = edges.groupby(id_col)[clin_col].transform("nunique")
    H_max = np.log(m.clip(lower=2))
    # normalised entropy in [0,1], define 0 when m < 2
    H_norm = np.where(m.values >= 2, (H / H_max).values, 0.0)

    # Sigmoid mapping -> entropy_scale in [s_min, s_max]
    sig = 1.0 / (1.0 + np.exp(-t * (H_norm - h0)))
    entropy_scale = s_min + (s_max - s_min) * sig
    edges["entropy"] = H.values
    edges["entropy_norm"] = H_norm
    edges["entropy_scale"] = entropy_scale

    # 4) Choose base weight and apply entropy blending
    valid_norms = {
        "weight","w_norm_sum","w_norm_deg","w_norm_bi","w_norm_col","w_norm_col_sqrt","w_norm_tfidf"
    }
    if normalization not in valid_norms:
        raise ValueError(f"normalization must be one of {sorted(valid_norms)}")

    weight_col = normalization
    weight_col_scaled = f"{weight_col}_entropy"

    if apply_entropy:
        edges[weight_col_scaled] = edges[weight_col] * ((1 - lambda_entropy) + lambda_entropy * edges["entropy_scale"])
    else:
        edges[weight_col_scaled] = edges[weight_col]

    # 5) Build bipartite graph
    patients = edges[id_col].unique()
    clinics  = edges[clin_col].unique()

    G_clin = graph_class()
    G_clin.add_nodes_from(patients, bipartite=0, kind="ID")
    G_clin.add_nodes_from(clinics,  bipartite=1, kind="CLIN_ID")

    for row in edges.itertuples(index=False):
        G_clin.add_edge(
            getattr(row, id_col),
            getattr(row, clin_col),
            weight=float(getattr(row, weight_col_scaled)),
            raw_weight=float(row.weight),
        )

    edges['raw_weight'] = edges['weight'].copy()
    edges['weight'] = edges[weight_col_scaled]

    return G_clin, edges


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



def collab_project_ids_fast_weighted(
    B, ids, clins, attr="weight",
    scheme="fractional",     # 'fractional' | 'cosine' | 'none'
    alpha=1.0,               # for 'fractional': use 1/(k-1)^alpha (alpha=1 is classic; 0.5 is softer)
    k_mode="binary",         # 'binary' (unique IDs per clinic) or 'weighted' (sum of weights into clinic)
    dtype=np.float64
):
    """
    Project a bipartite graph B (IDs x CLIN_IDs) onto IDs.

    Parameters
    ----------
    B : nx.Graph or nx.DiGraph
    ids, clins : list-like
        Node lists defining the two partitions (must be disjoint sets in B).
    attr : str
        Edge attribute to use as weight.
    scheme : str
        'fractional': S = M * diag(1/(k-1)^alpha) * M^T, with k from k_mode.
        'cosine'    : cosine similarity of rows (IDs).
        'none'      : plain S = M * M^T.
    alpha : float
        Exponent for fractional counting (alpha=1 classic, 0.5 is 1/sqrt(k-1)).
    k_mode : str
        'binary' uses k = number of distinct IDs per clinic (ignores edge weight).
        'weighted' uses k = sum of weights into clinic.
    dtype : type
        Numeric dtype for sparse matrices (use float64 to reduce underflow/rounding).

    Returns
    -------
    G_id : nx.Graph
        ID-only projection with 'weight' edges.
    S : csr_matrix
        The n_ids x n_ids sparse matrix of projected weights (symmetric, zero diagonal).
    """
    id_ix   = {n: i for i, n in enumerate(ids)}
    clin_ix = {n: j for j, n in enumerate(clins)}
    n, m = len(ids), len(clins)

    # Build COO triplets for M (IDs are rows, CLIN_IDs are columns)
    rows, cols, data = [], [], []
    for u, v, d in B.edges(data=True):
        # Enforce a single canonical direction: (ID, CLIN_ID)
        if u in id_ix and v in clin_ix:
            w = d.get(attr, 1.0)
            rows.append(id_ix[u]); cols.append(clin_ix[v]); data.append(float(w))
        # If your B might have edges in the reverse direction only, keep this branch.
        # Else, comment it out to avoid double-counting when both directions exist.
        elif v in id_ix and u in clin_ix:
            w = d.get(attr, 1.0)
            rows.append(id_ix[v]); cols.append(clin_ix[u]); data.append(float(w))

    if not rows:
        return nx.Graph(), csr_matrix((n, n), dtype=dtype)

    M = coo_matrix((np.array(data, dtype=dtype), (rows, cols)), shape=(n, m)).tocsr()
    M.sum_duplicates()  # make sure duplicates are summed

    # Compute S according to chosen scheme
    if scheme == "cosine":
        # L2 normalize rows: M_norm = D_r^{-1} M, then S = M_norm M_norm^T in [0,1]
        row_norm = np.sqrt(M.multiply(M).sum(axis=1)).A1
        row_scale = np.reciprocal(np.maximum(row_norm, 1e-12))
        M_norm = diags(row_scale) @ M
        S = (M_norm @ M_norm.T).tocsr()

    else:
        if scheme == "none":
            S = (M @ M.T).tocsr()
        elif scheme == "fractional":
            # k from binary membership or weighted in-sum
            if k_mode == "binary":
                k = np.asarray((M > 0).sum(axis=0)).ravel().astype(dtype)  # distinct IDs per clinic
            elif k_mode == "weighted":
                k = np.asarray(M.sum(axis=0)).ravel().astype(dtype)        # total incoming weight per clinic
            else:
                raise ValueError("k_mode must be 'binary' or 'weighted'")

            s = np.zeros_like(k)
            mask = k >= 2 if k_mode == "binary" else k > 0
            # 1/(k-1)^alpha (or 1/k^alpha if you prefer when k_mode='weighted')
            denom = (k[mask] - 1.0) if k_mode == "binary" else k[mask]
            s[mask] = 1.0 / np.power(denom, alpha)
            S = (M @ diags(s) @ M.T).tocsr()
        else:
            raise ValueError("scheme must be 'fractional', 'cosine', or 'none'")

    # Zero diagonal and drop zeros
    S.setdiag(0)
    S.eliminate_zeros()

    # Build NetworkX graph over IDs
    G_id = nx.Graph()
    G_id.add_nodes_from(ids)
    S_coo = S.tocoo()
    for i, j, w in zip(S_coo.row, S_coo.col, S_coo.data):
        if i < j:
            G_id.add_edge(ids[i], ids[j], weight=float(w))

    return G_id, S


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

