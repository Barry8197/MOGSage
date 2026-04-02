#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

from network_functions import (
    summarize_graph, 
    export_graph_csv, 
    sample_eids_with_phecode_min_counts,
    build_shared_phecode_graph_sparse,
    attach_phecode_flags_from_df,
    normalize_edges_similarity_from_node_attr,
    fix_phecode_onehot_in_graph,
    stratified_inductive_splits
)

TRAIN_SIZE = 0.7
VAL_SIZE = 0.1
TEST_SIZE = 0.2

def main():
    # Use default datadir, allow optional override via first CLI arg
    datadir = "."
    if len(sys.argv) > 1:
        datadir = sys.argv[1]

    PROJECT_ROOT = '.'
    if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    netdir = Path(datadir) / "Networks"
    netdir.mkdir(parents=True, exist_ok=True)

    prefix = 'Gphecode'
    out_train = netdir / f"{prefix}_tr.gpickle"
    out_val = netdir / f"{prefix}_val.gpickle"
    out_test = netdir / f"{prefix}_te.gpickle"

    # Load data
    with open(f"{datadir}/Phecode_ICD10_converted.pkl", "rb") as file:
        data = pickle.load(file)

    # Filter min phecode counts and patient counts (same logic as notebook)
    phecode_to_keep = data['phecode'].value_counts()[data['phecode'].value_counts() > 200].index
    data = data[data['phecode'].isin(phecode_to_keep)]

#    out = sample_eids_with_phecode_min_counts(data)
#    data = data[data['eid'].isin(out['selected_eids'])]

    # Build graph
    print('Building KG')
    G = build_shared_phecode_graph_sparse(data[['eid', 'phecode']])
    #print(G.number_of_nodes(), G.number_of_edges())

    # Normalize edges (set chosen metric as weight)
    G = normalize_edges_similarity_from_node_attr(G, metric='cosine', out_attr='cosine', replace_weight=True)

    # Attach flags/attributes
    G = attach_phecode_flags_from_df(G, data)

    # Summarize
    #summarize_graph(G)

    # Histogram of edge weights (saved to file)
    #weights = list(nx.get_edge_attributes(G, 'weight').values())
    #if len(weights) > 0:
    #    plt.hist(weights, bins=50)
    #    os.makedirs(f"{datadir}/Networks", exist_ok=True)
    #    plt.savefig(f"{datadir}/Networks/phecode_edge_weight_hist.png", dpi=150, bbox_inches="tight")
    #    plt.close()

    # Fix node attribute in place
    count = fix_phecode_onehot_in_graph(G)
    print(f"Converted 'phecode_onehot' to numpy arrays for {count} nodes.")

    # Quick sanity check
    try:
        any_type = next(iter(G.nodes(data=True)))[1].get("phecode_onehot", None)
        print(f"Example 'phecode_onehot' type: {type(any_type)}")
    except StopIteration:
        print("Graph has no nodes to check.")

    # Splits
    print("Creating stratified inductive splits ...")
    G_train, G_val, G_test, splits, stats = stratified_inductive_splits(
        G, train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE
    )
    print("Split stats:")
    print(stats)

    # Save results
    print(f"Saving train graph to {out_train}")
    with open(out_train, 'wb') as f:
        pickle.dump(G_train, f, pickle.HIGHEST_PROTOCOL)

    print(f"Saving val graph to {out_val}")
    with open(out_val, 'wb') as f:
        pickle.dump(G_val, f, pickle.HIGHEST_PROTOCOL)

    print(f"Saving test graph to {out_test}")
    with open(out_test, 'wb') as f:
        pickle.dump(G_test, f, pickle.HIGHEST_PROTOCOL)

    print("Done.")

if __name__ == "__main__":
    main()
