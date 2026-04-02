#!/usr/bin/env python3
# coding: utf-8
"""
Train MOGSage on UKB graphs with multimodal node features and multilabel targets.
Converted from the provided notebook to an executable script with minimal args.

Requirements:
- dgl, torch, numpy, pandas, networkx, scipy, tqdm, scikit-learn, matplotlib
- Your project modules (network_functions.py, gnn.py, helpers.py) must be importable
  (e.g., installed or on PYTHONPATH). This script does not modify sys.path.

Examples:
    ./train_mogsage.py \
        --datadir /work/gr-fe/bryan/data/UKB \
        --outdir ./results \
        --epochs 2500 \
        --alpha 0.5 \
        --graph-name Gphecode1000

    ./train_mogsage.py \
        --datadir /work/gr-fe/bryan/data/UKB \
        --outdir ./results \
        --graph-name Gphecode1000_tr
"""

import os
import gc
import argparse
import pickle
from functools import reduce
from pathlib import Path

# Keep DGL using PyTorch backend (mirrors notebook)
os.environ["DGLBACKEND"] = "pytorch"

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import trange

import torch
import torch.nn as nn

import dgl
from dgl.dataloading.negative_sampler import Uniform
from dgl.dataloading import DataLoader, NeighborSampler, as_edge_prediction_sampler


def construct_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the script.
    """
    p = argparse.ArgumentParser(description="Train MOGSage with UKB data")
    p.add_argument("--datadir", type=str, default="/work/gr-fe/bryan/data/UKB",
                   help="Base data directory containing Networks/ and 02_processed/")
    p.add_argument("--outdir", type=str, default="./results",
                   help="Output directory for results and artifacts")
    p.add_argument("--graph-name", type=str, default="Gphecode1000",
                   help="Base name for graphs in Networks/. "
                        "Examples: 'Gphecode1000' (will look for *_tr/_val/_psn) or "
                        "'Gphecode1000_tr' (will derive _val/_psn from this).")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--epochs", type=int, default=2500, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size for loaders")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay")
    p.add_argument("--alpha", type=float, default=0.5, help="Edge-vs-node loss tradeoff (alpha for edge loss)")
    p.add_argument("--fanout", type=int, default=15, help="Neighbor sampler fanout per GNN layer")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available")
    return p


def main(argv=None):
    parser = construct_parser()
    args = parser.parse_args(argv)

    # Imports for your project modules (must be importable without sys.path hacks)
    from network_functions import fix_phecode_onehot_in_graph
    from gnn import (
        MOGSage,
        MLPEdgePredictor,
        compute_loss,
        compute_auc,
        merge_dfs,
        evaluate,
        training_metric_plots
    )

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}")

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    networks_dir = datadir / "Networks"

    G_tr_path, G_val_path, G_psn_path = (networks_dir / f"{args.graph_name}_tr.gpickle",
                                  networks_dir / f"{args.graph_name}_val.gpickle",
                                  networks_dir / f"{args.graph_name}_psn.gpickle")
    
    print(f"Training graph:     {G_tr_path}")
    print(f"Validation graph:   {G_val_path}")
    print(f"PSN test graph:     {G_psn_path}")

    # ---------------------------
    # 1) Load graphs and fix labels
    # ---------------------------
    def load_and_fix_graph(path: Path):
        with open(path, "rb") as f:
            G = pickle.load(f)
        cnt = fix_phecode_onehot_in_graph(G)
        print(f"{path.name}: converted phecode_onehot to arrays for {cnt} nodes")
        return G

    g = load_and_fix_graph(G_tr_path)
    g_te = load_and_fix_graph(G_val_path)
    g_psn_te = load_and_fix_graph(G_psn_path)

    # Optional: export sorted phecodes
    unique_attr_names = {k for _, d in g.nodes(data=True) for k in d}
    all_phecodes = list(set(unique_attr_names) - set(["Label", "Label_x", "Label_y", "phecode_onehot"]))
    try:
        phecodes_int = [float(i.split("_")[1]) for i in all_phecodes]
        phecodes_final = pd.Series(phecodes_int).sort_values().reset_index(drop=True)
        phecodes_csv = outdir / f"phecodes_{args.graph_name}.csv"
        phecodes_final.to_csv(phecodes_csv, index=False)
        print(f"Saved phecodes to {phecodes_csv}")
    except Exception as e:
        print(f"Skipping phecodes.csv export: {e}")

    # ---------------------------
    # 2) Load omics
    # ---------------------------
    with open(datadir / "02_processed" / "metabolomics.pkl", "rb") as f:
        metabolomics = pickle.load(f)
    metabolomics["eid"] = metabolomics["eid"].astype(int)
    metabolomics.set_index("eid", inplace=True)

    with open(datadir / "02_processed" / "proteomics.pkl", "rb") as f:
        proteomics = pickle.load(f)
    proteomics = proteomics.fillna(proteomics.mean()).dropna(axis=1)
    proteomics["eid"] = proteomics["eid"].astype(int)
    proteomics.set_index("eid", inplace=True)

    # ---------------------------
    # 3) Align nodes across graphs and omics
    # ---------------------------
    all_nodes = set(g.nodes) | set(g_te.nodes) | set(g_psn_te.nodes)
    metabolomics = metabolomics.loc[list(set(metabolomics.index) & all_nodes)]
    proteomics = proteomics.loc[list(set(proteomics.index) & all_nodes)]

    datModalities = {
        "metabolomics": metabolomics,
        "proteomics": proteomics,
        # "genomics": data_geno,
    }

    # Build meta labels (phecode_onehot arrays -> DataFrame)
    meta_tr = pd.DataFrame(nx.get_node_attributes(g, "phecode_onehot")).T
    meta_te = pd.DataFrame(nx.get_node_attributes(g_te, "phecode_onehot")).T
    meta_psn = pd.DataFrame(nx.get_node_attributes(g_psn_te, "phecode_onehot")).T
    meta_tmp = pd.concat([meta_tr, meta_te, meta_psn], axis=0)
    meta = pd.DataFrame(np.float32(meta_tmp.values), index=meta_tmp.index)

    # Input dims for MOGSage
    MME_input_shapes = [datModalities[m].shape[1] for m in datModalities]

    # Merge modalities, align with meta and index order
    from gnn import merge_dfs  # ensure imported
    h = reduce(merge_dfs, list(datModalities.values()))
    h = h.loc[meta.index]
    h = h.loc[sorted(h.index)]
    meta = meta.loc[h.index]

    meta_csv = outdir / f"meta_phecodes_{args.graph_name}.csv"
    meta.to_csv(meta_csv)
    print(f"Saved meta labels to {meta_csv}")

    # ---------------------------
    # 4) Subgraph and convert to DGL
    # ---------------------------
    def prepare_dgl_graph(G_nx, h_df, meta_df):
        # Nodes to keep (intersection), preserve simple NX iteration order
        common = h_df.index.intersection(meta_df.index)
        nodes, idx = [], {}
        for n in G_nx:
            if n in common:
                idx[n] = len(nodes)
                nodes.append(n)
    
        # Build bidirectional edges over induced nodes
        src, dst = [], []
        for u, v in G_nx.edges():
            iu, iv = idx.get(u), idx.get(v)
            if iu is None or iv is None:
                continue
            src.append(iu); dst.append(iv)
            src.append(iv); dst.append(iu)
    
        # Free the NetworkX graph ASAP
        G_nx.clear()
        del G_nx
        import gc; gc.collect()
    
        # Create DGL graph
        if src:
            g = dgl.graph(
                (torch.tensor(src, dtype=torch.int64),
                 torch.tensor(dst, dtype=torch.int64)),
                num_nodes=len(nodes)
            )
        else:
            g = dgl.graph(([], []), num_nodes=len(nodes))
    
        # Attach features and one-hot labels as float32 tensors
        g.ndata["feat"] = torch.from_numpy(h_df.loc[nodes].to_numpy(dtype=np.float32, copy=False))
        g.ndata["label"] = torch.from_numpy(meta_df.loc[nodes].to_numpy(dtype=np.float32, copy=False))
        return g

    g_dgl = prepare_dgl_graph(g, h, meta)
    g_te_dgl = prepare_dgl_graph(g_te, h, meta)
    g_psn_te_dgl = prepare_dgl_graph(g_psn_te, h, meta)
    g_psn_te_dgl = g_psn_te_dgl.to(device)

    # ---------------------------
    # 5) Model, loaders, loss
    # ---------------------------
    from gnn import MOGSage, MLPEdgePredictor, compute_loss, compute_auc, evaluate, training_metric_plots

    num_classes = meta.shape[1]
    print(f"Detected num_classes (labels): {num_classes}")

    model = MOGSage(
        input_dims=MME_input_shapes,
        encoder_dims=[1000, 1000],
        latent_dims=[256, 256],
        decoder_dim=128,
        hidden_feats=[128],
        num_classes=num_classes
    )
    model.features = h.columns
    model = model.to(device)

    # DGL neighbor sampler
    sampler = NeighborSampler(
        [args.fanout for _ in range(len(model.gnnlayers))],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    neg_sampler = Uniform(k=1)
    edge_sampler = as_edge_prediction_sampler(sampler, negative_sampler=neg_sampler)

    # Note: following the original code that uses nodes() here.
    train_loader = DataLoader(
        g_dgl,
        g_dgl.nodes(),
        edge_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=False
    )
    val_loader = DataLoader(
        g_te_dgl,
        g_te_dgl.nodes(),
        edge_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=False
    )

    # Class imbalance handling with pos_weight from validation set labels
    pos = meta_tr.sum(axis=0) 
    neg = meta_tr.shape[0] - pos  
    pos_weight = torch.Tensor((neg / (pos + 1e-8)).values).clamp(1, 100).to(device)

    criterion_node = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pred = MLPEdgePredictor(model.hidden_feats[-1]).to(device)

    alpha = float(args.alpha)
    beta = 1.0 - alpha

    # ---------------------------
    # 6) Train
    # ---------------------------
    best_val_acc = 0.0
    best_state = None
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    epoch_bar = trange(1, args.epochs + 1, desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        running_acc_sum = 0.0
        seen = 0

        for it, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(train_loader):
            blocks = [b.to(device) for b in blocks]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]

            # Edge scores
            h_i = model.extract_embeddings(x, blocks)
            pos_score = pred(pos_graph, h_i)
            neg_score = pred(neg_graph, h_i)
            loss_edge = compute_loss(pos_score, neg_score)

            # Node logits
            logits = model(x, blocks)
            loss_node = criterion_node(logits, y)

            loss = alpha * loss_edge + beta * loss_node

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_curr = y.size(0)
            running_loss += loss.item() * batch_size_curr
            # Accuracy per-node averaged across labels
            with torch.no_grad():
                preds = (logits.sigmoid() >= 0.5)
                targets = (y > 0.5) if y.is_floating_point() else (y != 0)
                acc_per_node = (preds == targets).float().mean(dim=-1)  # average across labels
                running_acc_sum += acc_per_node.sum().item()
            seen += batch_size_curr

        train_loss = running_loss / max(seen, 1)
        train_acc = running_acc_sum / max(seen, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion_node, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}",
                              train_acc=f"{train_acc:.4f}",
                              val_loss=f"{val_loss:.4f}",
                              val_acc=f"{val_acc:.4f}")

    # Optional: plot training metrics
    try:
        training_metric_plots(history, best_val_acc)
    except Exception as e:
        print(f"training_metric_plots failed or not interactive: {e}")

    # ---------------------------
    # 7) Edge AUC on validation (first batch, as a quick check)
    # ---------------------------
    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    with torch.no_grad():
        for it, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(val_loader):
            blocks = [b.to(device) for b in blocks]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            x = blocks[0].srcdata["feat"]
            h_i = model.extract_embeddings(x, blocks)
            pos_score = pred(pos_graph, h_i)
            neg_score = pred(neg_graph, h_i)
            auc_val = compute_auc(pos_score, neg_score)
            print("-------- Edge Accuracy (val batch) --------------")
            print(f"AUC: {auc_val}")
            print("-------------------------------------------------")
            break

    # ---------------------------
    # 8) Inference on PSN test graph
    # ---------------------------
    with torch.no_grad():
        logits_psn = model(g_psn_te_dgl.ndata["feat"], [g_psn_te_dgl])
        y_psn = g_psn_te_dgl.ndata["label"]
        preds_psn = ((logits_psn.sigmoid() >= 0.5) ==
                     ((y_psn > 0.5) if y_psn.is_floating_point() else (y_psn != 0))).float()
        preds_np = preds_psn.detach().cpu().numpy()

    preds_path = outdir / f"results_{args.graph_name}_{round(alpha, 2)}.pkl"
    with open(preds_path, "wb") as f:
        pickle.dump(preds_np, f)
    print(f"Saved PSN predictions to {preds_path}")

    # ---------------------------
    # 9) Feature importance
    # ---------------------------
    try:
        fi = pd.DataFrame(model.feature_importance(val_loader, device).abs().mean(axis=0))[0]
        top_feats = {}
        for key, df in datModalities.items():
            top_k = fi.loc[df.columns].nlargest(20)
            top_feats[key] = top_k
        top_feats_path = outdir / f"top_feats_{args.graph_name}_{round(alpha, 2)}.pkl"
        with open(top_feats_path, "wb") as f:
            pickle.dump(top_feats, f)
        print(f"Saved top feature importances to {top_feats_path}")
    except Exception as e:
        print(f"Feature importance computation failed: {e}")

    # Save best model weights
    model_path = outdir / f"model_best_{args.graph_name}.pt"
    if best_state is not None:
        torch.save(best_state, model_path)
        print(f"Saved best model state_dict to {model_path}")


if __name__ == "__main__":
    main()
