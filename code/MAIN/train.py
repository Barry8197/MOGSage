"""
Utilities for edge prediction using GNN and Deep Graph Library

Functions exported
- compute_loss
- compute_auc
- evaluate
- training_metric_plots
- get_pos_neg_edges
- evaluate_multilabel
- normalise_adjs
- make_loader
- split_edges_inductive
- contrastive_distill_loss
- per_class_metrics_from_scores
- plot_per_class_confusion_matrices
"""

import os
import gc
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Optional

import seaborn as sns
import matplotlib.pyplot as plt
import palettable.wesanderson as wes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph, remove_self_loops, negative_sampling, train_test_split_edges
from torch_geometric.loader import NeighborSampler

from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    jaccard_score, classification_report, roc_auc_score
)

__all__ = [
    "compute_loss",
    "compute_auc",
    "evaluate",
    "training_metric_plots",
    "get_pos_neg_edges",
    "evaluate_multilabel",
    "normalise_adjs",
    "make_loader",
    "split_edges_inductive",
    "contrastive_distill_loss",
    "per_class_metrics_from_scores"
    "plot_per_class_confusion_matrices"
]

def plot_per_class_confusion_matrices(
    metrics_dict,
    class_labels=None,
    normalize="true",          # "true" | "pred" | "all" | None
    show_counts=False,         # if True, annotate "prop\n(count)"
    annotate_fmt=".2f",
    figsize_per_plot=(4.2, 3.6),
    cbar=True,
):
    """
    Plot one-vs-rest 2x2 confusion matrices per class as proportions,
    using the Wes Anderson 'Zissou' continuous colormap.

    Expects:
      metrics_dict['confusion'] with keys 'tp','fp','fn','tn' as arrays (n_classes,)
    """
    conf = metrics_dict["confusion"]
    tp = np.asarray(conf["tp"], dtype=float)
    fp = np.asarray(conf["fp"], dtype=float)
    fn = np.asarray(conf["fn"], dtype=float)
    tn = np.asarray(conf["tn"], dtype=float)

    n_classes = tp.shape[0]
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]
    if len(class_labels) != n_classes:
        raise ValueError(f"Expected {n_classes} class_labels, got {len(class_labels)}")

    # Wes Anderson palette via palettable: Zissou continuous colormap
    from palettable.wesanderson import Zissou_5_r as _Z
    cmap = _Z.mpl_colormap  # continuous matplotlib colormap

    figs = []
    for i, label in enumerate(class_labels):
        cm_counts = np.array([[tp[i], fn[i]],
                              [fp[i], tn[i]]], dtype=float)

        # ---- normalize to proportions ----
        if normalize is None:
            cm = cm_counts
        elif normalize == "true":
            denom = cm_counts.sum(axis=1, keepdims=True)
            cm = np.divide(cm_counts, denom, out=np.zeros_like(cm_counts), where=denom != 0)
        elif normalize == "pred":
            denom = cm_counts.sum(axis=0, keepdims=True)
            cm = np.divide(cm_counts, denom, out=np.zeros_like(cm_counts), where=denom != 0)
        elif normalize == "all":
            denom = cm_counts.sum()
            cm = cm_counts / denom if denom != 0 else np.zeros_like(cm_counts)
        else:
            raise ValueError('normalize must be one of: "true", "pred", "all", or None')

        # ---- annotations ----
        if show_counts and normalize is not None:
            ann = np.empty_like(cm, dtype=object)
            for r in range(2):
                for c in range(2):
                    ann[r, c] = f"{cm[r,c]:{annotate_fmt}}\n({int(cm_counts[r,c])})"
            annot = ann
            fmt = ""
        else:
            annot = True
            fmt = annotate_fmt if normalize is not None else "d"

        fig, ax = plt.subplots(figsize=figsize_per_plot)
        sns.heatmap(
            cm,
            ax=ax,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            cbar=cbar,
            vmin=0.0,
            vmax=1.0 if normalize is not None else None,
            square=True,
            linewidths=1,
            linecolor="white",
            xticklabels=["Pred +", "Pred -"],
            yticklabels=["True +", "True -"],
        )
        ax.set_title(f"{label} (one-vs-rest) — normalize={normalize}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        figs.append(fig)

    return figs


def per_class_metrics_from_scores(y_pred: np.ndarray,
                                 y_true: np.ndarray,
                                 threshold: float = 0.5):
    """
    Compute per-class accuracy, F1 (micro), and F1 (macro) for multi-label or one-vs-rest framing.

    Parameters
    ----------
    y_pred : (H, C) array
        Predicted scores/probabilities (or already-binary predictions).
    y_true : (H, C) array
        Ground-truth labels. If `assume_one_hot_true=True`, it should be one-hot (single-label).
        Otherwise it can be multi-hot (multi-label).
    threshold : float
        Threshold used to binarize y_pred if it is not already {0,1}.

    Returns
    -------
    metrics : dict
        {
          "per_class": {
             "accuracy": (C,),
             "f1_micro": (C,),
             "f1_macro": (C,),
             "precision": (C,),
             "recall": (C,),
             "support_pos": (C,),  # number of positives in y_true per class
          },
          "confusion": {
             "tp": (C,), "fp": (C,), "fn": (C,), "tn": (C,)
          }
        }

    Notes
    -----
    For each class k, we treat it as a binary problem (class k vs not-k):
      TP_k, FP_k, FN_k, TN_k
    Per-class micro-F1 equals the binary F1 for that class:
      F1_micro_k = 2*TP_k / (2*TP_k + FP_k + FN_k)
    Per-class macro-F1 is the average of positive-class F1 and negative-class F1:
      F1_macro_k = (F1_pos_k + F1_neg_k)/2
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape != y_true.shape or y_pred.ndim != 2:
        raise ValueError(f"Expected y_pred and y_true to have same shape (H, C). "
                         f"Got {y_pred.shape=} and {y_true.shape=}.")

    H, C = y_true.shape

    # Binarize predictions if needed
    if np.issubdtype(y_pred.dtype, np.floating) or np.any((y_pred != 0) & (y_pred != 1)):
        y_pred_bin = (y_pred >= threshold).astype(np.int64)
    else:
        y_pred_bin = y_pred.astype(np.int64)

    y_true_bin = y_true.astype(np.int64)
    if np.any((y_true_bin != 0) & (y_true_bin != 1)):
        raise ValueError("y_true must be binary (0/1)")

    # Confusion terms per class (vectorised)
    tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1), axis=0)
    fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0), axis=0)
    fn = np.sum((y_pred_bin == 0) & (y_true_bin == 1), axis=0)
    tn = np.sum((y_pred_bin == 0) & (y_true_bin == 0), axis=0)

    # Per-class accuracy
    acc = (tp + tn) / (tp + fp + fn + tn)

    # Precision/Recall for positive class (per class)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)

    # Per-class F1 (micro == binary F1 for this one-vs-rest class)
    f1_micro = np.divide(
        2 * tp, 2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) != 0
    )

    # Negative-class F1 (treat "not class k" as positive), then macro-average within the class.
    precision_neg = np.divide(tn, tn + fn, out=np.zeros_like(tn, dtype=float), where=(tn + fn) != 0)
    recall_neg    = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    f1_neg = np.divide(
        2 * precision_neg * recall_neg, precision_neg + recall_neg,
        out=np.zeros_like(precision_neg, dtype=float),
        where=(precision_neg + recall_neg) != 0
    )

    f1_macro = 0.5 * (f1_micro + f1_neg)

    return {
        "per_class": {
            "accuracy": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
            "support_pos": np.sum(y_true_bin == 1, axis=0),
        },
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

def contrastive_distill_loss(z_student, z_teacher, temperature=0.07):
    # Conceptual InfoNCE-style contrastive distillation loss
    # z_student, z_teacher: [B, D], L2-normalized
    z_s = F.normalize(z_student, dim=-1)
    z_t = F.normalize(z_teacher, dim=-1)
    
    # Similarities: each student embedding vs all teacher embeddings
    logits = (z_s @ z_t.T) / temperature  # [B, B]
    
    # Diagonal = positive pairs (same node)
    labels = torch.arange(len(z_s), device=z_s.device)
    
    return F.cross_entropy(logits, labels)

def split_edges_inductive(edge_index, val_ratio=0.1, seed=42):
    torch.manual_seed(seed)

    # Guard: empty graph
    if edge_index.size(1) == 0:
        empty = torch.zeros(2, 0, dtype=torch.long)
        return empty, empty, empty

    # Remove self-loops before splitting
    from torch_geometric.utils import remove_self_loops, to_undirected
    edge_index, _ = remove_self_loops(edge_index)

    # Make undirected so both directions are present
    edge_index = to_undirected(edge_index)

    E    = edge_index.size(1)
    perm = torch.randperm(E)

    n_sup  = max(int(E * val_ratio), 1)   # at least 1 supervision edge
    sup_idx = perm[:n_sup]
    mp_idx  = perm[n_sup:]

    # Guard: don't leave message passing empty
    if mp_idx.numel() == 0:
        mp_idx = perm   # fall back to using all edges for MP

    mp_edge_index = edge_index[:, mp_idx]
    sup_pos_edges = edge_index[:, sup_idx]

    num_nodes = int(edge_index.max().item()) + 1

    from torch_geometric.utils import negative_sampling
    sup_neg_edges = negative_sampling(
        edge_index=edge_index,          # avoid ALL known edges
        num_nodes=num_nodes,
        num_neg_samples=sup_idx.numel(),
        method='sparse',
    )

    # Guard: negative_sampling can return None if graph is too dense
    if sup_neg_edges is None or sup_neg_edges.size(1) == 0:
        print("Warning: negative sampling returned empty, using random pairs")
        src = torch.randint(0, num_nodes, (sup_idx.numel(),))
        dst = torch.randint(0, num_nodes, (sup_idx.numel(),))
        sup_neg_edges = torch.stack([src, dst], dim=0)

    print(f"split_edges_inductive: total={E}, mp={mp_idx.numel()}, "
          f"sup_pos={sup_pos_edges.size(1)}, sup_neg={sup_neg_edges.size(1)}, "
          f"num_nodes={num_nodes}")

    return mp_edge_index, sup_pos_edges, sup_neg_edges

def make_loader(ei, sampler_sizes, batch_size,  shuffle):
    return NeighborSampler(
        ei,
        node_idx=torch.arange(ei.max() + 1),
        sizes=sampler_sizes,
        batch_size=batch_size,
        shuffle=shuffle,
    )

def normalise_adjs(adjs, num_convs, device):
    if num_convs == 1:
        return [adjs.to(device)]
    return [(ei.to(device), e_id, size) for (ei, e_id, size) in adjs]

def evaluate_multilabel(y_true, y_pred, name='Model'):
    """Compute and display a suite of multi-label metrics."""
    hl  = hamming_loss(y_true, y_pred)
    em  = accuracy_score(y_true, y_pred)            # exact match
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_samp  = f1_score(y_true, y_pred, average='samples', zero_division=0)
    jac      = jaccard_score(y_true, y_pred, average='samples', zero_division=0)

    metrics = {
        'Model'        : name,
        'Hamming Loss' : round(hl,  4),
        'Exact Match'  : round(em,  4),
        'Micro F1'     : round(f1_micro, 4),
        'Macro F1'     : round(f1_macro, 4),
        'Sample F1'    : round(f1_samp,  4),
        'Jaccard'      : round(jac, 4),
    }
    return metrics

def get_pos_neg_edges(edge_index, target_nid):
    # Keep only edges among target_nid and relabel to local IDs 0..B-1
    pos_local, _ = subgraph(target_nid, edge_index, relabel_nodes=True)

    # Remove self-loops
    pos_local, _ = remove_self_loops(pos_local)

    # Sample negatives within the B local nodes, avoiding pos_local
    B = target_nid.numel()
    num_pos = pos_local.size(1)
    num_neg = max(num_pos, 1)
    neg_local = negative_sampling(
        edge_index=pos_local,
        num_nodes=B,
        num_neg_samples=num_neg,
        method='sparse'
    )
    return pos_local, neg_local

def training_metric_plots(history, best_val_acc) : 
    sns.set_theme(style="whitegrid")
    sns.set_palette(wes.Darjeeling2_5.mpl_colors)
    hist_df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Loss plot
    loss_df = hist_df.melt(id_vars="epoch", value_vars=["train_loss", "val_loss"],
                           var_name="set", value_name="loss")
    sns.lineplot(data=loss_df, x="epoch", y="loss", hue="set", ax=axes[0])
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(title="")
    
    # Accuracy plot
    acc_df = hist_df.melt(id_vars="epoch", value_vars=["train_acc", "val_acc"],
                          var_name="set", value_name="accuracy")
    sns.lineplot(data=acc_df, x="epoch", y="accuracy", hue="set", ax=axes[1])
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(title="")
    
    plt.show()
    
    # Print final metrics
    print(f"Final epoch - Train acc: {history['train_acc'][-1]:.4f}, "
          f"Val acc: {history['val_acc'][-1]:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

@torch.no_grad()
def evaluate_inductive(
    model, x_all, y_all,
    val_loader,
    edge_index_val,
    score_edges,
    criterion_node,
    device, alpha=0.5,
):
    model.eval()
    num_nodes_total = x_all.size(0)

    # --------------------------------------------------
    # Pass 1: cache embeddings for every val node
    # --------------------------------------------------
    h_cache = torch.zeros(num_nodes_total, model.hidden_feats[-1], device=device)

    for batch_size_cur, n_id, adjs in val_loader:
        adjs = normalise_adjs(adjs, len(model.convs), device)
        x    = x_all[n_id].to(device)

        h_i  = model.extract_embeddings(x, adjs)          # [batch_size_cur, dim]
        
        # Only store embeddings for the TARGET nodes (first batch_size_cur)
        h_cache[n_id[:batch_size_cur]] = h_i.detach()

    # --------------------------------------------------
    # Pass 2: score supervision edges from cached embeddings
    # --------------------------------------------------
    pos_val, neg_val = get_pos_neg_edges(
        edge_index_val.to(device),
        torch.arange(edge_index_val.max() + 1).to(device),
    )

    # Batch the edge scoring to avoid OOM on large graphs
    def score_in_batches(edge_index, batch_size=4096):
        scores = []
        for i in range(0, edge_index.size(1), batch_size):
            e     = edge_index[:, i:i+batch_size]
            src   = h_cache[e[0]]
            dst   = h_cache[e[1]]
            # Inline dot product to avoid re-building edge_index slices
            scores.append((src * dst).sum(dim=-1))
        return torch.cat(scores)

    pos_scores = score_in_batches(pos_val)
    neg_scores = score_in_batches(neg_val)
    val_auc    = compute_auc(pos_scores, neg_scores)

    # --------------------------------------------------
    # Pass 3: node classification loss + acc
    # --------------------------------------------------
    val_loss = val_acc = seen = 0

    for batch_size_cur, n_id, adjs in val_loader:
        adjs    = normalise_adjs(adjs, len(model.convs), device)
        x       = x_all[n_id].to(device)
        y_batch = y_all[n_id[:batch_size_cur]].to(device)

        logits   = model(x, adjs=adjs)
        loss_node = criterion_node(logits, y_batch)

        # Edge loss using cached embeddings for this batch
        local_nid = n_id[:batch_size_cur].to(device)
        src, dst  = pos_val
        mask_pos  = torch.isin(src, local_nid) & torch.isin(dst, local_nid)
        src, dst  = neg_val
        mask_neg  = torch.isin(src, local_nid) & torch.isin(dst, local_nid)

        if mask_pos.sum() > 0 and mask_neg.sum() > 0:
            global_to_local = torch.full((num_nodes_total,), -1, dtype=torch.long, device=device)
            global_to_local[local_nid] = torch.arange(batch_size_cur, device=device)

            pos_batch = global_to_local[pos_val[:, mask_pos]]
            neg_batch = global_to_local[neg_val[:, mask_neg]]

            h_i       = h_cache[local_nid]
            loss_edge = compute_loss(score_edges(h_i, pos_batch), score_edges(h_i, neg_batch))
        else:
            loss_edge = torch.tensor(0.0, device=device)

        loss     = alpha * loss_edge + (1 - alpha) * loss_node
        val_loss += loss.item() * y_batch.size(0)

        preds    = logits.sigmoid() >= 0.5
        y_bin    = y_batch > 0.5
        val_acc += (preds == y_bin).float().mean(dim=-1).sum().item()
        seen    += y_batch.size(0)

    return val_loss / max(seen, 1), val_acc / max(seen, 1), val_auc

@torch.no_grad()
def evaluate(model: nn.Module, 
             h : torch.Tensor,
             y : torch.Tensor,
             loader: DataLoader, 
             criterion_edge : nn.Module,
             criterion_node: nn.Module, 
             pos_val : nn.Tensor,
             neg_val : nn.Tensor,
             device: torch.device,
             alpha = None):
    model.eval()
    total_loss = 0.0
    total_acc = 0
    total = 0
    for batch_size_cur, n_id, adjs in loader:
        if len(model.convs) == 1 : 
            adjs = [adjs.to(device)]
        else : 
            adjs = [(edge_index.to(device), e_id, size) for (edge_index, e_id, size) in adjs]
        x = h[n_id].to(device)
        y_batch = y[n_id[:batch_size_cur]].to(device)

        # Edge logits and loss
        h_i = model.extract_embeddings(x, adjs)  # [B, hidden_dim]

        # Edge scores
        pos_score = criterion_edge(h_i, pos_val)
        neg_score = criterion_edge(h_i, neg_val)

        loss_edge = compute_loss(pos_score, neg_score)

        # Node logits and loss
        logits = model(x, adjs=adjs)                 # [B, num_classes]
        loss_node = criterion_node(logits, y_batch)

        # Combined loss
        loss = (alpha * loss_edge) + ((1-alpha) * loss_node)

        total_loss += loss.item() * y_batch.size(0)
        total_acc += ((logits.sigmoid() >= 0.5) == (y_batch > 0.5 if y_batch.is_floating_point() else y != 0)).float().mean(dim=-1).sum().item()  
        total += y_batch.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = total_acc / max(total, 1)
    return avg_loss, acc

def compute_loss(pos_score, neg_score , device='cuda'):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)