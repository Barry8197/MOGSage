"""
Utilities for edge prediction using GNN and Deep Graph Library

Functions exported
- keep_node_attrs_inplace
- train_node2vec_from_nx
- nx_to_pyg_data
- GraphSAGE
- MOGSage
- MLPEdgePredictor
- DotPredictor
- compute_loss
- compute_auc
- make_link_pred_split
- merge_dfs
- expand_col
- evaluate
- training_metric_plots
"""

import os
import gc
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from typing import Dict, Optional

import seaborn as sns
import matplotlib.pyplot as plt
import palettable.wesanderson as wes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SparseAdam

import dgl
import dgl.function as fn
from dgl.nn import SAGEConv
from dgl.dataloading import DataLoader

from sklearn.metrics import roc_auc_score

# Explicit import to avoid accidental shadowing by local files/modules named "node2vec"
from torch_geometric.nn.models.node2vec import Node2Vec as PyGNode2Vec
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce, from_networkx

import sys
sys.path.insert(0 , '.')
import layer_conductance

__all__ = [
    "keep_node_attrs_inplace",
    "train_node2vec_from_nx",
    "nx_to_pyg_data",
    "GraphSAGE",
    "MOGSage",
    "MLPEdgePredictor",
    "DotPredictor",
    "compute_loss",
    "compute_auc",
    "make_link_pred_split",
    "merge_dfs",
    "expand_col",
    "evaluate",
    "training_metric_plots"
]

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
def evaluate(model: nn.Module, 
             loader: DataLoader, 
             criterion_edge : nn.Module,
             criterion_node: nn.Module, 
             device: torch.device,
             alpha = None):
    model.eval()
    total_loss = 0.0
    total_acc = 0
    total = 0
    for it, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(loader):
        blocks = [b.to(device) for b in blocks]
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)
        
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]

        # Edge scores
        h_i = model.extract_embeddings(x, blocks)
        pos_score = criterion_edge(pos_graph, h_i)
        neg_score = criterion_edge(neg_graph, h_i)
        loss_edge = compute_loss(pos_score, neg_score)

        # Node logits
        logits = model(x, blocks)
        loss_node = criterion_node(logits, y)

        loss = (alpha * loss_edge) + ((1-alpha) * loss_node)

        total_loss += loss.item() * y.size(0)
        total_acc += ((logits.sigmoid() >= 0.5) == (y > 0.5 if y.is_floating_point() else y != 0)).float().mean(dim=-1).sum().item()  
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = total_acc / max(total, 1)
    return avg_loss, acc

def expand_col(
    df: pd.DataFrame,
    col: str = "phenotype",
    lowest_level: str = "none",
    positive_levels=("osteoporosis + osteopenia", "osteopenia"),
    drop_original: bool = True,
    dtype="int8",
) -> pd.DataFrame:
    """
    Convert a single phenotype column into binary indicator columns for specified phenotypes,
    fill NA with the lowest level, and optionally drop the original column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str
        Name of the phenotype column.
    lowest_level : str
        Label to use as the lowest level (also used to fill NA).
    positive_levels : tuple[str]
        Phenotype labels to one-hot encode (columns to create). Typically the non-lowest levels.
    drop_original : bool
        Whether to drop the original phenotype column.
    dtype : str or numpy dtype
        Integer dtype for the indicator columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new indicator columns (and without the original column if drop_original=True).
    """
    out = df.copy()

    # Fill NA with the lowest level
    ser = out[col].fillna(lowest_level).astype(str)

    # Normalise text for robust matching (trim, collapse spaces, casefold)
    norm = (
        ser.str.strip()
           .str.replace(r"\s+", " ", regex=True)
           .str.casefold()
    )

    def norm_str(s: str) -> str:
        return " ".join(str(s).split()).casefold()

    # Create indicator columns for each requested positive level
    for lvl in positive_levels:
        out[lvl] = (norm == norm_str(lvl)).astype(dtype)

    if drop_original:
        out = out.drop(columns=[col])

    return out


# Define the merge operation setup
def merge_dfs(left_df, right_df):
    """
    Merges two DataFrames on their indexes with an outer join method.

    Parameters:
        left_df (pd.DataFrame): The left DataFrame to merge.
        right_df (pd.DataFrame): The right DataFrame to merge.

    Returns:
        pd.DataFrame: The resulting DataFrame after merging.
    """    
    # Merging on 'key' and expanding with 'how=outer' to include all records
    return pd.merge(left_df, right_df, left_index=True, right_index=True, how='outer')


def make_link_pred_split(
    g: dgl.DGLGraph,
    test_ratio: float = 0.1,
    negative_ratio: float = 1.0,
    device: Optional[torch.device | str] = None,
    seed: Optional[int] = None,
) -> Dict[str, dgl.DGLGraph]:
    """
    Create random train/test splits for link prediction on a homogeneous DGLGraph.

    What it returns:
      - train_g: graph used for message passing with test edges removed
      - test_g: same as train_g (to avoid information leakage during testing)
      - train_pos_g: graph consisting only of positive training edges
      - train_neg_g: graph consisting only of negative training edges
      - test_pos_g: graph consisting only of positive test edges
      - test_neg_g: graph consisting only of negative test edges

    Notes:
      - Negative edges are sampled from the complement of the adjacency (no self-loops).
      - For very large graphs (many nodes), building a dense complement can be memory-heavy.
        Consider using a streaming negative sampler instead (e.g., DGL's UniformNegativeSampler).

    Args:
      g: Homogeneous DGLGraph (single node/edge type).
      test_ratio: Fraction of edges to hold out for testing (e.g., 0.1 for 10%).
      negative_ratio: Number of negatives per positive (e.g., 1.0 => equal counts).
      device: Target device for returned graphs (e.g., "cpu", "cuda", torch.device(...)).
      seed: Optional RNG seed for reproducibility.

    Returns:
      A dict with keys:
        {"train_g", "test_g", "train_pos_g", "train_neg_g", "test_pos_g", "test_neg_g"}.
    """
    if g.is_homogeneous is False:
        raise ValueError("This helper expects a homogeneous DGLGraph.")

    # Ensure reproducibility if seed is provided.
    if seed is not None:
        np.random.seed(seed)

    # Number of nodes/edges
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    # Get all edges (as torch tensors). Move to CPU for numpy/scipy operations.
    # u, v have shape [num_edges]
    u, v = g.edges()
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()

    # 1) Randomly split positive edges into train/test by shuffling edge IDs.
    eids = np.arange(num_edges)
    eids = np.random.permutation(eids)
    test_size = int(round(num_edges * test_ratio))
    test_eids = eids[:test_size]
    train_eids = eids[test_size:]

    # Select endpoints for train/test positives
    train_pos_u = u_np[train_eids]
    train_pos_v = v_np[train_eids]
    test_pos_u = u_np[test_eids]
    test_pos_v = v_np[test_eids]

    # 2) Build complement adjacency to sample negatives (no self-loops).
    #    Note: For very large graphs, this dense operation can be memory heavy.
    adj = sp.coo_matrix(
        (np.ones(num_edges, dtype=np.int8), (u_np, v_np)),
        shape=(num_nodes, num_nodes),
    )
    # Complement mask: 1 for non-edges, 0 for existing edges or diagonal
    dense_adj = adj.toarray().astype(np.int8)
    neg_mask = (1 - dense_adj)  # 1 where no edge exists
    # Remove self-loops from negative candidates
    np.fill_diagonal(neg_mask, 0)

    # Candidate negative pairs (u, v)
    neg_u_candidates, neg_v_candidates = np.where(neg_mask != 0)
    num_candidates = len(neg_u_candidates)

    # Determine how many negatives we need for train/test
    num_train_pos = len(train_eids)
    num_test_pos = len(test_eids)
    num_train_neg = int(round(num_train_pos * negative_ratio))
    num_test_neg = int(round(num_test_pos * negative_ratio))
    total_needed = num_train_neg + num_test_neg

    # Sample negatives (prefer without replacement when possible)
    if total_needed <= num_candidates:
        chosen_idx = np.random.choice(num_candidates, size=total_needed, replace=False)
    else:
        # Not enough unique negatives; fall back to sampling with replacement.
        chosen_idx = np.random.choice(num_candidates, size=total_needed, replace=True)

    # Split chosen negative indices into train/test portions
    test_neg_idx = chosen_idx[:num_test_neg]
    train_neg_idx = chosen_idx[num_test_neg:]

    train_neg_u = neg_u_candidates[train_neg_idx]
    train_neg_v = neg_v_candidates[train_neg_idx]
    test_neg_u = neg_u_candidates[test_neg_idx]
    test_neg_v = neg_v_candidates[test_neg_idx]

    # 3) Build graphs
    # Train graph: remove test edges to avoid leakage
    # dgl.remove_edges returns a new graph (non-inplace).
    test_eids_t = torch.from_numpy(test_eids).long()
    train_g = dgl.remove_edges(g, test_eids_t)

    # In link prediction we usually use the same edge-removed graph at test time
    # (message passing should not include the held-out test edges).
    test_g = train_g

    # Helper to build small subgraphs on CPU then move to device if requested.
    def edge_graph(src_np, dst_np, num_nodes, device):
        sg = dgl.graph(
            (torch.from_numpy(src_np).long(), torch.from_numpy(dst_np).long()),
            num_nodes=num_nodes,
        )
        return sg.to(device) if device is not None else sg

    train_pos_g = edge_graph(train_pos_u, train_pos_v, num_nodes, device)
    train_neg_g = edge_graph(train_neg_u, train_neg_v, num_nodes, device)
    test_pos_g = edge_graph(test_pos_u, test_pos_v, num_nodes, device)
    test_neg_g = edge_graph(test_neg_u, test_neg_v, num_nodes, device)

    # Move the structural graphs to the desired device as well
    if device is not None:
        train_g = train_g.to(device)
        test_g = test_g.to(device)

    return {
        "train_g": train_g,
        "test_g": test_g,
        "train_pos_g": train_pos_g,
        "train_neg_g": train_neg_g,
        "test_pos_g": test_pos_g,
        "test_neg_g": test_neg_g,
    }


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

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]

class MLPEdgePredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]

class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        aggregator: str = "mean",
        dropout: float = 0.5,
        norm: str | None = "batch",  # "batch", "layer", or None
        input_dropout: float = 0.5,  # optional dropout on input features
        residual: bool = False,      # optional residual connection on hidden layer
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator)
        self.conv2 = SAGEConv(hidden_feats, out_feats, aggregator)

        # Normalization on hidden layer
        if norm == "batch":
            self.norm1 = nn.BatchNorm1d(hidden_feats)
        elif norm == "layer":
            self.norm1 = nn.LayerNorm(hidden_feats)
        else:
            self.norm1 = None

        self.dropout = nn.Dropout(dropout)
        self.in_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.residual = residual and (in_feats == hidden_feats)  # ensure sizes match

    def forward(self, g, x):
        x = self.in_dropout(x)

        h = self.conv1(g, x)
        if self.norm1 is not None:
            h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        if self.residual:
            h = h + x  # only if hidden_feats == in_feats

        h = self.conv2(g, h)  # typically the output layer has no norm/dropout
        return h

class MOGSage(nn.Module):
    def __init__(self, input_dims, encoder_dims, latent_dims , decoder_dim, hidden_feats, num_classes, aggregator: str = "mean", dropout=0.5 , enc_dropout = 0.5) :        
        super().__init__()
        
        self.encoder_dims = nn.ModuleList()
        self.gnnlayers = nn.ModuleList()
        self.nnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = len(hidden_feats) + 1
        self.input_dims = input_dims
        self.hidden_feats = hidden_feats
        self.num_classes = num_classes

        # Encoder reduced dim input and pooling scheme
        for modality in range(len(input_dims)):  # excluding the input layer
            self.encoder_dims.append(Encoder(input_dims[modality], encoder_dims[modality] , latent_dims[modality] , decoder_dim , dropout=enc_dropout))
        
        # Combined Neural Network layers
        for layers in range(self.num_layers) :
            if layers < self.num_layers -1 :
                if layers == 0 : 
                    self.gnnlayers.append(
                        SAGEConv(decoder_dim, hidden_feats[layers], aggregator)
                    )
                else :
                    self.gnnlayers.append(
                        SAGEConv(hidden_feats[layers-1], hidden_feats[layers],aggregator)
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]))
            else : 
                self.nnlayers.append(
                    nn.Linear(hidden_feats[layers-1], num_classes)
                )
                
        self.drop = nn.Dropout(dropout)

    def _get_enc_median(self, i: int) -> torch.Tensor:
        return getattr(self, f"_enc_median_{i}")

    @torch.no_grad()
    def update_latent_medians_from_train(self, feat_train: torch.Tensor, device=None):
        """
        Compute per-modality median latent vector on TRAIN features once per epoch and cache it.
        Assumes each modality has at least one valid (non-NaN) row.
        feat_train: [N_train, sum(input_dims)] tensor for TRAIN nodes only.
        """
        was_training = self.training
        self.eval()
    
        if device is None:
            device = next(self.parameters()).device
    
        prev = 0
        for i, (Encoder, dim) in enumerate(zip(self.encoder_dims, self.input_dims)):
            h_mod = feat_train[:, prev:prev + dim].to(device)     # [N_train, dim]
            prev += dim
    
            valid = ~torch.isnan(h_mod).any(dim=1)                # [N_train]
            enc_valid = Encoder(h_mod[valid])                     # [N_valid, D]
            med = enc_valid.median(dim=0).values.detach()         # [D]
    
            name = f"_enc_median_{i}"
            if not hasattr(self, name):
                self.register_buffer(name, med.clone())
            else:
                getattr(self, name).copy_(med)
    
        if was_training:
            self.train()

    def forward(self, h, g):
        x = []
        prev = 0
        N = h.size(0)
    
        for i, (Encoder, dim) in enumerate(zip(self.encoder_dims, self.input_dims)):
            h_mod = h[:, prev:prev + dim]                         # [N, dim]
            prev += dim
    
            miss = torch.isnan(h_mod).any(dim=1)                  # [N]
            enc_valid = Encoder(h_mod[~miss])                     # [N_valid, D]
            D = enc_valid.shape[1]
    
            enc_full = h_mod.new_zeros((N, D))
            enc_full[~miss] = enc_valid
            enc_full[miss]  = getattr(self, f"_enc_median_{i}").to(h.device)
    
            x.append(enc_full)
    
        # Mean-pool across modalities
        x = torch.stack(x, dim=0).mean(dim=0)
        
        # Apply nn layers sequentially
        for l , (layer , g_layer) in enumerate(zip(self.gnnlayers , g)) :             
            x = layer(g_layer, x)
            x = self.batch_norms[l](x)
            x = F.relu(x)

        x = self.nnlayers[0](x)
            
        return x

    def extract_embeddings(self, h, g) : 
        """
        
        """
        prev_dim = 0
        x = []
        
        h = self.drop(h)

        # Process each modality through its respective encoder
        for i , (Encoder , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 

            # Encode data with no missing values in each modality
            n = h.shape[0]
            nan_rows = torch.isnan(h[: , prev_dim:dim+prev_dim]).any(dim=1).detach()
            x.append(Encoder(h[: , prev_dim:dim+prev_dim][~nan_rows]))

            # Impute missing values with median of the encoded features
            imputed_idx = torch.where(nan_rows)[0].cpu().detach()
            reindex = list(range(n))
            for imp_idx in imputed_idx :
                reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                del reindex[-1]

            # Concatenate the encoded features with the imputed values
            x[i] = torch.concat([x[i] , torch.median(x[i], dim=0).values.repeat(n).reshape(n , x[i].shape[1])])[reindex]
            
            # Update the previous dimension for the next modality
            prev_dim += dim

        # Stack the encoded features from all modalities and mean pool them
        x = torch.stack(x , dim = 0)
        x = torch.sum(x , dim = 0)/(i+1)
        
        # Apply nn layers sequentially
        for l , (layer , g_layer) in enumerate(zip(self.gnnlayers , g)) :             
            x = layer(g_layer, x)
            x = self.batch_norms[l](x)
            x = F.relu(x)
            
        return x   

    def feature_importance(self, test_dataloader , device):
        """
        Calculate feature importances using the Conductance algorithm through Captum.

        Args:
            test_dataset (torch.Tensor): The dataset for which to calculate importances.
            target_class (int): The target class index for which to calculate importances.

        Returns:
            pd.DataFrame: A dataframe containing the feature importances.
        """
        print('Feature Level Importance')
        feature_importances = torch.zeros((max(test_dataloader.indices)+1, np.sum(self.input_dims)) , device=device)

        prev_dim = 0
        for i , (enc , dim) in enumerate(zip(self.encoder_dims , self.input_dims)) : 
            cond = layer_conductance.LayerConductance(self, enc.encoder[0])

            for it, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(test_dataloader) : 
                output_nodes = blocks[0].dstdata['_ID']
                blocks = [b.to(device) for b in blocks]
                feature_importances_batch = feature_importances[output_nodes - 1]
                
                x = blocks[0].srcdata["feat"]
                y = self.forward(x , blocks).max(dim=1)[1]
                
                n = x.shape[0]
                nan_rows = torch.isnan(x[: , prev_dim:prev_dim + dim]).any(dim=1)
                    
                for target_class in y.unique() :
                    with torch.no_grad() : 
                        conductance = cond.attribute(x, target=target_class, additional_forward_args=blocks, internal_batch_size =128 , attribute_to_layer_input=True,n_steps=10)
                        
                    imputed_idx = torch.where(nan_rows)[0]
                    reindex = list(range(n))
                    for imp_idx in imputed_idx :
                        reindex.insert(imp_idx, reindex[-1])  # Insert the last index at the desired position
                        del reindex[-1]
    
                    cond_imputed = torch.concat([conductance , torch.zeros(n , conductance.shape[1], device=device)])[reindex]
                        
                    cond_output = cond_imputed[torch.isin(input_nodes , output_nodes)]

                    feature_importances_batch[y == target_class, prev_dim:prev_dim + dim] = cond_output[y == target_class]
                    del conductance
                    gc.collect()
                    torch.cuda.empty_cache()

                feature_importances[output_nodes] = feature_importances_batch
                    
            prev_dim += dim

        del cond
        gc.collect()
        torch.cuda.empty_cache()
        data_index = getattr(enc, 'data_index', np.arange(len(test_dataloader.indices)))

        feature_importances = feature_importances[test_dataloader.indices]

        feature_importances = pd.DataFrame(feature_importances.detach().cpu().numpy(),
                                           index=data_index,
                                           columns=self.features)
        
        self.feature_importances = feature_importances
        
        return self.feature_importances

# -----------------------------------------------------------------------------
# Optional: make runs more deterministic
# -----------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.backends.cuda.matmul.allow_tf32 = True  # speed-up on Ampere+

# -----------------------------------------------------------------------------
# Conversions and preprocessing
# -----------------------------------------------------------------------------
def nx_to_pyg_data(
    G: nx.Graph,
    node_attr_keys=None,
    edge_attr_keys=None,
):
    """
    Converts a NetworkX graph (with optional node/edge attributes) to a PyG Data object.

    node_attr_keys / edge_attr_keys:
      - None: do not attempt to pack attributes (safest for heterogeneous attrs)
      - list[str]: group/stack listed attributes (requires consistent shapes/dtypes)

    Returns:
      data: torch_geometric.data.Data with .edge_index (and optional attributes)
      node_id_map: dict mapping original NetworkX node -> integer ID (0..N-1)
    """
    # Ensure deterministic node ordering and build mapping
    nodes = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes)}

    # Relabel graph to 0..N-1
    G_int = nx.relabel_nodes(G, node_id_map, copy=True)

    # Convert to PyG Data, optionally packing attributes:
    if (node_attr_keys is None) and (edge_attr_keys is None):
        data = from_networkx(G_int)  # will include attributes as individual tensors/fields if consistent
    else:
        data = from_networkx(
            G_int,
            group_node_attrs=node_attr_keys,
            group_edge_attrs=edge_attr_keys,
        )

    # Ensure edge_index exists
    if not hasattr(data, "edge_index") or data.edge_index is None:
        # Build edge_index from NetworkX edges
        src, dst = zip(*G_int.edges())
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        data.edge_index = edge_index

    data.num_nodes = G_int.number_of_nodes()
    return data, node_id_map

def preprocess_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Clean, undirected, deduplicated edge_index (2, E).
    Matches your original pre-processing.
    """
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index

def train_node2vec_from_nx(
    G: nx.Graph,
    device: str = None,
    emb_dim: int = 128,
    p: float = 1.0,
    q: float = 1.0,
    walk_length: int = 80,
    window_size: int = 10,          # context_size in PyG
    num_walks_per_node: int = 10,   # walks_per_node in PyG
    neg_k: int = 5,                 # num_negative_samples in PyG
    batch_nodes: int = 1024,
    epochs: int = 5,
    lr: float = 0.01,
    sparse: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    average_in_out: bool = False,   # average input & context embeddings at the end
    max_steps_per_epoch: int = None # optional: cap steps per epoch to limit work
):
    """
    Trains Node2Vec on a NetworkX graph using PyTorch Geometric and returns the learned embeddings.

    Returns:
      - node_emb: [N, emb_dim] CPU tensor of node embeddings (aligned to integer IDs 0..N-1)
      - n2v: the trained PyG Node2Vec model
      - node_id_map: dict mapping original NetworkX node -> integer ID
    """
    # Device setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_cuda = device.startswith('cuda')

    # Convert NX -> PyG Data
    data, node_id_map = nx_to_pyg_data(G)
    num_nodes = data.num_nodes
    edge_index = preprocess_edge_index(data.edge_index, num_nodes=num_nodes)

    # Try to keep random-walk work on GPU if available; fall back to CPU if CUDA ops are missing.
    # Some builds of torch-cluster may not have CUDA random-walk kernels.
    rw_edge_index = edge_index
    try:
        if use_cuda:
            rw_edge_index = edge_index.to(device)
    except Exception:
        rw_edge_index = edge_index.cpu()

    # Build the Node2Vec model (explicit class import avoids "model is a string" issues)
    # Important: name the variable something other than "Node2Vec" or "node2vec" to avoid shadowing.
    try:
        n2v = PyGNode2Vec(
            edge_index=rw_edge_index,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=window_size,
            walks_per_node=num_walks_per_node,
            num_negative_samples=neg_k,
            p=p,
            q=q,
            sparse=sparse,
            num_nodes=num_nodes,
        ).to(device)
    except RuntimeError as e:
        # If putting edge_index on GPU failed (e.g., CUDA RW not available), retry with CPU edge_index
        print(f"Falling back to CPU random walks: {e}")
        n2v = PyGNode2Vec(
            edge_index=edge_index.cpu(),
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=window_size,
            walks_per_node=num_walks_per_node,
            num_negative_samples=neg_k,
            p=p,
            q=q,
            sparse=sparse,
            num_nodes=num_nodes,
        ).to(device)

    # Optimizer
    optimizer = SparseAdam(n2v.parameters(), lr=lr) if sparse else Adam(n2v.parameters(), lr=lr)

    # Random-walk pair loader
    loader = n2v.loader(
        batch_size=batch_nodes,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(pin_memory and use_cuda),
    )

    # Training loop
    n2v.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0

        for step, (pos_rw, neg_rw) in enumerate(loader):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break

            pos_rw = pos_rw.to(device, non_blocking=True)
            neg_rw = neg_rw.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = n2v.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            steps += 1

        avg = total_loss / max(1, steps)
        print(f"Epoch {epoch}/{epochs} | loss {avg:.4f}")

    # Final embeddings
    n2v.eval()
    with torch.no_grad():
        if average_in_out:
            node_emb = (n2v.embedding.weight + n2v.context_embedding.weight) / 2.0
        else:
            node_emb = n2v.embedding.weight
        node_emb = node_emb.detach().cpu().clone()

    return node_emb, n2v, node_id_map

def keep_node_attrs_inplace(G: nx.Graph, keep_keys):
    keep = set(keep_keys)
    for n, d in G.nodes(data=True):
        for k in list(d.keys()):
            if k not in keep:
                del d[k]
    return G


class Encoder(nn.Module):
    """
    Implements a simple encoder-decoder architecture using feed-forward neural networks.
    This module is structured with two linear layers for encoding, each followed by dropout
    and batch normalization. The decoding step is performed by a single linear transformation.

    Attributes:
        encoder (nn.ModuleList): A list of linear layers for the encoding part. It contains two
                                 linear transformations that progressively reduce the dimension
                                 from `input_dim` to `encoder_dim` and then to `latent_dim`.
        norm (nn.ModuleList): A list of batch normalization layers corresponding to each
                              encoder layer. These layers are used to stabilize and accelerate
                              the training process by normalizing the outputs of the linear layers.
        decoder (torch.nn.Sequential): A sequential container that holds the decoder module.
                                       It consists only of a single linear layer transforming
                                       the latent representation back to the output space with
                                       dimension `output_dim`.
        drop (nn.Dropout): A dropout layer applied after each encoding layer to prevent 
                           overfitting by randomly setting a fraction of input units to 0 
                           at each update during training time.

    Args:
        input_dim (int): The number of features in the input data.
        encoder_dim (int): The size of the first encoding layer, which defines the number
                           of neurons that produce the intermediate representation from the
                           input features.
        latent_dim (int): The size of the second encoding layer and the dimensionality of
                          the latent space where data is further compressed.
        output_dim (int): The size of the output layer, which denotes the number of features
                          in the reconstructed output from the latent representation.
        dropout (float): The dropout rate that defines the probability of setting a neuron 
                         to zero during training.
    
    Methods:
        forward(x):
            Defines the computation performed at every call of the encoder-decoder model.
            It takes an input tensor `x`, applies sequential encoding with dropout and 
            normalization, then decodes back to the target dimension.

            Parameters:
                x (Tensor): The input data tensor.

            Returns:
                Tensor: The decoded output tensor.
    """
    def __init__(self , input_dim, encoder_dim , latent_dim , output_dim , dropout):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.norm   = nn.ModuleList()
    
        self.encoder.extend([
            nn.Linear(input_dim , encoder_dim), 
            nn.Linear(encoder_dim, latent_dim)
        ])
        
        self.norm.extend([
            nn.BatchNorm1d(encoder_dim),
            nn.BatchNorm1d(latent_dim),
            nn.BatchNorm1d(output_dim)
        ])
        
        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_dim, output_dim),
        )
        
        self.drop = nn.Dropout(dropout) 

    def forward(self, x):
        encoded = x
        for layer in range(2) : 
            encoded = self.encoder[layer](encoded)
            encoded = self.drop(encoded)
            encoded = self.norm[layer](encoded)
            
        decoded = self.decoder(encoded)
        
        return self.norm[2](decoded)