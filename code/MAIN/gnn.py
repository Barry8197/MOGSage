"""
Utilities for node and edge prediction using GNN and PyTorch Geometric

Functions exported
- DotPredictor
- MLPEdgePredictor
- MOGSage
- Encoder
- EarlyStopping
- GraphSageSimple"
"""

import gc
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch_geometric.nn import SAGEConv

import layer_conductance

__all__ = [
    "DotPredictor",
    "MLPEdgePredictor",
    "MOGSage",
    "Encoder",
    "EarlyStopping",
    "GraphSageSimple"
]

from torch_geometric.nn import GCNConv

class GraphSageSimple(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, num_labels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1      = SAGEConv(in_dim, hidden_dim)
        self.conv2      = SAGEConv(hidden_dim, emb_dim)
        self.dropout    = dropout
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        logits = self.classifier(h)
        return h, logits

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.best_state = None
        self.stop       = False

    def step(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

class DotPredictor(nn.Module):
    """
    Dot-product edge scorer for PyG.

    forward(z, edge_index) -> logits [E]
      - z: [num_nodes, hidden_dims] node embeddings
      - edge_index: [2, E] edges (src, dst) to score
    """
    def forward(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)  # logits

class MLPEdgePredictor(nn.Module):
    """
    MLP edge scorer for PyG using concatenation [z_u || z_v].

    forward(z, edge_index) -> logits [E]
      - z: [num_nodes, hidden_dims]
      - edge_index: [2, E]
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.0,
                 apply_sigmoid: bool = False):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        self.apply_sigmoid = apply_sigmoid

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=-1)   # [E, 2*in_channels]
        h = F.relu(self.lin1(h))
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.lin2(h).view(-1)            # [E]
        return torch.sigmoid(logits) if self.apply_sigmoid else logits

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

class MOGSage(nn.Module):
    """
    Multi-Omic Graph Sage

    - update_latent_medians_from_train(feat_train): unchanged semantics.
    
    - extract_embeddings(x, adjs=None, edge_index=None): returns the penultimate GNN embeddings
    (i.e., after the last GraphSAGE layer and before the classifier). With 'adjs' it returns
    embeddings for target nodes only; with 'edge_index' it returns for all nodes.
    
    - feature_importance(loader, device): expects a NeighborSampler-like loader that yields
    (batch_size, n_id, adjs). It computes Captum LayerConductance per-feature at input level,
    accumulating into a [num_nodes, sum(input_dims)] tensor, then returns a pandas.DataFrame.
    
    """
    def __init__(
        self,
        input_dims: Sequence[int],
        encoder_dims: Sequence[int],
        latent_dims: Sequence[int],
        decoder_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        aggregator: str = "mean",
        dropout: float = 0.5,
        enc_dropout: float = 0.5,
        features: Optional[Sequence[str]] = None,   # optional column names
    ):
        super().__init__()
        self.input_dims = list(input_dims)
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.num_layers = len(hidden_dims) + 1  # same convention as original
        self.aggregator = aggregator
        self.features = list(features) if features is not None else None

        # Per-modality encoders (unchanged, assumed to return 'decoder_dim' features)
        self.encoder_dims = nn.ModuleList()
        for modality in range(len(self.input_dims)):
            self.encoder_dims.append(
                Encoder(
                    input_dim=self.input_dims[modality],
                    encoder_dim=encoder_dims[modality],
                    latent_dim=latent_dims[modality],
                    output_dim=decoder_dim,
                    dropout=enc_dropout,
                )
            )

        # GNN stack: SAGEConv + BN + ReLU; final linear classifier
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(self.num_layers):
            if l < self.num_layers - 1:
                in_c = decoder_dim if l == 0 else self.hidden_dims[l - 1]
                out_c = self.hidden_dims[l]
                self.convs.append(SAGEConv(in_channels=in_c, out_channels=out_c, aggr=self.aggregator))
                self.bns.append(nn.BatchNorm1d(out_c))
            else:
                # Final classifier (acts on output of last conv)
                self.classifier = nn.Linear(self.hidden_dims[l - 1], num_classes)

        self.drop = nn.Dropout(dropout)

    # --------- Utilities for cached medians ----------
    def _get_enc_median(self, i: int) -> torch.Tensor:
        return getattr(self, f"_enc_median_{i}")

    @torch.no_grad()
    def update_latent_medians_from_train(self, feat_train: torch.Tensor, device=None):
        """
        Compute per-modality median latent vector on TRAIN features once per epoch and cache it.
        Assumes each modality has at least one valid (non-NaN) row.

        feat_train: [N_train, sum(input_dims)] tensor for TRAIN nodes only (raw input features).
        """
        was_training = self.training
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        prev = 0
        for i, (enc, dim) in enumerate(zip(self.encoder_dims, self.input_dims)):
            h_mod = feat_train[:, prev:prev + dim].to(device)  # [N_train, dim]
            prev += dim

            valid = ~torch.isnan(h_mod).any(dim=1)
            # Only encode non-NaN rows to compute a clean median in latent space:
            enc_valid = enc(h_mod[valid])  # [N_valid, D]
            med = enc_valid.median(dim=0).values.detach()  # [D]

            name = f"_enc_median_{i}"
            if not hasattr(self, name):
                self.register_buffer(name, med.clone())
            else:
                getattr(self, name).copy_(med)

        if was_training:
            self.train()

    # --------- Core: Modality encoding + imputation ----------
    def _encode_modalities_with_cached_median(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, sum(input_dims)] for either all nodes (full graph) or the sampled nodes (neighbor sampling).
        Returns: [N, decoder_dim] mean-pooled across modalities, where per-modality missing feature rows
                 are replaced by the cached median (computed on TRAIN set).
        """
        N = x.size(0)
        chunks = []
        prev = 0
        device = x.device

        for i, (enc, dim) in enumerate(zip(self.encoder_dims, self.input_dims)):
            h_mod = x[:, prev:prev + dim]               # [N, dim]
            prev += dim

            miss = torch.isnan(h_mod).any(dim=1)        # [N]
            # encode only valid rows:
            enc_valid = enc(h_mod[~miss])               # [N_valid, D]
            D = enc_valid.shape[1]

            enc_full = h_mod.new_zeros((N, D))          # same device/dtype
            enc_full[~miss] = enc_valid
            # impute with cached median for this modality:
            enc_full[miss] = self._get_enc_median(i).to(device)

            chunks.append(enc_full)

        # Mean across modalities -> [N, decoder_dim]
        h = torch.stack(chunks, dim=0).mean(dim=0)
        return h

    # --------- Forward ----------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run encoders + GNN + classifier.

        x:
          - [N, sum(input_dims)] for all nodes.

        edge_index:
          - Full-graph edge_index. If provided (and adjs is None), forward returns predictions for all nodes.
        """
        if (edge_index is None):
            raise ValueError("Provide (edge_index).")

        # 1) Encode modalities with cached TRAIN medians for missing rows
        h = self._encode_modalities_with_cached_median(x)

        # 2) GNN propagation
        for l in range(len(self.convs)):
            h = self.convs[l](h, edge_index)
            h = self.bns[l](h)
            h = F.relu(h)

        # 3) Classifier
        logits = self.classifier(h)
        return h, logits

    # --------- Feature importance via Captum LayerConductance ----------
    def feature_importance(
        self,
        loader,     # A NeighborSampler-like loader yielding (batch_size, n_id, adjs)
        device: torch.device,
        feat_all: Optional[torch.Tensor] = None,  # Optional full feature matrix to slice from, else expects loader to supply
        indices: Optional[Sequence[int]] = None,  # Optional subset of nodes to report in final DataFrame
    ) -> pd.DataFrame:
        """
        Calculates input feature importances (per raw feature) using Conductance via Captum.

        Expected loader interface (NeighborSampler-style):
           for batch_size, n_id, adjs in loader:
               - batch_size: number of root (target) nodes
               - n_id: LongTensor of node indices for the union of subgraph nodes
               - adjs: list of (edge_index, e_id, size), where size=(n_src, n_dst) per layer

        Behavior:
           - For each modality 'i', we compute LayerConductance wrt enc.encoder[0] inputs.
           - Missing rows for that modality are imputed as zero importance (same as original code).
           - We accumulate per-node feature importances into a global tensor of shape
             [num_nodes, sum(input_dims)], then build a DataFrame.

        Parameters:
           - device: torch.device.
           - feat_all: Optional full feature matrix [N, sum(input_dims)]. If None, you should feed 'x'
             externally or adapt the loader to supply features (common pattern is to pass feat_all and index by n_id).
           - indices: Optional list/array of node indices to subset the final DataFrame rows (mirrors original).

        Returns:
           A pandas.DataFrame of shape [len(indices or N), sum(input_dims)].
        """
        self.eval()

        # Number of nodes
        if hasattr(loader, "data") and hasattr(loader.data, "num_nodes"):
            num_nodes = int(loader.data.num_nodes)
        elif hasattr(loader, "dataset") and hasattr(loader.dataset, "num_nodes"):
            num_nodes = int(loader.dataset.num_nodes)
        else:
            raise ValueError("Cannot infer number of nodes from loader. Please provide a loader with data.num_nodes.")

        in_dim_total = int(sum(self.input_dims))
        feature_importances = torch.zeros((num_nodes, in_dim_total), device=device)

        # Optional: feature column names
        if self.features is None:
            cols = []
            for i, d in enumerate(self.input_dims):
                cols += [f"mod{i}_f{j}" for j in range(d)]
        else:
            cols = list(self.features)

        # Offsets to slice modality features
        modality_offsets = np.cumsum([0] + list(self.input_dims))

        # Iterate modalities
        prev_dim = 0
        for i_mod, (enc, dim) in enumerate(zip(self.encoder_dims, self.input_dims)):
            # Captum Conductance wrt the first encoder layer's input
            cond = LayerConductance(self, enc.encoder[0])

            # Go over sampled mini-batches
            for batch in loader:
                # NeighborSampler yields (batch_size, n_id, adjs)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    batch_size, n_id, adjs = batch
                else:
                    raise ValueError(
                        "feature_importance expects a NeighborSampler-like loader yielding (batch_size, n_id, adjs)."
                    )

                # Prepare inputs
                n_id = n_id.to(device)
                adjs = [(edge_index.to(device), e_id, size) for (edge_index, e_id, size) in adjs]

                if feat_all is None:
                    raise ValueError("Please provide feat_all (full feature matrix) for feature_importance.")
                x = feat_all[n_id].to(device)  # features for ALL nodes in the sampled subgraph

                # Model predictions on current batch (target nodes only)
                with torch.no_grad():
                    logits = self.forward(x, adjs=adjs)  # [batch_size, num_classes]
                    y = logits.argmax(dim=1)            # [batch_size]

                # Identify rows with missing modality 'i_mod'
                slc = x[:, prev_dim:prev_dim + dim]         # [n_subgraph_nodes, dim] in subgraph order
                nan_rows = torch.isnan(slc).any(dim=1)      # marks missing rows across subgraph nodes

                # Compute conductance per target class separately (as in original)
                # For efficiency, you can also compute once over all classes and pick per-row,
                # but here we mirror the original approach.
                unique_targets = y.unique().tolist()

                # Pre-allocate cond_imputed for subgraph nodes (will fill per target class)
                # We'll only use the first 'batch_size' rows (targets) to write back.
                for target_class in unique_targets:
                    with torch.no_grad():
                        cond_attr = cond.attribute(
                            x,
                            target=int(target_class),
                            additional_forward_args=(adjs,),
                            internal_batch_size=128,
                            attribute_to_layer_input=True,
                            n_steps=10,
                        )
                        # cond_attr has shape [n_subgraph_nodes, D_layer_input] where
                        # D_layer_input == enc.encoder[0] input size == decoder path input for this modality.
                        # We impute zeros for missing rows of THIS modality:
                        cond_imputed = cond_attr.clone()
                        cond_imputed[nan_rows] = 0.0

                    # Keep only target (root) nodes in this mini-batch
                    cond_out = cond_imputed[:batch_size]    # [batch_size, dim0_of_layer] - NOTE: matches layer input
                    # We want to assign this conductance back to raw feature slice [prev_dim: prev_dim+dim].
                    # Since we computed attribution at the first layer input of the modality encoder
                    # (enc.encoder[0]), that input aligns with the raw modality features of size 'dim'.
                    # If your enc.encoder[0] layer changes dimension (e.g., Linear dim->hid), ensure
                    # attribute_to_layer_input=True so the attribution is w.r.t. its INPUT (size 'dim').

                    # Map target nodes back to global node indices:
                    out_nodes = n_id[:batch_size]  # global node IDs for root nodes
                    mask = (y == int(target_class))  # which root nodes had this predicted class
                    # Write into global importance tensor:
                    feature_importances[out_nodes[mask], prev_dim:prev_dim + dim] = cond_out[mask, :dim]

                    del cond_attr, cond_imputed, cond_out
                    gc.collect()
                    torch.cuda.empty_cache()

            prev_dim += dim

            del cond
            gc.collect()
            torch.cuda.empty_cache()

        # Optional row subsetting, like original (test_dataloader.indices)
        if indices is not None:
            idx = np.asarray(indices)
        else:
            idx = np.arange(num_nodes)

        # Build DataFrame
        df = pd.DataFrame(
            feature_importances[idx].detach().cpu().numpy(),
            index=idx,
            columns=cols,
        )
        self.feature_importances = df
        return df
