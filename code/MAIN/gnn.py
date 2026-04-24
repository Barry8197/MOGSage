#!/usr/bin/env python3
"""
Utilities for node and edge prediction using GNN and PyTorch Geometric

Functions exported
- DotPredictor
- MLPEdgePredictor
- MOGSage
- Encoder
- EarlyStopping
- GraphSageSimple
- LayerConductance
"""

import gc
import typing
from typing import List, Optional, Sequence, Tuple, Any, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch_geometric.nn import SAGEConv

from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
)
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.typing import BaselineType, Literal, TargetType
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
from captum.log import log_usage

__all__ = [
    "DotPredictor",
    "MLPEdgePredictor",
    "MOGSage",
    "Encoder",
    "EarlyStopping",
    "GraphSageSimple",
    "LayerConductance"
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

        for i in range(len(self.input_dims)):
            self.register_buffer(f"_enc_median_{i}", torch.zeros(decoder_dim))

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
        loader,                      # PyG NeighborLoader yielding sampled Data objects
        device: torch.device,
        feat_all: torch.Tensor | None = None,   # optional full feature matrix; if None uses batch.x
        indices=None,
        target_classes: torch.Tensor | None = None,
        target_label: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculates per-raw-feature importances using Captum LayerConductance.
    
        Compatible with torch_geometric.loader.NeighborLoader, which yields
        sampled Data objects with fields such as:
            - batch.x
            - batch.edge_index
            - batch.y
            - batch.n_id
            - batch.batch_size
    
        Assumes:
            - self.forward(x, edge_index) returns (h, logits)
            - attribution is computed wrt enc.encoder[0] input
            - only root nodes (first batch_size rows) are written back
    
        Parameters
        ----------
        loader : NeighborLoader
            PyG NeighborLoader over the graph.
        device : torch.device
            Device to run attribution on.
        feat_all : torch.Tensor | None
            Optional full feature matrix [N, sum(input_dims)].
            If provided, uses feat_all[batch.n_id].
            If None, uses batch.x directly.
        indices : optional sequence of ints
            Optional subset of node indices for final DataFrame rows.
    
        Returns
        -------
        pd.DataFrame
            DataFrame of shape [len(indices or N), sum(input_dims)] with
            per-node, per-feature importances.
        """
        self.eval()
    
        # Infer number of nodes
        if hasattr(loader, "data") and hasattr(loader.data, "num_nodes"):
            num_nodes = int(loader.data.num_nodes)
        elif hasattr(loader, "dataset") and hasattr(loader.dataset, "num_nodes"):
            num_nodes = int(loader.dataset.num_nodes)
        else:
            raise ValueError("Cannot infer number of nodes from loader. Please provide a loader with data.num_nodes.")
    
        in_dim_total = int(sum(self.input_dims))
        feature_importances = torch.zeros((num_nodes, in_dim_total), device=device)
    
        # Column names
        if self.features is None:
            cols = []
            for i, d in enumerate(self.input_dims):
                cols += [f"mod{i}_f{j}" for j in range(d)]
        else:
            cols = list(self.features)
    
        # Wrapper for Captum: return logits only
        def forward_for_captum(x, edge_index):
            _, logits = self.forward(x, edge_index=edge_index[:2])
            return logits
    
        prev_dim = 0
    
        for i_mod, (enc, dim) in enumerate(zip(self.encoder_dims, self.input_dims)):
            cond = LayerConductance(forward_for_captum, enc.encoder[0])
    
            for batch in loader:
                batch = batch.to(device)
    
                if not hasattr(batch, "batch_size"):
                    raise ValueError("NeighborLoader batch is missing 'batch_size'.")
                if not hasattr(batch, "n_id"):
                    raise ValueError("NeighborLoader batch is missing 'n_id'.")
                if not hasattr(batch, "edge_index"):
                    raise ValueError("NeighborLoader batch is missing 'edge_index'.")
    
                batch_size = int(batch.batch_size)
                n_id = batch.n_id
                edge_index = batch.edge_index
    
                # Features for all sampled nodes in this subgraph
                if feat_all is not None:
                    x = feat_all[n_id].to(device)
                else:
                    if not hasattr(batch, "x"):
                        raise ValueError("Batch has no x and feat_all was not provided.")
                    x = batch.x
    
                # Predict classes for root nodes only
                if target_classes is not None : 
                    y_pred = target_classes.to(device)[n_id[:batch_size]]
                elif target_label is not None : 
                    with torch.no_grad():
                        _, logits = self.forward(x, edge_index=edge_index)   # logits: [n_subgraph_nodes, num_classes]
                        y_pred = (torch.sigmoid(logits[:batch_size]) >= 0.5).to(torch.int32)[: , target_label]  # only root nodes
                else : 
                    raise ValueError("One of target_classes or target_label must be not None")
    
                # Current modality slice
                slc = x[:, prev_dim:prev_dim + dim]                      # [n_subgraph_nodes, dim]
                nan_rows = torch.isnan(slc).any(dim=1)                   # rows missing this modality
    
                unique_targets = y_pred.unique().tolist()

                for target_class in unique_targets:
                    cond_attr = cond.attribute(
                        inputs=x,
                        target=int(target_class),
                        additional_forward_args=edge_index,
                        internal_batch_size=min(128, x.size(0)),
                        attribute_to_layer_input=True,
                        n_steps=10,
                    )
                    # cond_attr shape should align with encoder layer input, i.e. raw modality dim
                    cond_attr = cond_attr.clone()
                    cond_attr[nan_rows] = 0.0
    
                    # keep only root nodes
                    cond_out = cond_attr[:batch_size, :dim]
    
                    out_nodes = n_id[:batch_size]                        # global IDs of root nodes
                    mask = (y_pred == int(target_class))
    
                    feature_importances[out_nodes[mask], prev_dim:prev_dim + dim] = cond_out[mask]
    
                    del cond_attr, cond_out
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
            prev_dim += dim
    
            del cond
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        if indices is not None:
            idx = np.asarray(indices)
        else:
            idx = np.arange(num_nodes)
    
        df = pd.DataFrame(
            feature_importances[idx].detach().cpu().numpy(),
            index=idx,
            columns=cols,
        )
        self.feature_importances = df
        return df

class LayerConductance(LayerAttribution, GradientAttribution):
    r"""
    Computes conductance with respect to the given layer. The
    returned output is in the shape of the layer's output, showing the total
    conductance of each hidden layer neuron.

    The details of the approach can be found here:
    https://arxiv.org/abs/1805.12233
    https://arxiv.org/abs/1807.09946

    Note that this provides the total conductance of each neuron in the
    layer's output. To obtain the breakdown of a neuron's conductance by input
    features, utilize NeuronConductance instead, and provide the target
    neuron index.
    """

    def __init__(
        self,
        forward_func: Callable,
        layer: nn.Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's input or
                          output dimensions, depending on whether we attribute to
                          the inputs or outputs of the layer, corresponding to
                          attribution of each neuron in the input or output of
                          this layer.
            device_ids (list[int]): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)

    def has_convergence_delta(self) -> bool:
        return True

    @typing.overload
    def attribute(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
        attribute_to_layer_input: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        ...

    @log_usage()
    def attribute(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        baselines: Union[
            None, int, float, torch.Tensor, Tuple[Union[int, float, torch.Tensor], ...]
        ] = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer
                        conductance is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define the starting point from which integral
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be repeated
                        for each of `n_steps` along the integrated path.
                        For all other types, the given argument is used for
                        all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_steps (int, optional): The number of steps used by the approximation
                        method. Default: 50.
            method (str, optional): Method for approximating the integral,
                        one of `riemann_right`, `riemann_left`, `riemann_middle`,
                        `riemann_trapezoid` or `gausslegendre`.
                        Default: `gausslegendre` if no method is provided.
            internal_batch_size (int, optional): Divides total #steps * #examples
                        data points into chunks of size at most internal_batch_size,
                        which are computed (forward / backward passes)
                        sequentially. internal_batch_size must be at least equal to
                        2 * #examples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain internal_batch_size / num_devices examples.
                        If internal_batch_size is None, then all evaluations are
                        processed in one batch.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer inputs, otherwise it will be computed with respect
                        to layer outputs.
                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Conductance of each neuron in given layer input or
                        output. Attributions will always be the same size as
                        the input or output of the given layer, depending on
                        whether we attribute to the inputs or outputs
                        of the layer which is decided by the input flag
                        `attribute_to_layer_input`.
                        Attributions are returned in a tuple if
                        the layer inputs / outputs contain multiple tensors,
                        otherwise a single tensor is returned.
            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                        The difference between the total
                        approximated and true conductance.
                        This is computed using the property that the total sum of
                        forward_func(inputs) - forward_func(baselines) must equal
                        the total sum of the attributions.
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        examples in inputs.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> layer_cond = LayerConductance(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer conductance for class 3.
            >>> # attribution size matches layer output, Nx12x32x32
            >>> attribution = layer_cond.attribute(input, target=3)
        """
        inputs, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inputs, baselines, n_steps, method)

        num_examples = inputs[0].shape[0]
        if internal_batch_size is not None:
            num_examples = inputs[0].shape[0]
            attrs = _batch_attribution(
                self,
                num_examples,
                internal_batch_size,
                n_steps + 1,
                include_endpoint=True,
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
                attribute_to_layer_input=attribute_to_layer_input,
            )

        else:
            attrs = self._attribute(
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
                attribute_to_layer_input=attribute_to_layer_input,
            )

        is_layer_tuple = isinstance(attrs, tuple)
        attributions = attrs if is_layer_tuple else (attrs,)

        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                target=target,
                additional_forward_args=additional_forward_args,
            )
            return _format_output(is_layer_tuple, attributions), delta
        return _format_output(is_layer_tuple, attributions)

    def _attribute(
        self,
        inputs: Tuple[torch.Tensor, ...],
        baselines: Tuple[Union[torch.Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        attribute_to_layer_input: bool = False,
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        num_examples = inputs[0].shape[0]
        if step_sizes_and_alphas is None:
            # Retrieve scaling factors for specified approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            alphas = alphas_func(n_steps + 1)
        else:
            _, alphas = step_sizes_and_alphas
        # Compute scaled inputs from baseline to final input.
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (#examples * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps + 1)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps + 1)

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        for i in range(len(alphas)) :
            scaled_features_tpl_tmp = tuple([scaled_features_tpl[0][i*num_examples:(i+1)*num_examples,:]])
            (layer_gradients_tmp, layer_evals_tmp,) = compute_layer_gradients_and_eval(
                forward_fn=self.forward_func,
                layer=self.layer,
                inputs=scaled_features_tpl_tmp,
                additional_forward_args=input_additional_args,
                target_ind=expanded_target,
                device_ids=self.device_ids,
                attribute_to_layer_input=attribute_to_layer_input,
            )
            if i == 0 :
                layer_gradients = layer_gradients_tmp[0]
                layer_evals     = layer_evals_tmp[0]
            else : 
                layer_gradients = tuple([torch.concat([layer_gradients,layer_gradients_tmp[0]])])
                layer_evals = tuple([torch.concat([layer_evals,layer_evals_tmp[0]])])
                num_examples = layer_gradients_tmp[0].shape[0]

        del scaled_features_tpl_tmp,layer_gradients_tmp,layer_evals_tmp
        # Compute differences between consecutive evaluations of layer_eval.
        # This approximates the total input gradient of each step multiplied
        # by the step size.
        grad_diffs = tuple(
            layer_eval[num_examples:] - layer_eval[:-num_examples]
            for layer_eval in layer_evals
        )
        # Element-wise multiply gradient of output with respect to hidden layer
        # and summed gradients with respect to input (chain rule) and sum
        # across stepped inputs.
        attributions = tuple(
            _reshape_and_sum(
                grad_diff * layer_gradient[:-num_examples],
                n_steps,
                num_examples,
                layer_eval.shape[1:],
            )
            for layer_gradient, layer_eval, grad_diff in zip(
                layer_gradients, layer_evals, grad_diffs
            )
        )
        return _format_output(len(attributions) > 1, attributions)

    @property
    def multiplies_by_inputs(self):
        return True