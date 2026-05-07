"""
mogsage.py
====================
CLI training script for MOGSage with Node2Vec teacher distillation,
TensorBoard logging, and structured output paths.

Output directory layout
-----------------------
<OUTDIR>/<RUN_TAG>/
    checkpoints/   mogsage_gnn_model_<RUN_TAG>.pt
    embeddings/    mogsage_gnn_embeddings_<RUN_TAG>.pkl
    ../../tb/      TensorBoard event files
    test_overall_<RUN_TAG>.csv
    test_perclass_<RUN_TAG>.csv

where RUN_TAG = <DECODER_DIM>_<HIDDEN_DIM>_<EMB_DIM>_<ALPHA_CONTRAST>

Usage
-----
# Default hyper-params (mirrors original notebook config)
python mogsage.py

# Custom hyper-params
python mogsage.py \\
    --hidden_dim 128 --emb_dim 64 --decoder_dim 128 \\
    --alpha_contrast 0.5 --lr 1e-3 --epochs 1000
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, 'code/MAIN/')
from helpers import (
    merge_dfs,
    get_split_indices_from_masks,
    extract_node_embeddings_fullgraph,
    save_gnn_embeddings_pickle,
    save_pytorch_model,
    median_impute_per_column,
)
from network_functions import (
    fix_pheno_onehot_in_graph,
    nx_to_pyg,
    load_graph,
    node_labels,
    edge_index_weighted_resample,
)
from train import (
    evaluate_multilabel,
    training_metric_plots,
    contrastive_distill_loss,
    per_class_metrics_from_scores,
)
from gnn import EarlyStopping, MOGSage


# ============================================================
# HELPERS
# ============================================================

def build_run_tag(args: argparse.Namespace) -> str:
    """Canonical run identifier used for all output paths and TensorBoard."""
    alpha_str  = str(args.alpha_contrast).replace(".", "p")
    lr_str     = str(args.lr).replace(".", "p")
    wd_str     = str(args.weight_decay).replace(".", "p")
    temp_str   = str(args.temperature).replace(".", "p")
    do_str     = str(args.dropout).replace(".", "p")
    enc_str    = "-".join(str(d) for d in args.enc_dims)
    lat_str    = "-".join(str(d) for d in args.latent_dims)
    return (
        f"dec{args.decoder_dim}"
        f"_hid{args.hidden_dim}"
        f"_emb{args.emb_dim}"
        f"_enc{enc_str}"
        f"_lat{lat_str}"
        f"_a{alpha_str}"
        f"_t{temp_str}"
        f"_lr{lr_str}"
        f"_wd{wd_str}"
        f"_do{do_str}"
    )


def make_output_dirs(base: str, tag: str) -> dict:
    """Create and return all output sub-directories for a run."""
    run_dir = os.path.join(base, tag)
    dirs = {
        "run":         run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "embeddings":  os.path.join(run_dir, "embeddings"),
        "tb":          os.path.join(run_dir, "../../tb"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def log(msg: str) -> None:
    """Timestamped print — works in terminals, SLURM logs, and nohup output."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# ARGUMENT PARSER  (all CONFIG section fields exposed)
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MOGSage (Node2Vec teacher) on SHCS patient-similarity network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- paths ----
    p.add_argument("--datadir",  default="/scratch/bryan/data/SHCS/MOGSage",
                   help="Root data directory")
    p.add_argument("--outdir",   default="/scratch/bryan/MOGSage/results/",
                   help="Root output directory (run sub-folder created automatically)")
    p.add_argument("--gprefix",  default="KGshcs",
                   help="Graph file prefix")

    # ---- data ----
    p.add_argument("--seed",       type=int,   default=7)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac",   type=float, default=0.2)
    p.add_argument("--omics",      type=str,   nargs="+",
                   default=["Proteomics", "Metabolomics", "PGS"],
                   help="Ordered list of omics modalities to load")
    p.add_argument("--class_labels",      type=str,   nargs="+",
                   default=['CAD', 'CKD', 'DMT2', 'Healthy', 'OST'],
                   help="Ordered list of class labels")

    # ---- model architecture ----
    p.add_argument("--hidden_dim",  type=int,   default=64)
    p.add_argument("--emb_dim",     type=int,   default=32)
    p.add_argument("--enc_dims",    type=int,   nargs="+", default=[64, 64, 32])
    p.add_argument("--latent_dims", type=int,   nargs="+", default=[32, 32, 12])
    p.add_argument("--decoder_dim", type=int,   default=64)
    p.add_argument("--dropout",     type=float, default=0.5)

    # ---- loaders ----
    p.add_argument("--train_batch",   type=int, default=1024)
    p.add_argument("--val_batch",     type=int, default=1024)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10])

    # ---- contrastive / distillation ----
    p.add_argument("--alpha_contrast", type=float, default=0.2)
    p.add_argument("--temperature",    type=float, default=0.07)

    # ---- optimiser ----
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--pos_weight_coef", type=int, default=1)

    # ---- training ----
    p.add_argument("--epochs",       type=int,   default=2000)
    p.add_argument("--es_patience",  type=int,   default=250)
    p.add_argument("--es_min_delta", type=float, default=1e-6)
    p.add_argument("--threshold",    type=float, default=0.5)

    # ---- outputs ----
    p.add_argument("--gen_emb",    action="store_true", default=False)
    p.add_argument("--save_model", action="store_true", default=False)

    return p.parse_args()


# ============================================================
# NODE2VEC TEACHER
# ============================================================

def train_node2vec(
    edge_index_w: torch.Tensor,
    args: argparse.Namespace,
    device: str,
    writer: SummaryWriter,
    run_tag: str,
) -> Node2Vec:
    """Train Node2Vec and log loss to TensorBoard."""
    node2vec = Node2Vec(
        edge_index=edge_index_w,
        embedding_dim=args.emb_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

    n2v_loader = node2vec.loader(
        batch_size=args.train_batch, shuffle=True, num_workers=0
    )
    n2v_optim = torch.optim.SparseAdam(node2vec.parameters(), lr=1e-3)

    log("--- Node2Vec teacher pre-training ---")
    for epoch in range(1, 500 + 1):
        node2vec.train()
        total_loss = 0.0

        for pos_rw, neg_rw in n2v_loader:
            n2v_optim.zero_grad(set_to_none=True)
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            n2v_optim.step()
            total_loss += float(loss.detach().cpu())

        avg_loss = total_loss / max(len(n2v_loader), 1)
        writer.add_scalar(f"Node2Vec/avg_loss/{run_tag}", avg_loss, epoch)

        if epoch % 20 == 0 or epoch == 1:
            log(f"Node2Vec  epoch {epoch:>4}/{500}  avg_loss={avg_loss:.4f}")

    log("Node2Vec training complete.")
    return node2vec


# ============================================================
# MOGSage TRAINING
# ============================================================

def train_MOGSage(
    model: nn.Module,
    train_loader,
    val_loader,
    data,
    criterion_node: nn.Module,
    args: argparse.Namespace,
    device: str,
    writer: SummaryWriter,
    run_tag: str,
) -> tuple[nn.Module, float]:
    """Train the MOGSage model; returns (best_model, best_val_loss)."""
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    es    = EarlyStopping(patience=args.es_patience, min_delta=args.es_min_delta)

    log("--- (MOGSage) training ---")

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        total_loss = total_seeds = 0
        tr_tp = tr_fp = tr_fn = 0.0

        for batch in train_loader:
            batch      = batch.to(device)
            seed_count = batch.batch_size
            optim.zero_grad(set_to_none=True)

            h_seed, logits = model(batch.x, batch.edge_index)
            logits   = logits[:seed_count]
            h_seed   = h_seed[:seed_count]
            h_t_seed = data.teacher_logits[batch.n_id[:seed_count]].to(device)
            y_seed   = batch.y[:seed_count]

            bce  = criterion_node(logits, y_seed)
            mse  = contrastive_distill_loss(h_seed, h_t_seed, temperature=args.temperature)
            loss = bce + args.alpha_contrast * mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss  += float(loss.detach().cpu()) * seed_count
            total_seeds += seed_count

            with torch.no_grad():
                y_pred  = (torch.sigmoid(logits) >= args.threshold).int()
                y_true  = batch.y[:seed_count].int()
                tr_tp  += float((y_pred &  y_true).sum())
                tr_fp  += float((y_pred & (1 - y_true)).sum())
                tr_fn  += float(((1 - y_pred) & y_true).sum())

        train_loss = total_loss / max(total_seeds, 1)
        train_f1   = (2 * tr_tp) / (2 * tr_tp + tr_fp + tr_fn + 1e-8)

        # ---- val ----
        model.eval()
        tp = fp = fn = 0.0
        val_total_loss = val_total_seeds = 0

        with torch.no_grad():
            for batch in val_loader:
                batch      = batch.to(device)
                seed_count = batch.batch_size

                h_seed, logits = model(batch.x, batch.edge_index)
                logits   = logits[:seed_count]
                h_seed   = h_seed[:seed_count]
                h_t_seed = data.teacher_logits[batch.n_id[:seed_count]].to(device)
                y_true   = batch.y[:seed_count]

                v_bce  = criterion_node(logits, y_true)
                v_mse  = contrastive_distill_loss(h_seed, h_t_seed, temperature=args.temperature)
                v_loss = v_bce + args.alpha_contrast * v_mse

                val_total_loss  += float(v_loss) * seed_count
                val_total_seeds += seed_count

                y_pred  = (torch.sigmoid(logits) >= args.threshold).int()
                y_true  = y_true.int()
                tp += float((y_pred &  y_true).sum())
                fp += float((y_pred & (1 - y_true)).sum())
                fn += float(((1 - y_pred) & y_true).sum())

        val_f1   = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        val_loss = val_total_loss / max(val_total_seeds, 1)

        # ---- TensorBoard ----
        writer.add_scalars(f"Loss/{run_tag}",       {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars(f"MicroF1/{run_tag}",    {"train": train_f1,   "val": val_f1},   epoch)
        writer.add_scalar( f"ES_counter/{run_tag}", es.counter, epoch)

        if epoch % 10 == 0 or epoch == 1:
            log(
                f"MOGSage  epoch {epoch:>5}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_F1={val_f1:.4f}  es={es.counter}/{es.patience}"
            )

        # Early stopping monitors val_loss (matching original notebook)
        es.step(val_loss, model)
        if es.stop:
            log(f"Early stopping at epoch {epoch} — best val loss: {es.best_score:.4f}")
            break

    es.restore(model)
    log(f"Restored best model — best val loss: {es.best_score:.4f}")

    # Log hyper-params + summary metric to TensorBoard HParams dashboard
    writer.add_hparams(
        hparam_dict={
            "hidden_dim":     args.hidden_dim,
            "emb_dim":        args.emb_dim,
            "enc_dims":       str(args.enc_dims),
            "latent_dims":    str(args.latent_dims),
            "decoder_dim":    args.decoder_dim,
            "alpha_contrast": args.alpha_contrast,
            "temperature":    args.temperature,
            "lr":             args.lr,
            "weight_decay":   args.weight_decay,
            "dropout":        args.dropout,
        },
        metric_dict={"hparam/best_val_loss": es.best_score},
        run_name=run_tag,
    )

    return model, es.best_score


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(
    model: nn.Module,
    test_loader,
    device: str,
    threshold: float,
    class_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch      = batch.to(device)
            seed_count = batch.batch_size
            _, logits  = model(batch.x, batch.edge_index)
            all_preds.append( (torch.sigmoid(logits[:seed_count]) >= threshold).int().cpu())
            all_labels.append(batch.y[:seed_count].int().cpu())

    y_true_np = torch.cat(all_labels).numpy()
    y_pred_np = torch.cat(all_preds ).numpy()

    overall   = pd.DataFrame([evaluate_multilabel(y_true_np, y_pred_np)]).set_index("Model")
    per_class = pd.DataFrame(per_class_metrics_from_scores(y_true_np, y_pred_np)["per_class"])
    per_class.index = class_labels

    return overall, per_class


# ============================================================
# MAIN
# ============================================================

def main():
    args         = parse_args()
    CLASS_LABELS  = args.class_labels
    ATTRS_TO_KEEP = {'pheno_onehot'}
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    run_tag       = build_run_tag(args)
    dirs          = make_output_dirs(args.outdir, run_tag)
    writer        = SummaryWriter(log_dir=dirs["tb"])

    log(f"Run tag   : {run_tag}")
    log(f"Device    : {device}")
    log(f"Output dir: {dirs['run']}")
    log("Hyperparameters:")
    for k, v in vars(args).items():
        log(f"  {k:25s} = {v}")

    # ==========================================================
    # 1) LOAD PSN GRAPH
    # ==========================================================
    log("Loading PSN graph …")
    g   = load_graph(os.path.join(args.datadir, "Networks", f"{args.gprefix}_psn.gpickle"), "PSN")
    cnt = fix_pheno_onehot_in_graph(g)
    log(f"Converted pheno_onehot to np.float32 for {cnt} nodes.")
    nx_to_pyg(g, ATTRS_TO_KEEP, get_weight=True)
    all_nodes = sorted(set(g.nodes))

    # ==========================================================
    # 2) LOAD OMICS FEATURES
    # ==========================================================
    log(f"Loading omics: {args.omics} …")
    expr_dfs = []
    for expr in args.omics:
        with open(os.path.join(args.datadir, "02_processed", f"{expr}.processed.pkl"), "rb") as f:
            data_expr = pickle.load(f)
        expr_df = data_expr["expr"]
        expr_dfs.append(expr_df.loc[expr_df.index.intersection(all_nodes)].copy())

    meta    = node_labels(g)
    meta    = pd.DataFrame(np.float32(meta.values), index=meta.index)
    h_df    = reduce(merge_dfs, expr_dfs).reindex(all_nodes)
    meta_df = meta.reindex(all_nodes)

    x = torch.tensor(h_df.values,    dtype=torch.float32)
    x = median_impute_per_column(x)
    y = torch.tensor(meta_df.values, dtype=torch.float32)

    INPUT_DIMS = [e.shape[1] for e in expr_dfs]
    log(f"Input dims per modality: {INPUT_DIMS}")

    # ==========================================================
    # 3) PyG DATA + TRAIN/VAL/TEST MASKS
    # ==========================================================
    log("Building PyG data object and splits …")
    data   = from_networkx(g)
    data.y = y

    N       = len(all_nodes)
    gen     = torch.Generator().manual_seed(args.seed)
    perm    = torch.randperm(N, generator=gen)
    n_train = int(args.train_frac * N)
    n_val   = int(args.val_frac   * N)

    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask   = torch.zeros(N, dtype=torch.bool)
    data.test_mask  = torch.zeros(N, dtype=torch.bool)
    data.train_mask[perm[:n_train]]                = True
    data.val_mask  [perm[n_train:n_train + n_val]] = True
    data.test_mask [perm[n_train + n_val:]]        = True

    scaler = StandardScaler()
    scaler.fit(x[perm[:n_train]])
    data.x = torch.tensor(scaler.transform(x), dtype=torch.float32)

    log(
        f"Nodes — train: {data.train_mask.sum()} | "
        f"val: {data.val_mask.sum()} | "
        f"test: {data.test_mask.sum()}"
    )

    # ==========================================================
    # 4) NEIGHBOR LOADERS
    # ==========================================================
    _lkw = dict(num_neighbors=args.num_neighbors, persistent_workers=False)
    train_loader = NeighborLoader(data, input_nodes=data.train_mask, batch_size=args.train_batch, shuffle=True,  **_lkw)
    val_loader   = NeighborLoader(data, input_nodes=data.val_mask,   batch_size=args.val_batch,   shuffle=False, **_lkw)
    test_loader  = NeighborLoader(data, input_nodes=data.test_mask,  batch_size=args.val_batch,   shuffle=False, **_lkw)

    # ==========================================================
    # 5) NODE2VEC TEACHER
    # ==========================================================
    log("Loading KG graph for Node2Vec teacher …")
    kg  = load_graph(os.path.join(args.datadir, "Networks", f"{args.gprefix}.gpickle"), "PSN")
    fix_pheno_onehot_in_graph(kg)
    nx_to_pyg(kg, ATTRS_TO_KEEP, get_weight=True)

    teacher_data   = from_networkx(kg)
    edge_index_w   = edge_index_weighted_resample(
        teacher_data.edge_index,
        teacher_data.weight,
        num_samples=teacher_data.edge_index.size(1),
    ).to(device)

    node2vec = train_node2vec(edge_index_w, args, device, writer, run_tag)

    node2vec.eval()
    with torch.no_grad():
        teacher_logits = node2vec()          # (num_kg_nodes, emb_dim)

    # teacher_logits are graph-structure embeddings used as distillation targets
    data.teacher_logits = teacher_logits

    # ==========================================================
    # 6) MOGSage MODEL + CLASS-WEIGHTED CRITERION
    # ==========================================================
    log("Initialising MOGSage …")
    GNN_DIMS = [args.hidden_dim, args.emb_dim]

    model = MOGSage(
        input_dims=INPUT_DIMS,
        encoder_dims=args.enc_dims,
        latent_dims=args.latent_dims,
        decoder_dim=args.decoder_dim,
        hidden_dims=GNN_DIMS,
        num_classes=y.shape[1],
        dropout=args.dropout,
    ).to(device)

    with torch.no_grad():
        pos        = y[perm[:n_train]].sum(dim=0)
        neg        = y[perm[:n_train]].shape[0] - pos
        pos_weight = ((neg / (pos + 1e-8)) ** args.pos_weight_coef).clamp(1, 100).to(device)

    log(f"pos_weight: {pos_weight.cpu().tolist()}")
    criterion_node = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ==========================================================
    # 7) MOGSage TRAINING
    # ==========================================================
    model, best_val_loss = train_MOGSage(
        model, train_loader, val_loader, data,
        criterion_node=criterion_node,
        args=args,
        device=device,
        writer=writer,
        run_tag=run_tag,
    )

    # ==========================================================
    # 8) TEST EVALUATION
    # ==========================================================
    log("Evaluating on test set …")
    overall_df, per_class_df = evaluate_test(
        model, test_loader, device, args.threshold, CLASS_LABELS
    )

    log("\n=== Overall Test Results ===")
    log(overall_df.to_string())
    log("\n=== Per-class Test Results ===")
    log(per_class_df.to_string())

    # Log scalar test metrics to TensorBoard
    for col in overall_df.columns:
        try:
            writer.add_scalar(f"Test/{col}/{run_tag}", float(overall_df[col].iloc[0]), 0)
        except Exception:
            pass
    for metric in per_class_df.columns:
        for cls in per_class_df.index:
            try:
                writer.add_scalar(
                    f"Test_PerClass/{metric}/{cls}/{run_tag}",
                    float(per_class_df.loc[cls, metric]), 0
                )
            except Exception:
                pass

    # Save CSV results alongside TensorBoard events
    overall_csv  = os.path.join(dirs["run"], f"test_overall_{run_tag}.csv")
    perclass_csv = os.path.join(dirs["run"], f"test_perclass_{run_tag}.csv")
    overall_df.to_csv(overall_csv)
    per_class_df.to_csv(perclass_csv)
    log(f"Saved test results   : {overall_csv}")
    log(f"Saved per-class results: {perclass_csv}")

    # ==========================================================
    # 9) SAVE EMBEDDINGS
    # ==========================================================
    if args.gen_emb:
        log("Extracting full-graph node embeddings …")
        train_idx, val_idx, test_idx = get_split_indices_from_masks(data)
        H = extract_node_embeddings_fullgraph(model, data, device=device, return_logits=False)
        log(f"Embeddings: {H.shape}  splits: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

        emb_path = os.path.join(
            dirs["embeddings"], f"mogsage_gnn_embeddings_{run_tag}.pkl"
        )
        save_gnn_embeddings_pickle(
            out_path=emb_path,
            H=H,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            x=data.x.cpu().numpy(),
            y=data.y.cpu().numpy(),
            label_names=CLASS_LABELS,
            extra={
                "model":        "MOGSage",
                "run_tag":      run_tag,
                "hidden_dim":   args.hidden_dim,
                "emb_dim":      args.emb_dim,
                "enc_dims":     args.enc_dims,
                "latent_dims":  args.latent_dims,
                "decoder_dim":  args.decoder_dim,
                "seed":         args.seed,
            },
        )
        log(f"Saved embeddings: {emb_path}")

    # ==========================================================
    # 10) SAVE MODEL
    # ==========================================================
    if args.save_model:
        model_path = os.path.join(
            dirs["checkpoints"], f"mogsage_gnn_model_{run_tag}"
        )
        save_pytorch_model(
            model=model,
            path=model_path,
            optimizer=None,
            extra={
                "model":        "MOGSage",
                "run_tag":      run_tag,
                "hidden_dim":   args.hidden_dim,
                "emb_dim":      args.emb_dim,
                "enc_dims":     args.enc_dims,
                "latent_dims":  args.latent_dims,
                "decoder_dim":  args.decoder_dim,
                "seed":         args.seed,
            },
            save_entire_model=False,
        )
        log(f"Saved model: {model_path}")

    writer.close()
    log("Done.")


if __name__ == "__main__":
    main()