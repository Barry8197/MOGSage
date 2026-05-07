# MOGSage

**Multi-Omic Graph Sage** — knowledge-graph-guided patient classification via GraphSAGE and Node2Vec teacher distillation.

MOGSage is the successor to [MOGDx](https://github.com/Barry8197/MOGDx), extending the multi-omic patient similarity network framework with scalable inductive graph learning and structural knowledge distillation. Where MOGDx uses a transductive GCN, MOGSage uses GraphSAGE with mini-batch neighbour sampling, enabling training on much larger patient networks without loading the full graph into GPU memory.

![Overview](https://github.com/Barry8197/MOGSage/releases/download/v1.0.0/MOGSage.png)

---

## Table of Contents

- [Background](#background)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Hyperparameter Grid Search](#hyperparameter-grid-search)
- [Results Explorer](#results-explorer)
- [Key Differences from MOGDx](#key-differences-from-mogdx)
- [Citation](#citation)
- [Contact](#contact)

---

## Background

Heterogeneous diseases such as cardiovascular disease, chronic kidney disease, and type 2 diabetes manifest across overlapping biological pathways and present significant diagnostic challenges. Integrating multi-omic data — proteomics, metabolomics, polygenic scores — with structured patient similarity networks offers a principled route to patient stratification.

MOGSage is applied to the **Swiss HIV Cohort Study (SHCS)**, a longitudinal cohort with rich multi-omic profiling. The model classifies patients into five disease phenotypes simultaneously (CAD, CKD, DMT2, Healthy, OST) as a multi-label problem, capturing comorbidity structure that single-label approaches miss.

---

## Architecture

MOGSage has three components trained end-to-end:

```
Multi-Omic Input                 Graph Structure
 [Proteomics]                       [PSN]
 [Metabolomics]   →  Multi-Modal  ←─────────────  Node2Vec
 [PGS]               Encoder          Teacher        (pre-trained)
       ↓               ↓               ↓
    Modality        Shared         Contrastive
    Encoders        Latent            Loss
       ↓            Space
    Decoder
       ↓
    Node Features
       ↓
  GraphSAGE Layers  ←──  Neighbour Sampling
       ↓
  Multi-Label Classification Head
       ↓
  BCE Loss + Contrastive Loss
```

### Multi-Modal Encoder (`MOGSage`)

Each omic modality is independently compressed through a two-layer encoder to a per-modality latent dimension. The latent vectors are then decoded into a shared embedding via mean pooling, creating a single unified node feature vector. This shared latent space is the input to the downstream GraphSAGE layers.

### Node2Vec 

A Node2Vec model is pre-trained on the **knowledge graph (KG)** — a richer, heterogeneous graph that encodes biological priors beyond the PSN alone. The resulting structural embeddings serve as contrastive targets during MOGSage training. The combined loss is:

```
L = L_BCE  +  α · L_contrastive(h_student, h_teacher)
```

where `α` (`--alpha_contrast`) controls the strength of structural knowledge transfer. This encourages the GNN to learn representations consistent with the underlying biology encoded in the KG, even when that graph is not directly used at inference time.

### GraphSAGE with Neighbour Sampling

MOGSage uses `NeighborLoader` from PyTorch Geometric for scalable mini-batch training. At each step, only a sampled neighbourhood is loaded (`--num_neighbors [15, 10]` by default), making the model trainable on large cohorts that would not fit in GPU memory under a full-graph transductive approach.

---

## Repository Structure

```
MOGSage/
├── mogsage.py                      # Main CLI training script
├── environment.yml
├── LICENSE
├── README.md
└── code/
    ├── MAIN/                        # Core Python implementation (model, training, utilities)
    │   ├── gnn.py                   # MOGSage model + EarlyStopping
    │   ├── train.py                 # Loss functions, evaluation, metrics
    │   ├── helpers.py               # Data utilities (imputation, embedding I/O)
    │   ├── network_functions.py     # Graph loading, PSN construction, PyG utils
    │   ├── preprocess_functions.py  # Omics processing functions
    │   └── applications.py          # RaKel multi-label predicitons, embedding clustering and survival curves
    ├── 01_meta_processing/          # Metadata format checks
    ├── 02_omics_processing/         # Omics preprocessing + sample overlap checks
    ├── 03_network_generation/       # Knowledge graph + PSN generation + embedding visualisation
    ├── 04_benchmark/                # Baselines (PyTorch, scikit-learn, RAkEL)
    ├── 05_MOGSageInteractive/       # Interactive notebooks (end-to-end experiments + grid search)
    └── 06_applications/             # Downstream analyses (clustering, signatures, survival)
```

---

## Installation

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU recommended (falls back to CPU)
- PyTorch ≥ 2.0
- PyTorch Geometric

### Setup

```bash
git clone https://github.com/Barry8197/MOGSage.git
cd MOGSage

conda env create -n mogsage -f environment.yml
conda activate mogsage
```

---

## Data Preparation

MOGSage expects the following directory layout under `--datadir`:

```
<datadir>/
├── Networks/
│   ├── KGshcs.gpickle          # Knowledge graph (used for Node2Vec teacher)
│   └── KGshcs_psn.gpickle      # Patient similarity network (used for GNN training)
└── 02_processed/
    ├── Proteomics.processed.pkl
    ├── Metabolomics.processed.pkl
    └── PGS.processed.pkl
```

Each `.processed.pkl` file should be a dictionary with key `"expr"` mapping to a `pd.DataFrame` with patients as rows and features as columns. The PSN and KG files are NetworkX graphs serialised with `pickle` (`.gpickle`), with node attributes including `pheno_onehot` (one-hot encoded disease label) and edge attribute `weight`.

---

## Usage

### Default run (mirrors original notebook configuration)

```bash
python mogsage.py
```

### Custom hyperparameters

```bash
python mogsage.py \
    --hidden_dim 128 \
    --emb_dim 64 \
    --decoder_dim 128 \
    --enc_dims 128 128 64 \
    --latent_dims 64 64 24 \
    --alpha_contrast 0.5 \
    --temperature 0.07 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --epochs 2000
```

### Save model and embeddings

```bash
python mogsage.py --save_model --gen_emb
```

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--datadir` | `/scratch/bryan/data/SHCS/MOGSage` | Root data directory |
| `--outdir` | `/scratch/bryan/MOGSage/results/` | Root output directory |
| `--gprefix` | `KGshcs` | Graph file prefix |
| `--seed` | `7` | Random seed |
| `--train_frac` | `0.6` | Train split fraction |
| `--val_frac` | `0.2` | Validation split fraction |
| `--omics` | `Proteomics Metabolomics PGS` | Omics modalities to load |
| `--class_labels` | `CAD CKD DMT2 Healthy OST` | Target class names |
| `--hidden_dim` | `64` | GraphSAGE hidden dimension |
| `--emb_dim` | `32` | GraphSAGE output / Node2Vec embedding dimension |
| `--enc_dims` | `64 64 32` | Per-modality encoder layer sizes |
| `--latent_dims` | `32 32 12` | Per-modality latent dimensions |
| `--decoder_dim` | `64` | Shared decoder output dimension |
| `--dropout` | `0.5` | Dropout rate |
| `--num_neighbors` | `15 10` | Neighbours sampled per GraphSAGE layer |
| `--train_batch` | `1024` | Training batch size |
| `--alpha_contrast` | `0.2` | Weight of distillation loss |
| `--temperature` | `0.07` | Temperature for contrastive loss |
| `--lr` | `5e-4` | AdamW learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--epochs` | `2000` | Maximum training epochs |
| `--es_patience` | `250` | Early stopping patience |
| `--threshold` | `0.5` | Sigmoid threshold for multi-label prediction |
| `--save_model` | `False` | Save model checkpoint |
| `--gen_emb` | `False` | Extract and save full-graph node embeddings |

---

## Outputs

All outputs are written to `<outdir>/<RUN_TAG>/` where `RUN_TAG` uniquely encodes the full hyperparameter configuration:

```
dec{decoder_dim}_hid{hidden_dim}_emb{emb_dim}_enc{enc_dims}_lat{latent_dims}
    _a{alpha}_t{temperature}_lr{lr}_wd{weight_decay}_do{dropout}
```

```
<outdir>/
├── <RUN_TAG>/
│   ├── test_overall_<RUN_TAG>.csv      # Micro/macro F1, precision, recall, AUROC
│   ├── test_perclass_<RUN_TAG>.csv     # Per-class F1, precision, recall
│   ├── checkpoints/                    # Saved model (if --save_model)
│   │   └── mogsage_gnn_model_<RUN_TAG>.pt
│   └── embeddings/                     # Node embeddings (if --gen_emb)
│       └── mogsage_gnn_embeddings_<RUN_TAG>.pkl
└── tb/
    └── <RUN_TAG>/                      # TensorBoard event files
```

### TensorBoard monitoring

All runs write to a shared `tb/` directory, with each run prefixed by its unique tag. This allows direct comparison across grid search runs in a single TensorBoard session:

```bash
tensorboard --logdir <outdir>/tb/
```

The HParams dashboard is populated automatically at the end of each run with all hyperparameters and the best validation loss, enabling side-by-side comparison.

---

## Hyperparameter Grid Search

`gridsearch.ipynb` generates a text file of CLI commands covering the full cartesian product of the parameter grid. Edit the `GRID` dict and run all cells — it writes one `python mogsage.py ...` command per line to `grid_commands.txt`, ready to feed into a cluster job array or sequential loop.

Default grid covers:

| Group | Parameters |
|---|---|
| Architecture | `hidden_dim`, `emb_dim`, `decoder_dim`, `enc_dims`, `latent_dims` |
| Distillation | `alpha_contrast`, `temperature` |
| Optimiser | `lr`, `weight_decay` |
| Regularisation | `dropout` |

---

## Results Explorer

`results_explorer.ipynb` provides an interactive post-hoc analysis of all completed runs. Point `RESULTS_DIR` at your results folder, set `RANK_BY` to your metric of interest, and run all cells. It produces:

- **Leaderboard** — styled table of the top-20 runs
- **Metric distributions** — histograms across all runs, with top-N highlighted
- **Top-5 comparison** — grouped bar chart across all metrics
- **Hyperparameter sensitivity** — box plots showing how each HP value affects the ranking metric
- **Per-class F1 heatmap** — identifies configs that win overall but collapse on specific phenotypes
- **HP–metric correlation** — Pearson correlation between numeric hyperparameters and test metrics
- **Best config summary card** — full hyperparameter and metric summary for the rank-1 run

---

## Key Differences from MOGDx

| | MOGDx | MOGSage |
|---|---|---|
| **GNN** | Transductive GCN (full graph) | Inductive GraphSAGE (mini-batch) |
| **Scalability** | Limited by GPU memory | Scales to large cohorts via neighbour sampling |
| **Graph input** | Fused PSN (SNF) | PSN for training + KG for teacher distillation |
| **Teacher signal** | None | Node2Vec pre-trained on knowledge graph |
| **Training loss** | Cross-entropy | BCE + contrastive distillation loss |
| **Classification** | Multi-class | Multi-label (handles comorbidities) |
| **Application** | TCGA cancer datasets | SHCS multi-disease cohort |
| **Preprocessing** | R pipeline (SNF) | Python pipeline |

---

## Citation

If you use MOGSage in your research, please cite the associated paper (forthcoming). For the underlying MOGDx framework:

```bibtex
@article{ryan2024mogdx,
  title   = {Multi-Omic Graph Diagnosis (MOGDx): A data integration tool to perform 
             classification tasks for heterogeneous diseases},
  author  = {Ryan, Barry and others},
  journal = {Bioinformatics},
  year    = {2024},
  doi     = {10.1093/bioinformatics/btae523}
}
```

---

## Contact

**Barry Ryan**  
Postdoctoral Researcher, École polytechnique fédérale de Lausanne   
barry.ryan@epfl.ch
· [LinkedIn](https://www.linkedin.com/in/barry-ryan/) · [Website](https://barry8197.github.io/barryryan/)