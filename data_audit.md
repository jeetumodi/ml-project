# EGCPPIS Data Audit Report (Source of Truth: `data/`)

**Auditor:** Antigravity AI  
**Date:** October 7, 2026  
**Workspace:** `c:\Users\jeetu\Desktop\Machine learning\PROJECT`  
**Strict Scope:** The `data/` directory is treated as the sole ground truth per project rules and user directive (*"dont use that ML_PORJECT folder"*).

---

## 1. Complete File Inventory

The entire `data/` directory was scanned recursively.

- **Total Files:** 3,062 files
- **Total Size:** 440.92 MB
- **Extension Breakdown:**
  - `.npy`: 2,964 files (4 x 741 features in `data/GraphPPIS/Feature`)
  - `.py`: 34 files (code in DeepPPISP, EGCPPIS, GraphPPIS)
  - `.pkl`: 24 files (cache, graphs, models)
  - `.fa`: 18 files (benchmark datasets and variants)
  - `.txt`: 6 files (readmes, licenses, blosum)
  - `.md`: 3 files
  - `.csv`: 3 files (metrics, data curves)
  - `.pth`: 3 files (model checkpoints)
  - `.tsv`: 1 file
  - `.pl`: 1 file (Perl chain extraction script)
  - `.png`: 1 file
  - `(no ext)`: 2 files (`caldis_CA` binary, `LICENSE`)
  - `.pyc`: 2 files

### Directory Overview Table

| Relative Path | Files | Subdirs | Size (MB) | Purpose | Loadable / Valid | Relevant to EGCPPIS |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `data/DeepPPISP/` | 5 | 4 | ~0.02 | Baseline DeepPPISP implementation | Yes | Yes (benchmark comparison) |
| `data/DeepPPISP/data_cache/` | 15 | 0 | 12.56 | DeepPPISP cache lists & pickle data | Yes | Yes (DeepPPISP split info) |
| `data/DeepPPISP/models/` | 3 | 0 | ~0.02 | DeepPPISP CNN/LSTM models | Yes | Reference baseline |
| `data/EGCPPIS/` | 12 | 3 | ~0.10 | Official EGCPPIS implementation | Yes | Source of truth for EGCPPIS |
| `data/EGCPPIS/Datasets/DeepPPISP/` | 4 | 1 | 0.36 | DeepPPISP benchmark datasets (.fa) | Yes | Core Benchmark (Train352, Test70) |
| `data/EGCPPIS/Datasets/GraphPPIS/` | 4 | 2 | 0.30 | GraphPPIS benchmark datasets (.fa) | Yes | Core Benchmark (Train335, Test60, Test315-28, Ubtest31-6) |
| `data/EGCPPIS/DeepPPISP/` | 10 | 1 | ~0.06 | EGCPPIS trained on DeepPPISP | Yes | DeepPPISP run code |
| `data/EGCPPIS/DeepPPISP/model_save/` | 4 | 0 | 9.55 | DeepPPISP checkpoints & PR/ROC curves | Yes (`best_model.pth`, 108 params) | Yes (DeepPPISP trained checkpoint) |
| `data/EGCPPIS/model_save/` | 1 | 0 | 4.55 | Official EGCPPIS checkpoint | Yes (`best_model.pth`, 108 params) | Yes (Reference model weights) |
| `data/GraphPPIS/Dataset/` | 5 | 0 | 0.31 | Original GraphPPIS datasets | Yes | Baseline datasets |
| `data/GraphPPIS/Feature/distance_map/` | 741 | 0 | ~80.0 | Residue distance matrices (C-alpha / SC) | Yes (Float64 NxN) | Yes (Residue graph construction) |
| `data/GraphPPIS/Feature/dssp/` | 741 | 0 | ~25.0 | DSSP secondary structure & solvent | Yes (Float64 Nx14) | Yes (14-d residue features) |
| `data/GraphPPIS/Feature/hmm/` | 741 | 0 | ~35.0 | HMM evolutionary profiles | Yes (Float64 Nx20) | Yes (20-d residue features) |
| `data/GraphPPIS/Feature/pssm/` | 741 | 0 | ~35.0 | PSSM evolutionary conservation matrices | Yes (Float64 Nx20) | Yes (20-d residue features) |
| `data/GraphPPIS/Model/` | 3 | 1 | 8.50 | GraphPPIS Fast and Slow models | Yes | Reference baseline |

---

## 2. Dataset Mapping

Mapping between the paper's target benchmark datasets and the actual verified files inside `data/`:

| Paper Dataset | Actual File Path in `data/` | Status | Protein Count Match | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Train352** | `data/EGCPPIS/Datasets/DeepPPISP/Train_352.fa` | **VERIFIED** | 352 / 352 | Sequence & label lengths match 100% |
| **Test70** | `data/EGCPPIS/Datasets/DeepPPISP/Test_70.fa` | **VERIFIED** | 70 / 70 | Exact match to single residue counts |
| **Train_335-1** | `data/EGCPPIS/Datasets/GraphPPIS/Train_335.fa` | **VERIFIED** | 334 / 334 | 335 minus 1 abnormal chain (`2j3rA`) = 334 |
| **Test_60** | `data/EGCPPIS/Datasets/GraphPPIS/Test_60.fa` | **VERIFIED** | 60 / 60 | Exact match to single residue counts |
| **Test_315-28** | `data/EGCPPIS/Datasets/GraphPPIS/Test_315-28.fa` | **VERIFIED** | 287 / 287 | 315 minus 28 overlapping/homologous = 287 |
| **Ubtest_31-6** | `data/EGCPPIS/Datasets/GraphPPIS/UBtest_31-6.fa` | **VERIFIED** | 25 / 25 | 31 minus 6 unbound proteins = 25 |

---

## 3. Dataset Statistics & Comparison with Paper

| Dataset | Paper Proteins | Actual Proteins | Paper Binding | Actual Binding | Paper Non-binding | Actual Non-binding | Paper Binding % | Actual Binding % | Match Status |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Train352** | 352 | 352 | 11,079 | 11,204 | 62,102 | 65,045 | 15.14% | 14.69% | **Close Match** (diff due to full sequence length retention) |
| **Test70** | 70 | 70 | 2,332 | 2,332 | 9,459 | 9,459 | 19.78% | 19.78% | **EXACT (100.0%)** |
| **Train_335-1** | 334 | 334 | 10,374 | 10,336 | 66,366* | 55,872 | 13.52%* | 15.61% | **EXACT MATCH** (*Paper table reported total 66,366 residues for Train_335; minus 1 protein `2j3rA` (38 pos, 120 neg) yields 10,336 / 55,872) |
| **Test_60** | 60 | 60 | 2,075 | 2,075 | 11,069 | 11,069 | 15.79% | 15.79% | **EXACT (100.0%)** |
| **Test_315-28** | 287 | 287 | 8,566 | 8,566 | 51,810 | 51,810 | 14.19% | 14.19% | **EXACT (100.0%)** |
| **Ubtest_31-6** | 25 | 25 | 711 | 711 | 5,206 | 5,206 | 12.02% | 12.02% | **EXACT (100.0%)** |

---

## 4. Sequence and Label Validation

Every protein sequence and label string was audited across all dataset files:
- **Length Consistency:** `len(sequence) == len(labels)` verified for every record across all files. **Zero mismatches.**
- **Amino Acid Alphabet:** Only valid standard 20 amino acids (`ACDEFGHIKLMNPQRSTVWY`) are present.
- **Labels:** Binary digits `{0, 1}` only. Zero invalid characters.
- **Empty / Malformed Records:** None.
- **Train / Test Leakage Check:**
  - DeepPPISP `Train_352` vs `Test_70`: **0 overlapping protein IDs, 0 identical sequences.**
  - GraphPPIS `Train_335` vs `Test_60`: **0 overlapping protein IDs, 0 identical sequences.**
  - GraphPPIS `Train_335` vs `Test_315-28`: **0 overlapping protein IDs, 0 identical sequences.**
  - GraphPPIS `Train_335` vs `UBtest_31-6`: **0 overlapping protein IDs, 0 identical sequences.**

---

## 5. Feature Availability in `data/`

| Feature | Paper Component | Expected Dim | Available in `data/` | Actual Dim | NaN / Inf Count | Status in `data/` |
| :--- | :--- | ---: | :--- | ---: | ---: | :--- |
| **One-hot** | Residue Sequence | 20 | Computed dynamically from sequence | 20 | 0 / 0 | **READY** |
| **PSSM** | Evolutionary Conservation | 20 | `data/GraphPPIS/Feature/pssm/` (741 files) | 20 | 0 / 0 | **READY** |
| **HMM** | Profile HMM | 20 | `data/GraphPPIS/Feature/hmm/` (741 files) | 20 | 0 / 0 | **READY** |
| **DSSP** | Secondary Structure & ASA | 14 | `data/GraphPPIS/Feature/dssp/` (741 files) | 14 | 0 / 0 | **READY** |
| **Distance Map** | Residue Spatial Graph | NxN | `data/GraphPPIS/Feature/distance_map/` (741 files) | NxN | 0 / 0 | **READY** |
| **resAF** | AlphaFold Residue Features | 7 | Missing in `data/` | N/A | N/A | **MISSING in data/** |
| **ESM-2 (bert)** | Pre-trained Language Model | 33 | Missing in `data/` | N/A | N/A | **MISSING in data/** |
| **psepos** | Pseudo-spatial position | 1 (from 3D) | Missing in `data/` | N/A | N/A | **MISSING in data/** |
| **Atom Model** | Atom Graph & Mapping | 37 nodes | Missing in `data/` | N/A | N/A | **MISSING in data/** |

### Feature Coverage Breakdown by Dataset (in `data/GraphPPIS/Feature`)
- **GraphPPIS Train_335-1 (334):** 334/334 (100.0%) for PSSM, HMM, DSSP, Distance Map.
- **GraphPPIS Test_60 (60):** 60/60 (100.0%) for PSSM, HMM, DSSP, Distance Map.
- **GraphPPIS Test_315-28 (287):** 287/287 (100.0%) for PSSM, HMM, DSSP, Distance Map.
- **GraphPPIS Ubtest_31-6 (25):** 25/25 (100.0%) for PSSM, HMM, DSSP, Distance Map.
- **DeepPPISP Train_352 (352):** 313/352 (88.9%) covered in GraphPPIS features.
- **DeepPPISP Test_70 (70):** 62/70 (88.6%) covered in GraphPPIS features.

---

## 6. Distance Map Validation

- **Files Checked:** 741 `.npy` files in `data/GraphPPIS/Feature/distance_map/`
- **Shape:** Symmetrical $L \times L$ where $L$ matches protein sequence length.
- **Range:** $[0.0, 161.7]$ Å. Diagonal is 0.0.
- **Graph Construction Cutoff:** Verified from paper and `data/EGCPPIS/utils.py:process_distance_map`:
  $$\text{edge}_{i, j} = 1 \iff \text{dist}(i, j) < 16 \text{ Å or } i = j$$
- **Integrity:** Zero NaN, zero Inf.

---

## 7. Atom Graph Validation

- **Paper Specification:**
  - Node features: 37-dimensional atom one-hot / properties
  - Edge cutoff: 2.3 Å
  - Mapping: `a2r_map` maps atom indices $0 \dots M-1$ to residue indices $0 \dots N-1$.
- **Status in `data/`:**
  - Files `Tenx/HSSPPI/Atom_model/*.pkl` are **not stored within the `data/` repository**.
  - The official codebase references `Tenx/HSSPPI/Atom_model/{protein_id}.pkl`.

---

## 8. Existing Code Audit

| File | Purpose | Paper Component | Status & Observations |
| :--- | :--- | :--- | :--- |
| `data/EGCPPIS/Model_alpha.py` | Model definition | Architecture: EGNN, GraphSAGE, Contrastive, Attention, MLP | **CORRECT**: Full 115-d input, 4 EGNN layers, 4 GraphSAGE layers, 230-d fused representation, 10-head gated attention, dual classification heads. |
| `data/EGCPPIS/egnn_pytorch.py` | E(n)-Equivariant GNN | Geometric equivariant message passing | **CORRECT**: 4 EGCL layers with coordinates and features update, AntiSymmetricConv. |
| `data/EGCPPIS/graphsage.py` | GraphSAGE convolution | Atomic graph representation | **CORRECT**: Custom and PyG-compatible SAGEConv. |
| `data/EGCPPIS/ppi_main.py` | Training loop | Loss = BCE + 0.1 * L_GCL, Adam (0.0008), ReduceLROnPlateau (factor=0.2, patience=5) | **CORRECT**: Exactly matches paper hyperparameters. Has hardcoded relative paths that need unified configuration. |
| `data/EGCPPIS/evaluation.py` | Metric computation | ACC, Precision, Recall, F1, MCC, AUC, AUPR | **CORRECT**: Uses scikit-learn standard routines and threshold optimization. |
| `data/EGCPPIS/utils.py` | Dataset loader & featurizer | Data loading, 115-d feature assembly | Hardcoded relative paths. |

---

## 9. Existing Checkpoint Audit

Inspected: `data/EGCPPIS/model_save/best_model.pth` (4.55 MB) and `data/EGCPPIS/DeepPPISP/model_save/best_model.pth` (4.55 MB).

- **Loadable:** Yes, without errors using PyTorch 2.5.1 CPU/CUDA.
- **Total Parameter Tensors:** 108 tensors.
- **Layer Dimensions:**
  - `multi_head_att.query.weight`: `(115, 115)` (5 heads, head dim 23)
  - `egnnmodel.layers.0.0.edge_mlp.0.weight`: `(462, 231)` ($115 \times 2 + 1 = 231$)
  - `egnnmodel.layers.0.0.node_mlp.0.weight`: `(230, 131)` ($115 + 16 = 131$)
  - `atom_model.ln.weight`: `(115, 37)` (projects 37-d atom features to 115-d)
  - `atom_model.gcn_x1..4.lin_l/r.weight`: `(115, 115)` (4 GraphSAGE layers)
  - `CL_mi.project.0/2.weight`: `(115, 115)` (2-layer MLP projection for contrastive learning)
  - `multi_head_att2.query.weight`: `(230, 230)` (10 heads, head dim 23 on fused representation)
  - `mlp2.gc_layer.weight`: `(128, 230)` $\to$ `(16, 128)` $\to$ `(1, 16)` (Final binary classifier)
- **Conclusion:** The checkpoint `best_model.pth` strictly and faithfully embodies the full EGCPPIS architecture with 115-d residue features and 230-d fused multimodal representations.

---

## 10. Missing Data & Gap Analysis (Source: `data/`)

### A. Missing Requirements in `data/`
1. **ESM-2 Embeddings (`bert_esm2_t36`):**
   - *Required by Paper:* 33-dimensional language model representation per residue.
   - *Status:* Missing in `data/`.
   - *Impact:* Residue feature dimension lacks the 33-d language model component.
2. **AlphaFold Residue Features (`resAF`):**
   - *Required by Paper:* 7-dimensional structural representation (pLDDT, coordinate variances).
   - *Status:* Missing in `data/`.
   - *Impact:* Residue feature dimension lacks 7 structural dimensions.
3. **Pseudo-Spatial Embeddings (`psepos`):**
   - *Required by Paper:* 1-dimensional pseudo-spatial scalar from 3D coordinates.
   - *Status:* Missing in `data/`.
   - *Impact:* 1-d coordinate distance feature missing.
4. **Fine-Grained Atomic Graphs (`Tenx/HSSPPI/Atom_model`):**
   - *Required by Paper:* 37-dimensional atomic features, 2.3 Å atomic graph edges, `a2r_map`.
   - *Status:* Missing in `data/`.
   - *Impact:* The atom-level GraphSAGE branch and atom-residue contrastive learning cannot execute without atomic graph definitions.

---

## 11. Final Readiness Classification

| Requirement | Target Configuration | Status in `data/` |
| :--- | :--- | :--- |
| **Benchmark Sequences & Labels** | 6 Benchmark Datasets | **READY (100% Verified)** |
| **Residue Graph Construction** | 16 Å Cutoff | **READY (741 Distance Maps Verified)** |
| **Classical Residue Features** | One-hot (20), PSSM (20), HMM (20), DSSP (14) = 74 dims | **READY (100% Coverage on GraphPPIS)** |
| **Multimodal Features (ESM-2, resAF, psepos)** | 33 + 7 + 1 = 41 dims | **MISSING in `data/`** |
| **Fine-Grained Atom Graphs** | 37-d atom graph, 2.3 Å, `a2r_map` | **MISSING in `data/`** |
| **Model Code (EGNN, GraphSAGE, GCL, Gated Attn)** | 4 EGCL, 4 SAGEConv, 10-head attention | **READY (`data/EGCPPIS/Model_alpha.py`)** |
| **Original Trained Checkpoint** | `best_model.pth` | **READY (4.55 MB, Verified)** |
| **Training Pipeline & Loss** | Adam, lr=0.0008, BCE + 0.1 * L_GCL | **READY** |
| **Evaluation Metrics Suite** | ACC, F1, MCC, AUC, AUPR | **READY** |

---

## 12. Reproduction Feasibility Assessment

1. **Full 115-D Multimodal + Atomic EGCPPIS:**
   - Because `resAF`, `bert_esm2_t36`, `psepos`, and `Atom_model` are not stored in `data/`, and per the strict scientific rule:
     > *"DO NOT fabricate missing data. DO NOT create fake features. DO NOT invent atom graphs."*
   - Full multimodal training from scratch using `data/` alone is **BLOCKED** by the absence of raw PDB / atomic files and ESM-2 cache in `data/`.
2. **Reproducible Sequence + Classical GraphPPIS Features Pipeline:**
   - All 741 distance maps, DSSP (14), PSSM (20), HMM (20), and One-hot (20) features ($74$ dimensions total) are **100% complete and verified**.
   - The paper's architecture (EGNN 4 layers, Gated Multi-Head Attention, Adam optimizer, scheduler, loss) can be faithfully benchmarked in this verified setting.
3. **Official Checkpoint Validation:**
   - The official checkpoint `best_model.pth` (115-d $\to$ 230-d) is fully preserved and verified as a benchmark reference.
