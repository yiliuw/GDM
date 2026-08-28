# Gumbel Dynamical Model (GDM)

[![arXiv](https://img.shields.io/badge/arXiv-2509.21578-b31b1b.svg)](https://arxiv.org/abs/2509.21578)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyTorch implementation and experiments for the paper **[Interpretable time series analysis with Gumbel dynamics](https://arxiv.org/abs/2509.21578)**.

GDM is a switching dynamical system defined over a *relaxed-discrete* state space. Instead of forcing each time step into exactly one hard discrete state which produces spurious rapid switching on noisy real-world data, GDM places Gumbel-driven soft states at the core of the generative model. States link directly to observations, with no additional Gaussian latent trajectory blurring the dynamics.

**Highlights**

- **Soft, sticky, overlapping transitions.** Models smooth, variable-speed, and stochastic regime changes that hard-switching models (AR-HMM, SLDS, rSLDS) miss.
- **Fully differentiable.** The Gumbel-Softmax relaxation makes the whole model trainable end-to-end with standard gradient descent (GS-BBVI) — no variational EM or per-sequence E-steps.
- **Amortized inference.** A reusable inference network maps observations to state logits, so new sequences are processed in a single forward pass without re-optimization.
- **Interpretable primitives.** Learned dynamics remain a small set of analyzable linear (or RNN-parameterized) motifs; inferred states align with expert labels substantially better than benchmarks on real data.

## Repository structure

```
├── code/                        # Core library
│   ├── bbvi_infer_2l.py         #   GDM + GS-BBVI, single-trial version
│   ├── bbvi_infer_2lB.py        #   Batched version (multiple trials)
│   └── bbvi_infer_RNNB.py       #   Batched version with RNN transitions / BiGRU posterior
├── experiments/
│   ├── NASCAR/                  # Synthetic benchmark (standard + soft-sticky variants)
│   │   ├── F2 - Standard.ipynb / F2 - SoftSticky.ipynb     → paper Figure 2
│   │   ├── T1 - Standard.ipynb / T1 - SoftSticky.ipynb     → paper Table 1
│   │   └── nascarsoft.py        #   Data generator for both NASCAR variants
│   ├── Formula1/                # F1 telemetry (position- and velocity-based)
│   │   ├── F3 - Overview / Acc2022 / Acc2024 / InferredTrack.ipynb  → paper Figure 3
│   │   └── f1.py                #   Telemetry retrieval and plotting via FastF1
│   └── CalMS/                   # CalMS21 mouse social behavior
│       ├── F4 - GDM.ipynb / F4 - GDMrnn.ipynb / F4 - rSLDS.ipynb    → paper Figure 4
│       └── calms.py             #   Data loading and preprocessing
├── environment/                 # Docker environment (CodeOcean capsule)
└── LICENSE                      # MIT
```

The three files in `code/` are self-contained: each defines the generative model, the variational posterior, the GS-BBVI training loop, and plotting/evaluation utilities. Experiment folders carry a local copy of the version they use.

## Installation

Core requirements:

```bash
pip install torch numpy pandas matplotlib scipy scikit-learn tqdm
```

Experiment-specific extras:

```bash
pip install fastf1                                 # Formula 1 telemetry retrieval
pip install git+https://github.com/lindermanlab/ssm   # rSLDS/SLDS baselines only
```

A GPU is recommended but not required. To reproduce the exact environment, `environment/Dockerfile` builds the CodeOcean capsule image (CUDA 11.8, `numpy 1.26.4`, `pandas 2.1.4`, `matplotlib 3.10.0`).


## Reproducing the paper

| Paper result | Notebook(s) | Data |
|---|---|---|
| Figure 2, Table 1 (NASCAR, standard + soft-sticky) | `experiments/NASCAR/F2 - *.ipynb`, `T1 - *.ipynb` | Generated locally by `nascarsoft.py` |
| Figure 3 (Formula 1, position- and velocity-based) | `experiments/Formula1/F3 - *.ipynb` | Fetched at runtime via [FastF1](https://github.com/theOehrly/Fast-F1) |
| Figure 4 (CalMS21 mouse social behavior) | `experiments/CalMS/F4 - *.ipynb` | Download [CalMS21 Task 1](https://data.caltech.edu/records/s0vdx-0k302) |


