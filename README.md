# HMM-GNN-Parkinson-Human-Balance

Hidden Markov Model (HMM)-based human balance estimation with an improved Graph Neural Network (GNN) approach for Parkinson's disease classification.

## Overview

This repository contains the HMM-based human balance processing pipeline and the proposed HMM + GNN classification approach.

The work extends HMM-based human balance representation with a graph-based deep learning approach for Parkinson's disease classification.

## Repository Structure

```text
HMM-GNN-Parkinson-Human-Balance/
│
├── ExampleArticle/
│   └── C source code and data for the practical example
│
├── Oliveira/
│   └── C source code and data related to the Oliveira dataset
│
├── Santos_and_Duarte/
│   └── C source code and data related to the Santos & Duarte dataset
│
├── Oliveira + Santos_and_Duarte/
│   └── Combined C source code and data processing for both datasets
│
├── Improved_HMM_GNN/
│   ├── GNN baseline implementation
│   ├── Improved GNN V2
│   ├── Multi-seed experiments
│   ├── Label permutation test
│   ├── Result comparison
│   └── demo_run.py
│
├── final_comparison.csv
│   └── Comparison of the evaluated approaches
│
└── README.md
```

## Datasets

The project uses publicly available human balance datasets:

- **Oliveira et al. (2022)** — Ground reaction force data from individuals with Parkinson's disease.
- **Santos & Duarte (2016)** — Public dataset of human balance evaluations.

The HMM pipeline processes the balance measurements and generates HMM-based features used by the classification experiments.

## Quick Demo

A short demonstration of the proposed HMM + GNN pipeline is available in:

```text
Improved_HMM_GNN/demo_run.py
```

Run from the `Improved_HMM_GNN` directory:

```bash
python demo_run.py
```

The demo uses a shortened configuration:

```text
3-fold cross-validation
20 epochs
```

This is intended for quickly demonstrating that the complete pipeline runs successfully.

The full research experiments use a more extensive evaluation setup.

## Research Documentation & Results

Detailed information about:

- Research methodology
- HMM feature generation
- Graph construction
- Improved GNN architecture
- Experimental setup
- Results and comparisons
- Robustness experiments
- Research figures
- Paper documentation

is available on the project website.

**Project Website:**  
`[ADD WEBSITE LINK HERE]`

## References

### Original HMM Research

Denkeng, A. T., Mourad, A. M., Iloga, S., Mba, R. M., Baazaoui, H., Ndié, T. D., & Romain, O. (2025).

*Efficient Characterization Of The Human Balance Using HMMS.*

IEEE Access, 13, 183456–183479.

DOI: 10.1109/ACCESS.2025.3622375

### Oliveira Dataset

Oliveira, C. E. N., Souza, C., Treza, R. d. C., Hondo, S. M., Los Angeles, E., Bernardo, C., Shida, T. K. F., Oliveira, L., Novaes, T. M., Campos, D. d. S. F., et al. (2022).

*A public data set with ground reaction forces of human balance in individuals with Parkinson's disease.*

Frontiers in Neuroscience, 16, 865882.

### Santos & Duarte Dataset

Santos, D. A. & Duarte, M. (2016).

*A public data set of human balance evaluations.*

PeerJ, 4, 2648.

## Purpose

This repository is maintained for academic and research purposes and contains the implementation and supporting material for the HMM-based human balance and HMM + GNN research work.
