# CAMELS-SIMBA GNN Thesis

This repository contains the codebase for a thesis project applying Graph Neural Networks to CAMELS-SIMBA halo catalogs.

The goal is to convert CAMELS-SIMBA halo catalogs into graph-based datasets and train graph neural network models to predict cosmological parameters, starting with `Omega_m`.

---

## Project Goal

The project studies whether graph-based representations of dark matter halo catalogs can capture useful cosmological information.

Each universe is represented as a graph or a temporal sequence of graphs:

- **Universe** = one simulation sample
- **Snapshot** = one graph at one cosmic time
- **Nodes** = halos
- **Node features** = halo physical properties
- **Edges** = spatial k-nearest-neighbor connections
- **Target** = cosmological parameter `Omega_m`

---

## Current Validated Pipeline

The current preprocessing pipeline includes:

1. Raw CAMELS-SIMBA halo catalog inspection
2. Real `Omega_m` target extraction from file metadata
3. Halo selection using top halos by raw `Mvir`
4. Node feature construction using:

```text
[log10(Mvir), X, Y, Z, VX, VY, VZ]

