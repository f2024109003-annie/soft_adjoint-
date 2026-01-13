# Soft Adjoint Operators for the Lorenz–96 Model

This repository provides a **minimal, reproducible Python/JAX implementation**
of the *soft adjoint framework* proposed in the accompanying research paper.

The code demonstrates how ensemble-based uncertainty can be aggregated
into a single soft adjoint gradient computation for a chaotic dynamical system.

---

##  Purpose of this Repository

- Verify numerical claims made in the paper
- Provide transparent and reproducible code
- Demonstrate the soft adjoint concept on the Lorenz–96 system

This repository focuses **only on the Lorenz–96 benchmark**.
Quadrotor experiments are provided in a separate repository.

---

##  Repository Contents

- `soft_adjoint_lorenz96.py`  
  Python/JAX implementation of:
  - Lorenz–96 dynamics
  - Ensemble (soft set) forcing
  - Soft adjoint gradient computation

---

##  Model Description

The Lorenz–96 system is a standard low-order atmospheric proxy model used to
test scalability and robustness of data assimilation and sensitivity methods
in chaotic regimes.

Soft adjoints aggregate gradients over an ensemble of forcings,
avoiding multiple independent adjoint solves.

---

##  Relation to Paper Results

This code supports:
- RMSE values reported for the Lorenz–96 experiment
- Linear ensemble scaling behavior
- Timing comparisons with classical and ensemble adjoint baselines

Exact numerical values may vary slightly depending on hardware and random
initialization.

---

##  Data and Forcing

The ensemble forcing used here is **synthetic** and designed to emulate
ECMWF ERA5 ensemble variability.

Raw ERA5 NetCDF files are not included due to size and licensing constraints.
Users may download ERA5 data directly from:

https://cds.climate.copernicus.eu

---

##  How to Run

1. Clone the repository:
```bash
git clone https://github.com/f2024109003-annie/soft-adjoint-lorenz96.git
cd soft-adjoint-lorenz96

2. Install Dependencies:
pip install jax jaxlib numpy matplotlib

3. Run the experiments:
python soft_adjoint_lorenz96.py

---

## Reproducibility

The numerical results reported in the paper are generated using this codebase.
In particular, the Lorenz–96 experiments can be reproduced by running the
provided script with default parameters.

The implementation uses fixed model settings and ensemble configurations
to ensure consistent behavior across runs. The reported RMSE and timing
results correspond to averaged outcomes over multiple realizations, as
described in the paper.

## Results Mapping (Paper ↔ Code)

The following components of the paper are supported by this repository:

- **Lorenz–96 Experiments**  
  Implemented in `soft_adjoint_lorenz96.py`.  
  This script reproduces the numerical results reported in:
  - Ensemble scaling analysis
  - RMSE comparisons between classical and soft adjoint methods
  - Computational timing measurements

- **Soft Adjoint Gradient Evaluation**  
  The function `soft_adjoint_grad` implements the aggregated gradient
  corresponding to the soft Gramian operator defined in the paper.

Quadrotor experiments described in the paper are simulation-based and are
not included in this repository. These are discussed as conceptual and
numerical illustrations rather than fully reproducible benchmarks.

## Data Availability

The atmospheric uncertainty structure used in this work is motivated by the
ERA5 ensemble reanalysis dataset provided by the European Centre for
Medium-Range Weather Forecasts (ECMWF), which is publicly available via the
Copernicus Climate Data Store:
https://cds.climate.copernicus.eu/
Due to data volume and licensing constraints, raw ERA5 ensemble files are not
included in this repository. Instead, the numerical experiments use
programmatically generated ensemble forcings that are consistent with the
statistical properties described in the paper.
All numerical results reported in the paper can be reproduced using the code
provided in this repository.



