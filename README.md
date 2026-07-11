# Simulating Concepts of Physics in Python using Numerical Methods

A collection of Python scripts that numerically simulate classical, electromagnetic, and quantum systems — each paired with an animation and, wherever possible, a comparison against the analytic solution. Written for a project in the 6th semester Experiential Laboratory.

## Contents

### Classical Mechanics
| Script | Description |
|---|---|
| [`01-sho.py`](simulations/01-sho.py) | Simple harmonic oscillator, solved with `solve_ivp` and compared to the analytic solution |
| [`02-damped-sho.py`](simulations/02-damped-sho.py) | Damped harmonic oscillator across under-, critically-, and over-damped regimes |
| [`03-simple-pendulum.py`](simulations/03-simple-pendulum.py) | Simple pendulum at large angle, beyond the small-angle approximation |
| [`04-orbital-motion.py`](simulations/04-orbital-motion.py) | Two-body orbital motion under an inverse-square gravitational force |
| [`05-collision.py`](simulations/05-collision.py) | 1D elastic/inelastic collision with a configurable coefficient of restitution |

### Electromagnetism
| Script | Description |
|---|---|
| [`06-lap-field-solver.py`](simulations/06-lap-field-solver.py) | Electrostatic dipole field solved on a grid via Gauss-Seidel relaxation |
| [`07-lap-particle-tracker.py`](simulations/07-lap-particle-tracker.py) | Charged particle trajectory through the field produced by script 06 |
| [`08-em-fields.py`](simulations/08-em-fields.py) | Oscillating electric dipole field and the AC magnetic field around a current-carrying wire |
| [`09-lorentz.py`](simulations/09-lorentz.py) | Charged particle motion under the Lorentz force in crossed E and B fields |
| [`10-double-slit.py`](simulations/10-double-slit.py) | Double-slit interference pattern from wave superposition |

### Quantum Mechanics
| Script | Description |
|---|---|
| [`11-qho.py`](simulations/11-qho.py) | 1D quantum harmonic oscillator via the shooting method |
| [`12-qmtunneling.py`](simulations/12-qmtunneling.py) | Quantum tunnelling through potential wells and barriers via matrix diagonalization |

## Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Setup

Using `uv` (recommended):

```bash
git clone https://github.com/adi-pandya/numerical-simulation-physics.git
cd numerical-simulation-physics
uv sync
```

Using `pip`:

```bash
git clone https://github.com/adi-pandya/numerical-simulation-physics.git
cd numerical-simulation-physics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a simulation

```bash
uv run simulations/01-sho.py
# or, with pip/venv:
python simulations/01-sho.py
```

> **Note:** `07-lap-particle-tracker.py` reads a `dipole_field.npz` file that is meant to be exported by `06-lap-field-solver.py`. That export block is currently commented out in `06-lap-field-solver.py` — uncomment it (and the matching `np.savez_compressed(...)` call) and run script 06 once before running script 07.

## Repository structure

```
.
├── simulations/        # one script per concept, numbered in the order above
├── pyproject.toml       # project metadata + dependencies (for uv)
├── requirements.txt     # dependencies (for pip)
└── README.md
```

## License

Not yet specified. Add a `LICENSE` file (MIT is a common default for teaching/demo code like this) if you want to make reuse terms explicit.
