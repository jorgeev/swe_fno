# FNO for the Shallow-Water Equations (1D & 2D)

Train a Fourier Neural Operator (FNO) on shallow-water equation data with
**variable bottom topography**, **f-plane Coriolis** (in 2D) and **periodic
boundary conditions**. Data is generated on the fly with a pseudo-spectral
RK4 solver.

## Equations

**1D**
```
h_t + (h u)_x = 0
u_t + u u_x + g (h + b)_x = 0
```

**2D (f-plane)**
```
h_t + (h u)_x + (h v)_y = 0
u_t + u u_x + v u_y - f v + g (h + b)_x = 0
v_t + u v_x + v v_y + f u + g (h + b)_y = 0
```

`h` is the water depth, `b(x[,y])` is bottom elevation, `f` is the constant
Coriolis parameter (2D only, sampled per trajectory).

## File layout

| file              | role                                                      |
|-------------------|-----------------------------------------------------------|
| `config.py`       | dataclasses with all hyperparameters                      |
| `fno.py`          | `SpectralConv1d/2d`, `FNO1d`, `FNO2d`                     |
| `solver_1d.py`    | pseudo-spectral 1D SWE solver + dataset generator         |
| `solver_2d.py`    | pseudo-spectral 2D SWE solver + dataset generator         |
| `dataset.py`      | `SWEDataset1D`, `SWEDataset2D` (PyTorch `Dataset`s)       |
| `utils.py`        | relative L² loss, channel-wise normalizer                 |
| `train.py`        | dimension-agnostic train / eval loop                      |
| `main_1d.py`      | entry point: generate data → train → evaluate (1D)        |
| `main_2d.py`      | entry point: generate data → train → evaluate (2D)        |

## Operator being learned

* **1D** : `(h₀, u₀, b) ↦ (h(T), u(T))`  — shape `[3, nx] → [2, nx]`
* **2D** : `(h₀, u₀, v₀, b, f) ↦ (h(T), u(T), v(T))` — shape `[5, ny, nx] → [3, ny, nx]`

The Coriolis parameter `f` is constant per sample; it is broadcast to a
constant field so the model can use it as a parameter channel.

## Usage

```bash
pip install torch numpy
python main_1d.py     # 1D experiment
python main_2d.py     # 2D experiment
```

Hyperparameters live in `config.py`. Reduce `n_train`, `epochs`, `nx`, or
`width` for a quick sanity run; defaults are sized so a single GPU finishes
the 1D job in a couple of minutes and the 2D job in roughly 10–30 minutes
depending on hardware.

## Notes

* Periodic BCs are enforced naturally by the pseudo-spectral solver and by
  the FFTs inside the FNO.
* The solver applies 2/3-rule dealiasing to the nonlinear products and a
  small `(k²)²` hyperviscosity for stability. `dt` is set from a CFL
  condition based on `√(g·H₀) + |u|`.
* Trajectories where `h` becomes non-positive or non-finite are rejected
  and re-drawn, so amplitudes can be tuned without crashing the run.
* The trained checkpoint stores both the model weights and the input/output
  normalization statistics, so inference is self-contained.
